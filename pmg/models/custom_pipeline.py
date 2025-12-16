"""
Custom Stable Diffusion Pipeline for PMG
"""
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from safetensors.torch import load_file
from collections import defaultdict


def load_lora_weights(pipeline, checkpoint_path, multiplier, device, dtype):
    """Load LoRA weights into a Stable Diffusion pipeline"""
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device=device)

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    # directly update weight in diffusers model
    for layer, elems in updates.items():
        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        # get elements for this layer
        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        alpha = elems['alpha']
        if alpha:
            alpha = alpha.item() / weight_up.shape[1]
        else:
            alpha = 1.0

        # update weight
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(
                weight_up.squeeze(3).squeeze(2), 
                weight_down.squeeze(3).squeeze(2)
            ).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

    return pipeline


class SDPipeline(torch.nn.Module):
    """
    Custom Stable Diffusion Pipeline for PMG
    
    Args:
        weight_dtype: Data type for model weights (e.g., torch.bfloat16)
        model_id: HuggingFace model ID or local path
        height: Output image height
        width: Output image width
    """
    def __init__(self, weight_dtype, model_id="runwayml/stable-diffusion-v1-5", height=512, width=512):
        super().__init__()
        self.height = height
        self.width = width
        self.weight_dtype = weight_dtype
        
        # Load components
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder"
        ).to(weight_dtype)
        
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(weight_dtype)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(weight_dtype)
        
        # Freeze base model
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
    
    def forward(self, embs, pixel_values):
        """
        Training forward pass
        
        Args:
            embs: Text embeddings from LLaMA encoder
            pixel_values: Target images
            
        Returns:
            loss: Diffusion loss
        """
        embs = embs.to(self.weight_dtype)
        latents = self.vae.encode(pixel_values.to(self.weight_dtype)).latent_dist.sample().detach()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents, dtype=self.weight_dtype)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        )
        timesteps = timesteps.long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps).to(self.weight_dtype)
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=embs).sample

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss
    
    def textEncode(self, text=None, tokens=None, num_tokens=None, return_tokens=False, **kw):
        """
        Encode text to embeddings
        
        Args:
            text: Input text
            tokens: Pre-tokenized input
            num_tokens: Max length for tokenization
            return_tokens: Whether to return tokens instead of embeddings
            
        Returns:
            Text embeddings or tokens
        """
        if tokens is None:
            if num_tokens is None:
                tokens = self.tokenizer(
                    text,
                    padding="longest",
                    return_tensors="pt",
                    **kw
                ).input_ids
            else:
                tokens = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=num_tokens,
                    return_tensors="pt",
                    **kw
                ).input_ids
        
        if return_tokens:
            return tokens
        
        tokens = tokens.to(self.text_encoder.device)
        hidden_states = self.text_encoder(tokens)
        return hidden_states.last_hidden_state
    
    def generate(self, prompt_embeds, nega_embs=None, negative_prompt="", 
                 generator=torch.manual_seed(0), show_processbar=False, 
                 num_inference_steps=50, guidance_scale=6):
        """
        Generate images from prompt embeddings
        
        Args:
            prompt_embeds: Text embeddings
            nega_embs: Negative prompt embeddings (optional)
            negative_prompt: Negative prompt text
            generator: Random generator
            show_processbar: Whether to show progress bar
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            Generated images as numpy arrays (uint8)
        """
        embs = prompt_embeds.to(self.weight_dtype)
        batch_size = embs.shape[0]
        
        # Prepare negative embeddings
        if nega_embs is None:
            nega_tokens = self.tokenizer(
                [negative_prompt] * batch_size,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
                max_length=embs.shape[1]
            ).input_ids.to(embs.device)
            nega_embs = self.text_encoder(nega_tokens)[0]
        text_embeddings = torch.cat([nega_embs, embs])
        
        # Prepare latents
        if type(generator) == list:
            latents = [torch.randn(
                (self.unet.in_channels, self.height // 8, self.width // 8),
                generator=g,
                dtype=self.weight_dtype,
            ).to(embs.device) for g in generator]
            latents = torch.stack(latents)
        else:
            latents = torch.randn(
                (batch_size, self.unet.in_channels, self.height // 8, self.width // 8),
                generator=generator,
                dtype=self.weight_dtype,
            ).to(embs.device)
        
        self.noise_scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.noise_scheduler.init_noise_sigma

        # Denoising loop
        process_bar = tqdm(self.noise_scheduler.timesteps) if show_processbar else self.noise_scheduler.timesteps
            
        for t in process_bar:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, timestep=t)

            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample

            noise_pred_nega, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_nega + guidance_scale * (noise_pred_text - noise_pred_nega)
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            images = self.vae.decode(latents).sample
            
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach()

        # Normalize to [0, 1]
        if images.min() < 0.0 or images.max() > 1.0:
            images = (images.clamp(-1, 1) + 1) / 2

        # Convert to float32 (numpy doesn't support bfloat16)
        images = images.to(torch.float32)

        # Convert to NHWC and numpy
        images = images.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        images = (images * 255).round().astype("uint8")
        
        return images
    
    def __call__(self, *args, **kwargs):
        """Allow calling the pipeline directly"""
        return self.generate(*args, **kwargs)

