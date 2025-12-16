"""
Soft Prompt and Prefix Encoder for PMG (PyTorch Implementation)
"""
import torch
import torch.nn as nn


class PrefixEncoder(nn.Module):
    """
    Prefix Encoder that generates past_key_values for LLaMA
    
    Args:
        num_hidden_layers: Number of transformer layers
        hidden_size: Hidden size of the model
        pre_seq_len: Length of prefix sequence
        prefix_projection: Whether to use projection
        prefix_hidden_size: Hidden size for projection
    """
    def __init__(self, num_hidden_layers, hidden_size, pre_seq_len, 
                 prefix_projection=False, prefix_hidden_size=4096):
        super().__init__()
        self.prefix_projection = prefix_projection
        
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = nn.Embedding(pre_seq_len, hidden_size)
            self.trans = nn.Sequential(
                nn.Linear(hidden_size, prefix_hidden_size),
                nn.Tanh(),
                nn.Linear(prefix_hidden_size, num_hidden_layers * 2 * hidden_size)
            )
        else:
            self.embedding = nn.Embedding(pre_seq_len, num_hidden_layers * 2 * hidden_size)

    def forward(self, prefix):
        """
        Forward pass
        
        Args:
            prefix: Prefix token indices
            
        Returns:
            past_key_values: Encoded prefix for transformer
        """
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class InferenceModel(nn.Module):
    """
    PMG Inference Model with trainable prompts and prefix encoder
    
    Args:
        layer_num: Number of transformer layers
        num_image_prompt: Number of image prompt tokens
        num_prefix_prompt: Number of prefix prompt tokens
        emb_dim: Embedding dimension (default: 4096 for LLaMA)
        sd_hidden_state_dim: Stable Diffusion hidden state dimension
    """
    def __init__(self, layer_num, num_image_prompt, num_prefix_prompt, 
                 emb_dim=4096, sd_hidden_state_dim=768):
        super().__init__()
        self.layer_num = layer_num
        self.num_image_prompt = num_image_prompt
        self.num_prefix_prompt = num_prefix_prompt
        self.emb_dim = emb_dim
        
        # Mapping layer from LLaMA embedding space to SD embedding space
        self.mapping_layer = nn.Linear(emb_dim, sd_hidden_state_dim)
        
        # Trainable image prompt embeddings
        self.trainable_prompt = nn.Parameter(
            torch.randn(1, num_image_prompt, emb_dim), 
            requires_grad=True
        )
        
        # Prefix tokens and encoder
        self.register_buffer('prefix_tokens', torch.arange(num_prefix_prompt).long())
        self.prefix_encoder = PrefixEncoder(layer_num, emb_dim, num_prefix_prompt)
    
    def forward(self, llama_tokenizer, llama_model, token, token_len):
        """
        Forward pass through the inference model
        
        Args:
            llama_tokenizer: LLaMA tokenizer
            llama_model: LLaMA model
            token: Input token IDs
            token_len: Length of valid tokens for each sample
            
        Returns:
            encoder_hidden_states: Hidden states for SD conditioning
        """
        bsz = token.shape[0]
        attention_mask = (token != llama_tokenizer.pad_token_id)
        
        # Get embeddings and insert trainable prompts
        emb = llama_model.model.embed_tokens(token)
        for i in range(bsz):
            l = token_len[i].item()
            emb[i, l:l+self.num_image_prompt] = self.trainable_prompt
            attention_mask[i, l:l+self.num_image_prompt] = 1
        
        # Add prefix attention mask
        attention_mask = torch.cat([
            torch.ones((bsz, self.num_prefix_prompt), device=attention_mask.device, dtype=attention_mask.dtype), 
            attention_mask
        ], dim=1)
        
        # Generate past_key_values from prefix encoder
        num_head = llama_model.model.layers[0].self_attn.num_heads
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(bsz, -1).to(token.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        
        # Reshape past_key_values
        # [bsz, num_prefix_prompt, layer_num * 2 * hidden_size]
        past_key_values = past_key_values.view(
            bsz, self.num_prefix_prompt, self.layer_num, 2, num_head, -1
        )
        # [layer_num, 2, bsz, num_head, num_prefix_prompt, head_dim]
        past_key_values = past_key_values.permute(2, 3, 0, 4, 1, 5)
        
        # Convert to tuple format for HuggingFace
        past_key_values = tuple([
            (past_key_values[i, 0], past_key_values[i, 1]) 
            for i in range(self.layer_num)
        ])
        
        # Forward through LLaMA
        outputs = llama_model.model.forward(
            inputs_embeds=emb,
            output_hidden_states=True,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        
        # Extract image prompt hidden states
        encoder_hidden_states = []
        for i in range(bsz):
            l = token_len[i].item()
            encoder_hidden_states.append(
                outputs.last_hidden_state[i, l:l+self.num_image_prompt]
            )
        encoder_hidden_states = torch.stack(encoder_hidden_states)
        
        # Map to SD embedding space
        encoder_hidden_states = self.mapping_layer(encoder_hidden_states)
        
        return encoder_hidden_states

