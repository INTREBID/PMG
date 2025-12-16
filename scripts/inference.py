#!/usr/bin/env python3
"""
PMG Unified Inference Script

Usage:
    python scripts/inference.py --config configs/flickr_train.yaml --checkpoint outputs/flickr/final_model
    python scripts/inference.py --config configs/pog_train.yaml --checkpoint outputs/pog/checkpoint-5000 --output_dir results/pog
"""
import os
import sys
import json
import yaml
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from accelerate import Accelerator
from transformers import LlamaForCausalLM, LlamaTokenizer

from pmg.models.custom_pipeline import SDPipeline
from pmg.models.soft_prompt import InferenceModel
from pmg.data.dataset import PMGInferenceDataset


def load_config(config_path: str):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="PMG Inference Script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save generated images (default: dataset eval_outputs)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="Number of images to generate per sample (default: from config)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    print(f"Dataset: {config['dataset']['name']}")
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = config['dataset']['data_dir'] + "/eval_outputs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Determine weight dtype
    weight_dtype = torch.bfloat16
    
    # Setup accelerator
    accelerator = Accelerator()
    
    # Load LLaMA model
    print(f"Loading LLaMA from {config['model']['llama']}")
    llama_tokenizer = LlamaTokenizer.from_pretrained(config['model']['llama'])
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    
    llama_model = LlamaForCausalLM.from_pretrained(
        config['model']['llama'],
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    llama_model.requires_grad_(False)
    llama_model.eval()
    
    # Load Stable Diffusion pipeline
    print(f"Loading Stable Diffusion from {config['model']['stable_diffusion']}")
    sd_pipeline = SDPipeline(
        weight_dtype=weight_dtype,
        model_id=config['model']['stable_diffusion'],
        height=config['pmg']['image_size'],
        width=config['pmg']['image_size']
    )
    sd_pipeline.eval()
    
    # Create PMG model
    print("Creating PMG model...")
    pmg_model = InferenceModel(
        layer_num=len(llama_model.model.layers),
        num_image_prompt=config['pmg']['num_image_prompt'],
        num_prefix_prompt=config['pmg']['num_prefix_prompt'],
        emb_dim=4096,
        sd_hidden_state_dim=768
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint_path = os.path.join(args.checkpoint, "pytorch_model.bin")
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=accelerator.device)
        pmg_model.load_state_dict(state_dict)
    else:
        # Try loading with accelerator
        accelerator.load_state(args.checkpoint)
    
    pmg_model.to(accelerator.device)
    pmg_model.eval()
    
    # Load test data
    print(f"Loading test data from {config['dataset']['test_json']}")
    with open(config['dataset']['test_json'], 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples")
    
    # Create dataset
    test_dataset = PMGInferenceDataset(
        data=test_data,
        tokenizer=llama_tokenizer,
        max_len=config['pmg']['max_sequence_length'],
        dataset_name=config['dataset']['name']
    )
    
    # Inference settings
    num_inference_steps = config['inference']['num_inference_steps']
    guidance_scale = config['inference']['guidance_scale']
    negative_prompt = config['inference']['negative_prompt']
    num_images_per_sample = args.num_images or config['inference']['num_images_per_sample']
    
    print(f"\nInference settings:")
    print(f"  Steps: {num_inference_steps}")
    print(f"  Guidance scale: {guidance_scale}")
    print(f"  Images per sample: {num_images_per_sample}")
    print()
    
    # Generate images
    print("Generating images...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Generating"):
            sample = test_dataset[idx]
            
            # Create output directory for this sample
            sample_dir = os.path.join(output_dir, f"sample_{idx:04d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Get PMG embeddings
            input_ids = sample['input_ids'].unsqueeze(0).to(accelerator.device)
            token_len = torch.tensor([sample['token_len']]).to(accelerator.device)
            
            encoder_hidden_states = pmg_model(
                llama_tokenizer,
                llama_model,
                input_ids,
                token_len
            )
            
            # Generate multiple images
            for img_idx in range(num_images_per_sample):
                generator = torch.Generator(device=accelerator.device).manual_seed(42 + img_idx)
                
                # Generate image
                images = sd_pipeline.generate(
                    prompt_embeds=encoder_hidden_states,
                    negative_prompt=negative_prompt,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    show_processbar=False
                )
                
                # Save image
                image = Image.fromarray(images[0])
                image_path = os.path.join(sample_dir, f"gen_{img_idx}.jpg")
                image.save(image_path)
            
            # Save sample info
            info = {
                'sample_idx': sample['sample_idx'],
                'user_id': sample['user_id'],
                'target_item_id': sample['target_item_id'],
                'target_caption': sample['target_caption'],
                'num_generated': num_images_per_sample
            }
            
            info_path = os.path.join(sample_dir, "info.json")
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Generation complete! Images saved to {output_dir}")


if __name__ == "__main__":
    main()

