#!/usr/bin/env python3
"""
PMG Unified Training Script

Usage:
    python scripts/train.py --config configs/flickr_train.yaml
    python scripts/train.py --config configs/pog_train.yaml --resume_from outputs/pog/checkpoint-1000
"""
import os
import sys
import json
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from transformers import LlamaForCausalLM, LlamaTokenizer

from pmg.models.custom_pipeline import SDPipeline
from pmg.models.soft_prompt import InferenceModel
from pmg.data.dataset import PMGDataset

logger = get_logger(__name__)


def load_config(config_path: str):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="PMG Training Script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from checkpoint directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    print(f"Dataset: {config['dataset']['name']}")
    
    # Override resume_from if provided
    if args.resume_from:
        config['training']['resume_from'] = args.resume_from
    
    # Set seed
    set_seed(config['training'].get('seed', 42))
    
    # Enable TF32 for faster training
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Determine weight dtype
    weight_dtype = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }[config['training'].get('mixed_precision', 'bf16')]
    
    # Setup accelerator
    output_dir = config['training']['output_dir']
    logging_dir = os.path.join(output_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)
    
    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir,
        logging_dir=logging_dir
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        mixed_precision=config['training']['mixed_precision'],
        log_with=config['training'].get('report_to', 'tensorboard'),
        project_config=accelerator_project_config,
    )
    
    logger.info(f"Accelerator device: {accelerator.device}")
    
    # Load LLaMA model
    logger.info(f"Loading LLaMA from {config['model']['llama']}")
    llama_tokenizer = LlamaTokenizer.from_pretrained(config['model']['llama'])
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    
    llama_model = LlamaForCausalLM.from_pretrained(
        config['model']['llama'],
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    llama_model.requires_grad_(False)
    llama_model = accelerator.prepare(llama_model)
    
    # Load Stable Diffusion pipeline
    logger.info(f"Loading Stable Diffusion from {config['model']['stable_diffusion']}")
    sd_pipeline = SDPipeline(
        weight_dtype=weight_dtype,
        model_id=config['model']['stable_diffusion'],
        height=config['pmg']['image_size'],
        width=config['pmg']['image_size']
    )
    torch.cuda.empty_cache()
    sd_pipeline = accelerator.prepare(sd_pipeline)
    
    # Load training and validation data
    logger.info("Loading training data...")
    with open(config['dataset']['train_json'], 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    logger.info(f"Loaded {len(train_data)} training samples")
    
    logger.info("Loading validation data...")
    with open(config['dataset']['val_json'], 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    logger.info(f"Loaded {len(val_data)} validation samples")
    
    # Create datasets
    train_dataset = PMGDataset(
        data=train_data,
        tokenizer=llama_tokenizer,
        sd_pipeline=sd_pipeline,
        dataset_name=config['dataset']['name'],
        max_len=config['pmg']['max_sequence_length'],
        num_image_prompt=config['pmg']['num_image_prompt'],
        image_size=config['pmg']['image_size'],
        mode='train',
        repeats=1
    )
    
    val_dataset = PMGDataset(
        data=val_data,
        tokenizer=llama_tokenizer,
        sd_pipeline=sd_pipeline,
        dataset_name=config['dataset']['name'],
        max_len=config['pmg']['max_sequence_length'],
        num_image_prompt=config['pmg']['num_image_prompt'],
        image_size=config['pmg']['image_size'],
        mode='val',
        repeats=1
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['train_batch_size'],
        shuffle=True,
        num_workers=config['training']['dataloader_num_workers']
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['training']['val_batch_size'],
        shuffle=False,
        num_workers=config['training']['dataloader_num_workers']
    )
    
    train_dataloader = accelerator.prepare(train_dataloader)
    val_dataloader = accelerator.prepare(val_dataloader)
    
    # Create PMG inference model
    logger.info("Creating PMG model...")
    pmg_model = InferenceModel(
        layer_num=len(llama_model.model.layers),
        num_image_prompt=config['pmg']['num_image_prompt'],
        num_prefix_prompt=config['pmg']['num_prefix_prompt'],
        emb_dim=4096,
        sd_hidden_state_dim=768
    )
    pmg_model = accelerator.prepare(pmg_model)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        pmg_model.parameters(),
        lr=config['training']['learning_rate'],
        betas=(config['training']['adam_beta1'], config['training']['adam_beta2']),
        weight_decay=config['training']['weight_decay'],
        eps=config['training']['adam_epsilon']
    )
    
    # Calculate training steps
    num_train_epochs = config['training']['num_train_epochs']
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    num_update_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    
    # Setup scheduler
    lr_scheduler = get_scheduler(
        config['training']['lr_scheduler'],
        optimizer=optimizer,
        num_warmup_steps=config['training']['lr_warmup_steps'],
        num_training_steps=max_train_steps,
        num_cycles=config['training'].get('lr_num_cycles', 1)
    )
    
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    
    if config['training']['resume_from']:
        resume_path = config['training']['resume_from']
        logger.info(f"Resuming from checkpoint: {resume_path}")
        accelerator.load_state(resume_path)
        
        # Extract epoch and step from checkpoint name
        checkpoint_name = os.path.basename(resume_path)
        if 'checkpoint' in checkpoint_name:
            global_step = int(checkpoint_name.split('-')[-1])
            start_epoch = global_step // num_update_steps_per_epoch
            logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config['training']['train_batch_size']}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    progress_bar = tqdm(
        range(max_train_steps),
        initial=global_step,
        desc="Training",
        disable=not accelerator.is_local_main_process
    )
    
    pmg_model.train()
    
    for epoch in range(start_epoch, num_train_epochs):
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(pmg_model):
                # Get PMG embeddings
                encoder_hidden_states = pmg_model(
                    llama_tokenizer,
                    llama_model,
                    batch['input_ids'],
                    batch['token_len']
                )
                
                # Convert pixel values to tensor (handle both numpy and tensor inputs)
                if isinstance(batch['pixel_values'][0], torch.Tensor):
                    # Data is already tensor, move to CPU if needed before stacking
                    pixel_values = torch.stack([pv.cpu() if pv.is_cuda else pv for pv in batch['pixel_values']])
                else:
                    # Data is numpy array
                    pixel_values = torch.from_numpy(np.stack(batch['pixel_values']))
                
                pixel_values = pixel_values.float() / 255.0
                pixel_values = pixel_values.permute(0, 3, 1, 2)  # NHWC -> NCHW
                pixel_values = (pixel_values * 2.0) - 1.0  # [0, 1] -> [-1, 1]
                pixel_values = pixel_values.to(accelerator.device)
                
                # Calculate loss (use forward method for training)
                loss = sd_pipeline.forward(encoder_hidden_states, pixel_values)
                
                train_loss += loss.detach().item()
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        pmg_model.parameters(),
                        config['training'].get('max_grad_norm', 1.0)
                    )
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Logging
                if global_step % config['training'].get('logging_steps', 10) == 0:
                    avg_loss = train_loss / config['training']['logging_steps']
                    lr = lr_scheduler.get_last_lr()[0]
                    logger.info(f"Step {global_step}: loss={avg_loss:.4f}, lr={lr:.2e}")
                    
                    if accelerator.is_main_process:
                        accelerator.log({
                            "train_loss": avg_loss,
                            "learning_rate": lr,
                            "epoch": epoch
                        }, step=global_step)
                    
                    train_loss = 0.0
                
                # Save checkpoint
                if global_step % config['training'].get('save_steps', 1000) == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")
                
                # Validation
                if global_step % config['training'].get('validation_steps', 100) == 0:
                    logger.info("Running validation...")
                    pmg_model.eval()
                    val_loss = 0.0
                    
                    with torch.no_grad():
                        for val_batch in val_dataloader:
                            encoder_hidden_states = pmg_model(
                                llama_tokenizer,
                                llama_model,
                                val_batch['input_ids'],
                                val_batch['token_len']
                            )
                            
                            # Convert pixel values to tensor (handle both numpy and tensor inputs)
                            if isinstance(val_batch['pixel_values'][0], torch.Tensor):
                                # Data is already tensor, move to CPU if needed before stacking
                                pixel_values = torch.stack([pv.cpu() if pv.is_cuda else pv for pv in val_batch['pixel_values']])
                            else:
                                # Data is numpy array
                                pixel_values = torch.from_numpy(np.stack(val_batch['pixel_values']))
                            
                            pixel_values = pixel_values.float() / 255.0
                            pixel_values = pixel_values.permute(0, 3, 1, 2)
                            pixel_values = (pixel_values * 2.0) - 1.0
                            pixel_values = pixel_values.to(accelerator.device)
                            
                            # Calculate loss (use forward method for training)
                            loss = sd_pipeline.forward(encoder_hidden_states, pixel_values)
                            val_loss += loss.item()
                    
                    val_loss /= len(val_dataloader)
                    logger.info(f"Validation loss: {val_loss:.4f}")
                    
                    if accelerator.is_main_process:
                        accelerator.log({"val_loss": val_loss}, step=global_step)
                    
                    pmg_model.train()
    
    # Save final model
    if accelerator.is_main_process:
        final_save_path = os.path.join(output_dir, "final_model")
        accelerator.save_state(final_save_path)
        logger.info(f"Training complete! Final model saved to {final_save_path}")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()

