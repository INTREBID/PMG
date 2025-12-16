# Training Guide

Complete guide for training PMG models.

## Quick Start

```bash
# Train on FLICKR dataset
python scripts/train.py --config configs/flickr_train.yaml

# Train on POG dataset
python scripts/train.py --config configs/pog_train.yaml

# Train on SER dataset
python scripts/train.py --config configs/ser_train.yaml
```

---

## Configuration

Training is controlled via YAML configuration files in `configs/`.

### Key Parameters

```yaml
dataset:
  name: FLICKR
  train_json: ./datasets/FLICKR/processed_dataset/train.json
  val_json: ./datasets/FLICKR/processed_dataset/val.json

model:
  stable_diffusion: runwayml/stable-diffusion-v1-5
  llama: meta-llama/Llama-2-7b-chat-hf
  clip: ./checkpoints/clip-vit-base-patch32

training:
  num_train_epochs: 3
  train_batch_size: 6
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-6
  mixed_precision: bf16  # or fp16, fp32

pmg:
  num_image_prompt: 2    # Number of learnable image tokens
  num_prefix_prompt: 2   # Number of prefix tokens
  max_sequence_length: 600
  image_size: 512
```

---

## Training Process

### What Gets Trained?

PMG only trains the **soft prompt components**:
- ✅ Trainable image prompt embeddings
- ✅ Prefix encoder network
- ✅ Mapping layer (LLaMA → Stable Diffusion space)

Frozen components:
- ❄️ LLaMA-2 weights
- ❄️ Stable Diffusion (VAE, UNet, Text Encoder)

This results in only **~22M trainable parameters**.

### Training Loop

1. **Forward Pass**:
   - Encode user history with LLaMA + soft prompts
   - Extract image prompt embeddings
   - Map to SD embedding space
   - Generate noise predictions with SD

2. **Loss Calculation**:
   - MSE loss between predicted and actual noise

3. **Optimization**:
   - AdamW optimizer with gradient clipping
   - Linear warmup scheduler

---

## Advanced Options

### Resume Training

```bash
python scripts/train.py \
    --config configs/flickr_train.yaml \
    --resume_from outputs/flickr/checkpoint-5000
```

### Custom Output Directory

Edit config file:
```yaml
training:
  output_dir: ./my_custom_output
```

### Hyperparameter Tuning

Key parameters to adjust:

**Learning Rate**:
```yaml
training:
  learning_rate: 5.0e-6  # Default
  # Try: 1e-6, 3e-6, 1e-5
```

**Batch Size & Accumulation**:
```yaml
training:
  train_batch_size: 6
  gradient_accumulation_steps: 4
  # Effective batch size = 6 * 4 = 24
```

**Training Duration**:
```yaml
training:
  num_train_epochs: 3  # Default
  save_steps: 1000
  validation_steps: 100
```

---

## Multi-GPU Training

PMG uses Accelerate for distributed training.

### Single Machine, Multiple GPUs

```bash
# Automatically uses all available GPUs
python scripts/train.py --config configs/flickr_train.yaml
```

### Custom Accelerate Config

Create `accelerate_config.yaml`:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 4
mixed_precision: bf16
```

Run with:
```bash
accelerate launch --config_file accelerate_config.yaml \
    scripts/train.py --config configs/flickr_train.yaml
```

---

## Memory Optimization

### Mixed Precision

Enable BF16 or FP16:
```yaml
training:
  mixed_precision: bf16  # Recommended for A100/H100
  # or fp16 for older GPUs
```

### Gradient Checkpointing

For limited VRAM, modify the code to enable gradient checkpointing:
```python
# In scripts/train.py
llama_model.gradient_checkpointing_enable()
```

### Batch Size Guidelines

| GPU Memory | Batch Size | Grad Accum | Effective BS |
|------------|------------|------------|--------------|
| 24GB (3090)| 2          | 12         | 24           |
| 40GB (A100)| 6          | 4          | 24           |
| 80GB (A100)| 12         | 2          | 24           |

---

## Monitoring Training

### TensorBoard

Training logs are automatically saved:

```bash
tensorboard --logdir outputs/flickr/logs
```

Metrics logged:
- `train_loss`: Training loss per step
- `val_loss`: Validation loss
- `learning_rate`: Current LR

### Checkpoints

Checkpoints are saved every `save_steps`:

```
outputs/flickr/
├── checkpoint-1000/
├── checkpoint-2000/
├── final_model/
└── logs/
```

---

## Training Time Estimates

Approximate training times on A100 (40GB):

| Dataset | Samples | Epochs | Time   |
|---------|---------|--------|--------|
| FLICKR  | ~15K    | 3      | ~6h    |
| POG     | ~20K    | 3      | ~8h    |
| SER     | ~25K    | 3      | ~10h   |

*Times vary based on GPU, batch size, and dataset size.*

---

## Troubleshooting

### CUDA Out of Memory

**Solutions**:
1. Reduce `train_batch_size`
2. Increase `gradient_accumulation_steps`
3. Use mixed precision (`bf16` or `fp16`)
4. Reduce `image_size` (e.g., 512 → 256)

### Slow Training

**Solutions**:
1. Enable TF32: Already enabled in code
2. Reduce `dataloader_num_workers` if I/O bound
3. Use faster storage (SSD) for datasets

### Loss Not Decreasing

**Solutions**:
1. Check learning rate (try lower: 1e-6)
2. Verify data preprocessing is correct
3. Ensure models are loaded correctly
4. Check if validation loss decreases

### Model Divergence

**Solutions**:
1. Reduce learning rate
2. Add gradient clipping (already enabled)
3. Use warmup steps

---

## Best Practices

1. **Start Small**: Test on a subset of data first
2. **Monitor Metrics**: Watch both train and validation loss
3. **Save Frequently**: Use reasonable `save_steps`
4. **Validate Often**: Set appropriate `validation_steps`
5. **Use Version Control**: Track config changes
6. **Document Experiments**: Keep notes on hyperparameters

---

## Example Training Script

Complete example with all options:

```bash
python scripts/train.py \
    --config configs/flickr_train.yaml \
    --device cuda
```

For custom configs:

```bash
# Copy and modify
cp configs/flickr_train.yaml configs/my_experiment.yaml
# Edit configs/my_experiment.yaml
python scripts/train.py --config configs/my_experiment.yaml
```

---

## Next Steps

After training:
1. [Run inference](INFERENCE.md) to generate images
2. [Evaluate results](EVALUATION.md) using metrics
3. Compare with baselines
4. Fine-tune hyperparameters if needed

