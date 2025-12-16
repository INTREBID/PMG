# Inference Guide

Complete guide for generating images with trained PMG models.

## Quick Start

```bash
python scripts/inference.py \
    --config configs/flickr_train.yaml \
    --checkpoint outputs/flickr/final_model \
    --output_dir results/flickr_generated
```

---

## Command Line Options

### Required Arguments

- `--config`: Path to configuration YAML file
- `--checkpoint`: Path to trained model checkpoint directory

### Optional Arguments

- `--output_dir`: Directory to save generated images (default: from config)
- `--device`: Device to use (default: `cuda`)
- `--num_images`: Number of images per sample (default: from config)

---

## Configuration

Inference settings in YAML:

```yaml
inference:
  num_inference_steps: 50      # Denoising steps
  guidance_scale: 6.0          # CFG scale
  negative_prompt: "blurry, low quality"
  num_images_per_sample: 1     # Images per test sample
```

### Guidance Scale

Controls how strongly the model follows the learned preferences:

- **Low (3-5)**: More creative, diverse outputs
- **Medium (6-8)**: Balanced (recommended)
- **High (9-12)**: Stronger adherence to preferences

### Inference Steps

Number of denoising steps:

- **25 steps**: Fast, lower quality
- **50 steps**: Balanced (recommended)
- **100 steps**: Slow, highest quality

---

## Output Structure

Generated images are organized by sample:

```
output_dir/
├── sample_0000/
│   ├── gen_0.jpg      # Generated image
│   ├── gen_1.jpg      # (if num_images > 1)
│   └── info.json      # Sample metadata
├── sample_0001/
│   ├── gen_0.jpg
│   └── info.json
...
```

### info.json Format

```json
{
  "sample_idx": 0,
  "user_id": "user_123",
  "target_item_id": "item_456",
  "target_caption": "A stylish blue jacket...",
  "num_generated": 1
}
```

---

## Examples

### Generate with Custom Settings

```bash
python scripts/inference.py \
    --config configs/flickr_train.yaml \
    --checkpoint outputs/flickr/checkpoint-5000 \
    --output_dir results/exp1 \
    --num_images 3 \
    --device cuda
```

### Generate on CPU

```bash
python scripts/inference.py \
    --config configs/flickr_train.yaml \
    --checkpoint outputs/flickr/final_model \
    --device cpu
```

**Note**: CPU inference is very slow (100x slower than GPU).

---

## Performance

### Speed

Approximate generation times per image (512x512, 50 steps):

| Hardware | Time/Image |
|----------|------------|
| A100 (40GB) | ~2s |
| RTX 3090 (24GB) | ~3s |
| V100 (32GB) | ~4s |
| CPU | ~5min |

### Memory Usage

| Resolution | VRAM Required |
|------------|---------------|
| 256x256 | ~8GB |
| 512x512 | ~12GB |
| 768x768 | ~20GB |
| 1024x1024 | ~30GB |

---

## Batch Processing

To process large test sets efficiently, the script automatically handles all test samples.

Monitor progress:
```bash
python scripts/inference.py ... 2>&1 | tee inference.log
```

---

## Tips and Tricks

### Quality Improvement

1. **Use more steps**: Increase `num_inference_steps`
2. **Adjust guidance**: Try different `guidance_scale` values
3. **Better checkpoint**: Use checkpoint with lower validation loss

### Speed Optimization

1. **Reduce steps**: Use 25-30 steps for faster generation
2. **Batch size**: Modify code to process multiple samples together
3. **Lower resolution**: Reduce `image_size` in config

### Diversity

Generate multiple images per sample:
```bash
--num_images 5
```

Each will use a different random seed.

---

## Troubleshooting

### CUDA Out of Memory

**Solutions**:
1. Reduce `image_size` in config
2. Use CPU (slow): `--device cpu`
3. Process fewer samples at once (modify code)

### Poor Quality Images

**Possible causes**:
1. Checkpoint not fully trained
2. Wrong checkpoint loaded
3. Data preprocessing issues

**Solutions**:
1. Use later checkpoint or final model
2. Verify checkpoint path
3. Check if training completed successfully

### Images Don't Match Preferences

**Possible causes**:
1. Model not converged
2. Guidance scale too low
3. User preferences not loaded

**Solutions**:
1. Train for more epochs
2. Increase `guidance_scale`
3. Verify `user_preferences` path in config

---

## Advanced Usage

### Custom Negative Prompts

Edit config:
```yaml
inference:
  negative_prompt: "blurry, low quality, distorted, watermark, text, bad anatomy"
```

Dataset-specific negatives:
- **FLICKR**: Add "drawing, painting, cartoon"
- **POG**: Add "face, portrait, landscape"
- **SER**: Add "photorealistic, 3d, realistic"

### Deterministic Generation

Modify the code to use fixed seeds:

```python
generator = torch.Generator(device=device).manual_seed(42)
```

---

## Post-Processing

### Upscaling

Use external tools to upscale generated images:

```bash
# Using Real-ESRGAN
python inference_realesrgan.py \
    -i results/flickr_generated \
    -o results/flickr_upscaled \
    -s 2
```

### Filtering

Filter images by quality metrics:

```python
import json

# Load evaluation results
with open('results/metrics_results.json') as f:
    results = json.load(f)

# Filter high-quality images
high_quality = [
    s for s in results['per_sample_results']
    if s['metrics']['laion_aesthetic'] > 10.5
]

print(f"Found {len(high_quality)} high-quality images")
```

---

## Next Steps

1. [Evaluate generated images](EVALUATION.md)
2. Compare with baseline methods
3. Visualize results
4. Conduct user studies

