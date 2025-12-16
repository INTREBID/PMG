# Evaluation Guide

Complete guide for evaluating PMG-generated images.

## Quick Start

```bash
# Generate images first (if not done)
python scripts/inference.py \
    --config configs/flickr_train.yaml \
    --checkpoint outputs/flickr/final_model

# Then evaluate
python scripts/evaluate.py \
    --config configs/flickr_train.yaml \
    --eval_dir datasets/FLICKR/eval_outputs
```

---

## Evaluation Metrics

PMG uses 8 comprehensive metrics to evaluate personalized image generation:

### 1. LPIPS (Learned Perceptual Image Patch Similarity)

**Range**: [0, 1] (lower is better)

**Description**: Measures perceptual similarity using deep VGG features.

**Variants**:
- `lpips_target`: Distance to target image
- `lpips_history_avg`: Average distance to historical user images

**Interpretation**:
- Lower values = more similar
- Good: 0.5-0.7
- Excellent: < 0.5

---

### 2. SSIM (Structural Similarity Index)

**Range**: [-1, 1] (higher is better)

**Description**: Compares luminance, contrast, and structure.

**Variants**:
- `ssim_target`: Similarity to target image
- `ssim_history_avg`: Average similarity to history

**Interpretation**:
- Higher values = more similar structure
- Good: 0.3-0.5
- Excellent: > 0.5

---

### 3. CPS (CLIP Preference Similarity)

**Range**: [-1, 1] (higher is better)

**Description**: Measures alignment between generated image and user preference text using CLIP.

**Calculation**: Cosine similarity between CLIP image and text embeddings

**Interpretation**:
- Higher = better alignment with user preferences
- Good: 0.2-0.3
- Excellent: > 0.3

---

### 4. CPIS (CLIP Preference Image Similarity)

**Range**: [-1, 1] (higher is better)

**Description**: Measures visual style consistency with user history using CLIP image embeddings.

**Variant**:
- `cpis_history_avg`: Average CLIP similarity to history

**Interpretation**:
- Higher = more consistent with user's visual style
- Good: 0.5-0.7
- Excellent: > 0.7

---

### 5. HPSv2 (Human Preference Score v2)

**Range**: [0, 1] (higher is better)

**Description**: Predicts human aesthetic preferences for text-to-image generation.

**Training**: Model trained on human preference data

**Interpretation**:
- Higher = better predicted human preference
- Good: 0.2-0.25
- Excellent: > 0.25

---

### 6. LAION Aesthetic Score

**Range**: Typically [0, 10+] (higher is better)

**Description**: Predicts aesthetic quality using LAION-Aesthetics trained predictor.

**Calculation**: Linear projection from CLIP features

**Interpretation**:
- 5-7: Average quality
- 7-9: Good quality
- 9-11: Excellent quality
- PMG achieves: **10.16-10.78**

---

### 7. Verifier Score (FLICKR only)

**Range**: [0, 1] (higher is better)

**Description**: Worker-specific personalization verification.

**Training**: Learned from worker-specific aesthetic preferences

**Interpretation**:
- Higher = better personalization to individual worker

---

## Running Evaluation

### Basic Usage

```bash
python scripts/evaluate.py \
    --config configs/flickr_train.yaml \
    --device cuda
```

### Custom Paths

```bash
python scripts/evaluate.py \
    --config configs/flickr_train.yaml \
    --eval_dir results/my_generation \
    --output results/my_metrics.json \
    --device cuda
```

---

## Output Format

Evaluation results are saved as JSON:

```json
{
  "dataset": "FLICKR",
  "total_samples": 2000,
  "evaluated_samples": 1998,
  "average_metrics": {
    "lpips_target": 0.7487,
    "lpips_history_avg": 0.7955,
    "ssim_target": 0.3102,
    "ssim_history_avg": 0.2889,
    "cps": 0.1907,
    "cpis_history_avg": 0.5002,
    "hpsv2": 0.2255,
    "laion_aesthetic": 10.4664
  },
  "std_metrics": {
    "lpips_target": 0.0613,
    ...
  },
  "per_sample_results": [
    {
      "sample_idx": 0,
      "user_id": "user_123",
      "target_item_id": "item_456",
      "metrics": {
        "lpips_target": 0.7234,
        ...
      }
    },
    ...
  ]
}
```

---

## Understanding Results

### PMG Performance Summary

| Metric | FLICKR | POG | SER | Best Value |
|--------|--------|-----|-----|------------|
| LPIPS (Target) ↓ | 0.749 | 0.703 | 0.784 | Lower |
| SSIM (Target) ↑ | 0.310 | 0.235 | 0.269 | Higher |
| CPS ↑ | 0.191 | 0.255 | 0.236 | Higher |
| CPIS (History) ↑ | 0.500 | 0.702 | 0.571 | Higher |
| HPSv2 ↑ | 0.225 | 0.190 | 0.201 | Higher |
| **Aesthetic** ↑ | **10.47** | **10.78** | **10.16** | Higher |

Key observations:
- **Excellent Aesthetic Scores**: 10+ consistently
- **Good Personalization**: CPIS shows style consistency
- **Competitive Similarity**: LPIPS and SSIM competitive with baselines

---

## Comparison with Baselines

### vs. Textual Inversion

PMG advantages:
- ✅ **Much higher aesthetic scores** (10+ vs 5-6)
- ✅ Better CLIP-based preference alignment
- ⚖️ Similar perceptual similarity

### vs. IP-Adapter

PMG advantages:
- ✅ **Significantly better aesthetics** (10+ vs 5-6)
- ✅ Better human preference scores (HPSv2)
- ⚖️ IP-Adapter has slightly better CPIS (history matching)

---

## Troubleshooting

### Missing Metrics

**Problem**: Some metrics return `None`

**Causes**:
1. HPSv2 not installed
2. Missing CLIP model
3. Missing user preferences file

**Solutions**:
```bash
# Install HPSv2
pip install hpsv2

# Check CLIP model path
ls checkpoints/clip-vit-base-patch32/

# Verify user preferences
ls datasets/FLICKR/FLICKR_styles.json
```

### Slow Evaluation

**Problem**: Evaluation takes too long

**Solutions**:
1. Use GPU: `--device cuda`
2. Reduce image resolution in config
3. Disable HPSv2 if not needed

### Memory Issues

**Problem**: CUDA out of memory

**Solutions**:
1. Process in smaller batches (modify code)
2. Use CPU for some metrics
3. Reduce image size

---

## Advanced Usage

### Evaluate Subset

Modify test.json to include only desired samples:

```python
import json

with open('datasets/FLICKR/processed_dataset/test.json') as f:
    data = json.load(f)

# Take first 100 samples
subset = data[:100]

with open('datasets/FLICKR/processed_dataset/test_subset.json', 'w') as f:
    json.dump(subset, f)
```

Then evaluate:
```bash
# Temporarily modify config to use test_subset.json
python scripts/evaluate.py --config configs/flickr_train.yaml
```

### Custom Metrics

Add custom metrics in `pmg/evaluation/metrics.py`:

```python
class MetricsEvaluator:
    def calculate_custom_metric(self, image1, image2):
        # Your custom metric
        pass
```

---

## Visualization

### Generate Results Summary

```python
import json
import pandas as pd

# Load results
with open('results/metrics_results.json') as f:
    results = json.load(f)

# Create DataFrame
metrics_df = pd.DataFrame([
    {
        'Metric': k,
        'Mean': v,
        'Std': results['std_metrics'][k]
    }
    for k, v in results['average_metrics'].items()
])

print(metrics_df.to_markdown(index=False))
```

### Plot Distributions

```python
import matplotlib.pyplot as plt

# Extract per-sample metric values
lpips_values = [
    s['metrics']['lpips_target']
    for s in results['per_sample_results']
    if s['metrics']['lpips_target'] is not None
]

plt.hist(lpips_values, bins=50)
plt.xlabel('LPIPS (vs Target)')
plt.ylabel('Count')
plt.title('Distribution of LPIPS Scores')
plt.savefig('lpips_distribution.png')
```

---

## Metric Interpretation Guide

### What Makes Good Personalized Generation?

**Similarity Metrics** (LPIPS, SSIM):
- Should match target reasonably well
- But not too close (avoid copying)
- Balance between novelty and fidelity

**Preference Metrics** (CPS, CPIS):
- Higher is better for personalization
- CPS: Text-based preference alignment
- CPIS: Visual style consistency

**Quality Metrics** (HPSv2, Aesthetic):
- Directly measure generation quality
- Higher = better quality
- Most important for practical use

### Ideal Score Profile

For personalized generation:
- ✅ High Aesthetic (9+)
- ✅ High CPS (0.25+)
- ✅ Moderate CPIS (0.5-0.7)
- ✅ Moderate LPIPS (0.6-0.8)
- ✅ High HPSv2 (0.2+)

---

## Next Steps

1. Analyze per-sample results to identify failure cases
2. Compare with baseline methods
3. Adjust model or training if needed
4. Visualize generated images
5. Conduct user studies for qualitative evaluation

