# Installation Guide

Complete installation instructions for PMG.

## System Requirements

### Hardware

- **GPU**: NVIDIA GPU with 24GB+ VRAM (recommended)
  - RTX 3090 (24GB): Minimum
  - L40 (48GB): Recommended
  - V100 (32GB): Supported
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ free disk space
  - Models: ~30GB
  - Datasets: ~50GB (varies by dataset)
  - Training outputs: ~20GB

### Software

- **OS**: Linux (Ubuntu 20.04/22.04 recommended)
- **Python**: 3.8, 3.9, or 3.10
- **CUDA**: 11.7 or 12.1
- **cuDNN**: Compatible with CUDA version

---

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/INTREBID/PMG.git
cd PMG
```

### 2. Create Environment


```bash
conda create -n pmg python=3.9
conda activate pmg
```

### 3. Install PyTorch

Install PyTorch with CUDA support:

```bash
pip install torch torchvision
```

Verify installation:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `transformers` (LLaMA, CLIP)
- `diffusers` (Stable Diffusion)
- `accelerate` (Distributed training)
- `lpips` (Perceptual loss)
- `torchmetrics` (SSIM)
- Other utilities

### 5. Install PMG Package

```bash
pip install -e .
```

This allows you to import `pmg` from anywhere.

### 6. Verify Installation

```bash
python -c "import pmg; print('PMG installed successfully!')"
```

---

## Optional Dependencies

### HPSv2 (Human Preference Score)

```bash
pip install hpsv2
```

### Development Tools

```bash
pip install pytest black flake8 isort
```

---

## Download Pre-trained Models

### Automatic Download

```bash
cd checkpoints
./download_models.sh
```

This will download:
1. Stable Diffusion v1.5 (~10GB)
2. LLaMA-2-7B-Chat (~14GB)
3. CLIP ViT-B/32 (~500MB)

### Manual Download

If automatic download fails:

**Stable Diffusion v1.5**:
```bash
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 checkpoints/stable-diffusion-v1-5
```

**LLaMA-2-7B-Chat**:
1. Request access at: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
2. Once approved:
```bash
huggingface-cli login
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf checkpoints/Llama-2-7b-chat-hf
```

**CLIP ViT-B/32**:
```bash
git clone https://huggingface.co/openai/clip-vit-base-patch32 checkpoints/clip-vit-base-patch32
```

---

## Download Datasets

See [docs/DATASET.md](docs/DATASET.md) for detailed instructions.

### Quick Download

**FLICKR-AES**:
```bash
# Download from: https://github.com/alanspike/personalizedImageAesthetics
# Extract to: datasets/FLICKR/
```

**POG**:
```bash
# Download from: https://github.com/wenyuer/POG
# Extract to: datasets/POG/
```

**SER30K**:
```bash
# Download from: https://github.com/nku-shengzheliu/SER30K
# Extract to: datasets/SER/
```

### Verify Datasets

```bash
python data/download_datasets.py --all --data_root datasets/
```

---

## Configuration

### Update Model Paths

If using local model paths, edit config files:

```yaml
# configs/flickr_train.yaml
model:
  stable_diffusion: ./checkpoints/stable-diffusion-v1-5
  llama: ./checkpoints/Llama-2-7b-chat-hf
  clip: ./checkpoints/clip-vit-base-patch32
```

### Update Dataset Paths

By default, paths are relative. If using absolute paths:

```yaml
dataset:
  data_dir: /absolute/path/to/datasets/FLICKR
```

---



## Testing Installation

Run quick tests:

```bash
# Test imports
python -c "from pmg.models import SDPipeline; print('✓ Models')"
python -c "from pmg.data import PMGDataset; print('✓ Data')"
python -c "from pmg.evaluation import MetricsEvaluator; print('✓ Evaluation')"

# Test CUDA
python -c "import torch; assert torch.cuda.is_available(); print('✓ CUDA')"

# Check models
ls checkpoints/stable-diffusion-v1-5/
ls checkpoints/Llama-2-7b-chat-hf/
ls checkpoints/clip-vit-base-patch32/
```

---

## Next Steps

After installation:

1. [Preprocess a dataset](docs/DATASET.md)
2. [Start training](docs/TRAINING.md)
3. [Generate images](docs/INFERENCE.md)
4. [Evaluate results](docs/EVALUATION.md)

---

## Support

If you encounter issues:

1. Check [troubleshooting](#troubleshooting) section
2. Search existing GitHub issues
3. Open a new issue with:
   - System information
   - Error message
   - Steps to reproduce

