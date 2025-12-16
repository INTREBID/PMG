# PMG: Personalized Multimodal Generation with Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official PyTorch implementation of **PMG: Personalized Multimodal Generation with Large Language Models** (WWW 2024).

> **Abstract:** PMG transforms user-interacted and reference images into textual descriptions, using pre-trained LLMs to extract user preferences through keywords and implicit embeddings to condition the image generator.

[[Paper](https://arxiv.org/abs/2404.08677)] [[Original MindSpore Code](https://github.com/mindspore-lab/models/tree/master/research/huawei-noah/PMG)]

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Results](#results)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Overview

PMG is the first method for personalized multimodal generation using Large Language Models. It leverages:

1. **LLaMA-2** to extract and encode user preferences from historical interactions
2. **Soft Prompt Learning** with trainable prefix and image tokens
3. **Stable Diffusion** conditioned on learned user representations

### Architecture

```mermaid
graph LR
    A[Historical Items] --> B[LLaMA-2 Encoder]
    B --> C[Soft Prompt Learning]
    C --> D[User Preference<br/>Embeddings]
    D --> E[Stable Diffusion]
    E --> F[Personalized Images]
```

---

## Features

- ✅ **Unified Training Pipeline**: Single script for all three datasets (FLICKR, POG, SER)
- ✅ **YAML Configuration System**: Easy hyperparameter management
- ✅ **Comprehensive Evaluation**: 8 metrics including LPIPS, SSIM, CLIP-based, HPSv2, Aesthetic
- ✅ **Flexible Inference**: Support for batch and single-sample generation
- ✅ **Modular Design**: Clean package structure with reusable components
- ✅ **Mixed Precision Training**: FP16/BF16 support for faster training
- ✅ **Accelerate Integration**: Multi-GPU and distributed training ready

---

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.7 (for GPU support)

### Step 1: Clone the Repository

```bash
git clone https://github.com/INTREBID/PMG.git
cd PMG
```

### Step 2: Create Environment

```bash
# Using conda (recommended)
conda create -n pmg python=3.9
conda activate pmg

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### Step 4: Download Pre-trained Models

```bash
cd checkpoints
./download_models.sh
```

Models required:
- Stable Diffusion v1.5
- LLaMA-2-7B-Chat (requires HuggingFace access approval)
- CLIP ViT-B/32

---

## Quick Start

### 1. Prepare Dataset

Download and preprocess one of the supported datasets:

```bash
# FLICKR-AES
python data/preprocess_flickr.py --data_dir datasets/FLICKR

# POG
python data/preprocess_pog.py --data_dir datasets/POG

# SER30K
python data/preprocess_ser.py --data_dir datasets/SER
```

See [docs/DATASET.md](docs/DATASET.md) for detailed dataset information.

### 2. Train Model

```bash
# Train on FLICKR
python scripts/train.py --config configs/flickr_train.yaml

# Train on POG
python scripts/train.py --config configs/pog_train.yaml

# Train on SER
python scripts/train.py --config configs/ser_train.yaml
```

### 3. Generate Images

```bash
python scripts/inference.py \
    --config configs/flickr_train.yaml \
    --checkpoint outputs/flickr/final_model \
    --output_dir results/flickr_generated
```

### 4. Evaluate Results

```bash
python scripts/evaluate.py \
    --config configs/flickr_train.yaml \
    --eval_dir results/flickr_generated
```

---

## Datasets

PMG supports three diverse datasets:

| Dataset | Domain | Images | Users | Description |
|---------|--------|--------|-------|-------------|
| **FLICKR-AES** | Aesthetic Photos | 40,988 | 210 | Creative Commons photos rated for aesthetic quality |
| **POG** | Fashion | 16,100 | 2,000 | Polyvore outfit items and user interactions |
| **SER30K** | Stickers | 30,000+ | - | Emotion-labeled sticker dataset |

### Data Format

All datasets follow a unified JSON structure:

```json
{
  "user_id": "user_123",
  "history_items_info": [
    {"item_id": "item_1", "caption": "...", "image_path": "..."},
    {"item_id": "item_2", "caption": "...", "image_path": "..."}
  ],
  "target_item_info": {
    "item_id": "target", "caption": "...", "image_path": "..."
  }
}
```

See [docs/DATASET.md](docs/DATASET.md) for details.

---

## Training

### Configuration

Edit YAML config files in `configs/` to customize hyperparameters:

```yaml
training:
  num_train_epochs: 3
  train_batch_size: 6
  learning_rate: 5.0e-6
  gradient_accumulation_steps: 4

pmg:
  num_image_prompt: 2
  num_prefix_prompt: 2
  image_size: 512
```

### Multi-GPU Training

The training script automatically uses all available GPUs via Accelerate:

```bash
# Will use all available GPUs
python scripts/train.py --config configs/flickr_train.yaml
```

### Resume Training

```bash
python scripts/train.py \
    --config configs/flickr_train.yaml \
    --resume_from outputs/flickr/checkpoint-5000
```

See [docs/TRAINING.md](docs/TRAINING.md) for advanced training options.

---

## Inference

### Batch Generation

```bash
python scripts/inference.py \
    --config configs/flickr_train.yaml \
    --checkpoint outputs/flickr/final_model \
    --num_images 5
```

### Custom Settings

Modify `inference` section in config YAML:

```yaml
inference:
  num_inference_steps: 50
  guidance_scale: 6.0
  negative_prompt: "blurry, low quality, distorted"
```

See [docs/INFERENCE.md](docs/INFERENCE.md) for details.

---

## Evaluation

PMG provides comprehensive evaluation metrics:

### Metrics

1. **LPIPS (vs Target)**: Perceptual similarity to target image
2. **LPIPS (vs History Avg)**: Perceptual similarity to user history
3. **SSIM (vs Target)**: Structural similarity to target
4. **SSIM (vs History Avg)**: Structural similarity to history
5. **CPS**: CLIP Preference Similarity (image-text alignment)
6. **CPIS (vs History Avg)**: CLIP Preference Image Similarity
7. **HPSv2**: Human Preference Score v2
8. **LAION Aesthetic**: Aesthetic quality score

### Run Evaluation

```bash
python scripts/evaluate.py \
    --config configs/flickr_train.yaml \
    --eval_dir results/flickr_generated \
    --output results/flickr_metrics.json
```

See [docs/EVALUATION.md](docs/EVALUATION.md) for metric details.

---

## Results

### Performance on Three Datasets

| Method | Dataset | LPIPS↓ | SSIM↑ | CPS↑ | CPIS↑ | HPSv2↑ | Aesthetic↑ |
|--------|---------|--------|-------|------|-------|--------|------------|
| PMG | FLICKR | 0.749 | 0.310 | 0.191 | 0.500 | 0.225 | **10.47** |
| PMG | POG | 0.703 | 0.235 | 0.255 | 0.702 | 0.190 | **10.78** |
| PMG | SER | 0.784 | 0.269 | 0.236 | 0.571 | 0.201 | **10.16** |

PMG consistently achieves superior aesthetic quality across all datasets.

---

## Project Structure

```
PMG/
├── pmg/                    # Core package
│   ├── models/            # SD pipeline, soft prompt
│   ├── data/              # Dataset classes
│   ├── utils/             # Image/prompt utilities
│   └── evaluation/        # Metrics evaluator
├── scripts/               # Training/inference/evaluation
├── configs/               # YAML configurations
├── data/                  # Data preprocessing scripts
├── experiments/           # Training scripts
├── docs/                  # Documentation
├── checkpoints/           # Pre-trained models
└── datasets/              # Dataset storage
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{pmg2024,
  title={PMG: Personalized Multimodal Generation with Large Language Models},
  author={Shen, Xiaoteng and Zhang, Rui and Zhao, Xiaoyan and Zhu, Jieming and Xiao, Xi},
  booktitle={Proceedings of the ACM Web Conference 2024},
  year={2024}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Original paper and MindSpore implementation: [PMG](https://github.com/mindspore-lab/models/tree/master/research/huawei-noah/PMG)
- Pre-trained models: [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5), [LLaMA-2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [CLIP](https://huggingface.co/openai/clip-vit-base-patch32)
- Datasets: FLICKR-AES, POG, SER30K

---

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

Happy generating! 🎨✨

