# Pre-trained Models

This directory contains pre-trained models required for PMG.

## Required Models

### 1. Stable Diffusion v1.5

```bash
# The model will be automatically downloaded from HuggingFace
# Or you can download manually:
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```

### 2. LLaMA-2-7B-Chat

```bash
# Request access from Meta and download from HuggingFace:
# https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```

### 3. CLIP ViT-B/32

```bash
# Download CLIP model
git clone https://huggingface.co/openai/clip-vit-base-patch32
```

## Directory Structure

After downloading, your directory should look like:

```
checkpoints/
├── stable-diffusion-v1-5/
├── Llama-2-7b-chat-hf/
└── clip-vit-base-patch32/
```

## Alternative: Automatic Download

Models will be automatically downloaded from HuggingFace Hub on first use if not found locally. Make sure you have:

1. HuggingFace account and accepted model licenses
2. Installed `huggingface-hub`: `pip install huggingface-hub`
3. Logged in: `huggingface-cli login`

## Pre-trained PMG Checkpoints

We provide pre-trained PMG checkpoints for all three datasets:

- **FLICKR**: [Download Link TBA]
- **POG**: [Download Link TBA]
- **SER**: [Download Link TBA]

Place the downloaded checkpoints in their respective directories:

```
checkpoints/
├── pmg_flickr/
├── pmg_pog/
└── pmg_ser/
```

