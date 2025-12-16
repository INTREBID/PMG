# Changelog

## Version 1.0.0 (2024-12-16)

### Initial Release - PyTorch Implementation

This is the first official PyTorch implementation of PMG (Personalized Multimodal Generation), converted from the original MindSpore version.

#### Features

- ✅ **Unified Training Pipeline**: Single script for all datasets (FLICKR, POG, SER)
- ✅ **YAML Configuration System**: Easy hyperparameter management
- ✅ **Comprehensive Evaluation**: 8 metrics (LPIPS, SSIM, CPS, CPIS, HPSv2, Aesthetic, Verifier)
- ✅ **Modular Package Structure**: Clean `pmg/` package with reusable components
- ✅ **Complete Documentation**: README, Dataset, Training, Inference, Evaluation guides
- ✅ **Mixed Precision Training**: FP16/BF16 support
- ✅ **Accelerate Integration**: Multi-GPU ready
- ✅ **Data Preprocessing Scripts**: For all three datasets
- ✅ **Model Download Scripts**: Automated pre-trained model setup
- ✅ **Experiment Scripts**: Quick start bash scripts

#### Project Structure

```
PMG_Release/
├── pmg/                    # Core package
│   ├── models/            # Custom SD pipeline, soft prompt
│   ├── data/              # Dataset classes
│   ├── utils/             # Utilities
│   └── evaluation/        # Metrics
├── scripts/               # Main scripts (train, inference, evaluate)
├── configs/               # YAML configurations
├── data/                  # Data preprocessing
├── experiments/           # Training scripts
├── docs/                  # Documentation
├── checkpoints/           # Pre-trained models
└── datasets/              # Dataset storage
```

#### Datasets Supported

1. **FLICKR-AES**: 40,988 aesthetic photos, 210 workers
2. **POG**: 16,100 fashion items, 2,000 users
3. **SER30K**: 30,000+ stickers with emotions

#### Key Components

**Models** (`pmg/models/`):
- `custom_pipeline.py`: Custom Stable Diffusion pipeline with LoRA support
- `soft_prompt.py`: Prefix encoder and inference model (PyTorch version)

**Data** (`pmg/data/`):
- `dataset.py`: Unified PMGDataset for all datasets

**Utils** (`pmg/utils/`):
- `image_utils.py`: Image processing utilities
- `prompt_utils.py`: Prompt generation utilities

**Evaluation** (`pmg/evaluation/`):
- `metrics.py`: Complete MetricsEvaluator with 8 metrics

**Scripts** (`scripts/`):
- `train.py`: Unified training script
- `inference.py`: Unified inference script
- `evaluate.py`: Unified evaluation script

**Data Processing** (`data/`):
- `preprocess_flickr.py`: FLICKR-AES preprocessing
- `preprocess_pog.py`: POG preprocessing
- `preprocess_ser.py`: SER30K preprocessing
- `download_datasets.py`: Dataset validation

**Utilities**:
- `checkpoints/download_models.sh`: Model download script
- `experiments/*.sh`: Quick start training scripts

#### Configuration Files

- `configs/flickr_train.yaml`: FLICKR training configuration
- `configs/pog_train.yaml`: POG training configuration
- `configs/ser_train.yaml`: SER training configuration

#### Documentation

- `README.md`: Main project documentation
- `docs/DATASET.md`: Dataset guide
- `docs/TRAINING.md`: Training guide
- `docs/INFERENCE.md`: Inference guide
- `docs/EVALUATION.md`: Evaluation guide

#### Installation

```bash
pip install -r requirements.txt
pip install -e .
```

#### Quick Start

```bash
# Download models
cd checkpoints && ./download_models.sh

# Preprocess data
python data/preprocess_flickr.py --data_dir datasets/FLICKR

# Train
python scripts/train.py --config configs/flickr_train.yaml

# Inference
python scripts/inference.py --config configs/flickr_train.yaml --checkpoint outputs/flickr/final_model

# Evaluate
python scripts/evaluate.py --config configs/flickr_train.yaml
```

#### Known Issues

- HPSv2 may require manual installation: `pip install hpsv2`
- LLaMA-2 requires HuggingFace access approval
- Large datasets require significant disk space (POG: ~20GB)

#### Future Work

- [ ] Add support for custom datasets
- [ ] Implement real-time inference API
- [ ] Add web UI for interactive generation
- [ ] Support for larger models (LLaMA-13B, SD-XL)
- [ ] Implement additional evaluation metrics
- [ ] Add model compression options

---

## Migration from MindSpore

This PyTorch version maintains compatibility with the original paper's methodology while providing:

1. **Better Ecosystem**: PyTorch/HuggingFace integration
2. **Easier Installation**: Standard pip requirements
3. **More Flexibility**: Easy to extend and customize
4. **Better Documentation**: Comprehensive guides
5. **Modern Tools**: Accelerate for distributed training

### Key Differences from Original

- Framework: MindSpore → PyTorch
- Configuration: Python args → YAML configs
- Dataset: Hardcoded → Unified format
- Evaluation: Separate → Integrated metrics
- Documentation: Minimal → Comprehensive

---

For detailed information, see the [README](README.md) and documentation in `docs/`.

