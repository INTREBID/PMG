# Dataset Guide

This document provides detailed information about the three datasets supported by PMG.

## Overview

PMG supports three diverse datasets across different domains:

1. **FLICKR-AES**: Aesthetic photo evaluation
2. **POG (Polyvore Outfits Generation)**: Fashion recommendation
3. **SER30K**: Sticker and emoji generation

---

## FLICKR-AES Dataset

### Description

FLICKR-AES contains 40,988 Creative Commons-licensed photos from FLICKR, rated for aesthetic quality on a scale of 1-5.

### Statistics

- **Total Images**: 40,988
- **Workers**: 210 annotators
- **Total Ratings**: 193,208 (5 ratings per image)
- **Data Collection**: Amazon Mechanical Turk

### Download

1. Visit: [https://github.com/alanspike/FLICKR-AES](https://github.com/alanspike/FLICKR-AES)
2. Download and extract to `datasets/FLICKR/`

### Data Files

```
datasets/FLICKR/
├── 40K/                                        # Images
├── FLICKR-AES_image_labeled_by_each_worker.csv # Per-worker ratings
├── FLICKR-AES_image_score.txt                 # Aggregated scores
├── FLICKR_captions.json                       # Image captions
└── FLICKR_styles.json                         # Worker style preferences
```

### Preprocessing

```bash
python data/preprocess_flickr.py --data_dir datasets/FLICKR
```

This generates:
- `processed_dataset/train.json`
- `processed_dataset/val.json` (15 samples)
- `processed_dataset/test.json` (max 2000 samples)

---

## POG (Polyvore Outfits Generation) Dataset

### Description

POG is a multimodal fashion dataset containing outfit combinations and user purchase histories.

### Statistics

- **Selected Users**: 2,000
- **Items**: 16,100 fashion products
- **Domain**: Clothing, accessories

### Download

1. Visit: [https://github.com/xthan/polyvore](https://github.com/xthan/polyvore)
2. Download and extract to `datasets/POG/`

### Data Files

```
datasets/POG/
├── images/            # Product images
├── user_data.txt      # User interaction sequences
├── outfit_data.txt    # Outfit combinations
├── item_data.txt      # Item metadata
├── captions_sampled.json
└── user_styles.json   # User style preferences (7,994 entries)
```

### Preprocessing

```bash
python data/preprocess_pog.py --data_dir datasets/POG
```

---

## SER30K Dataset

### Description

SER30K is a large-scale dataset of stickers, each categorized by theme and annotated with emotion labels.

### Statistics

- **Total Stickers**: 30,000+
- **Categories**: Multiple emotion and theme categories
- **Annotations**: Emotion labels per sticker

### Download

1. Visit: [https://github.com/LizhenWangXDU/SER30K](https://github.com/LizhenWangXDU/SER30K)
2. Download and extract to `datasets/SER/`

### Data Files

```
datasets/SER/
├── Images/                   # Sticker images (organized by theme)
├── Annotations/              # Emotion and category labels
├── ser30k_captions.json     # Sticker descriptions
├── user_preferences.json    # User preference keywords
└── id_map.csv               # ID mapping
```

### Preprocessing

```bash
python data/preprocess_ser.py --data_dir datasets/SER
```

---

## Unified Data Format

All datasets are converted to a unified JSON format:

### Training/Val/Test JSON Structure

```json
[
  {
    "user_id": "user_abc123",           # or "worker_id" for FLICKR
    "history_item_ids": ["item1", "item2", "item3"],
    "history_items_info": [
      {
        "item_id": "item1",
        "caption": "A stylish red crossbody bag...",
        "image_path": "/path/to/image1.jpg",
        "score": 4,                     # FLICKR only
        "aesthetic_score": 0.75         # FLICKR only
      },
      ...
    ],
    "target_item_id": "target_item",
    "target_item_info": {
      "item_id": "target_item",
      "caption": "A blue denim jacket...",
      "image_path": "/path/to/target.jpg"
    },
    "user_style": "elegant, vibrant, classic",  # POG, SER
    "worker_style": "nature, landscapes",        # FLICKR
    "num_interactions": 3,
    "window_position": 154,             # Position in sequence
    "total_sequence_length": 783
  },
  ...
]
```

### User Preferences JSON

**FLICKR format (list of objects)**:
```json
[
  {
    "worker": "WORKER_ABC123",
    "style": "Nature, landscapes, vibrant colors, serene"
  }
]
```

**POG format (list of objects)**:
```json
[
  {
    "user": "USER_HASH_123",
    "style": "Elegant, vibrant, classic, sophisticated"
  }
]
```

**SER format (nested objects)**:
```json
{
  "topic-name": {
    "topic": "topic-name",
    "num_history_items": 25,
    "keywords": ["keyword1", "keyword2", ...],
    "sample_captions": ["caption1", "caption2", ...]
  }
}
```

---

## Data Validation

Verify your dataset setup:

```bash
# Check single dataset
python data/download_datasets.py --dataset FLICKR --data_dir datasets/FLICKR

# Check all datasets
python data/download_datasets.py --all --data_root datasets/
```

---

## Dataset Statistics Summary

| Dataset | Train | Val | Test | Total Images | Avg History Len |
|---------|-------|-----|------|--------------|-----------------|
| FLICKR  | TBA   | 15  | 2000 | 40,988       | 3               |
| POG     | TBA   | TBA | TBA  | 16,100       | 5               |
| SER     | TBA   | TBA | TBA  | 30,000+      | Variable        |

---

## License and Usage

Each dataset has its own license. Please refer to the original dataset sources for terms of use:

- **FLICKR-AES**: Creative Commons licensed images
- **POG**: Academic use (check original repository)
- **SER30K**: Research purposes (check original repository)

When publishing results using these datasets, please cite the original papers.

