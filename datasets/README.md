# Datasets

This directory contains the datasets used for training and evaluation.

## Dataset Structure

After downloading and preprocessing, your directory should look like:

```
datasets/
├── FLICKR/
│   ├── 40K/                          # Raw images
│   ├── FLICKR_captions.json          # Image captions
│   ├── FLICKR_styles.json            # Worker style preferences
│   ├── FLICKR-AES_image_labeled_by_each_worker.csv
│   ├── FLICKR-AES_image_score.txt
│   └── processed_dataset/
│       ├── train.json
│       ├── val.json
│       └── test.json
│
├── POG/
│   ├── images/                       # Raw images
│   ├── user_data.txt
│   ├── outfit_data.txt
│   ├── item_data.txt
│   ├── captions_sampled.json
│   ├── user_styles.json
│   └── processed_dataset/
│       ├── train.json
│       ├── val.json
│       └── test.json
│
└── SER/
    ├── Images/                       # Raw stickers
    ├── Annotations/
    ├── ser30k_captions.json
    ├── user_preferences.json
    └── processed_dataset/
        ├── train.json
        ├── val.json
        └── test.json
```

## Download Instructions

### 1. FLICKR-AES Dataset

1. Download from: [FLICKR-AES GitHub](https://github.com/alanspike/FLICKR-AES)
2. Extract to `datasets/FLICKR/`
3. Run preprocessing: `python data/preprocess_flickr.py --data_dir datasets/FLICKR`

### 2. POG Dataset

1. Download from: [Polyvore Outfits Dataset](https://github.com/xthan/polyvore)
2. Extract to `datasets/POG/`
3. Download images: `python data/preprocess_pog.py --download --data_dir datasets/POG`
4. Run preprocessing: `python data/preprocess_pog.py --data_dir datasets/POG`

### 3. SER30K Dataset

1. Download from: [SER30K](https://github.com/LizhenWangXDU/SER30K)
2. Extract to `datasets/SER/`
3. Run preprocessing: `python data/preprocess_ser.py --data_dir datasets/SER`

## Dataset Statistics

| Dataset | Train | Val | Test | Total Images |
|---------|-------|-----|------|--------------|
| FLICKR  | TBA   | 15  | 2000 | 40,988       |
| POG     | TBA   | TBA | TBA  | 16,100       |
| SER     | TBA   | TBA | TBA  | 30,000+      |

## Data Format

All datasets follow a unified JSON format. See `docs/DATASET.md` for detailed format specifications.

