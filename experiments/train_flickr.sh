#!/bin/bash
# Train PMG on FLICKR dataset

python scripts/train.py \
    --config configs/flickr_train.yaml \
    --device cuda

