#!/bin/bash
# Train PMG on POG dataset

python scripts/train.py \
    --config configs/pog_train.yaml \
    --device cuda

