#!/bin/bash
# Train PMG on SER dataset

python scripts/train.py \
    --config configs/ser_train.yaml \
    --device cuda

