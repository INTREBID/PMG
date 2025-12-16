#!/usr/bin/env python3
"""
PMG Unified Evaluation Script

Usage:
    python scripts/evaluate.py --config configs/flickr_train.yaml --eval_dir datasets/FLICKR/eval_outputs
    python scripts/evaluate.py --config configs/pog_train.yaml
    python scripts/evaluate.py --config configs/ser_train.yaml --output results/ser_metrics.json
"""
import os
import sys
import yaml
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pmg.evaluation.metrics import evaluate_dataset


def load_config(config_path: str):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="PMG Evaluation Script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default=None,
        help="Directory containing generated images (default: from config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results JSON (default: eval_dir/metrics_results.json)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for evaluation"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    print(f"Dataset: {config['dataset']['name']}")
    
    # Set paths
    dataset_name = config['dataset']['name']
    test_json_path = config['dataset']['test_json']
    
    if args.eval_dir:
        eval_output_dir = args.eval_dir
    else:
        eval_output_dir = os.path.join(config['dataset']['data_dir'], "eval_outputs")
    
    if args.output:
        output_json_path = args.output
    else:
        output_json_path = os.path.join(eval_output_dir, "metrics_results.json")
    
    user_preferences_path = config['dataset']['user_preferences']
    
    print(f"\nEvaluation settings:")
    print(f"  Test data: {test_json_path}")
    print(f"  Generated images: {eval_output_dir}")
    print(f"  User preferences: {user_preferences_path}")
    print(f"  Output: {output_json_path}")
    print()
    
    # Run evaluation
    evaluate_dataset(
        dataset_name=dataset_name,
        test_json_path=test_json_path,
        eval_output_dir=eval_output_dir,
        output_json_path=output_json_path,
        user_preferences_path=user_preferences_path,
        device=args.device
    )


if __name__ == "__main__":
    main()

