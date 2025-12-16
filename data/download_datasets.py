#!/usr/bin/env python3
"""
Download and validate datasets for PMG

Usage:
    python data/download_datasets.py --dataset FLICKR --data_dir datasets/FLICKR
    python data/download_datasets.py --dataset POG --data_dir datasets/POG
    python data/download_datasets.py --dataset SER --data_dir datasets/SER
    python data/download_datasets.py --all --data_root datasets/
"""
import os
import argparse
from pathlib import Path


def print_section(title):
    """Print a section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def validate_flickr(data_dir):
    """Validate FLICKR dataset"""
    print_section("Validating FLICKR Dataset")
    
    required_files = [
        "FLICKR-AES_image_labeled_by_each_worker.csv",
        "FLICKR-AES_image_score.txt",
        "FLICKR_captions.json",
        "FLICKR_styles.json",
    ]
    
    required_dirs = ["40K"]
    
    missing = []
    for file in required_files:
        path = os.path.join(data_dir, file)
        if not os.path.exists(path):
            missing.append(file)
        else:
            print(f"✓ Found: {file}")
    
    for dir_name in required_dirs:
        path = os.path.join(data_dir, dir_name)
        if not os.path.exists(path):
            missing.append(f"{dir_name}/")
        else:
            # Count images
            try:
                num_images = len([f for f in os.listdir(path) if f.endswith('.jpg')])
                print(f"✓ Found: {dir_name}/ ({num_images} images)")
            except:
                print(f"✓ Found: {dir_name}/")
    
    if missing:
        print(f"\n❌ Missing files/directories:")
        for item in missing:
            print(f"  - {item}")
        print(f"\n📥 Download FLICKR-AES dataset from:")
        print(f"   https://github.com/alanspike/FLICKR-AES")
        return False
    else:
        print(f"\n✅ FLICKR dataset is complete!")
        return True


def validate_pog(data_dir):
    """Validate POG dataset"""
    print_section("Validating POG Dataset")
    
    required_files = [
        "user_data.txt",
        "outfit_data.txt",
        "item_data.txt",
        "captions_sampled.json",
        "user_styles.json",
    ]
    
    required_dirs = ["images"]
    
    missing = []
    for file in required_files:
        path = os.path.join(data_dir, file)
        if not os.path.exists(path):
            missing.append(file)
        else:
            print(f"✓ Found: {file}")
    
    for dir_name in required_dirs:
        path = os.path.join(data_dir, dir_name)
        if not os.path.exists(path):
            missing.append(f"{dir_name}/")
        else:
            try:
                num_images = len([f for f in os.listdir(path) if f.endswith('.jpg')])
                print(f"✓ Found: {dir_name}/ ({num_images} images)")
            except:
                print(f"✓ Found: {dir_name}/")
    
    if missing:
        print(f"\n❌ Missing files/directories:")
        for item in missing:
            print(f"  - {item}")
        print(f"\n📥 Download POG dataset from:")
        print(f"   https://github.com/xthan/polyvore")
        return False
    else:
        print(f"\n✅ POG dataset is complete!")
        return True


def validate_ser(data_dir):
    """Validate SER dataset"""
    print_section("Validating SER Dataset")
    
    required_files = [
        "ser30k_captions.json",
        "user_preferences.json",
        "id_map.csv",
    ]
    
    required_dirs = ["Images", "Annotations"]
    
    missing = []
    for file in required_files:
        path = os.path.join(data_dir, file)
        if not os.path.exists(path):
            missing.append(file)
        else:
            print(f"✓ Found: {file}")
    
    for dir_name in required_dirs:
        path = os.path.join(data_dir, dir_name)
        if not os.path.exists(path):
            missing.append(f"{dir_name}/")
        else:
            print(f"✓ Found: {dir_name}/")
    
    if missing:
        print(f"\n❌ Missing files/directories:")
        for item in missing:
            print(f"  - {item}")
        print(f"\n📥 Download SER30K dataset from:")
        print(f"   https://github.com/LizhenWangXDU/SER30K")
        return False
    else:
        print(f"\n✅ SER dataset is complete!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Download and validate PMG datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["FLICKR", "POG", "SER"],
        help="Dataset to validate"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all datasets"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing the dataset"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="datasets",
        help="Root directory containing all datasets (for --all)"
    )
    
    args = parser.parse_args()
    
    if args.all:
        # Validate all datasets
        print_section("Validating All Datasets")
        results = {}
        
        flickr_dir = os.path.join(args.data_root, "FLICKR")
        pog_dir = os.path.join(args.data_root, "POG")
        ser_dir = os.path.join(args.data_root, "SER")
        
        results["FLICKR"] = validate_flickr(flickr_dir)
        results["POG"] = validate_pog(pog_dir)
        results["SER"] = validate_ser(ser_dir)
        
        # Summary
        print_section("Summary")
        for dataset, valid in results.items():
            status = "✅ Complete" if valid else "❌ Incomplete"
            print(f"{dataset}: {status}")
        
        if all(results.values()):
            print("\n🎉 All datasets are ready!")
        else:
            print("\n⚠️  Some datasets need to be downloaded.")
    
    elif args.dataset:
        if not args.data_dir:
            print("Error: --data_dir is required when --dataset is specified")
            return
        
        # Validate specific dataset
        if args.dataset == "FLICKR":
            validate_flickr(args.data_dir)
        elif args.dataset == "POG":
            validate_pog(args.data_dir)
        elif args.dataset == "SER":
            validate_ser(args.data_dir)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

