"""
AgroDetect - Dataset Download & Verification Helper
Allows easy downloading and inspection of the PlantVillage dataset.
"""

import os
import sys
import argparse
from pathlib import Path


def verify_dataset(dataset_dir: str):
    """Inspects dataset folder and prints summary statistics."""
    base_path = Path(dataset_dir)
    print(f"\nInspecting dataset directory at: {base_path.resolve()}\n")

    if not base_path.exists():
        print(f"[!] Directory '{dataset_dir}' does not exist.")
        print("Please follow instructions in dataset/README.md to download the dataset.")
        return False

    subdirs = [p.name for p in base_path.iterdir() if p.is_dir()]
    print(f"Found subdirectories: {subdirs}")

    split_dirs = [d for d in ["train", "valid", "test", "val"] if (base_path / d).is_dir()]
    if split_dirs:
        print(f"[+] Detected split dataset structure: {split_dirs}")
        for s in split_dirs:
            split_path = base_path / s
            classes = [p.name for p in split_path.iterdir() if p.is_dir()]
            total_images = sum(
                len(list((split_path / c).glob("*.*")))
                for c in classes
            )
            print(f"  -> {s}: {len(classes)} classes, {total_images} total image files.")
        return True
    else:
        # Check if base_path directly has class folders
        classes = [p.name for p in base_path.iterdir() if p.is_dir() and not p.name.startswith(".")]
        if classes:
            total_images = sum(
                len(list((base_path / c).glob("*.*")))
                for c in classes
            )
            print(f"[+] Detected flat dataset with {len(classes)} classes and {total_images} images.")
            print("Note: training/dataset.py can automatically split this into train/val/test splits.")
            return True
        else:
            print("[!] No class subfolders detected. Please extract your images into dataset/train and dataset/valid.")
            return False


def download_via_kagglehub(target_dir: str):
    """Downloads dataset using kagglehub if installed."""
    try:
        import kagglehub
        print("[+] Downloading 'vipoooool/new-plant-diseases-dataset' using kagglehub...")
        path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
        print(f"[+] Download complete! Files stored at: {path}")
        print(f"Please copy the train and valid directories to: {target_dir}")
    except ImportError:
        print("[!] 'kagglehub' is not installed. You can install it with: pip install kagglehub")
        print("Alternatively, download manually following dataset/README.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgroDetect Dataset Tool")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--verify", action="store_true", help="Verify dataset integrity")
    parser.add_argument("--download", action="store_true", help="Attempt download via kagglehub")

    args = parser.parse_args()

    if args.download:
        download_via_kagglehub(args.data_dir)
    else:
        verify_dataset(args.data_dir)
