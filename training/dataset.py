"""
AgroDetect - Dataset Loading & Data Augmentation Module
Supports standard PlantVillage structures (train/valid/test) and automatic splits.
Adapts dynamically to the dataset's actual classes.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ResNet50 input specifications
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_data_transforms() -> Dict[str, transforms.Compose]:
    """
    Returns data transforms for training, validation, and testing.
    Includes rich data augmentation for training to prevent overfitting.
    """
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return {
        "train": train_transforms,
        "valid": eval_transforms,
        "test": eval_transforms,
    }


def save_class_mapping(class_names: List[str], output_path: str = "models/class_names.json") -> None:
    """Saves the detected class names list to a JSON file."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(class_names)} class mappings to {output_path}")


def load_class_mapping(file_path: str = "models/class_names.json") -> List[str]:
    """Loads class names list from a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Class names mapping file not found at: {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class SafeImageFolder(datasets.ImageFolder):
    """
    Extension of torchvision ImageFolder that safely skips corrupted or unreadable images
    instead of crashing during a long training epoch.
    """
    def __getitem__(self, index: int) -> Optional[Tuple[torch.Tensor, int]]:
        try:
            return super().__getitem__(index)
        except Exception as exc:
            logger.warning(f"Error loading image at index {index}: {exc}. Skipping sample.")
            # Return adjacent sample
            alt_idx = (index + 1) % len(self.samples)
            return super().__getitem__(alt_idx)


def get_data_loaders(
    data_dir: str = "dataset",
    batch_size: int = 32,
    num_workers: int = 0,
    val_split: float = 0.15,
    test_split: float = 0.15,
    save_mapping: bool = True,
    mapping_path: str = "models/class_names.json"
) -> Tuple[Dict[str, DataLoader], List[str]]:
    """
    Inspects data_dir and builds PyTorch DataLoaders.
    
    Supports two formats:
      Case A: Pre-split directory with 'train', 'valid' (or 'val'), and optionally 'test'.
      Case B: Flat directory containing class folders directly (automatically splits).
      
    Returns:
      (loaders_dict, class_names_list)
    """
    base_path = Path(data_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {data_dir}")

    transforms_dict = get_data_transforms()

    # Check for subdirectories like 'train' and 'valid'
    has_train = (base_path / "train").is_dir()
    has_valid = (base_path / "valid").is_dir() or (base_path / "val").is_dir()
    val_dirname = "valid" if (base_path / "valid").is_dir() else "val"
    has_test = (base_path / "test").is_dir()

    loaders = {}
    class_names = []

    if has_train and has_valid:
        logger.info(f"Detected pre-split dataset at: {data_dir}")
        train_dataset = SafeImageFolder(
            root=str(base_path / "train"),
            transform=transforms_dict["train"]
        )
        val_dataset = SafeImageFolder(
            root=str(base_path / val_dirname),
            transform=transforms_dict["valid"]
        )

        class_names = train_dataset.classes
        logger.info(f"Loaded {len(class_names)} classes from train/ directory.")

        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        loaders["valid"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        if has_test:
            test_dataset = SafeImageFolder(
                root=str(base_path / "test"),
                transform=transforms_dict["test"]
            )
            loaders["test"] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
        else:
            logger.info("No dedicated test/ directory found; validation set will be used for final checks.")
            loaders["test"] = loaders["valid"]

    else:
        # Case B: Flat directory with class folders
        logger.info(f"Inspecting flat dataset directory: {data_dir}")
        full_dataset = SafeImageFolder(
            root=str(base_path),
            transform=transforms_dict["valid"]  # default transform
        )
        class_names = full_dataset.classes
        total_samples = len(full_dataset)

        if total_samples == 0:
            raise ValueError(f"No valid image files found inside class folders in: {data_dir}")

        logger.info(f"Total samples: {total_samples} across {len(class_names)} classes.")

        val_size = int(total_samples * val_split)
        test_size = int(total_samples * test_split)
        train_size = total_samples - val_size - test_size

        # Deterministic generator for reproducibility
        generator = torch.Generator().manual_seed(42)
        train_indices, val_indices, test_indices = random_split(
            range(total_samples),
            [train_size, val_size, test_size],
            generator=generator
        )

        # Apply specific transforms
        train_subset = datasets.ImageFolder(root=str(base_path), transform=transforms_dict["train"])
        val_subset = datasets.ImageFolder(root=str(base_path), transform=transforms_dict["valid"])
        test_subset = datasets.ImageFolder(root=str(base_path), transform=transforms_dict["test"])

        train_data = Subset(train_subset, train_indices.indices)
        val_data = Subset(val_subset, val_indices.indices)
        test_data = Subset(test_subset, test_indices.indices)

        loaders["train"] = DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )
        loaders["valid"] = DataLoader(
            val_data, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )
        loaders["test"] = DataLoader(
            test_data, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )

        logger.info(f"Split completed: Train={train_size}, Valid={val_size}, Test={test_size}")

    if save_mapping and class_names:
        save_class_mapping(class_names, mapping_path)

    return loaders, class_names


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AgroDetect Dataset Loader Tool")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Dataset root directory")
    parser.add_argument("--verify", action="store_true", help="Verify dataset and print class counts")
    args = parser.parse_args()

    try:
        loaders, classes = get_data_loaders(data_dir=args.data_dir, batch_size=8)
        print("\n" + "="*50)
        print(f"Dataset Verification Successful!")
        print(f"Classes Found: {len(classes)}")
        for idx, cls in enumerate(classes):
            print(f"  [{idx:02d}] {cls}")
        print("="*50 + "\n")
    except Exception as e:
        print(f"\n[!] Verification encountered an issue: {e}")
