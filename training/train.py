"""
AgroDetect - ResNet50 Model Training Pipeline
Performs transfer learning with ResNet50 for plant disease classification.
Tracks metrics, saves the best checkpoint as best_model.pth, and plots training curves.
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Import dataset module
from training.dataset import get_data_loaders, save_class_mapping

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AgroDetect-Train")


def build_resnet50_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Constructs a ResNet50 model with pretrained weights and replaces
    the final classification layer to match num_classes.
    """
    logger.info(f"Building ResNet50 architecture for {num_classes} classes (pretrained={pretrained})...")
    if pretrained:
        try:
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        except Exception:
            # Fallback for older torchvision versions
            model = models.resnet50(pretrained=True)
    else:
        model = models.resnet50(weights=None)

    # Replace final linear layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


def plot_training_curves(history: Dict[str, List[float]], output_path: str = "outputs/training_curves.png") -> None:
    """Plots and saves loss and accuracy curves over epochs."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#e74c3c", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Validation Loss", color="#3498db", linewidth=2, linestyle="--")
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=11)

    # Accuracy plot
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc (%)", color="#27ae60", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Validation Acc (%)", color="#9b59b6", linewidth=2, linestyle="--")
    ax2.set_title("Classification Accuracy vs. Epochs", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved training curves plot to: {output_path}")


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Runs a validation epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = correct / total if total > 0 else 0.0
    return val_loss, val_acc


def train_model(
    data_dir: str = "dataset",
    model_save_path: str = "models/best_model.pth",
    mapping_save_path: str = "models/class_names.json",
    outputs_dir: str = "outputs",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    freeze_backbone: bool = False,
    device_name: Optional[str] = None
) -> None:
    """
    Full training orchestration routine:
    1. Loads dataset and classes
    2. Builds ResNet50
    3. Runs training and validation epochs
    4. Saves best_model.pth based on validation accuracy
    5. Saves training curves and JSON metrics
    """
    # 1. Device configuration
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target compute device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 2. Data loaders
    logger.info(f"Loading dataset from: {data_dir} ...")
    loaders, class_names = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        save_mapping=True,
        mapping_path=mapping_save_path
    )
    num_classes = len(class_names)
    logger.info(f"Configured training pipeline for {num_classes} plant disease classes.")

    # 3. Model construction
    model = build_resnet50_model(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    # Optionally freeze earlier feature extraction layers
    if freeze_backbone:
        logger.info("Freezing ResNet50 convolutional backbone (training fc layer only)...")
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    # 4. Optimization setup
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    best_val_acc = 0.0
    start_time = time.time()

    save_dir = Path(model_save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Starting ResNet50 training loop...")
    print("\n" + "=" * 70)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<12} | {'Val Loss':<12} | {'Val Acc':<12} | {'Time':<8}")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, loaders["valid"], criterion, device)

        # Update scheduler based on validation accuracy
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        elapsed = time.time() - epoch_start
        print(f"{epoch:<8} | {train_loss:<12.4f} | {train_acc * 100:<11.2f}% | {val_loss:<12.4f} | {val_acc * 100:<11.2f}% | {elapsed:<6.1f}s")

        # Checkpoint if best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.info(f"[*] New best validation accuracy: {best_val_acc * 100:.2f}%. Saving checkpoint to {model_save_path} ...")
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_names": class_names,
                "architecture": "resnet50"
            }
            torch.save(checkpoint, model_save_path)

    total_time = time.time() - start_time
    print("=" * 70)
    logger.info(f"Training completed in {total_time / 60:.2f} minutes.")
    logger.info(f"Highest Validation Accuracy: {best_val_acc * 100:.2f}%")

    # 5. Save training curves
    plot_training_curves(history, output_path=os.path.join(outputs_dir, "training_curves.png"))

    # 6. Save training metrics JSON
    metrics_path = os.path.join(outputs_dir, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "epochs": epochs,
            "batch_size": batch_size,
            "initial_lr": lr,
            "device": str(device),
            "best_val_accuracy": best_val_acc,
            "total_time_seconds": total_time,
            "history": history
        }, f, indent=2)
    logger.info(f"Saved training history to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AgroDetect ResNet50 Model")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Dataset directory path")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze initial ResNet50 layers")
    parser.add_argument("--model-save-path", type=str, default="models/best_model.pth", help="Checkpoint save path")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda' or 'cpu')")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Output directory for plots/logs")

    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        model_save_path=args.model_save_path,
        outputs_dir=args.outputs_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_backbone=args.freeze_backbone,
        device_name=args.device
    )
