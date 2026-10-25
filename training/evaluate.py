"""
AgroDetect - Model Evaluation Module
Evaluates trained ResNet50 checkpoint on the test dataset.
Generates classification report, confusion matrix plot, accuracy, precision, recall, and F1-score.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from training.dataset import get_data_loaders, load_class_mapping
from training.train import build_resnet50_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AgroDetect-Eval")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str = "outputs/confusion_matrix.png"
) -> None:
    """Plots and saves a formatted confusion matrix heatmap."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    num_classes = len(class_names)

    # Scale figure size dynamically with number of classes
    fig_size = max(10, min(24, num_classes * 0.55))
    plt.figure(figsize=(fig_size, fig_size))

    # Shorten class names for readability if too long
    clean_labels = [c.replace("___", " - ").replace("_", " ") for c in class_names]

    sns.heatmap(
        cm,
        annot=num_classes <= 20,  # only annotate numbers if class count is manageable
        fmt="d",
        cmap="Greens",
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        cbar=True
    )
    plt.title("AgroDetect ResNet50 - Test Confusion Matrix", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=12, fontweight="bold")
    plt.ylabel("Actual True Class", fontsize=12, fontweight="bold")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to: {output_path}")


def evaluate_model(
    model_path: str = "models/best_model.pth",
    mapping_path: str = "models/class_names.json",
    data_dir: str = "dataset",
    outputs_dir: str = "outputs",
    batch_size: int = 32,
    device_name: str = None
) -> Dict:
    """
    Evaluates the saved model checkpoint on the test set.
    """
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Checkpoint file '{model_path}' not found! Please train a model first with training/train.py"
        )

    # 1. Device configuration
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    # 2. Load classes and data
    class_names = load_class_mapping(mapping_path)
    num_classes = len(class_names)
    logger.info(f"Loaded {num_classes} classes from {mapping_path}")

    loaders, _ = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        save_mapping=False
    )
    test_loader = loaders.get("test", loaders.get("valid"))

    # 3. Build model architecture & load checkpoint weights
    model = build_resnet50_model(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    logger.info("Model weights loaded successfully into ResNet50.")

    # 4. Run inference over test set
    all_preds = []
    all_targets = []
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    total = 0

    logger.info("Running evaluation on test samples...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            total += targets.size(0)

    test_loss = running_loss / total if total > 0 else 0.0
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 5. Compute performance metrics
    accuracy = (all_preds == all_targets).mean()
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )

    report_str = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    report_dict = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(all_targets, all_preds)

    # 6. Print summary
    print("\n" + "=" * 65)
    print("           AgroDetect ResNet50 Evaluation Results")
    print("=" * 65)
    print(f"Total Test Samples:   {total}")
    print(f"Test Cross-Entropy:   {test_loss:.4f}")
    print(f"Overall Accuracy:     {accuracy * 100:.2f}%")
    print(f"Macro Precision:      {precision_macro * 100:.2f}%")
    print(f"Macro Recall:         {recall_macro * 100:.2f}%")
    print(f"Macro F1-Score:       {f1_macro * 100:.2f}%")
    print(f"Weighted F1-Score:    {f1_weighted * 100:.2f}%")
    print("=" * 65)
    print("\nDetailed Classification Report:\n")
    print(report_str)

    # 7. Save confusion matrix and JSON metrics
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)
    cm_path = os.path.join(outputs_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, output_path=cm_path)

    metrics_path = os.path.join(outputs_dir, "test_metrics.json")
    results = {
        "total_samples": int(total),
        "test_loss": float(test_loss),
        "accuracy": float(accuracy),
        "macro_metrics": {
            "precision": float(precision_macro),
            "recall": float(recall_macro),
            "f1_score": float(f1_macro)
        },
        "weighted_metrics": {
            "precision": float(precision_weighted),
            "recall": float(recall_weighted),
            "f1_score": float(f1_weighted)
        },
        "classification_report": report_dict
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation metrics to: {metrics_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AgroDetect ResNet50 Model")
    parser.add_argument("--model-path", type=str, default="models/best_model.pth", help="Trained model path")
    parser.add_argument("--mapping-path", type=str, default="models/class_names.json", help="Class names mapping")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Dataset root directory")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Outputs directory for plots")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda' or 'cpu')")

    args = parser.parse_args()

    try:
        evaluate_model(
            model_path=args.model_path,
            mapping_path=args.mapping_path,
            data_dir=args.data_dir,
            outputs_dir=args.outputs_dir,
            batch_size=args.batch_size,
            device_name=args.device
        )
    except FileNotFoundError as err:
        logger.error(str(err))
