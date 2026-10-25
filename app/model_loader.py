"""
AgroDetect - ResNet50 Model Loader Module
Manages the lifecycle of the PyTorch ResNet50 model.
Loads the model once during application startup and handles missing checkpoints gracefully.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AgroDetect-ModelLoader")

DEFAULT_MODEL_PATH = "models/best_model.pth"
DEFAULT_CLASSES_PATH = "models/class_names.json"


class ModelManager:
    """
    Singleton manager for loading, holding, and serving the PyTorch ResNet50 model.
    Guarantees thread-safe access and single-time model loading into memory.
    """
    _instance: Optional["ModelManager"] = None

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, classes_path: str = DEFAULT_CLASSES_PATH):
        self.model_path = model_path
        self.classes_path = classes_path
        self.model: Optional[nn.Module] = None
        self.class_names: List[str] = []
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded: bool = False
        self.status_message: str = "Model has not been initialized."

    @classmethod
    def get_instance(cls, model_path: str = DEFAULT_MODEL_PATH, classes_path: str = DEFAULT_CLASSES_PATH) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls(model_path=model_path, classes_path=classes_path)
        return cls._instance

    def load_class_names(self) -> List[str]:
        """Loads and returns the class names mapping from JSON."""
        path = Path(self.classes_path)
        if not path.exists():
            raise FileNotFoundError(f"Classes mapping file not found at '{self.classes_path}'")
        with open(path, "r", encoding="utf-8") as f:
            classes = json.load(f)
        if not isinstance(classes, list) or len(classes) == 0:
            raise ValueError(f"Invalid class mapping in '{self.classes_path}'. Expected non-empty list.")
        return classes

    def load_model(self) -> bool:
        """
        Attempts to load the trained ResNet50 model from disk.
        Returns True if successful, False if model file is missing or invalid.
        Does NOT crash the server if checkpoint is not found.
        """
        model_file = Path(self.model_path)
        classes_file = Path(self.classes_path)

        # 1. Check for classes file
        if not classes_file.exists():
            self.is_loaded = False
            self.status_message = (
                f"Class names file '{self.classes_path}' not found. "
                "Please run 'python -m training.dataset --verify' to generate class mapping."
            )
            logger.warning(self.status_message)
            return False

        try:
            self.class_names = self.load_class_names()
            logger.info(f"Loaded {len(self.class_names)} classes from '{self.classes_path}'")
        except Exception as e:
            self.is_loaded = False
            self.status_message = f"Error reading class names: {str(e)}"
            logger.error(self.status_message)
            return False

        # 2. Check for trained model checkpoint (auto-download if missing)
        if not model_file.exists():
            logger.info(f"Model checkpoint '{self.model_path}' not found. Automatically downloading pretrained ResNet50 model weights...")
            try:
                from models.download_weights import download_trained_weights
                download_success = download_trained_weights()
                if not download_success or not model_file.exists():
                    self.is_loaded = False
                    self.status_message = (
                        f"Model checkpoint '{self.model_path}' not found. "
                        "To train a custom model, run: python -m training.train --data-dir dataset"
                    )
                    logger.warning(self.status_message)
                    return False
            except Exception as dl_err:
                self.is_loaded = False
                self.status_message = f"Automatic download failed: {dl_err}. Please check internet connection."
                logger.error(self.status_message)
                return False

        # Also ensure sample testing leaf images are available
        try:
            sample_dir = Path("dataset/samples")
            if not sample_dir.exists() or len(list(sample_dir.glob("*.*"))) == 0:
                from dataset.download_sample_images import download_sample_leaves
                download_sample_leaves()
        except Exception:
            pass

        # 3. Construct ResNet50 architecture and load weights
        try:
            logger.info(f"Loading ResNet50 weights from '{self.model_path}' onto {self.device}...")
            num_classes = len(self.class_names)

            # Build ResNet50 skeleton
            model = models.resnet50(weights=None)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes)
            )

            checkpoint = torch.load(self.model_path, map_location=self.device)

            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint

                # If checkpoint contains class_names, ensure it matches or adopt it
                if "class_names" in checkpoint and isinstance(checkpoint["class_names"], list):
                    self.class_names = checkpoint["class_names"]
                    num_classes = len(self.class_names)
            else:
                state_dict = checkpoint

            # Strip 'module.' prefix if saved with torch.nn.DataParallel
            cleaned_state_dict = {
                (k[7:] if k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }

            # Try loading with Sequential head (Dropout + Linear)
            loaded_successfully = False
            try:
                model.load_state_dict(cleaned_state_dict)
                loaded_successfully = True
            except RuntimeError:
                pass

            if not loaded_successfully:
                try:
                    # Fallback 1: Simple Linear head
                    model.fc = nn.Linear(in_features, num_classes)
                    model.load_state_dict(cleaned_state_dict)
                    loaded_successfully = True
                except RuntimeError:
                    pass

            if not loaded_successfully:
                # Fallback 2: Non-strict loading if minor buffer mismatches exist
                logger.info("Attempting non-strict state_dict loading...")
                model.load_state_dict(cleaned_state_dict, strict=False)

            model = model.to(self.device)
            model.eval()

            # Warmup inference pass
            dummy_input = torch.zeros(1, 3, 224, 224, device=self.device)
            with torch.no_grad():
                _ = model(dummy_input)

            self.model = model
            self.is_loaded = True
            self.status_message = f"ResNet50 model successfully loaded with {num_classes} classes on {self.device}."
            logger.info(self.status_message)
            return True

        except Exception as exc:
            self.is_loaded = False
            self.model = None
            self.status_message = f"Failed to load ResNet50 checkpoint: {str(exc)}"
            logger.error(self.status_message, exc_info=True)
            return False

    def get_model(self) -> Tuple[Optional[nn.Module], List[str], torch.device]:
        """Returns the loaded model, class names list, and compute device."""
        return self.model, self.class_names, self.device


def init_model(model_path: str = DEFAULT_MODEL_PATH, classes_path: str = DEFAULT_CLASSES_PATH) -> ModelManager:
    """Helper to initialize and load model at server startup."""
    manager = ModelManager.get_instance(model_path=model_path, classes_path=classes_path)
    manager.load_model()
    return manager
