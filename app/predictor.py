"""
AgroDetect - Leaf Disease Predictor Module
Executes PyTorch inference, computes top predictions, and maps results to disease knowledge.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import torch

from app.model_loader import ModelManager
from app.preprocessing import preprocess_image
from app.schemas import DiseaseDetail, PredictionResponse, TopPrediction

logger = logging.getLogger("AgroDetect-Predictor")

DISEASE_INFO_PATH = "app/disease_info.json"


class ModelNotLoadedError(Exception):
    """Raised when inference is requested but model checkpoint is not loaded."""
    pass


class DiseaseKnowledgeBase:
    """Manages retrieval of structured disease advice in multiple languages."""
    _info_cache: Optional[Dict] = None

    @classmethod
    def load_info(cls, path: str = DISEASE_INFO_PATH) -> Dict:
        if cls._info_cache is None:
            info_file = Path(path)
            if info_file.exists():
                try:
                    with open(info_file, "r", encoding="utf-8") as f:
                        cls._info_cache = json.load(f)
                except Exception as exc:
                    logger.error(f"Failed to read disease_info.json: {exc}")
                    cls._info_cache = {}
            else:
                logger.warning(f"Disease info file '{path}' not found.")
                cls._info_cache = {}
        return cls._info_cache

    @classmethod
    def get_details(cls, class_name: str, lang: str = "en") -> Optional[DiseaseDetail]:
        """Retrieves localized disease details for a given class label."""
        data = cls.load_info()
        entry = data.get(class_name)
        if not entry:
            return None

        crop = entry.get("crop", "Unknown Plant")
        condition = entry.get("condition", class_name)
        is_healthy = entry.get("is_healthy", "healthy" in class_name.lower())

        translations = entry.get("translations", {})
        # Support requested lang or fallback to English
        trans = translations.get(lang) or translations.get("en", {})

        return DiseaseDetail(
            disease_name=trans.get("disease_name", f"{crop} {condition}"),
            affected_plant=trans.get("affected_plant", crop),
            is_healthy=is_healthy,
            symptoms=trans.get("symptoms", []),
            preventive_measures=trans.get("preventive_measures", []),
            general_management=trans.get("general_management", [])
        )


from app.translations_data import get_localized_disease_name


def format_class_name(class_name: str, lang: str = "en") -> Tuple[str, str, str, bool]:
    """
    Parses raw class names like 'Tomato___Early_blight' into (display_name, crop, condition, is_healthy)
    localized in the requested language.
    """
    return get_localized_disease_name(class_name, lang=lang)


def predict_disease(
    image_input: Union[bytes, Image.Image],
    lang: str = "en",
    top_k: int = 3,
    filename: str = ""
) -> PredictionResponse:
    """
    Primary prediction entry point:
    1. Verifies model readiness
    2. Preprocesses uploaded image (OpenCV/Pillow)
    3. Runs PyTorch forward pass on ResNet50
    4. Computes softmax probabilities
    5. Retrieves localized disease information in the requested language
    """
    manager = ModelManager.get_instance()
    if not manager.is_loaded or manager.model is None:
        raise ModelNotLoadedError(
            "Trained model checkpoint is not loaded. "
            "Please train the model first by running 'python -m training.train --data-dir dataset' "
            "or place 'best_model.pth' in the models/ directory."
        )

    model = manager.model
    class_names = manager.class_names
    device = manager.device

    # 1. Preprocess image
    start_time = time.time()
    tensor = preprocess_image(image_input, filename=filename)
    tensor = tensor.to(device)

    # 2. ResNet50 Forward inference
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    inference_time = (time.time() - start_time) * 1000.0  # in ms

    # 3. Extract top-k predictions localized in the selected language
    k = min(top_k, len(class_names))
    top_probs, top_indices = torch.topk(probabilities, k=k)

    top_predictions: List[TopPrediction] = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        cls_name = class_names[idx]
        d_name, crop, cond, is_h = get_localized_disease_name(cls_name, lang=lang)
        top_predictions.append(
            TopPrediction(
                class_name=cls_name,
                display_name=d_name,
                crop=crop,
                condition=cond,
                confidence=round(prob * 100.0, 2),
                is_healthy=is_h
            )
        )

    # Primary (top-1) prediction
    primary = top_predictions[0]

    # 4. Fetch localized disease advice
    disease_info = DiseaseKnowledgeBase.get_details(primary.class_name, lang=lang)

    return PredictionResponse(
        success=True,
        predicted_class=primary.class_name,
        display_name=primary.display_name,
        crop=primary.crop,
        condition=primary.condition,
        confidence=primary.confidence,
        is_healthy=primary.is_healthy,
        top_predictions=top_predictions,
        disease_info=disease_info,
        language=lang,
        inference_time_ms=round(inference_time, 2)
    )
