"""
AgroDetect - Pydantic Data Schemas
Defines request and response structures for the FastAPI application.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    """Details for an individual predicted class."""
    class_name: str = Field(..., description="Raw class name from dataset, e.g. Tomato___Early_blight")
    display_name: str = Field(..., description="Formatted user-friendly name, e.g. Tomato - Early Blight")
    crop: str = Field(..., description="Plant crop name, e.g. Tomato")
    condition: str = Field(..., description="Plant condition or disease name, e.g. Early Blight")
    confidence: float = Field(..., description="Confidence percentage (0.0 to 100.0)")
    is_healthy: bool = Field(..., description="Whether this class represents a healthy plant")


class DiseaseDetail(BaseModel):
    """Structured disease knowledge and preventive guidance."""
    disease_name: str
    affected_plant: str
    is_healthy: bool
    symptoms: List[str] = Field(default_factory=list, description="Common observable visual symptoms")
    preventive_measures: List[str] = Field(default_factory=list, description="Recommended preventive agricultural practices")
    general_management: List[str] = Field(default_factory=list, description="General guidance and cultural management")
    disclaimer: str = Field(
        default="General guidance only. Confirm with a local agricultural extension officer or agronomist before applying chemical treatments.",
        description="Safety and liability disclaimer"
    )


class PredictionResponse(BaseModel):
    """Response returned by the /predict endpoint."""
    success: bool
    predicted_class: str
    display_name: str
    crop: str
    condition: str
    confidence: float
    is_healthy: bool
    top_predictions: List[TopPrediction]
    disease_info: Optional[DiseaseDetail] = None
    language: str = "en"
    inference_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """Response returned by the /health endpoint."""
    status: str
    model_loaded: bool
    device: str
    total_classes: int
    model_path: str
    architecture: str = "ResNet50"
    version: str = "1.0.0"
    message: Optional[str] = None


class ClassListResponse(BaseModel):
    """Response returned by the /classes endpoint."""
    total_classes: int
    classes: List[str]
    crops: List[str]


class ErrorResponse(BaseModel):
    """Standardized error response."""
    detail: str
    error_code: str
    hint: Optional[str] = None
