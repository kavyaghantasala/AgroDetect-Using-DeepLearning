"""
AgroDetect - FastAPI Web Application
Provides RESTful APIs for plant disease detection and serves the interactive frontend.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

from app.model_loader import ModelManager, init_model
from app.predictor import (
    predict_disease,
    ModelNotLoadedError,
    DiseaseKnowledgeBase,
    format_class_name
)
from app.preprocessing import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES
from app.schemas import (
    HealthResponse,
    PredictionResponse,
    ClassListResponse,
    DiseaseDetail,
    ErrorResponse
)

from app.translations_data import SUPPORTED_LANGUAGES


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager:
    Initializes the model on application startup once.
    """
    print("\n" + "=" * 60)
    print("  AgroDetect - Plant Disease Detection System Initializing")
    print("=" * 60)
    manager = init_model()
    if manager.is_loaded:
        print(f"[OK] Model loaded successfully: {len(manager.class_names)} classes on {manager.device}.")
    else:
        print(f"[NOTICE] {manager.status_message}")
    print("=" * 60 + "\n")
    yield
    print("[INFO] AgroDetect application shutting down.")


app = FastAPI(
    title="AgroDetect API",
    description=(
        "Plant Disease Detection System using Deep Learning (ResNet50). "
        "Allows uploading plant leaf images for automated disease classification, "
        "confidence estimation, symptoms analysis, and preventive agricultural recommendations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for cross-origin frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------
@app.exception_handler(ModelNotLoadedError)
async def handle_model_not_loaded(request: Request, exc: ModelNotLoadedError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": str(exc),
            "error_code": "MODEL_NOT_LOADED",
            "hint": (
                "To train the model on your dataset, run: "
                "python -m training.train --data-dir dataset --epochs 15"
            )
        }
    )


@app.exception_handler(ValueError)
async def handle_value_error(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": str(exc),
            "error_code": "INVALID_INPUT",
            "hint": "Please upload a valid leaf image in JPG, JPEG, or PNG format under 10 MB."
        }
    )


# ---------------------------------------------------------
# Web Routes
# ---------------------------------------------------------
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_path = STATIC_DIR / "images" / "logo.svg"
    if favicon_path.exists():
        return FileResponse(str(favicon_path), media_type="image/svg+xml")
    return JSONResponse(status_code=204, content={})


@app.get("/", response_class=HTMLResponse, summary="Serve Homepage")
async def index_page(request: Request):
    """Serves the AgroDetect interactive web interface."""
    manager = ModelManager.get_instance()
    context = {
        "request": request,
        "model_loaded": manager.is_loaded,
        "status_message": manager.status_message,
        "languages": SUPPORTED_LANGUAGES,
        "classes_count": len(manager.class_names)
    }
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=context
    )


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and Readiness Check",
    tags=["System"]
)
async def health_check():
    """
    Checks whether the application and PyTorch model are ready for predictions.
    Does not crash if model is not loaded; reports status transparently.
    """
    manager = ModelManager.get_instance()
    return HealthResponse(
        status="healthy" if manager.is_loaded else "setup_mode",
        model_loaded=manager.is_loaded,
        device=str(manager.device),
        total_classes=len(manager.class_names),
        model_path=manager.model_path,
        architecture="ResNet50",
        version="1.0.0",
        message=manager.status_message
    )


@app.get(
    "/classes",
    response_model=ClassListResponse,
    summary="Supported Disease Classes",
    tags=["Classes"]
)
async def get_supported_classes():
    """Returns the list of plant disease classes currently supported by the model."""
    manager = ModelManager.get_instance()
    classes = manager.class_names
    if not classes:
        # Fallback to loading from disk if model has not been initialized
        try:
            classes = manager.load_class_names()
        except Exception:
            classes = []

    crops = sorted(list(set(format_class_name(c)[1] for c in classes))) if classes else []

    return ClassListResponse(
        total_classes=len(classes),
        classes=classes,
        crops=crops
    )


@app.get(
    "/disease/{class_name}",
    response_model=DiseaseDetail,
    summary="Get Disease Information",
    tags=["Disease Info"],
    responses={
        404: {"model": ErrorResponse, "description": "Disease class not found"}
    }
)
async def get_disease_info(
    class_name: str,
    lang: str = Query("en", description="Language code: en, hi, ta, te")
):
    """
    Retrieves symptoms, preventive measures, and general management guidance
    for a specific plant disease class.
    """
    detail = DiseaseKnowledgeBase.get_details(class_name, lang=lang)
    if not detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Information for disease class '{class_name}' is not available."
        )
    return detail


@app.get(
    "/languages",
    summary="Supported Languages",
    tags=["Localization"]
)
async def get_languages():
    """Returns supported languages for UI and disease advisories."""
    return SUPPORTED_LANGUAGES


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Plant Disease from Leaf Image",
    tags=["Inference"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image file"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    }
)
async def predict(
    file: UploadFile = File(..., description="Plant leaf image (JPG, JPEG, or PNG)"),
    lang: str = Query("en", description="Language code: en (English), hi (Hindi), ta (Tamil), te (Telugu)")
):
    """
    Accepts an uploaded plant leaf image:
    1. Validates file extension, MIME type, and file size.
    2. Decodes and preprocesses image with OpenCV and Pillow.
    3. Runs transfer-learned ResNet50 inference.
    4. Returns predicted disease, confidence percentage, top-3 predictions,
       symptoms, and actionable preventive recommendations in the requested language.
    """
    # 1. Validate file extension
    filename = file.filename or "uploaded_leaf.jpg"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension '{ext}'. Only JPG, JPEG, and PNG images are supported."
        )

    # 2. Read bytes
    try:
        contents = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file contents: {str(exc)}"
        )

    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty. Please select a valid leaf image."
        )

    if len(contents) > MAX_FILE_SIZE_BYTES:
        size_mb = len(contents) / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded image ({size_mb:.1f} MB) exceeds maximum limit of 10 MB."
        )

    # 3. Predict disease
    try:
        result = predict_disease(
            image_input=contents,
            lang=lang,
            top_k=3,
            filename=filename
        )
        return result
    except ModelNotLoadedError as mne:
        raise mne
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during image processing: {str(exc)}"
        )
