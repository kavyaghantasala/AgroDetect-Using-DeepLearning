"""
AgroDetect - FastAPI Integration Tests
Tests endpoints, health checks, error handling, multilingual queries, and prediction workflow.
"""

import io
import pytest
from PIL import Image
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from app.main import app
from app.model_loader import ModelManager


@pytest.fixture(scope="module")
def client():
    """Provides FastAPI test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_image_bytes():
    """Creates a sample RGB leaf image bytes in memory."""
    img = Image.new("RGB", (300, 300), color=(60, 160, 70))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestEndpoints:

    def test_homepage_loads(self, client):
        """Tests that GET / serves the HTML homepage with 200 OK."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "AgroDetect" in response.text

    def test_health_check_endpoint(self, client):
        """Tests that GET /health returns 200 and system readiness data."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "architecture" in data
        assert data["architecture"] == "ResNet50"

    def test_classes_endpoint(self, client):
        """Tests that GET /classes returns supported plant disease classes."""
        response = client.get("/classes")
        assert response.status_code == 200
        data = response.json()
        assert "total_classes" in data
        assert "classes" in data
        assert "crops" in data
        assert isinstance(data["classes"], list)
        assert data["total_classes"] >= 10  # Standard PlantVillage has 38 classes

    def test_get_disease_info_valid(self, client):
        """Tests that GET /disease/{class_name} returns disease information."""
        response = client.get("/disease/Tomato___Early_blight")
        assert response.status_code == 200
        data = response.json()
        assert data["disease_name"] != ""
        assert data["affected_plant"] != ""
        assert len(data["symptoms"]) > 0
        assert len(data["preventive_measures"]) > 0
        assert "disclaimer" in data

    def test_get_disease_info_multilingual(self, client):
        """Tests disease information localization with query parameter ?lang=hi."""
        response = client.get("/disease/Tomato___Early_blight?lang=hi")
        assert response.status_code == 200
        data = response.json()
        # Verify Hindi disease name
        assert "टमाटर" in data["disease_name"] or "झुलसा" in data["disease_name"]

    def test_get_disease_info_invalid_class_returns_404(self, client):
        """Tests that requesting an unknown class returns 404 Not Found."""
        response = client.get("/disease/NonExistentPlant___FakeDisease")
        assert response.status_code == 404

    def test_predict_without_model_returns_503(self, client, sample_image_bytes):
        """
        When best_model.pth is not loaded, /predict must return 503 Service Unavailable
        with a clear error explanation, without crashing.
        """
        manager = ModelManager.get_instance()
        original_loaded = manager.is_loaded
        original_model = manager.model

        try:
            # Force model to be unloaded
            manager.is_loaded = False
            manager.model = None

            files = {"file": ("leaf.jpg", sample_image_bytes, "image/jpeg")}
            response = client.post("/predict", files=files)

            assert response.status_code == 503
            data = response.json()
            assert data["error_code"] == "MODEL_NOT_LOADED"
            assert "training" in data["hint"].lower()
        finally:
            manager.is_loaded = original_loaded
            manager.model = original_model

    def test_predict_with_invalid_extension_returns_400(self, client):
        """Tests that uploading an invalid file format (e.g. .txt) returns 400 Bad Request."""
        files = {"file": ("test.txt", b"plain text content", "text/plain")}
        response = client.post("/predict", files=files)
        assert response.status_code == 400
        assert "Invalid file extension" in response.json()["detail"]

    def test_predict_with_corrupted_image_returns_400(self, client):
        """Tests that uploading corrupted image bytes returns 400 Bad Request."""
        manager = ModelManager.get_instance()
        original_loaded = manager.is_loaded
        original_model = manager.model

        try:
            # Temporarily simulate loaded model
            manager.is_loaded = True
            manager.model = nn.Identity()

            files = {"file": ("damaged.jpg", b"\xFF\xD8\xFFcorrupted garbage", "image/jpeg")}
            response = client.post("/predict", files=files)
            assert response.status_code == 400
            assert "not a valid or readable image" in response.json()["detail"]
        finally:
            manager.is_loaded = original_loaded
            manager.model = original_model

    def test_predict_with_active_model_produces_prediction(self, client, sample_image_bytes):
        """
        Tests that when an initialized model is present,
        /predict produces a full prediction with confidence, top predictions, and disease info.
        """
        manager = ModelManager.get_instance()
        original_loaded = manager.is_loaded
        original_model = manager.model
        original_classes = manager.class_names

        try:
            # Create a mock classification model for integration testing
            test_classes = ["Tomato___Early_blight", "Tomato___healthy", "Potato___Early_blight"]
            
            class MockClassifier(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Simple linear layer to project [1, 3, 224, 224] to 3 classes
                    self.pool = nn.AdaptiveAvgPool2d((1, 1))
                    self.fc = nn.Linear(3, 3)
                    # Fixed weights to make Tomato___Early_blight highest
                    with torch.no_grad():
                        self.fc.weight.fill_(0.1)
                        self.fc.bias.copy_(torch.tensor([5.0, 1.0, 0.5]))

                def forward(self, x):
                    x = self.pool(x).view(x.size(0), -1)
                    return self.fc(x)

            manager.is_loaded = True
            manager.model = MockClassifier()
            manager.class_names = test_classes
            manager.device = torch.device("cpu")

            files = {"file": ("leaf.jpg", sample_image_bytes, "image/jpeg")}
            response = client.post("/predict?lang=en", files=files)

            assert response.status_code == 200
            res = response.json()
            assert res["success"] is True
            assert res["predicted_class"] == "Tomato___Early_blight"
            assert res["crop"] == "Tomato"
            assert res["condition"] == "Early Blight"
            assert res["confidence"] > 50.0
            assert res["disease_info"] is not None
            assert len(res["disease_info"]["symptoms"]) > 0

            # Test Telugu localized prediction
            response_te = client.post("/predict?lang=te", files={"file": ("leaf.jpg", sample_image_bytes, "image/jpeg")})
            assert response_te.status_code == 200
            res_te = response_te.json()
            assert "టమోటా" in res_te["crop"]
            assert len(res_te["top_predictions"]) == 3
            assert "టమోటా" in res_te["top_predictions"][0]["display_name"]

        finally:
            manager.is_loaded = original_loaded
            manager.model = original_model
            manager.class_names = original_classes
