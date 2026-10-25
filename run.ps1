# AgroDetect - PowerShell Launcher
Set-Location -Path $PSScriptRoot

Write-Host "=====================================================================" -ForegroundColor Green
Write-Host "               AgroDetect - Plant Disease Detection System           " -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green

# 1. Create venv if not present
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "[INFO] Creating virtual environment in .venv ..." -ForegroundColor Yellow
    python -m venv .venv
}

# 2. Activate venv
Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

# 3. Check requirements
python -c "import fastapi, torch, torchvision, PIL, cv2" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[INFO] Installing dependencies from requirements.txt ..." -ForegroundColor Yellow
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
}

# 4. Check model weights
if (-not (Test-Path "models\best_model.pth")) {
    Write-Host "[INFO] Downloading pretrained ResNet50 model weights..." -ForegroundColor Cyan
    python models\download_weights.py
    if (-not (Test-Path "dataset\samples")) {
        python dataset\download_sample_images.py
    }
}

Write-Host "`nStarting AgroDetect on http://127.0.0.1:8000 ...`n" -ForegroundColor Green
Start-Process "http://127.0.0.1:8000"

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
