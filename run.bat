@echo off
setlocal enabledelayedexpansion

:: Set working directory to the batch script folder
cd /d "%~dp0"

echo =====================================================================
echo                AgroDetect - Plant Disease Detection System
echo =====================================================================
echo.

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not found in system PATH.
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

:: 2. Create virtual environment if it does not exist
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Creating Python virtual environment in .venv ...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [INFO] Virtual environment created successfully.
)

:: 3. Activate the virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

:: 4. Check if required packages are installed
python -c "import fastapi, torch, torchvision, PIL, cv2" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing required dependencies from requirements.txt ...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
)

:: 5. Automatically download model weights if not present
if not exist "models\best_model.pth" (
    echo [INFO] models\best_model.pth was not found.
    echo [INFO] Automatically downloading pretrained ResNet50 model weights...
    python models\download_weights.py
    if not exist "dataset\samples" (
        python dataset\download_sample_images.py
    )
) else (
    echo [INFO] Model checkpoint detected: models\best_model.pth
)

echo.
echo =====================================================================
echo   Starting AgroDetect Web Application...
echo   URL: http://127.0.0.1:8000
echo   API Documentation: http://127.0.0.1:8000/docs
echo   Press Ctrl+C to terminate the server.
echo =====================================================================
echo.

:: Open browser automatically
start "" http://127.0.0.1:8000

:: Start Uvicorn server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

if %errorlevel% neq 0 (
    echo.
    echo [NOTICE] If Uvicorn stopped, port 8000 may already be in use by another running window.
    echo Please close any other running python or AgroDetect windows.
)

pause
