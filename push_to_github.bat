@echo off
setlocal enabledelayedexpansion
title AgroDetect - Push to GitHub
cd /d "%~dp0"

echo =====================================================================
echo           AgroDetect - Automated GitHub Push Utility
echo =====================================================================
echo Target: https://github.com/kavyaghantasala/AgroDetect-Using-DeepLearning
echo =====================================================================
echo.

where git >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not found in your system PATH!
    echo Please install Git from https://git-scm.com/download/win and restart this script.
    echo.
    pause
    exit /b 1
)

echo [*] Checking local Git repository...
if not exist ".git" (
    echo [*] Running git init...
    git init
)

echo [*] Configuring remote origin to:
echo     https://github.com/kavyaghantasala/AgroDetect-Using-DeepLearning.git
git remote remove origin >nul 2>&1
git remote add origin https://github.com/kavyaghantasala/AgroDetect-Using-DeepLearning.git

echo [*] Staging all project files...
git add .

echo [*] Creating commit...
git commit -m "Complete AgroDetect Plant Disease Detection System with ResNet50, FastAPI, and 10 Languages"

echo [*] Setting primary branch to main...
git branch -M main

echo.
echo =====================================================================
echo Pushing files to GitHub main branch...
echo (If a GitHub login or browser window opens, please sign in to authorize)
echo =====================================================================
echo.

git push -u origin main

if %errorlevel% neq 0 (
    echo.
    echo [NOTICE] If the remote repository already contains a README or initial commit,
    echo pushing normally can be rejected.
    echo.
    set /p FORCE_CHOICE="Do you want to force push your complete AgroDetect code to main? [Y/N]: "
    if /i "!FORCE_CHOICE!"=="Y" (
        echo [*] Force pushing to main branch...
        git push -u origin main --force
    ) else (
        echo [*] Attempting pull with rebase then pushing...
        git pull origin main --rebase --allow-unrelated-histories
        git push -u origin main
    )
)

echo.
echo =====================================================================
echo   DONE! Check your repository online at:
echo   https://github.com/kavyaghantasala/AgroDetect-Using-DeepLearning
echo =====================================================================
echo.
pause
