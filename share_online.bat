@echo off
setlocal
title AgroDetect - Public Online Sharing
cd /d "%~dp0"

echo =====================================================================
echo           AgroDetect - Instant Public URL Generator
echo =====================================================================
echo.
echo Make sure AgroDetect is already running via run.bat (on port 8000).
echo.
echo Checking for Cloudflare Tunnel (cloudflared)...

where cloudflared >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Cloudflared found! Creating a secure public HTTPS link...
    echo Share the resulting https://....trycloudflare.com URL with anyone.
    echo.
    cloudflared tunnel --url http://127.0.0.1:8000
    goto end
)

where npx >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Using localtunnel via npx...
    echo.
    npx localtunnel --port 8000
    goto end
)

echo [NOTICE] Neither 'cloudflared' nor 'npx' was detected on your PATH.
echo.
echo To get an instant public URL directly from your laptop:
echo   Run this in PowerShell:
echo   winget install --id Cloudflare.cloudflared -e
echo.
echo Or read DEPLOYMENT.md for step-by-step instructions to deploy
echo permanently for free on Hugging Face Spaces (Recommended) or Render.
echo =====================================================================
echo.
pause
:end
