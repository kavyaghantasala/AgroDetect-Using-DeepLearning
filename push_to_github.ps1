# AgroDetect - Push to GitHub PowerShell Script
Set-Location -Path $PSScriptRoot

Write-Host "=====================================================================" -ForegroundColor Green
Write-Host "          AgroDetect - Automated GitHub Push Utility" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host "Target: https://github.com/kavyaghantasala/AgroDetect-Using-DeepLearning`n" -ForegroundColor Cyan

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Git is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Git from https://git-scm.com/download/win"
    pause
    exit 1
}

if (-not (Test-Path ".git")) {
    Write-Host "[*] Initializing local Git repository..." -ForegroundColor Yellow
    git init
}

Write-Host "[*] Configuring remote origin..." -ForegroundColor Yellow
git remote remove origin 2>$null
git remote add origin https://github.com/kavyaghantasala/AgroDetect-Using-DeepLearning.git

Write-Host "[*] Staging all files..." -ForegroundColor Yellow
git add .

Write-Host "[*] Creating commit..." -ForegroundColor Yellow
git commit -m "Complete AgroDetect Plant Disease Detection System with ResNet50, FastAPI, and 10 Languages"

Write-Host "[*] Setting branch to main..." -ForegroundColor Yellow
git branch -M main

Write-Host "`n[*] Pushing to GitHub main branch..." -ForegroundColor Cyan
git push -u origin main

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[NOTICE] Push was rejected (repository might already have initial files)." -ForegroundColor Yellow
    $choice = Read-Host "Do you want to force push to overwrite with this complete project? (Y/N)"
    if ($choice -match '^[Yy]') {
        Write-Host "[*] Force pushing to main..." -ForegroundColor Magenta
        git push -u origin main --force
    } else {
        Write-Host "[*] Pulling and rebasing..." -ForegroundColor Yellow
        git pull origin main --rebase --allow-unrelated-histories
        git push -u origin main
    }
}

Write-Host "`n=====================================================================" -ForegroundColor Green
Write-Host "  DONE! Visit your repository:" -ForegroundColor Green
Write-Host "  https://github.com/kavyaghantasala/AgroDetect-Using-DeepLearning" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Green
pause
