# 🚀 AgroDetect Deployment Guide

This guide details **3 straightforward ways** to deploy your **AgroDetect Plant Disease Detection System** to the cloud or get an instant public URL.

---

## 🌟 Method 1: Hugging Face Spaces (Recommended — 100% Free with 16 GB RAM)

> **Why Hugging Face?**
> Deep learning models (PyTorch + ResNet50) need around 1 GB of memory. Most free tiers (Render/Railway) only offer 512 MB and crash with Out-Of-Memory errors. **Hugging Face Spaces gives you 16 GB RAM and 2 vCPUs completely free with no credit card required.**

### Step-by-Step Instructions:

1. **Create an Account:**
   - Sign up for free at [huggingface.co](https://huggingface.co).

2. **Create a New Space:**
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space).
   - **Space Name:** `agrodetect` (or your preferred name)
   - **License:** `mit` or `openrail`
   - **Select the Space SDK:** Choose **Docker** 🐳
   - **Choose a Docker template:** Choose **Blank**
   - **Space Hardware:** Select **Free (CPU basic · 2 vCPU · 16 GB RAM)**
   - Click **Create Space**.

3. **Upload Your Project Files:**
   *You can upload via Web UI or via Git:*
   
   **Via Web Browser (Easiest):**
   - In your newly created Space, click the **Files** tab.
   - Click **Add file** ➔ **Upload files**.
   - Drag and drop the following files & folders from `AgroDetect`:
     - `app/` (all files & subfolders)
     - `models/` (`download_weights.py` and `class_names.json`)
     - `Dockerfile`
     - `.dockerignore`
     - `requirements.txt`
     - `README.md`
   - Scroll to the bottom and click **Commit changes to main**.

   **Via Git CLI:**
   ```bash
   git init
   git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/agrodetect
   git add .
   git commit -m "Deploy AgroDetect"
   git push -u origin main
   ```

4. **Automatic Build & Launch:**
   - Hugging Face automatically detects `Dockerfile`, installs CPU PyTorch, downloads the model checkpoint, and boots FastAPI on port `7860`.
   - In 2–3 minutes, your live public URL will be ready at:  
     `https://YOUR_USERNAME-agrodetect.hf.space`

---

## 🌐 Method 2: Render.com (Free Web Service)

1. Push your AgroDetect code to your **GitHub** account.
2. Sign up / Log in to [render.com](https://render.com).
3. Click **New +** ➔ **Web Service**.
4. Connect your GitHub repository.
5. In settings:
   - **Environment:** `Docker` (Render will automatically read `render.yaml` and `Dockerfile`)
   - **Instance Type:** `Free`
6. Click **Create Web Service**.
7. Render will build the container and provide a live URL like:  
   `https://agrodetect-xxxx.onrender.com`

---

## ⚡ Method 3: Instant Public URL from Your Laptop (Zero Setup)

If you need a live public HTTPS URL **right now** for viva, college demo, or mobile testing without uploading to GitHub:

### Using Cloudflare Tunnel (Free & No Account Needed):
1. Make sure your local AgroDetect app is running:
   ```cmd
   run.bat
   ```
2. Open a second terminal and run:
   ```cmd
   winget install --id Cloudflare.cloudflared -e
   cloudflared tunnel --url http://127.0.0.1:8000
   ```
3. Cloudflare will output a public HTTPS link (e.g. `https://random-words-xyz.trycloudflare.com`). Anyone can open this link on their smartphone or laptop anywhere in the world!

### Or Using Localtunnel (Requires Node.js):
```cmd
npx localtunnel --port 8000
```
