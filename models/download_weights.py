"""
AgroDetect - Model Weights Downloader
Downloads a trained ResNet50 PlantVillage checkpoint from Hugging Face into models/best_model.pth.
"""

import os
import sys
import urllib.request
from pathlib import Path
import torch

MODEL_URL = "https://huggingface.co/mykkularathne/plant-disease-resnet50/resolve/main/resnet50_best.pth"
TARGET_PATH = Path("models/best_model.pth")


def download_progress_hook(block_num, block_size, total_size):
    """Prints a friendly download progress bar."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, (downloaded / total_size) * 100)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        bar_len = 40
        filled = int(bar_len * (percent / 100))
        bar = "=" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r[{bar}] {percent:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\rDownloaded {downloaded / (1024 * 1024):.1f} MB...")
        sys.stdout.flush()


def download_trained_weights():
    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("      AgroDetect - Downloading Pretrained ResNet50 Weights")
    print("=" * 65)
    print(f"Source URL:  {MODEL_URL}")
    print(f"Destination: {TARGET_PATH.resolve()}")
    print("Downloading (~95 MB)... Please wait.\n")

    try:
        # User-agent header to avoid bot filtering
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AgroDetect/1.0")]
        urllib.request.install_opener(opener)

        urllib.request.urlretrieve(MODEL_URL, str(TARGET_PATH), reporthook=download_progress_hook)
        print("\n\n[+] Download completed successfully!")

        # Verify file integrity with torch.load
        print("[*] Verifying PyTorch checkpoint integrity...")
        checkpoint = torch.load(str(TARGET_PATH), map_location="cpu")
        print("[+] Checkpoint verified! ResNet50 model weights are valid.")
        print("\n=============================================================")
        print("  SUCCESS! Model is installed at: models/best_model.pth")
        print("  Your running server will automatically reload and is ready")
        print("  to analyze plant disease leaves on http://127.0.0.1:8000")
        print("=============================================================\n")
        return True

    except Exception as exc:
        print(f"\n[!] Download failed: {exc}")
        print("\nAlternative options:")
        print("1. Download manually in your browser from:")
        print(f"   {MODEL_URL}")
        print("   and save it as: models/best_model.pth")
        print("2. Or train a model on your dataset with:")
        print("   python -m training.train --data-dir dataset --epochs 15")
        return False


if __name__ == "__main__":
    download_trained_weights()
