"""
AgroDetect - Sample Leaf Images Downloader
Downloads a few sample leaf photos for quick testing and demonstration.
"""

import os
import urllib.request
from pathlib import Path

SAMPLE_IMAGES = {
    "tomato_early_blight.jpg": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Tomato___Early_blight/0012b9d2-2130-4a06-a834-b1f3af34f57e___RS_Erly.B%209472.JPG",
    "tomato_healthy.jpg": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Tomato___healthy/000146ff-92a4-4db6-90c8-e729311230e3___GH_HLLeaf%20259.1.JPG",
    "potato_late_blight.jpg": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Potato___Late_blight/0051e5e8-d1c4-4a84-bf3a-a4aa571f8b82___RS_LB%204640.JPG"
}

OUTPUT_DIR = Path("dataset/samples")


def download_sample_leaves():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nDownloading sample leaf images for testing...")

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AgroDetect/1.0")]
    urllib.request.install_opener(opener)

    for filename, url in SAMPLE_IMAGES.items():
        dest = OUTPUT_DIR / filename
        if not dest.exists():
            try:
                print(f"[*] Downloading {filename}...")
                urllib.request.urlretrieve(url, str(dest))
                print(f"[+] Saved to: {dest}")
            except Exception as e:
                print(f"[!] Could not download {filename}: {e}")
        else:
            print(f"[i] Already exists: {dest}")

    print(f"\nSample leaves are ready in: {OUTPUT_DIR.resolve()}\n")


if __name__ == "__main__":
    download_sample_leaves()
