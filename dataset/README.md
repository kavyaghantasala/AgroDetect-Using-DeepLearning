# AgroDetect Dataset Guide

AgroDetect trains on leaf images of healthy and diseased agricultural crops. The standard benchmark is the **PlantVillage** dataset (or the **New Plant Diseases Dataset (Augmented)**).

---

## Recommended Directory Structure

Your `dataset` folder should follow this layout:

```text
dataset/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   ├── Tomato___Early_blight/
│   ├── Tomato___healthy/
│   └── ... (one subfolder per class)
├── valid/
│   ├── Apple___Apple_scab/
│   └── ...
└── test/
    ├── Apple___Apple_scab/
    └── ...
```

> **Flexible Split Support**:
> If your dataset only has a single folder with class directories (e.g. `dataset/raw/Class_1/...`), our training loader (`training/dataset.py`) can automatically split your data into 70% Train, 15% Validation, and 15% Test!

---

## Option 1: Download via Kaggle CLI (Recommended)

1. Install the Kaggle CLI:
   ```powershell
   pip install kaggle
   ```
2. Place your `kaggle.json` API token in `C:\Users\<YourUser>\.kaggle\kaggle.json`.
3. Download the New Plant Diseases Dataset (Augmented):
   ```powershell
   kaggle datasets download -d vipoooool/new-plant-diseases-dataset
   ```
4. Extract the zip file into the `dataset/` directory:
   ```powershell
   Expand-Archive -Path new-plant-diseases-dataset.zip -DestinationPath dataset/temp
   ```
5. Move the extracted `New Plant Diseases Dataset(Augmented)/train` and `valid` folders into `dataset/train` and `dataset/valid`.

---

## Option 2: Download via Browser

1. Visit: [Kaggle - New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
   or [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
2. Click **Download** (approx. 800 MB - 1.5 GB).
3. Unzip the archive into this `dataset/` folder.

---

## Option 3: Use the Download Script

We provide a helper script in this directory:
```powershell
python dataset/download_dataset.py --verify
```
Run `python dataset/download_dataset.py --help` for automated download options using `kagglehub`.

---

## Class Verification

To verify that your dataset folders are properly detected before running a full training loop:
```powershell
python -m training.dataset --data-dir dataset --verify
```
This script will list all detected classes, sample image counts, and report any missing or corrupted image files.
