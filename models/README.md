# AgroDetect Models Directory

This folder stores model weights and class mapping configurations.

## Expected Files

- `class_names.json`: A JSON list of all plant disease class labels in alphabetical order matching the output logits of the trained neural network.
- `best_model.pth`: The trained PyTorch ResNet50 model weights checkpoint.

> **Note on `best_model.pth`:**
> As per project requirements, placeholder or random weights are **not** committed to this repository.
> When you train the model with `python -m training.train`, the checkpoint with the highest validation accuracy is automatically saved here as `best_model.pth`.

## Model Specification

- **Base Architecture**: ResNet50 (`torchvision.models.resnet50`)
- **Transfer Learning**: Pretrained on ImageNet-1K (`ResNet50_Weights.DEFAULT`)
- **Classifier Head**:
  - Linear layer (`in_features=2048`, `out_features=num_classes`)
- **Input Dimensions**: 3 x 224 x 224 (RGB)
- **Normalization**:
  - Mean: `[0.485, 0.456, 0.406]`
  - Standard Deviation: `[0.229, 0.224, 0.225]`

## Training the Model

To train the model on your dataset, run:
```powershell
python -m training.train --data-dir dataset --epochs 15 --batch-size 32
```

Once training finishes, `best_model.pth` will be placed in this folder and the web application will automatically detect and load it on startup.
