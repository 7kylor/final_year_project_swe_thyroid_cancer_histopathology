# Thyroid Cancer Histopathology Analysis

Deep learning framework for thyroid cancer detection and analysis using histopathological images.

## Features

- **Nuclei Segmentation**: UNet, SegFormer, Faster R-CNN, Ensemble models
- **Cancer Classification**: ResNet152, DenseNet121, EfficientNet ensemble
- **Feature Extraction**: 36 morphological, textural, and spatial features
- **Web Interface**: Interactive analysis with real-time progress tracking

## Quick Start

### Web Interface

```bash
cd src/deployment
python web_app.py
```

Opens at `http://localhost:5001` with drag-and-drop image upload.

### Command Line

```bash
# Unified pipeline
python src/deployment/segmentation_classification/unified_pipeline.py --image path/to/image.png

# Feature-based pipeline  
python src/deployment/feature_based_classification/feature_based_pipeline.py --image path/to/image.png
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch 1.9+

## Model Checkpoints

Place trained models in `models_checkpoints/`:

- `unet_model.pth` - UNet segmentation
- `segformer_model.pth` - SegFormer segmentation  
- `fastercnn_model.pth` - Faster R-CNN segmentation
- `ensemble_model.pth` - Ensemble segmentation
- `model_best.pt` - Classification ensemble

## Project Structure

```
thyroid_cancer_histopathology/
├── src/
│   ├── segmentation/         # Segmentation models
│   ├── classification/       # Classification models
│   └── deployment/          # Web app and pipelines
├── models_checkpoints/      # Trained model files
├── tests/                   # Performance tests
└── data/                    # Image datasets
```

## Performance

- **Segmentation**: Dice 0.80, IoU 0.67 (Ensemble)
- **Classification**: 94.17% accuracy (Weighted Fusion)
- **Inference**: 42-205ms depending on model

## Supported Formats

PNG, JPG, JPEG, TIFF histopathology images

---

**Research Prototype** - For educational/research use only
