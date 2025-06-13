# Deployment

Thyroid Cancer Histopathology Analysis deployment components.

## Usage

### Web Interface

```bash
python web_app.py
```

- Runs at `http://localhost:5001`
- Drag-and-drop image upload
- Real-time progress tracking

### Command Line

```bash
# Unified pipeline
python segmentation_classification/unified_pipeline.py --image path/to/image.png

# Feature-based pipeline
python feature_based_classification/feature_based_pipeline.py --image path/to/image.png
```

## Files

- `web_app.py` - Web interface
- `segmentation_classification/` - End-to-end pipeline
- `feature_based_classification/` - Feature extraction pipeline
- `templates/` - HTML templates

## Required Models

Place in `../../models_checkpoints/`:

- `unet_model.pth`
- `segformer_model.pth`  
- `fastercnn_model.pth`
- `ensemble_model.pth`
- `model_best.pt`

## Testing

```bash
python ../../tests/test_pipelines.py
```
