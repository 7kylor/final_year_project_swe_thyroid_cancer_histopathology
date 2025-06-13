# Deployment

Thyroid Cancer Histopathology Analysis deployment components.

## Usage

### Web Interface

```bash
# Auto-detects available port starting from 5001
python web_app.py

# Or specify a port manually
python web_app.py --port 8080

# Using the launcher scripts (recommended)
./run_web.sh          # Unix/Linux/macOS
run_web.bat            # Windows
```

- Automatically finds available port (5001, 5002, 5003, etc.)
- Drag-and-drop image upload
- Real-time progress tracking
- Browser opens automatically

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
