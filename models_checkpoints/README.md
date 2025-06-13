# Model Checkpoints

Trained model files for thyroid cancer histopathology analysis.

## Required Files

### Segmentation Models

- `unet_model.pth`
- `segformer_model.pth`  
- `fastercnn_model.pth`
- `ensemble_model.pth`

### Classification Model

- `model_best.pt`

## Notes

- Models are automatically loaded by deployment pipelines
- Missing models will use fallback synthetic data
- All deployment code points to this centralized directory
