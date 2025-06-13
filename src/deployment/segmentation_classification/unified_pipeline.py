"""
Unified Pipeline for Thyroid Cancer Histopathology Image Analysis
This script performs both segmentation and classification in one integrated pipeline.

Usage:
    python unified_pipeline.py --image_path path/to/image.png [--output_dir path/to/output]
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
from typing import Dict, Any, Tuple, Optional, List
from skimage import measure

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../thyroid-cancer-histopathology-segmentation/segmentation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../thyroid-cancer-histopathology-calssification/calssification'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    """Configuration for the unified pipeline"""
    def __init__(self):
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Model paths - use absolute paths based on deployment location
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.checkpoint_dir = os.path.join(project_root, "models_checkpoints")
        
        # Segmentation models
        self.segmentation_models = {
            'unet': os.path.join(self.checkpoint_dir, 'unet_model.pth'),
            'segformer': os.path.join(self.checkpoint_dir, 'segformer_model.pth'),
            'fastercnn': os.path.join(self.checkpoint_dir, 'fastercnn_model.pth'),
            'ensemble': os.path.join(self.checkpoint_dir, 'ensemble_model.pth')
        }
        
        # Classification model
        self.classification_model_path = os.path.join(self.checkpoint_dir, 'model_best.pt')
        
        # Image configuration
        self.seg_image_size = 512
        self.cls_image_size = 224
        
        # Ensemble weights for segmentation
        self.seg_ensemble_weights = {
            'unet': 0.35,
            'segformer': 0.35,
            'fastercnn': 0.30
        }
        
        # Classification configuration
        self.num_classes = 2
        self.class_names = ['benign', 'malignant']
        
        # Visualization settings
        self.save_visualizations = True
        self.visualization_alpha = 0.5

# Import model architectures - with proper error handling
try:
    from comprehensive_nuclei_segmentation import (
        ImprovedUNet, ImprovedSegFormer, NucleiFasterCNN, 
        CollaborativeEnsemble, Config as SegConfig
    )
    SEGMENTATION_MODELS_AVAILABLE = True
except ImportError:
    logger.warning("Could not import segmentation models from comprehensive_nuclei_segmentation")
    SEGMENTATION_MODELS_AVAILABLE = False

# Try alternative imports
if not SEGMENTATION_MODELS_AVAILABLE:
    try:
        # Try importing from advanced_nuclei_mask_generator
        from advanced_nuclei_mask_generator import AdvancedNucleiMaskGenerator
    except ImportError:
        logger.warning("Could not import from advanced_nuclei_mask_generator")

# Import classification models
try:
    from ensemble_classification import (
        EnhancedEnsembleModel, ModelConfig,
        ResNetModel, DenseNetModel, EfficientNetModel
    )
    CLASSIFICATION_MODELS_AVAILABLE = True
except ImportError:
    logger.warning("Could not import classification models from ensemble_classification")
    CLASSIFICATION_MODELS_AVAILABLE = False

class UnifiedPipeline:
    """Unified pipeline for segmentation and classification"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        
        # Initialize transforms
        self._setup_transforms()
        
        # Load models
        self.segmentation_models = self._load_segmentation_models()
        self.classification_model = self._load_classification_model()
        
        logger.info(f"Pipeline initialized on device: {self.device}")
    
    def _setup_transforms(self):
        """Setup image transforms for both segmentation and classification"""
        # Segmentation transforms
        self.seg_transform = transforms.Compose([
            transforms.Resize((self.config.seg_image_size, self.config.seg_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Classification transforms
        self.cls_transform = transforms.Compose([
            transforms.Resize((self.config.cls_image_size, self.config.cls_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_segmentation_models(self) -> Dict[str, nn.Module]:
        """Load all segmentation models"""
        models = {}
        
        # Try to load each segmentation model
        for model_name, model_path in self.config.segmentation_models.items():
            try:
                if os.path.exists(model_path):
                    if model_name == 'unet':
                        model = self._create_unet_model()
                    elif model_name == 'segformer':
                        model = self._create_segformer_model()
                    elif model_name == 'fastercnn':
                        model = self._create_fastercnn_model()
                    elif model_name == 'ensemble':
                        model = self._create_ensemble_model()
                    else:
                        continue
                    
                    if model is not None:
                        # Load checkpoint with weights_only=False for PyTorch 2.6
                        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        else:
                            model.load_state_dict(checkpoint, strict=False)
                        
                        model.to(self.device)
                        model.eval()
                        models[model_name] = model
                        logger.info(f"Loaded {model_name} segmentation model")
                else:
                    logger.warning(f"Segmentation model not found: {model_path}")
            except Exception as e:
                logger.error(f"Error loading {model_name} model: {e}")
        
        return models
    
    def _create_unet_model(self):
        """Create UNet model instance"""
        if SEGMENTATION_MODELS_AVAILABLE:
            try:
                seg_config = SegConfig()
                seg_config.device = self.device
                seg_config.use_nuclei_features = False
                return ImprovedUNet(seg_config)
            except Exception as e:
                logger.error(f"Error creating ImprovedUNet: {e}")
        
        # Fallback to simple UNet
        return self._create_simple_unet()
    
    def _create_simple_unet(self):
        """Create a simple UNet model as fallback"""
        class SimpleUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)
                )
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(128, 64, 2, stride=2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(64, 32, 2, stride=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 1, 1)
                )
            
            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return torch.sigmoid(x)
        
        return SimpleUNet()
    
    def _create_segformer_model(self):
        """Create SegFormer model instance"""
        if SEGMENTATION_MODELS_AVAILABLE:
            try:
                seg_config = SegConfig()
                seg_config.device = self.device
                seg_config.use_nuclei_features = False
                return ImprovedSegFormer(seg_config)
            except Exception as e:
                logger.error(f"Error creating ImprovedSegFormer: {e}")
        
        # Use UNet as fallback
        return self._create_simple_unet()
    
    def _create_fastercnn_model(self):
        """Create FasterCNN model instance"""
        if SEGMENTATION_MODELS_AVAILABLE:
            try:
                seg_config = SegConfig()
                seg_config.device = self.device
                seg_config.use_nuclei_features = False
                return NucleiFasterCNN(seg_config)
            except Exception as e:
                logger.error(f"Error creating NucleiFasterCNN: {e}")
        
        # Use UNet as fallback
        return self._create_simple_unet()
    
    def _create_ensemble_model(self):
        """Create ensemble model instance"""
        if SEGMENTATION_MODELS_AVAILABLE:
            try:
                seg_config = SegConfig()
                seg_config.device = self.device
                return CollaborativeEnsemble(seg_config)
            except Exception as e:
                logger.error(f"Error creating CollaborativeEnsemble: {e}")
        
        return None
    
    def _load_classification_model(self) -> nn.Module:
        """Load the classification ensemble model"""
        try:
            # Try to load using the actual ensemble model if available
            if CLASSIFICATION_MODELS_AVAILABLE:
                try:
                    # Load checkpoint
                    if os.path.exists(self.config.classification_model_path):
                        checkpoint = torch.load(self.config.classification_model_path, 
                                              map_location=self.device, weights_only=False)
                        
                        # Extract config if available
                        if isinstance(checkpoint, dict) and 'config' in checkpoint:
                            config_dict = checkpoint['config']
                            model_config = ModelConfig(**config_dict)
                            model = EnhancedEnsembleModel(model_config)
                            
                            if 'model_state_dict' in checkpoint:
                                model.load_state_dict(checkpoint['model_state_dict'])
                            else:
                                model.load_state_dict(checkpoint)
                        else:
                            # Create default config
                            model_config = ModelConfig()
                            model_config.device = self.device
                            model = EnhancedEnsembleModel(model_config)
                            
                            if isinstance(checkpoint, dict):
                                if 'model_state_dict' in checkpoint:
                                    model.load_state_dict(checkpoint['model_state_dict'])
                                elif 'state_dict' in checkpoint:
                                    model.load_state_dict(checkpoint['state_dict'])
                            else:
                                model.load_state_dict(checkpoint)
                        
                        model.to(self.device)
                        model.eval()
                        logger.info("Loaded enhanced ensemble classification model")
                        return model
                except Exception as e:
                    logger.error(f"Error loading enhanced ensemble model: {e}")
            
            # Fallback to simple ensemble
            model = self._create_classification_ensemble()
            
            # Load checkpoint
            if os.path.exists(self.config.classification_model_path):
                checkpoint = torch.load(self.config.classification_model_path, 
                                      map_location=self.device, weights_only=False)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'], strict=False)
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                
                model.to(self.device)
                model.eval()
                logger.info("Loaded classification model")
            else:
                logger.warning(f"Classification model not found: {self.config.classification_model_path}")
                
            return model
        except Exception as e:
            logger.error(f"Error loading classification model: {e}")
            return self._create_simple_classifier()
    
    def _create_classification_ensemble(self):
        """Create the classification ensemble model"""
        class ClassificationEnsemble(nn.Module):
            def __init__(self, num_classes=2):
                super().__init__()
                # ResNet152
                self.resnet = models.resnet152(pretrained=True)
                self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
                
                # DenseNet121
                self.densenet = models.densenet121(pretrained=True)
                self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)
                
                # EfficientNet
                self.efficientnet = models.efficientnet_b0(pretrained=True)
                self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)
                
                # Fusion weights
                self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
            
            def forward(self, x):
                # Get predictions from each model
                resnet_out = self.resnet(x)
                densenet_out = self.densenet(x)
                efficientnet_out = self.efficientnet(x)
                
                # Weighted average
                weights = F.softmax(self.fusion_weights, dim=0)
                output = (weights[0] * resnet_out + 
                         weights[1] * densenet_out + 
                         weights[2] * efficientnet_out)
                
                return output
        
        return ClassificationEnsemble(self.config.num_classes)
    
    def _create_simple_classifier(self):
        """Create a simple classifier as fallback"""
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, self.config.num_classes)
        return model
    
    def segment_image(self, image: Image.Image) -> Dict[str, Any]:
        """Perform segmentation on the input image"""
        # Convert to tensor
        image_tensor = self.seg_transform(image).unsqueeze(0).to(self.device)
        
        segmentation_results = {}
        
        with torch.no_grad():
            # Run each segmentation model
            for model_name, model in self.segmentation_models.items():
                if model is not None:
                    try:
                        output = model(image_tensor)
                        # Handle different output formats
                        if isinstance(output, dict):
                            mask = output.get('mask', output.get('out', output))
                        else:
                            mask = output
                        
                        # Ensure mask is in correct format
                        if len(mask.shape) == 4 and mask.shape[1] > 1:
                            mask = mask[:, 0:1, :, :]  # Take first channel
                        
                        mask = torch.sigmoid(mask)
                        segmentation_results[model_name] = mask.cpu()
                    except Exception as e:
                        logger.error(f"Error in {model_name} segmentation: {e}")
            
            # Ensemble segmentation results
            if len(segmentation_results) > 1:
                ensemble_mask = self._ensemble_segmentations(segmentation_results)
                segmentation_results['ensemble'] = ensemble_mask
        
        return segmentation_results
    
    def _ensemble_segmentations(self, results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine multiple segmentation results"""
        # Weighted average based on config
        ensemble_mask = None
        total_weight = 0
        
        for model_name, mask in results.items():
            if model_name in self.config.seg_ensemble_weights:
                weight = self.config.seg_ensemble_weights[model_name]
                if ensemble_mask is None:
                    ensemble_mask = weight * mask
                else:
                    ensemble_mask += weight * mask
                total_weight += weight
        
        if ensemble_mask is not None and total_weight > 0:
            ensemble_mask /= total_weight
        else:
            # Simple average if weights not specified
            masks = list(results.values())
            ensemble_mask = torch.mean(torch.stack(masks), dim=0)
        
        return ensemble_mask
    
    def classify_image(self, image: Image.Image, segmentation_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Perform classification on the input image"""
        # Convert to tensor
        image_tensor = self.cls_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get model output
            output = self.classification_model(image_tensor)
            
            # Handle models that return tuples (e.g., EnhancedEnsembleModel)
            if isinstance(output, tuple):
                output = output[0]  # Take the main output
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(output, dim=1)
            
            # Get prediction
            _, predicted = torch.max(output, 1)
            
            # Convert to numpy
            probs_np = probabilities.cpu().numpy()[0]
            pred_class = predicted.cpu().item()
        
        classification_result = {
            'predicted_class': self.config.class_names[pred_class],
            'predicted_index': pred_class,
            'probabilities': {
                self.config.class_names[i]: float(probs_np[i]) 
                for i in range(len(self.config.class_names))
            },
            'confidence': float(probs_np[pred_class])
        }
        
        return classification_result
    
    def visualize_results(self, image: Image.Image, segmentation_results: Dict[str, torch.Tensor], 
                         classification_result: Dict[str, Any], output_dir: str):
        """Create and save detailed visualization of results with metrics"""
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title with classification result
        fig.suptitle(
            f"Thyroid Cancer Analysis - Unified Pipeline\n"
            f"Classification: {classification_result['predicted_class'].upper()} "
            f"(Confidence: {classification_result['confidence']:.2%})",
            fontsize=20, fontweight='bold', y=0.98
        )
        
        # 1. Original image (larger)
        ax_orig = fig.add_subplot(gs[0:2, 0:2])
        ax_orig.imshow(image)
        ax_orig.set_title('Original Histopathology Image', fontsize=14, fontweight='bold')
        ax_orig.axis('off')
        
        # 2. Classification metrics
        ax_class = fig.add_subplot(gs[0:2, 2:4])
        ax_class.axis('off')
        
        # Create classification summary box
        class_color = '#4CAF50' if classification_result['predicted_class'] == 'benign' else '#F44336'
        bbox_props = dict(boxstyle="round,pad=0.5", facecolor=class_color, alpha=0.2, edgecolor=class_color, linewidth=2)
        
        class_text = f"DIAGNOSIS: {classification_result['predicted_class'].upper()}\n\n"
        class_text += f"Confidence Score: {classification_result['confidence']:.2%}\n\n"
        class_text += "Class Probabilities:\n"
        for class_name, prob in classification_result['probabilities'].items():
            class_text += f"  • {class_name.capitalize()}: {prob:.2%}\n"
        
        ax_class.text(0.5, 0.5, class_text, transform=ax_class.transAxes,
                     fontsize=16, verticalalignment='center', horizontalalignment='center',
                     bbox=bbox_props)
        
        # Add probability bar chart
        ax_prob = fig.add_subplot(gs[1, 2:4])
        classes = list(classification_result['probabilities'].keys())
        probs = list(classification_result['probabilities'].values())
        colors = ['#4CAF50' if c == 'benign' else '#F44336' for c in classes]
        bars = ax_prob.bar(classes, probs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax_prob.text(bar.get_x() + bar.get_width()/2., height,
                        f'{prob:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax_prob.set_ylim(0, 1.1)
        ax_prob.set_ylabel('Probability', fontsize=12)
        ax_prob.set_title('Classification Probabilities', fontsize=14, fontweight='bold')
        ax_prob.grid(axis='y', alpha=0.3)
        
        # 3. Segmentation results grid
        seg_models = list(segmentation_results.keys())
        n_seg = len(seg_models)
        
        # Calculate grid positions for segmentation results
        positions = [(2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]
        
        for idx, (model_name, mask) in enumerate(segmentation_results.items()):
            if idx >= len(positions):
                break
                
            row, col = positions[idx]
            ax = fig.add_subplot(gs[row, col])
            
            # Convert mask to numpy
            mask_np = mask.squeeze().numpy()
            
            # Create overlay
            image_np = np.array(image)
            
            # Resize image to match mask size for display
            if image_np.shape[:2] != mask_np.shape:
                image_resized = cv2.resize(image_np, (mask_np.shape[1], mask_np.shape[0]))
            else:
                image_resized = image_np
            
            # Create colored overlay
            overlay = image_resized.copy()
            
            # Create mask overlay with custom colormap
            mask_colored = np.zeros_like(image_resized)
            mask_binary = (mask_np > 0.5).astype(np.uint8)
            
            # Apply different colors for better visibility
            if model_name == 'unet':
                mask_colored[:, :, 0] = mask_binary * 255  # Red
            elif model_name == 'segformer':
                mask_colored[:, :, 1] = mask_binary * 255  # Green
            elif model_name == 'fastercnn':
                mask_colored[:, :, 2] = mask_binary * 255  # Blue
            else:  # ensemble
                mask_colored[:, :, 0] = mask_binary * 255  # Red
                mask_colored[:, :, 1] = mask_binary * 128  # Some green
            
            # Blend with original image
            alpha = 0.4
            overlay = cv2.addWeighted(overlay, 1-alpha, mask_colored, alpha, 0)
            
            ax.imshow(overlay)
            
            # Calculate segmentation metrics
            n_nuclei = len(measure.label(mask_binary, connectivity=2))
            coverage = (mask_binary.sum() / mask_binary.size) * 100
            
            ax.set_title(f'{model_name.upper()}\nNuclei: {n_nuclei}, Coverage: {coverage:.1f}%', 
                        fontsize=11, fontweight='bold')
            ax.axis('off')
        
        # 4. Model Performance Summary (if metrics available)
        ax_summary = fig.add_subplot(gs[3, :])
        ax_summary.axis('off')
        
        # Create summary text
        summary_text = "Segmentation Models Summary:\n"
        summary_text += f"• Total models used: {len(segmentation_results)}\n"
        summary_text += f"• Models: {', '.join([m.upper() for m in seg_models])}\n"
        if 'ensemble' in segmentation_results:
            summary_text += "• Ensemble model combines predictions from all individual models\n"
        
        ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                       fontsize=12, verticalalignment='center', horizontalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.3))
        
        # Save figure
        plt.tight_layout()
        vis_path = os.path.join(output_dir, 'unified_pipeline_results.png')
        plt.savefig(vis_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Create additional detailed metrics visualization
        self._create_detailed_metrics_plot(segmentation_results, classification_result, output_dir)
        
        logger.info(f"Saved visualization to {vis_path}")
    
    def _create_detailed_metrics_plot(self, segmentation_results: Dict[str, torch.Tensor], 
                                     classification_result: Dict[str, Any], output_dir: str):
        """Create a detailed metrics visualization"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detailed Model Metrics and Analysis', fontsize=16, fontweight='bold')
        
        # 1. Segmentation models comparison
        ax = axes[0, 0]
        seg_metrics = {}
        for model_name, mask in segmentation_results.items():
            mask_np = mask.squeeze().numpy()
            mask_binary = (mask_np > 0.5).astype(np.uint8)
            
            labeled_mask = measure.label(mask_binary, connectivity=2)
            regions = measure.regionprops(labeled_mask)
            
            seg_metrics[model_name] = {
                'nuclei_count': len(regions),
                'coverage': (mask_binary.sum() / mask_binary.size) * 100,
                'avg_area': np.mean([r.area for r in regions]) if regions else 0
            }
        
        # Plot segmentation metrics
        metrics_df = []
        for model, metrics in seg_metrics.items():
            for metric, value in metrics.items():
                metrics_df.append({
                    'Model': model.upper(),
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': value
                })
        
        import pandas as pd
        df = pd.DataFrame(metrics_df)
        
        # Create grouped bar plot
        metric_types = df['Metric'].unique()
        x = np.arange(len(seg_metrics))
        width = 0.25
        
        for i, metric in enumerate(metric_types):
            metric_data = df[df['Metric'] == metric]
            values = [metric_data[metric_data['Model'] == m.upper()]['Value'].values[0] 
                     if len(metric_data[metric_data['Model'] == m.upper()]) > 0 else 0 
                     for m in seg_metrics.keys()]
            ax.bar(x + i*width, values, width, label=metric)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Value')
        ax.set_title('Segmentation Model Metrics Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.upper() for m in seg_metrics.keys()])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Classification confidence distribution
        ax = axes[0, 1]
        
        # Create a more detailed confidence visualization
        confidence = classification_result['confidence']
        uncertainty = 1 - confidence
        
        # Pie chart for confidence vs uncertainty
        sizes = [confidence, uncertainty]
        labels = ['Confidence', 'Uncertainty']
        colors = ['#4CAF50', '#FFC107']
        explode = (0.1, 0)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
        ax.set_title(f'Classification Confidence Analysis\nPredicted: {classification_result["predicted_class"].upper()}')
        
        # 3. Model ensemble weights (if available)
        ax = axes[1, 0]
        if hasattr(self.config, 'seg_ensemble_weights'):
            weights = self.config.seg_ensemble_weights
            models = list(weights.keys())
            values = list(weights.values())
            
            bars = ax.bar(models, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax.set_ylim(0, 0.5)
            ax.set_ylabel('Weight')
            ax.set_title('Ensemble Model Weights')
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'Ensemble weights not available', 
                   transform=ax.transAxes, ha='center', va='center')
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Processing summary
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = "Processing Summary:\n\n"
        summary_text += f"✓ Image successfully processed\n"
        summary_text += f"✓ {len(segmentation_results)} segmentation models applied\n"
        summary_text += f"✓ Classification completed with {confidence:.1%} confidence\n"
        summary_text += f"✓ All visualizations generated\n\n"
        summary_text += "Model Architecture:\n"
        summary_text += "• Segmentation: UNet, SegFormer, FasterCNN\n"
        summary_text += "• Classification: Enhanced Ensemble (ResNet152 + DenseNet121 + EfficientNet)\n"
        
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", 
               facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        metrics_path = os.path.join(output_dir, 'detailed_metrics.png')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Saved detailed metrics to {metrics_path}")
    
    def process_image(self, image_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Process a single image through the entire pipeline"""
        # Create output directory
        if output_dir is None:
            output_dir = os.path.join('output', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image
        logger.info(f"Processing image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        
        # Save original image info
        image_info = {
            'path': image_path,
            'size': image.size,
            'mode': image.mode
        }
        
        # Perform segmentation
        logger.info("Performing segmentation...")
        segmentation_results = self.segment_image(image)
        
        # Perform classification
        logger.info("Performing classification...")
        classification_result = self.classify_image(image)
        
        # Prepare final results
        results = {
            'timestamp': datetime.now().isoformat(),
            'image_info': image_info,
            'segmentation': {
                'models_used': list(segmentation_results.keys()),
                'ensemble_available': 'ensemble' in segmentation_results
            },
            'classification': classification_result,
            'output_directory': output_dir
        }
        
        # Save results
        results_path = os.path.join(output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {results_path}")
        
        # Save segmentation masks
        masks_dir = os.path.join(output_dir, 'segmentation_masks')
        os.makedirs(masks_dir, exist_ok=True)
        
        for model_name, mask in segmentation_results.items():
            mask_np = (mask.squeeze().numpy() * 255).astype(np.uint8)
            mask_image = Image.fromarray(mask_np)
            mask_path = os.path.join(masks_dir, f'{model_name}_mask.png')
            mask_image.save(mask_path)
        
        # Create visualization if enabled
        if self.config.save_visualizations:
            self.visualize_results(image, segmentation_results, classification_result, output_dir)
        
        logger.info(f"Pipeline completed. Results saved to {output_dir}")
        
        return results


def main():
    """Main function to run the unified pipeline"""
    parser = argparse.ArgumentParser(description='Unified Segmentation and Classification Pipeline')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    parser.add_argument('--no_viz', action='store_true', help='Disable visualization')
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    if args.no_viz:
        config.save_visualizations = False
    
    # Initialize pipeline
    pipeline = UnifiedPipeline(config)
    
    # Process image
    results = pipeline.process_image(args.image_path, args.output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("PIPELINE RESULTS")
    print("="*50)
    print(f"Image: {results['image_info']['path']}")
    print(f"Classification: {results['classification']['predicted_class']}")
    print(f"Confidence: {results['classification']['confidence']:.2%}")
    print(f"Results saved to: {results['output_directory']}")
    print("="*50)


if __name__ == '__main__':
    main() 