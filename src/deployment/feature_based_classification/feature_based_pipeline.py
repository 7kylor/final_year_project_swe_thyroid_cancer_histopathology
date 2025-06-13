"""
Feature-Based Classification Pipeline for Thyroid Cancer Histopathology
This script extracts features from segmented images and uses them as inputs for classification.

Usage:
    python feature_based_pipeline.py --image_path path/to/image.png [--output_dir path/to/output]
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
from scipy import ndimage
from skimage import measure, morphology, feature
from typing import Dict, Any, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../thyroid-cancer-histopathology-segmentation/segmentation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../thyroid-cancer-histopathology-calssification/calssification'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    """Configuration for the feature-based pipeline"""
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
        
        # Feature extraction configuration
        self.extract_shape_features = True
        self.extract_texture_features = True
        self.extract_intensity_features = True
        self.extract_spatial_features = True
        
        # Classification configuration
        self.num_classes = 2
        self.class_names = ['benign', 'malignant']
        
        # Visualization settings
        self.save_visualizations = True
        self.visualization_alpha = 0.5

# Import model architectures
try:
    from comprehensive_nuclei_segmentation import (
        ImprovedUNet, ImprovedSegFormer, NucleiFasterCNN, 
        CollaborativeEnsemble, Config as SegConfig
    )
    SEGMENTATION_MODELS_AVAILABLE = True
except ImportError:
    logger.warning("Could not import segmentation models from comprehensive_nuclei_segmentation")
    SEGMENTATION_MODELS_AVAILABLE = False

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

class FeatureExtractor:
    """Extract meaningful features from segmentation masks and images"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_all_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Extract all features from image and mask"""
        features = {}
        
        # Ensure mask is binary
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Extract different feature categories
        if binary_mask.sum() > 0:  # Only extract features if mask has content
            shape_features = self._extract_shape_features(binary_mask)
            texture_features = self._extract_texture_features(image, binary_mask)
            intensity_features = self._extract_intensity_features(image, binary_mask)
            spatial_features = self._extract_spatial_features(binary_mask)
            
            # Combine all features
            features.update(shape_features)
            features.update(texture_features)
            features.update(intensity_features)
            features.update(spatial_features)
        else:
            # Return zero features if mask is empty
            features = self._get_zero_features()
        
        # Store feature names
        self.feature_names = list(features.keys())
        
        return features
    
    def _extract_shape_features(self, mask: np.ndarray) -> Dict[str, float]:
        """Extract shape-based features from segmentation mask"""
        features = {}
        
        # Label connected components
        labeled_mask = measure.label(mask)
        regions = measure.regionprops(labeled_mask)
        
        if len(regions) > 0:
            # Aggregate features across all regions
            areas = [r.area for r in regions]
            perimeters = [r.perimeter for r in regions]
            eccentricities = [r.eccentricity for r in regions]
            solidity_values = [r.solidity for r in regions]
            
            # Basic statistics
            features['nuclei_count'] = len(regions)
            features['mean_area'] = np.mean(areas)
            features['std_area'] = np.std(areas) if len(areas) > 1 else 0
            features['mean_perimeter'] = np.mean(perimeters)
            features['std_perimeter'] = np.std(perimeters) if len(perimeters) > 1 else 0
            features['mean_eccentricity'] = np.mean(eccentricities)
            features['mean_solidity'] = np.mean(solidity_values)
            
            # Size distribution features
            features['area_coefficient_variation'] = features['std_area'] / features['mean_area'] if features['mean_area'] > 0 else 0
            features['max_area'] = max(areas)
            features['min_area'] = min(areas)
            
            # Shape irregularity
            circularities = [4 * np.pi * r.area / (r.perimeter ** 2) if r.perimeter > 0 else 0 for r in regions]
            features['mean_circularity'] = np.mean(circularities)
            features['std_circularity'] = np.std(circularities) if len(circularities) > 1 else 0
        else:
            # No regions found
            features = {
                'nuclei_count': 0, 'mean_area': 0, 'std_area': 0,
                'mean_perimeter': 0, 'std_perimeter': 0, 'mean_eccentricity': 0,
                'mean_solidity': 0, 'area_coefficient_variation': 0,
                'max_area': 0, 'min_area': 0, 'mean_circularity': 0, 'std_circularity': 0
            }
        
        return features
    
    def _extract_texture_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Extract texture features from the masked region"""
        features = {}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        
        # Apply mask
        masked_region = gray_image * mask
        masked_pixels = gray_image[mask > 0]
        
        if len(masked_pixels) > 0:
            # GLCM (Gray Level Co-occurrence Matrix) features
            # Simplified version - compute basic texture measures
            features['texture_contrast'] = np.std(masked_pixels)
            features['texture_homogeneity'] = 1 / (1 + np.var(masked_pixels))
            features['texture_energy'] = np.sum(masked_pixels ** 2) / len(masked_pixels)
            features['texture_entropy'] = self._compute_entropy(masked_pixels)
            
            # Local Binary Pattern features
            try:
                lbp = feature.local_binary_pattern(masked_region, 8, 1, method='uniform')
                lbp_masked = lbp[mask > 0]
                features['lbp_mean'] = np.mean(lbp_masked)
                features['lbp_std'] = np.std(lbp_masked)
            except:
                features['lbp_mean'] = 0
                features['lbp_std'] = 0
            
            # Edge features
            edges = cv2.Canny(gray_image.astype(np.uint8), 50, 150)
            edge_density = np.sum(edges * mask) / np.sum(mask)
            features['edge_density'] = edge_density
        else:
            features = {
                'texture_contrast': 0, 'texture_homogeneity': 0,
                'texture_energy': 0, 'texture_entropy': 0,
                'lbp_mean': 0, 'lbp_std': 0, 'edge_density': 0
            }
        
        return features
    
    def _extract_intensity_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Extract intensity-based features"""
        features = {}
        
        # Work with each color channel
        if len(image.shape) == 3:
            for i, channel in enumerate(['red', 'green', 'blue']):
                channel_data = image[:, :, i]
                masked_pixels = channel_data[mask > 0]
                
                if len(masked_pixels) > 0:
                    features[f'{channel}_mean'] = np.mean(masked_pixels)
                    features[f'{channel}_std'] = np.std(masked_pixels)
                    features[f'{channel}_skewness'] = self._compute_skewness(masked_pixels)
                    features[f'{channel}_kurtosis'] = self._compute_kurtosis(masked_pixels)
                else:
                    features[f'{channel}_mean'] = 0
                    features[f'{channel}_std'] = 0
                    features[f'{channel}_skewness'] = 0
                    features[f'{channel}_kurtosis'] = 0
        else:
            # Grayscale image
            masked_pixels = image[mask > 0]
            if len(masked_pixels) > 0:
                features['intensity_mean'] = np.mean(masked_pixels)
                features['intensity_std'] = np.std(masked_pixels)
                features['intensity_skewness'] = self._compute_skewness(masked_pixels)
                features['intensity_kurtosis'] = self._compute_kurtosis(masked_pixels)
            else:
                features['intensity_mean'] = 0
                features['intensity_std'] = 0
                features['intensity_skewness'] = 0
                features['intensity_kurtosis'] = 0
        
        return features
    
    def _extract_spatial_features(self, mask: np.ndarray) -> Dict[str, float]:
        """Extract spatial distribution features"""
        features = {}
        
        # Calculate centroid and spatial moments
        if mask.sum() > 0:
            # Find center of mass
            center_y, center_x = ndimage.center_of_mass(mask)
            features['centroid_x'] = center_x / mask.shape[1]  # Normalize
            features['centroid_y'] = center_y / mask.shape[0]  # Normalize
            
            # Calculate spatial spread
            y_coords, x_coords = np.where(mask > 0)
            if len(x_coords) > 0:
                features['spatial_spread_x'] = np.std(x_coords) / mask.shape[1]
                features['spatial_spread_y'] = np.std(y_coords) / mask.shape[0]
                
                # Density features
                bbox_area = (x_coords.max() - x_coords.min() + 1) * (y_coords.max() - y_coords.min() + 1)
                features['density'] = mask.sum() / bbox_area if bbox_area > 0 else 0
            else:
                features['spatial_spread_x'] = 0
                features['spatial_spread_y'] = 0
                features['density'] = 0
        else:
            features = {
                'centroid_x': 0, 'centroid_y': 0,
                'spatial_spread_x': 0, 'spatial_spread_y': 0,
                'density': 0
            }
        
        return features
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        """Compute Shannon entropy"""
        if len(data) == 0:
            return 0
        hist, _ = np.histogram(data, bins=256)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0
        prob = hist / hist.sum()
        return -np.sum(prob * np.log2(prob + 1e-10))
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness"""
        if len(data) < 3:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis"""
        if len(data) < 4:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _get_zero_features(self) -> Dict[str, float]:
        """Return zero-valued features when mask is empty"""
        return {
            # Shape features
            'nuclei_count': 0, 'mean_area': 0, 'std_area': 0,
            'mean_perimeter': 0, 'std_perimeter': 0, 'mean_eccentricity': 0,
            'mean_solidity': 0, 'area_coefficient_variation': 0,
            'max_area': 0, 'min_area': 0, 'mean_circularity': 0, 'std_circularity': 0,
            # Texture features
            'texture_contrast': 0, 'texture_homogeneity': 0,
            'texture_energy': 0, 'texture_entropy': 0,
            'lbp_mean': 0, 'lbp_std': 0, 'edge_density': 0,
            # Intensity features
            'red_mean': 0, 'red_std': 0, 'red_skewness': 0, 'red_kurtosis': 0,
            'green_mean': 0, 'green_std': 0, 'green_skewness': 0, 'green_kurtosis': 0,
            'blue_mean': 0, 'blue_std': 0, 'blue_skewness': 0, 'blue_kurtosis': 0,
            # Spatial features
            'centroid_x': 0, 'centroid_y': 0,
            'spatial_spread_x': 0, 'spatial_spread_y': 0,
            'density': 0
        }


class FeatureBasedClassifier(nn.Module):
    """Neural network classifier that uses extracted features"""
    
    def __init__(self, input_features: int, num_classes: int = 2):
        super().__init__()
        
        # Feature processing layers
        self.feature_processor = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.feature_processor(x)


class FeatureBasedPipeline:
    """Feature-based pipeline for classification using segmentation features"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self._setup_transforms()
        
        # Load models
        self.segmentation_models = self._load_segmentation_models()
        self.classification_model = self._load_classification_model()
        self.feature_classifier = self._load_feature_classifier()
        
        logger.info(f"Feature-based pipeline initialized on device: {self.device}")
    
    def _setup_transforms(self):
        """Setup image transforms"""
        self.seg_transform = transforms.Compose([
            transforms.Resize((self.config.seg_image_size, self.config.seg_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.cls_transform = transforms.Compose([
            transforms.Resize((self.config.cls_image_size, self.config.cls_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_segmentation_models(self) -> Dict[str, nn.Module]:
        """Load segmentation models"""
        models = {}
        
        # Try to load actual models first
        if SEGMENTATION_MODELS_AVAILABLE:
            for model_name, model_path in self.config.segmentation_models.items():
                if os.path.exists(model_path):
                    try:
                        if model_name == 'unet':
                            seg_config = SegConfig()
                            seg_config.device = self.device
                            seg_config.use_nuclei_features = False
                            model = ImprovedUNet(seg_config)
                        elif model_name == 'segformer':
                            seg_config = SegConfig()
                            seg_config.device = self.device
                            seg_config.use_nuclei_features = False
                            model = ImprovedSegFormer(seg_config)
                        elif model_name == 'fastercnn':
                            seg_config = SegConfig()
                            seg_config.device = self.device
                            seg_config.use_nuclei_features = False
                            model = NucleiFasterCNN(seg_config)
                        else:
                            continue
                        
                        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        else:
                            model.load_state_dict(checkpoint, strict=False)
                        
                        model.to(self.device)
                        model.eval()
                        models[model_name] = model
                        logger.info(f"Loaded {model_name} segmentation model")
                        
                        # Only load first available model for feature extraction
                        break
                    except Exception as e:
                        logger.error(f"Error loading {model_name} model: {e}")
        
        # Fallback to simple model if no models loaded
        if not models:
            # Create simple UNet as base model
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
            
            # Try to load UNet model
            unet_path = self.config.segmentation_models['unet']
            if os.path.exists(unet_path):
                try:
                    model = SimpleUNet()
                    checkpoint = torch.load(unet_path, map_location=self.device, weights_only=False)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                    model.to(self.device)
                    model.eval()
                    models['unet'] = model
                    logger.info("Loaded UNet segmentation model")
                except Exception as e:
                    logger.error(f"Error loading UNet model: {e}")
                    # Create default model
                    model = SimpleUNet()
                    model.to(self.device)
                    model.eval()
                    models['unet'] = model
        
        return models
    
    def _load_classification_model(self) -> Optional[nn.Module]:
        """Load the original classification model (optional for comparison)"""
        try:
            if CLASSIFICATION_MODELS_AVAILABLE:
                # Try to load enhanced ensemble model
                if os.path.exists(self.config.classification_model_path):
                    checkpoint = torch.load(self.config.classification_model_path, 
                                          map_location=self.device, weights_only=False)
                    
                    if isinstance(checkpoint, dict) and 'config' in checkpoint:
                        config_dict = checkpoint['config']
                        model_config = ModelConfig(**config_dict)
                        model = EnhancedEnsembleModel(model_config)
                        
                        if 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            model.load_state_dict(checkpoint)
                            
                        model.to(self.device)
                        model.eval()
                        logger.info("Loaded enhanced ensemble classification model for comparison")
                        return model
            
            # Fallback to simple ResNet
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, self.config.num_classes)
            
            if os.path.exists(self.config.classification_model_path):
                checkpoint = torch.load(self.config.classification_model_path, 
                                      map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict):
                    # Try to load state dict
                    try:
                        if 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        elif 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'], strict=False)
                    except:
                        logger.warning("Could not load classification model weights")
                
                model.to(self.device)
                model.eval()
                logger.info("Loaded original classification model for comparison")
                return model
            else:
                logger.warning("Original classification model not found")
                return None
        except Exception as e:
            logger.error(f"Error loading classification model: {e}")
            return None
    
    def _load_feature_classifier(self) -> nn.Module:
        """Create and initialize the feature-based classifier"""
        # Determine number of features
        num_features = len(self.feature_extractor._get_zero_features())
        
        # Create classifier
        classifier = FeatureBasedClassifier(num_features, self.config.num_classes)
        classifier.to(self.device)
        classifier.eval()
        
        # Note: In a real deployment, you would load pre-trained weights here
        # For demonstration, we're using a randomly initialized model
        logger.info(f"Created feature-based classifier with {num_features} input features")
        
        return classifier
    
    def segment_image(self, image: Image.Image) -> Tuple[np.ndarray, Dict[str, torch.Tensor]]:
        """Perform segmentation and return the best mask"""
        # Convert to tensor
        image_tensor = self.seg_transform(image).unsqueeze(0).to(self.device)
        
        segmentation_results = {}
        
        with torch.no_grad():
            # Run segmentation models
            for model_name, model in self.segmentation_models.items():
                if model is not None:
                    try:
                        output = model(image_tensor)
                        mask = torch.sigmoid(output)
                        segmentation_results[model_name] = mask.cpu()
                    except Exception as e:
                        logger.error(f"Error in {model_name} segmentation: {e}")
        
        # Use the first available mask
        if segmentation_results:
            best_mask = next(iter(segmentation_results.values()))
            best_mask_np = best_mask.squeeze().numpy()
        else:
            # Create empty mask if segmentation failed
            best_mask_np = np.zeros((self.config.seg_image_size, self.config.seg_image_size))
        
        return best_mask_np, segmentation_results
    
    def extract_features(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract features from image and segmentation mask"""
        # Resize mask to match original image size if needed
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Extract features
        features_dict = self.feature_extractor.extract_all_features(image, mask)
        
        # Convert to numpy array
        feature_values = np.array(list(features_dict.values()), dtype=np.float32)
        
        return feature_values, features_dict
    
    def classify_with_features(self, features: np.ndarray) -> Dict[str, Any]:
        """Classify using extracted features"""
        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get prediction from feature-based classifier
            output = self.feature_classifier(features_tensor)
            probabilities = F.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            probs_np = probabilities.cpu().numpy()[0]
            pred_class = predicted.cpu().item()
        
        result = {
            'predicted_class': self.config.class_names[pred_class],
            'predicted_index': pred_class,
            'probabilities': {
                self.config.class_names[i]: float(probs_np[i]) 
                for i in range(len(self.config.class_names))
            },
            'confidence': float(probs_np[pred_class])
        }
        
        return result
    
    def visualize_results(self, image: Image.Image, mask: np.ndarray, 
                         features: Dict[str, float], classification_result: Dict[str, Any],
                         output_dir: str):
        """Create comprehensive visualization of results with detailed metrics"""
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import seaborn as sns
        
        # Convert PIL Image to numpy array for consistency
        image_np = np.array(image)
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(24, 16))
        gs = gridspec.GridSpec(4, 6, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title with classification result
        fig.suptitle(
            f"Thyroid Cancer Analysis - Feature-Based Pipeline\n"
            f"Classification: {classification_result['predicted_class'].upper()} "
            f"(Confidence: {classification_result['confidence']:.2%})",
            fontsize=20, fontweight='bold', y=0.98
        )
        
        # 1. Original image
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax1.imshow(image_np)
        ax1.set_title('Original Histopathology Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 2. Segmentation mask
        ax2 = fig.add_subplot(gs[0:2, 2:4])
        ax2.imshow(mask, cmap='viridis')
        ax2.set_title('Nuclei Segmentation Mask', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Add segmentation statistics
        mask_binary = (mask > 0.5).astype(np.uint8)
        n_nuclei = len(measure.label(mask_binary, connectivity=2))
        coverage = (mask_binary.sum() / mask_binary.size) * 100
        
        ax2.text(0.02, 0.98, f'Nuclei Count: {n_nuclei}\nCoverage: {coverage:.1f}%', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Overlay
        ax3 = fig.add_subplot(gs[0:2, 4:6])
        overlay = image_np.copy()
        
        # Resize mask to match image size if needed
        if mask.shape != image_np.shape[:2]:
            mask_resized = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
        else:
            mask_resized = mask
        
        # Create colored overlay with better visibility
        mask_colored = np.zeros_like(image_np)
        mask_binary_resized = (mask_resized > 0.5).astype(np.uint8)
        
        # Use a distinctive color for nuclei
        mask_colored[:, :, 0] = mask_binary_resized * 255  # Red channel
        mask_colored[:, :, 1] = mask_binary_resized * 50   # Slight green
        
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        ax3.imshow(overlay)
        ax3.set_title('Segmentation Overlay', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # 4. Classification results
        ax4 = fig.add_subplot(gs[2, 0:2])
        
        # Create classification summary
        class_color = '#4CAF50' if classification_result['predicted_class'] == 'benign' else '#F44336'
        
        # Probability bars
        classes = list(classification_result['probabilities'].keys())
        probs = list(classification_result['probabilities'].values())
        colors = ['#4CAF50' if c == 'benign' else '#F44336' for c in classes]
        bars = ax4.bar(classes, probs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{prob:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax4.set_ylim(0, 1.1)
        ax4.set_ylabel('Probability', fontsize=12)
        ax4.set_title('Classification Probabilities', fontsize=14, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Top features by category
        ax5 = fig.add_subplot(gs[2, 2:6])
        
        # Group features by category
        shape_features = {k: v for k, v in features.items() 
                         if any(x in k for x in ['area', 'perimeter', 'nuclei', 'circularity', 'solidity', 'eccentricity'])}
        texture_features = {k: v for k, v in features.items() 
                           if any(x in k for x in ['texture', 'lbp', 'edge', 'entropy', 'contrast', 'homogeneity'])}
        intensity_features = {k: v for k, v in features.items() 
                             if any(x in k for x in ['red_', 'green_', 'blue_', 'intensity_', 'mean', 'std', 'skewness', 'kurtosis'])
                             and not any(x in k for x in ['area', 'perimeter'])}
        spatial_features = {k: v for k, v in features.items() 
                           if any(x in k for x in ['centroid', 'spatial', 'density']) 
                           and 'edge' not in k}
        
        # Select top features from each category
        top_features = []
        categories = []
        
        for cat_name, cat_features in [('Shape', shape_features), ('Texture', texture_features), 
                                       ('Intensity', intensity_features), ('Spatial', spatial_features)]:
            if cat_features:
                sorted_features = sorted(cat_features.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                for feat_name, feat_val in sorted_features:
                    top_features.append((feat_name, feat_val))
                    categories.append(cat_name)
        
        # Plot top features
        if top_features:
            feat_names = [f[0].replace('_', ' ').title() for f in top_features]
            feat_values = [f[1] for f in top_features]
            
            # Color by category
            category_colors = {'Shape': '#FF6B6B', 'Texture': '#4ECDC4', 
                              'Intensity': '#45B7D1', 'Spatial': '#96CEB4'}
            bar_colors = [category_colors[cat] for cat in categories]
            
            bars = ax5.barh(feat_names, feat_values, color=bar_colors)
            ax5.set_xlabel('Feature Value', fontsize=12)
            ax5.set_title('Top Features by Category', fontsize=14, fontweight='bold')
            ax5.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, feat_values):
                width = bar.get_width()
                ax5.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{val:.3f}', ha='left' if val >= 0 else 'right', 
                        va='center', fontsize=10)
        
        # 6. Feature distribution by category
        ax6 = fig.add_subplot(gs[3, 0:3])
        
        # Calculate category statistics
        category_stats = {
            'Shape': {'count': len(shape_features), 'mean': np.mean(list(shape_features.values())) if shape_features else 0},
            'Texture': {'count': len(texture_features), 'mean': np.mean(list(texture_features.values())) if texture_features else 0},
            'Intensity': {'count': len(intensity_features), 'mean': np.mean(list(intensity_features.values())) if intensity_features else 0},
            'Spatial': {'count': len(spatial_features), 'mean': np.mean(list(spatial_features.values())) if spatial_features else 0}
        }
        
        # Create stacked bar chart
        categories = list(category_stats.keys())
        counts = [category_stats[cat]['count'] for cat in categories]
        means = [abs(category_stats[cat]['mean']) for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax6.bar(x - width/2, counts, width, label='Feature Count', color='#3498db')
        bars2 = ax6.bar(x + width/2, means, width, label='Mean |Value|', color='#e74c3c')
        
        ax6.set_xlabel('Feature Category')
        ax6.set_ylabel('Value')
        ax6.set_title('Feature Statistics by Category')
        ax6.set_xticks(x)
        ax6.set_xticklabels(categories)
        ax6.legend()
        ax6.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom')
        
        # 7. Feature importance heatmap
        ax7 = fig.add_subplot(gs[3, 3:6])
        
        # Create a matrix of features for heatmap
        feature_matrix = []
        feature_labels = []
        
        for cat_name, cat_features in [('Shape', shape_features), ('Texture', texture_features), 
                                       ('Intensity', intensity_features), ('Spatial', spatial_features)]:
            if cat_features:
                # Normalize features within category
                values = list(cat_features.values())
                if max(abs(v) for v in values) > 0:
                    normalized = [v / max(abs(v) for v in values) for v in values]
                else:
                    normalized = values
                
                # Take top 5 features per category
                sorted_items = sorted(zip(cat_features.keys(), normalized), 
                                    key=lambda x: abs(x[1]), reverse=True)[:5]
                
                for feat_name, norm_val in sorted_items:
                    feature_matrix.append([norm_val])
                    feature_labels.append(f"{cat_name[:3]}: {feat_name[:15]}")
        
        if feature_matrix:
            sns.heatmap(np.array(feature_matrix), 
                       yticklabels=feature_labels,
                       xticklabels=['Normalized Value'],
                       cmap='RdBu_r', center=0, cbar=True,
                       annot=True, fmt='.2f', ax=ax7)
            ax7.set_title('Top Features Heatmap (Normalized)', fontsize=14, fontweight='bold')
        
        # Save figure
        plt.tight_layout()
        vis_path = os.path.join(output_dir, 'feature_based_results.png')
        plt.savefig(vis_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Saved visualization to {vis_path}")
        
        # Create additional detailed metrics
        self._create_detailed_feature_analysis(features, classification_result, mask, output_dir)
    
    def _create_detailed_feature_analysis(self, features: Dict[str, float], 
                                        classification_result: Dict[str, Any],
                                        mask: np.ndarray, output_dir: str):
        """Create detailed feature analysis visualization"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig = plt.figure(figsize=(20, 16))
        gs = plt.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Detailed Feature Analysis and Metrics', fontsize=18, fontweight='bold')
        
        # 1. Feature correlation matrix
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        
        # Create feature matrix for correlation
        feature_names = list(features.keys())[:20]  # Top 20 features
        feature_values = [features[name] for name in feature_names]
        
        # Create correlation matrix (simplified - showing feature relationships)
        corr_matrix = np.outer(feature_values, feature_values)
        if np.max(np.abs(corr_matrix)) > 0:
            corr_matrix = corr_matrix / np.max(np.abs(corr_matrix))
        
        sns.heatmap(corr_matrix, 
                   xticklabels=[n[:10] for n in feature_names],
                   yticklabels=[n[:10] for n in feature_names],
                   cmap='coolwarm', center=0, 
                   cbar_kws={'label': 'Feature Correlation'},
                   ax=ax1)
        ax1.set_title('Feature Correlation Matrix (Top 20)', fontsize=14, fontweight='bold')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax1.get_yticklabels(), rotation=0)
        
        # 2. Feature distribution plots
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[1, 2])
        
        # Shape features distribution
        shape_features = {k: v for k, v in features.items() 
                         if any(x in k for x in ['area', 'perimeter', 'nuclei', 'circularity'])}
        if shape_features:
            values = list(shape_features.values())
            ax2.hist(values, bins=20, color='#FF6B6B', alpha=0.7, edgecolor='black')
            ax2.set_title('Shape Features Distribution', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Feature Value')
            ax2.set_ylabel('Frequency')
            ax2.grid(axis='y', alpha=0.3)
        
        # Texture features distribution
        texture_features = {k: v for k, v in features.items() 
                           if any(x in k for x in ['texture', 'lbp', 'edge', 'entropy'])}
        if texture_features:
            values = list(texture_features.values())
            ax3.hist(values, bins=20, color='#4ECDC4', alpha=0.7, edgecolor='black')
            ax3.set_title('Texture Features Distribution', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Feature Value')
            ax3.set_ylabel('Frequency')
            ax3.grid(axis='y', alpha=0.3)
        
        # 3. Segmentation quality metrics
        ax4 = fig.add_subplot(gs[2, :])
        
        # Calculate segmentation metrics
        mask_binary = (mask > 0.5).astype(np.uint8)
        labeled_mask = measure.label(mask_binary, connectivity=2)
        regions = measure.regionprops(labeled_mask)
        
        if regions:
            areas = [r.area for r in regions]
            perimeters = [r.perimeter for r in regions]
            eccentricities = [r.eccentricity for r in regions]
            solidities = [r.solidity for r in regions]
            
            # Create box plots
            data = [areas, perimeters, eccentricities, solidities]
            labels = ['Area', 'Perimeter', 'Eccentricity', 'Solidity']
            
            bp = ax4.boxplot(data, labels=labels, patch_artist=True)
            
            # Color the boxes
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax4.set_title('Nuclei Morphological Features Distribution', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Value')
            ax4.grid(axis='y', alpha=0.3)
        
        # 4. Classification confidence analysis
        ax5 = fig.add_subplot(gs[3, 0])
        
        # Confidence vs uncertainty pie chart
        confidence = classification_result['confidence']
        uncertainty = 1 - confidence
        
        sizes = [confidence, uncertainty]
        labels = ['Confidence', 'Uncertainty']
        colors = ['#4CAF50', '#FFC107']
        explode = (0.1, 0)
        
        ax5.pie(sizes, explode=explode, labels=labels, colors=colors, 
               autopct='%1.1f%%', shadow=True, startangle=90)
        ax5.set_title(f'Classification Confidence\n{classification_result["predicted_class"].upper()}', 
                     fontsize=12, fontweight='bold')
        
        # 5. Feature importance summary
        ax6 = fig.add_subplot(gs[3, 1:3])
        
        # Create summary statistics table
        summary_data = []
        summary_data.append(['Total Features Extracted', len(features)])
        summary_data.append(['Nuclei Detected', len(regions) if regions else 0])
        summary_data.append(['Segmentation Coverage', f"{(mask_binary.sum() / mask_binary.size) * 100:.1f}%"])
        summary_data.append(['Classification Confidence', f"{confidence:.1%}"])
        summary_data.append(['Predicted Class', classification_result['predicted_class'].upper()])
        
        # Create table
        table = ax6.table(cellText=summary_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(summary_data) + 1):
            for j in range(2):
                cell = table[i, j]
                if i == 0:  # Header row
                    cell.set_facecolor('#667eea')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
        
        ax6.axis('off')
        ax6.set_title('Analysis Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        analysis_path = os.path.join(output_dir, 'feature_analysis.png')
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Saved detailed feature analysis to {analysis_path}")
    
    def _create_feature_report(self, features: Dict[str, float], output_dir: str):
        """This method is now replaced by _create_detailed_feature_analysis"""
        pass  # Functionality moved to _create_detailed_feature_analysis
    
    def process_image(self, image_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Process a single image through the feature-based pipeline"""
        # Create output directory
        if output_dir is None:
            output_dir = os.path.join('output', 'feature_based_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image
        logger.info(f"Processing image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Save original image info
        image_info = {
            'path': image_path,
            'size': image.size,
            'mode': image.mode
        }
        
        # Step 1: Segmentation
        logger.info("Performing segmentation...")
        mask, segmentation_results = self.segment_image(image)
        
        # Save segmentation mask
        mask_path = os.path.join(output_dir, 'segmentation_mask.png')
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(mask_path)
        
        # Step 2: Feature extraction
        logger.info("Extracting features from segmentation...")
        feature_vector, features_dict = self.extract_features(image_np, mask)
        
        # Step 3: Classification with features
        logger.info("Performing feature-based classification...")
        classification_result = self.classify_with_features(feature_vector)
        
        # Optional: Also run original classification for comparison
        comparison_result = None
        if self.classification_model is not None:
            logger.info("Running original classification for comparison...")
            image_tensor = self.cls_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.classification_model(image_tensor)
                
                # Handle models that return tuples (e.g., EnhancedEnsembleModel)
                if isinstance(output, tuple):
                    output = output[0]  # Take the main output
                
                probs = F.softmax(output, dim=1)
                _, pred = torch.max(output, 1)
                comparison_result = {
                    'predicted_class': self.config.class_names[pred.item()],
                    'confidence': float(probs[0, pred.item()])
                }
        
        # Prepare final results
        results = {
            'timestamp': datetime.now().isoformat(),
            'image_info': image_info,
            'segmentation': {
                'models_used': list(segmentation_results.keys()),
                'mask_saved_to': mask_path
            },
            'features': {
                'num_features': len(feature_vector),
                'feature_names': list(features_dict.keys()),
                'feature_values': {k: float(v) for k, v in features_dict.items()}
            },
            'classification': {
                'feature_based': classification_result,
                'original_model': comparison_result
            },
            'output_directory': output_dir
        }
        
        # Save results
        results_path = os.path.join(output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {results_path}")
        
        # Save features as CSV for analysis
        features_csv_path = os.path.join(output_dir, 'extracted_features.csv')
        import csv
        with open(features_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Feature', 'Value'])
            for k, v in features_dict.items():
                writer.writerow([k, v])
        
        # Create visualization if enabled
        if self.config.save_visualizations:
            self.visualize_results(image, mask, features_dict, classification_result, output_dir)
        
        logger.info(f"Pipeline completed. Results saved to {output_dir}")
        
        return results


def main():
    """Main function to run the feature-based pipeline"""
    parser = argparse.ArgumentParser(description='Feature-Based Classification Pipeline')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    parser.add_argument('--no_viz', action='store_true', help='Disable visualization')
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    if args.no_viz:
        config.save_visualizations = False
    
    # Initialize pipeline
    pipeline = FeatureBasedPipeline(config)
    
    # Process image
    results = pipeline.process_image(args.image_path, args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("FEATURE-BASED PIPELINE RESULTS")
    print("="*60)
    print(f"Image: {results['image_info']['path']}")
    print(f"Features extracted: {results['features']['num_features']}")
    print(f"\nFeature-based classification:")
    print(f"  Predicted: {results['classification']['feature_based']['predicted_class']}")
    print(f"  Confidence: {results['classification']['feature_based']['confidence']:.2%}")
    
    if results['classification']['original_model']:
        print(f"\nOriginal model (for comparison):")
        print(f"  Predicted: {results['classification']['original_model']['predicted_class']}")
        print(f"  Confidence: {results['classification']['original_model']['confidence']:.2%}")
    
    print(f"\nResults saved to: {results['output_directory']}")
    print("="*60)


if __name__ == '__main__':
    main() 