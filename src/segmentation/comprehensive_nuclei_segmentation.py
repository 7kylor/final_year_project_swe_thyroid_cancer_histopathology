import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging
import json
import matplotlib.pyplot as plt
from datetime import datetime
import random
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
import time
from sklearn.metrics import jaccard_score, accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import directed_hausdorff

"""
run this script to train the comprehensive nuclei segmentation model
"""
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the AdvancedNucleiMaskGenerator from the other file
from advanced_nuclei_mask_generator import AdvancedNucleiMaskGenerator

class Config:
    def __init__(self):
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Basic configuration
        self.seed = 42
        
        # Dataset paths
        self.data_root = "data"
        self.train_dir = os.path.join(self.data_root, "train/images")
        self.val_dir = os.path.join(self.data_root, "val/images")
        self.test_dir = os.path.join(self.data_root, "test/images")
        
        # Output directories
        self.output_dir = 'output'
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.output_dir, f"comprehensive_nuclei_segmentation_{current_time}")
        
        # Create output directories
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, 'predictions'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "training_logs"), exist_ok=True)
        
        # Input/output configuration
        self.input_channels = 3  # RGB images
        self.use_nuclei_features = True  # Whether to use nuclei features
        self.num_nuclei_features = 5  # Number of features per nuclei
        
        # Model configuration
        self.use_fastercnn = True  # Use FasterCNN by default instead of FastCNN
        self.batch_size = 4
        self.quick_test_samples = 32  # Number of samples to use for quick testing
        self.mask_method = 'adaptive_threshold'  # Default mask generation method
        
        # Ensemble weights configuration
        self.ensemble_weights = {
            'unet': 0.35, 
            'fastcnn': 0.30, 
            'fastercnn': 0.30,
            'segformer': 0.35
        }

    def _get_device(self):
        """
        Determine the best available device for processing
        
        Returns:
            torch.device: The device to use
        """
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using Apple Silicon (MPS)")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        
        return device

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Evaluation metrics
def dice_coefficient(pred, target, smooth=1e-6):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def precision_score(pred, target, smooth=1e-6):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return (intersection + smooth) / (pred_flat.sum() + smooth)

def recall_score(pred, target, smooth=1e-6):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return (intersection + smooth) / (target_flat.sum() + smooth)

def f1_score(precision, recall, smooth=1e-6):
    return 2 * (precision * recall) / (precision + recall + smooth)

class NucleiDataset(Dataset):
    """Dataset for nuclei segmentation with benign and malignant classes"""
    
    def __init__(self, data_dir, mode='train', transform=None, mask_transform=None, 
                 split_ratio=(0.7, 0.15, 0.15), mask_generator=None, use_nuclei_features=False, force_balanced=False):
        """
        Initialize the dataset
        
        Args:
            data_dir: Root directory containing 'benign' and 'malignant' subdirectories with images
            mode: One of 'train', 'val', 'test'
            transform: Image transforms
            mask_transform: Mask transforms
            split_ratio: Tuple of (train, val, test) ratios
            mask_generator: Optional mask generator to use at runtime
            use_nuclei_features: Whether to include nuclei features
            force_balanced: Whether to force a balanced dataset
        """
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.mask_transform = mask_transform
        self.mask_generator = mask_generator
        self.use_nuclei_features = use_nuclei_features
        self.force_balanced = force_balanced
        
        # Check if data_dir exists and has class directories
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory {data_dir} does not exist")
            
        # Find all images
        self.benign_dir = os.path.join(data_dir, 'benign')
        self.malignant_dir = os.path.join(data_dir, 'malignant')
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        if self.mask_transform is None:
            self.mask_transform = transforms.Compose([
                transforms.ToTensor()
            ])
        
        # List all image files for each class
        self.benign_imgs = []
        self.malignant_imgs = []
        
        if os.path.exists(self.benign_dir):
            self.benign_imgs = [os.path.join(self.benign_dir, f) for f in os.listdir(self.benign_dir) 
                               if os.path.isfile(os.path.join(self.benign_dir, f)) and 
                               (f.endswith('.png') or f.endswith('.tif') or f.endswith('.tiff'))]
        
        if os.path.exists(self.malignant_dir):
            self.malignant_imgs = [os.path.join(self.malignant_dir, f) for f in os.listdir(self.malignant_dir)
                                  if os.path.isfile(os.path.join(self.malignant_dir, f)) and
                                  (f.endswith('.png') or f.endswith('.tif') or f.endswith('.tiff'))]
        
        logger.info(f"Found {len(self.benign_imgs)} benign images and {len(self.malignant_imgs)} malignant images")
        
        # Find corresponding masks
        self.benign_masks = {}
        self.malignant_masks = {}
        
        # Check for binary masks
        binary_mask_dir = os.path.join(os.path.dirname(os.path.dirname(data_dir)), "mask_binary")
        
        if os.path.exists(binary_mask_dir):
            # Check for benign masks
            benign_mask_dir = os.path.join(binary_mask_dir, "benign")
            if os.path.exists(benign_mask_dir):
                for mask_file in os.listdir(benign_mask_dir):
                    if mask_file.endswith('.png') or mask_file.endswith('.tif') or mask_file.endswith('.tiff'):
                        # Extract image name
                        img_name = os.path.basename(mask_file).replace('_mask', '')
                        # Find corresponding image
                        for img_path in self.benign_imgs:
                            if os.path.basename(img_path) == img_name:
                                self.benign_masks[img_path] = os.path.join(benign_mask_dir, mask_file)
                                break
            
            # Check for malignant masks
            malignant_mask_dir = os.path.join(binary_mask_dir, "malignant")
            if os.path.exists(malignant_mask_dir):
                for mask_file in os.listdir(malignant_mask_dir):
                    if mask_file.endswith('.png') or mask_file.endswith('.tif') or mask_file.endswith('.tiff'):
                        # Extract image name
                        img_name = os.path.basename(mask_file).replace('_mask', '')
                        # Find corresponding image
                        for img_path in self.malignant_imgs:
                            if os.path.basename(img_path) == img_name:
                                self.malignant_masks[img_path] = os.path.join(malignant_mask_dir, mask_file)
                                break
        
        # Check if we found any masks or will need to generate them
        benign_mask_count = len(self.benign_masks)
        malignant_mask_count = len(self.malignant_masks)
        
        logger.info(f"Found {benign_mask_count} benign masks and {malignant_mask_count} malignant masks")
        
        # For images without corresponding masks, we'll generate them at runtime using mask_generator
        if (benign_mask_count < len(self.benign_imgs) or 
            malignant_mask_count < len(self.malignant_imgs)) and self.mask_generator is None:
            logger.warning("Not all images have corresponding masks and no mask_generator provided!")
        
        # Split datasets
        self.benign_indices = list(range(len(self.benign_imgs)))
        self.malignant_indices = list(range(len(self.malignant_imgs)))
        
        # Shuffle indices
        random.shuffle(self.benign_indices)
        random.shuffle(self.malignant_indices)
        
        # Calculate split points
        benign_train_size = int(len(self.benign_indices) * split_ratio[0])
        benign_val_size = int(len(self.benign_indices) * split_ratio[1])
        
        malignant_train_size = int(len(self.malignant_indices) * split_ratio[0])
        malignant_val_size = int(len(self.malignant_indices) * split_ratio[1])
        
        # Split indices
        if self.mode == 'train':
            self.benign_indices = self.benign_indices[:benign_train_size]
            self.malignant_indices = self.malignant_indices[:malignant_train_size]
        elif self.mode == 'val':
            self.benign_indices = self.benign_indices[benign_train_size:benign_train_size+benign_val_size]
            self.malignant_indices = self.malignant_indices[malignant_train_size:malignant_train_size+malignant_val_size]
        else:  # test
            self.benign_indices = self.benign_indices[benign_train_size+benign_val_size:]
            self.malignant_indices = self.malignant_indices[malignant_train_size+malignant_val_size:]
        
        # Make sure we have at least some images from each class for testing
        if self.mode == 'test' and (len(self.benign_indices) == 0 or len(self.malignant_indices) == 0):
            # If not enough images for both classes in test, take at least 2 from each class
            if len(self.benign_imgs) > 0:
                self.benign_indices = list(range(min(2, len(self.benign_imgs))))
            if len(self.malignant_imgs) > 0:
                self.malignant_indices = list(range(min(2, len(self.malignant_imgs))))
                
        # Force balanced dataset if requested
        if force_balanced and len(self.benign_imgs) > 0 and len(self.malignant_imgs) > 0:
            logger.info("Forcing balanced dataset...")
            # Take same number of samples from each class
            min_samples = min(len(self.benign_indices), len(self.malignant_indices))
            if min_samples == 0:
                min_samples = 2  # At least 2 samples from each class
            else:
                min_samples = max(min_samples, 2)  # At least 2 samples
                
            # Limit to the minimum number of samples
            self.benign_indices = self.benign_indices[:min_samples]
            self.malignant_indices = self.malignant_indices[:min_samples]
            
            logger.info(f"Balanced dataset: {min_samples} samples from each class")
        
        # Record class indices and filenames
        self.imgs = []
        self.labels = []
        
        for idx in self.benign_indices:
            if idx < len(self.benign_imgs):
                self.imgs.append(self.benign_imgs[idx])
                self.labels.append(0)  # 0 for benign
        
        for idx in self.malignant_indices:
            if idx < len(self.malignant_imgs):
                self.imgs.append(self.malignant_imgs[idx])
                self.labels.append(1)  # 1 for malignant
        
        # Log dataset statistics
        logger.info(f"{mode.capitalize()} dataset loaded with {len(self.imgs)} images")
        logger.info(f"Class distribution - Benign: {len(self.benign_indices)}, Malignant: {len(self.malignant_indices)}")
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # Get image path
        img_path = self.imgs[idx]
        label = self.labels[idx]
        
        # Determine class and index within class
        if label == 0:  # Benign
            class_name = "benign"
            mask_dict = self.benign_masks
        else:  # Malignant
            class_name = "malignant"
            mask_dict = self.malignant_masks
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get or generate mask
        if img_path in mask_dict:
            # Load existing mask
            mask_path = mask_dict[img_path]
            mask = Image.open(mask_path).convert('L')
        elif self.mask_generator is not None:
            # Generate mask on the fly
            try:
                # Convert PIL image to numpy array
                image_np = np.array(image)
                # Generate mask
                result = self.mask_generator.generate_mask(image_np)
                
                # Handle different return types
                mask_np = None
                if isinstance(result, tuple) and len(result) >= 1:
                    mask_np = result[0]  # First element is the mask
                else:
                    mask_np = result
                    
                # Ensure mask is valid
                if mask_np is not None and isinstance(mask_np, np.ndarray):
                    # Convert to binary mask
                    mask_np = (mask_np > 0).astype(np.float32)
                    # Convert back to PIL
                    mask = Image.fromarray((mask_np * 255).astype(np.uint8))
                else:
                    # Create empty mask
                    mask = Image.new('L', image.size, 0)
            except Exception as e:
                logger.error(f"Error generating mask for {img_path}: {e}")
                # Create empty mask
                mask = Image.new('L', image.size, 0)
        
        # Store original image for visualization
        original_image = np.array(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # Convert mask to tensor
            mask = torch.tensor(np.array(mask) / 255.0, dtype=torch.float32).unsqueeze(0)
        
        # Create sample
        sample = {
            'image': image,
            'mask': mask,
            'label': label,
            'class': class_name,
            'path': img_path,
            'original_image': original_image,
        }
        
        # Add nuclei features if needed
        if self.use_nuclei_features:
            try:
                # Generate fixed nuclei features - simplified version for testing
                # Create a fixed-size tensor with basic image statistics
                # This ensures compatibility with model expectations
                h, w = image.shape[1], image.shape[2]
                
                # Create a tensor with 5 feature maps (same size as image)
                nuclei_features = torch.zeros((5, h, w), dtype=torch.float32)
                
                # Try to extract real features if possible
                if self.mask_generator is not None:
                    try:
                        # Use a simplified mask to extract features
                        mask_np = mask.cpu().numpy()[0]
                        img_np = original_image
                        
                        # Extract simple features
                        # 1. Mean color channels
                        nuclei_features[0, :, :] = torch.tensor(np.mean(img_np, axis=2) / 255.0, dtype=torch.float32)
                        
                        # 2. Edge detection
                        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                        edges = cv2.Canny(gray, 50, 150) / 255.0
                        nuclei_features[1, :, :] = torch.tensor(cv2.resize(edges, (w, h)), dtype=torch.float32)
                        
                        # 3. Gaussian blur
                        blurred = cv2.GaussianBlur(gray, (5, 5), 0) / 255.0
                        nuclei_features[2, :, :] = torch.tensor(cv2.resize(blurred, (w, h)), dtype=torch.float32)
                        
                        # 4. Local binary pattern
                        from skimage.feature import local_binary_pattern
                        radius = 1
                        n_points = 8
                        lbp = local_binary_pattern(gray, n_points, radius, method='uniform') / 255.0
                        nuclei_features[3, :, :] = torch.tensor(cv2.resize(lbp, (w, h)), dtype=torch.float32)
                        
                        # 5. Binary mask
                        nuclei_features[4, :, :] = mask
                    except Exception as e:
                        logger.error(f"Error extracting detailed features: {e}")
                        # Keep the default zeroed features
                
                sample['nuclei_features'] = nuclei_features
            except Exception as e:
                logger.error(f"Error creating nuclei features for {img_path}: {e}")
                # Create empty features
                nuclei_features = torch.zeros((5, image.shape[1], image.shape[2]), dtype=torch.float32)
                sample['nuclei_features'] = nuclei_features
        
        return sample

# Model architecture components
class DimensionAwareModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def ensure_dimensions_match(self, x, target, mode='bilinear'):
        if x.shape[2:] != target.shape[2:]:
            x = nn.functional.interpolate(x, size=target.shape[2:], mode=mode, align_corners=True if mode=='bilinear' else None)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# UNet Architecture
class ImprovedUNet(DimensionAwareModule):
    def __init__(self, config):
        super().__init__()
        self.n_channels = config.input_channels
        self.use_nuclei_features = config.use_nuclei_features
        self.num_nuclei_features = config.num_nuclei_features if config.use_nuclei_features else 0
        
        # Input layers
        self.inc = DoubleConv(self.n_channels, 64)
        
        # Encoder
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        
        # Output layer
        self.outc = nn.Conv2d(64, 1, kernel_size=1)
        
        # Nuclei feature processing (if enabled)
        if self.use_nuclei_features:
            self.feature_fusion = nn.Sequential(
                nn.Linear(self.num_nuclei_features, 64),
                nn.ReLU(),
                nn.Linear(64, 256),
                nn.ReLU()
            )
            self.feature_spatial = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x):
        nuclei_features = None
        if isinstance(x, dict):
            nuclei_features = x.get('nuclei_features')
            x = x['image']
        
        # Initial convolution
        x1 = self.inc(x)
        
        # Encoder path
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Integrate nuclei features if available
        if self.use_nuclei_features and nuclei_features is not None:
            # Process nuclei features
            batch_size = x.shape[0]
            features = self.feature_fusion(nuclei_features)
            features = features.view(batch_size, 256, 1, 1)
            
            # Ensure features are broadcast correctly
            if features.shape[1] != x5.shape[1]:
                # If dimensions don't match, adjust the feature tensor
                features_processed = self.feature_spatial(features)
                # Use adaptive pooling to match dimensions if needed
                features_processed = nn.functional.adaptive_avg_pool2d(features_processed, output_size=x5.shape[2:])
                # Repeat the features to match the channel dimension
                features_processed = features_processed.repeat(1, x5.shape[1] // features_processed.shape[1], 1, 1)
                x5 = x5 + features_processed
            else:
                # If dimensions match, use as is
                features = features.expand(-1, -1, x5.shape[2], x5.shape[3])
                x5 = x5 + self.feature_spatial(features)
        
        # Decoder path
        x = self.up1(x5)
        x = self.ensure_dimensions_match(x, x4)
        x = self.conv1(torch.cat([x4, x], dim=1))
        
        x = self.up2(x)
        x = self.ensure_dimensions_match(x, x3)
        x = self.conv2(torch.cat([x3, x], dim=1))
        
        x = self.up3(x)
        x = self.ensure_dimensions_match(x, x2)
        x = self.conv3(torch.cat([x2, x], dim=1))
        
        x = self.up4(x)
        x = self.ensure_dimensions_match(x, x1)
        x = self.conv4(torch.cat([x1, x], dim=1))
        
        # Output
        x = self.outc(x)
        return torch.sigmoid(x)

# FastCNN Architecture
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = self.relu(out)
        return out

class ImprovedFastCNN(DimensionAwareModule):
    def __init__(self, config):
        super().__init__()
        self.n_channels = config.input_channels
        self.use_nuclei_features = config.use_nuclei_features
        self.num_nuclei_features = config.num_nuclei_features if config.use_nuclei_features else 0
        
        # Encoder blocks
        self.encoder1 = nn.Sequential(
            nn.Conv2d(self.n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64)
        )
        
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128)
        )
        
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256)
        )
        
        # Bridge
        self.bridge = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResidualBlock(512)
        )
        
        # Decoder blocks
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.outc = nn.Conv2d(64, 1, kernel_size=1)
        
        # Nuclei feature processing
        if self.use_nuclei_features:
            self.feature_fusion = nn.Sequential(
                nn.Linear(self.num_nuclei_features, 64),
                nn.ReLU(),
                nn.Linear(64, 512),
                nn.ReLU()
            )

    def forward(self, x):
        nuclei_features = None
        if isinstance(x, dict):
            nuclei_features = x.get('nuclei_features')
            x = x['image']
        
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        # Bridge
        b = self.bridge(e3)
        
        # Apply nuclei features if available
        if self.use_nuclei_features and nuclei_features is not None:
            batch_size = x.shape[0]
            features = self.feature_fusion(nuclei_features)
            
            # Handle potential dimension mismatch
            features = features.view(batch_size, 512, 1, 1)
            if features.shape[1] == b.shape[1]:
                features = features.expand(-1, -1, b.shape[2], b.shape[3])
                b = b + features
        else:
                # Adjust the features to match dimensions
                features = nn.functional.adaptive_avg_pool2d(
                    features.expand(-1, -1, 1, 1), 
                    output_size=(b.shape[2], b.shape[3])
                )
                features = features.repeat(1, b.shape[1] // features.shape[1], 1, 1)
                b = b + features
        
        # Decoder
        d1 = self.decoder1(b)
        d2 = self.decoder2(d1)
        d3 = self.decoder3(d2)
        
        # Output
        out = self.outc(d3)
        return torch.sigmoid(out)

# SegFormer-inspired Architecture
class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, C//num_heads]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ImprovedSegFormer(DimensionAwareModule):
    def __init__(self, config):
        super().__init__()
        self.n_channels = config.input_channels
        self.use_nuclei_features = config.use_nuclei_features
        self.num_nuclei_features = config.num_nuclei_features if config.use_nuclei_features else 0
        
        # Patch Embedding
        self.patch_embed1 = nn.Sequential(
            nn.Conv2d(self.n_channels, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.patch_embed2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.patch_embed3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.patch_embed4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Transformer blocks
        self.transformer1 = self._make_transformer_block(64, 2)
        self.transformer2 = self._make_transformer_block(128, 2)
        self.transformer3 = self._make_transformer_block(256, 2)
        self.transformer4 = self._make_transformer_block(512, 2)
        
        # MLP decode head
        self.linear_c4 = nn.Linear(512, 256)
        self.linear_c3 = nn.Linear(256, 128)
        self.linear_c2 = nn.Linear(128, 64)
        self.linear_c1 = nn.Linear(64, 64)
        
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Final layers
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(256, 1, kernel_size=1)
        
        # Nuclei feature processing
        if self.use_nuclei_features:
            self.feature_fusion = nn.Sequential(
                nn.Linear(self.num_nuclei_features, 64),
                nn.ReLU(),
                nn.Linear(64, 512),
                nn.ReLU()
            )
    
    def _make_transformer_block(self, dim, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            ))
        return nn.ModuleList(layers)
    
    def _apply_transformer_blocks(self, x, blocks):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        
        for block in blocks:
            x = x + block(x)
        
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

    def forward(self, x):
        nuclei_features = None
        if isinstance(x, dict):
            nuclei_features = x.get('nuclei_features')
            x = x['image']
        
        # Stage 1
        x1 = self.patch_embed1(x)  # [B, 64, H/4, W/4]
        x1 = self._apply_transformer_blocks(x1, self.transformer1)
        
        # Stage 2
        x2 = self.patch_embed2(x1)  # [B, 128, H/8, W/8]
        x2 = self._apply_transformer_blocks(x2, self.transformer2)
        
        # Stage 3
        x3 = self.patch_embed3(x2)  # [B, 256, H/16, W/16]
        x3 = self._apply_transformer_blocks(x3, self.transformer3)
        
        # Stage 4
        x4 = self.patch_embed4(x3)  # [B, 512, H/32, W/32]
        x4 = self._apply_transformer_blocks(x4, self.transformer4)
        
        # Apply nuclei features if available
        if self.use_nuclei_features and nuclei_features is not None:
            batch_size = x.shape[0]
            features = self.feature_fusion(nuclei_features)
            
            # Handle potential dimension mismatch
            features = features.view(batch_size, 512, 1, 1)
            if features.shape[1] == x4.shape[1]:
                features = features.expand(-1, -1, x4.shape[2], x4.shape[3])
                x4 = x4 + features
            else:
                # Adjust the features to match dimensions
                features = nn.functional.adaptive_avg_pool2d(
                    features.expand(-1, -1, 1, 1), 
                    output_size=(x4.shape[2], x4.shape[3])
                )
                features = features.repeat(1, x4.shape[1] // features.shape[1], 1, 1)
                x4 = x4 + features
        
        # Decode Head - with dimension alignment
        c4 = self.linear_c4(x4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B, 256, H/32, W/32]
        c4_upsampled = nn.functional.interpolate(c4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        
        c3 = self.linear_c3(x3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B, 128, H/16, W/16]
        # Match dimensions for addition
        if c3.shape[1] != c4_upsampled.shape[1]:
            # Adjust channels to match
            c4_adapted = nn.functional.conv2d(
                c4_upsampled, 
                weight=torch.ones(c3.shape[1], c4_upsampled.shape[1], 1, 1).to(c4_upsampled.device) / c4_upsampled.shape[1],
                bias=None
            )
            c3 = c3 + c4_adapted
        else:
            c3 = c3 + c4_upsampled
        
        c3_upsampled = nn.functional.interpolate(c3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        
        c2 = self.linear_c2(x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B, 64, H/8, W/8]
        # Match dimensions for addition
        if c2.shape[1] != c3_upsampled.shape[1]:
            # Adjust channels to match
            c3_adapted = nn.functional.conv2d(
                c3_upsampled, 
                weight=torch.ones(c2.shape[1], c3_upsampled.shape[1], 1, 1).to(c3_upsampled.device) / c3_upsampled.shape[1],
                bias=None
            )
            c2 = c2 + c3_adapted
        else:
            c2 = c2 + c3_upsampled
        
        c2_upsampled = nn.functional.interpolate(c2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
        c1 = self.linear_c1(x1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B, 64, H/4, W/4]
        # Match dimensions for addition
        if c1.shape[1] != c2_upsampled.shape[1]:
            # Adjust channels to match
            c2_adapted = nn.functional.conv2d(
                c2_upsampled, 
                weight=torch.ones(c1.shape[1], c2_upsampled.shape[1], 1, 1).to(c2_upsampled.device) / c2_upsampled.shape[1],
                bias=None
            )
            c1 = c1 + c2_adapted
        else:
            c1 = c1 + c2_upsampled
        
        # Fuse features
        fused = torch.cat([
            c1,
            nn.functional.interpolate(c2, size=c1.shape[2:], mode='bilinear', align_corners=False),
            nn.functional.interpolate(c3, size=c1.shape[2:], mode='bilinear', align_corners=False),
            nn.functional.interpolate(c4, size=c1.shape[2:], mode='bilinear', align_corners=False)
        ], dim=1)
        
        fused = self.linear_fuse(fused)
        fused = self.dropout(fused)
        fused = nn.functional.interpolate(fused, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Output
        seg = self.conv_seg(fused)
        return torch.sigmoid(seg)

class ResidualConnection(nn.Module):
    def __init__(self, downsample=None):
        super().__init__()
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        return x + identity

# Helper class for residual connections in FasterCNN
class AddLayerWithDownsample(nn.Module):
    def __init__(self, downsample=None):
        super().__init__()
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        return x + identity

# FasterCNN architecture optimized for nuclei segmentation
class NucleiFasterCNN(DimensionAwareModule):
    """Simplified FasterCNN-inspired architecture for nuclei segmentation"""
    
    def __init__(self, config):
        super().__init__()
        self.n_channels = config.input_channels
        self.use_nuclei_features = config.use_nuclei_features
        self.num_nuclei_features = config.num_nuclei_features if config.use_nuclei_features else 0
        
        # Initial layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(self.n_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Encoder blocks
        self.encoder1 = self._make_layer(64, 64, 2)
        self.encoder2 = self._make_layer(64, 128, 2, stride=2)
        self.encoder3 = self._make_layer(128, 256, 2, stride=2)
        self.encoder4 = self._make_layer(256, 512, 2, stride=2)
        
        # Decoder blocks with skip connections
        self.decoder1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.decoder3 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.decoder4 = nn.Sequential(
            nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # Final layers
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        
        # Nuclei feature processor
        if self.use_nuclei_features:
            self.feature_processor = nn.Sequential(
                nn.Linear(self.num_nuclei_features, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 256),
                nn.LayerNorm(256),
                nn.ReLU()
            )
            
            # Fusion modules for nuclei features
            self.feature_fusion1 = nn.Sequential(
                nn.Conv2d(512 + 256, 512, kernel_size=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # First block with potential downsampling
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers.append(self._make_block(in_channels, out_channels, stride, downsample))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(self._make_block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _make_block(self, in_channels, out_channels, stride=1, downsample=None):
        class Block(nn.Module):
            def __init__(self, in_ch, out_ch, stride, downsample):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_ch)
                self.downsample = downsample
            
            def forward(self, x):
                identity = x
                
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                
                out = self.conv2(out)
                out = self.bn2(out)
                
                if self.downsample is not None:
                    identity = self.downsample(x)
                
                out += identity
                out = self.relu(out)
                
                return out
        
        return Block(in_channels, out_channels, stride, downsample)
    
    def forward(self, x):
        nuclei_features = None
        if isinstance(x, dict):
            nuclei_features = x.get('nuclei_features')
            x = x['image']
        
        # Get input dimensions for later use
        input_h, input_w = x.shape[2], x.shape[3]
        
        # Encoder path
        x0 = self.input_layer(x)  # [B, 64, H/4, W/4]
        x1 = self.encoder1(x0)    # [B, 64, H/4, W/4]
        x2 = self.encoder2(x1)    # [B, 128, H/8, W/8]
        x3 = self.encoder3(x2)    # [B, 256, H/16, W/16]
        x4 = self.encoder4(x3)    # [B, 512, H/32, W/32]
        
        # Process nuclei features if available
        if self.use_nuclei_features and nuclei_features is not None:
            processed_features = self.feature_processor(nuclei_features)
            processed_features = processed_features.view(x4.shape[0], 256, 1, 1)
            processed_features = processed_features.expand(-1, -1, x4.shape[2], x4.shape[3])
            
            # Fuse with encoder features
            x4 = torch.cat([x4, processed_features], dim=1)
            x4 = self.feature_fusion1(x4)
        
        # Decoder path with skip connections
        d1 = self.decoder1(x4)  # [B, 256, H/16, W/16]
        
        # Match dimensions for skip connections using interpolation
        if d1.shape[2:] != x3.shape[2:]:
            x3_resized = F.interpolate(x3, size=d1.shape[2:], mode='bilinear', align_corners=False)
        else:
            x3_resized = x3
        d1 = torch.cat([d1, x3_resized], dim=1)  # [B, 512, H/16, W/16]
        
        d2 = self.decoder2(d1)  # [B, 128, H/8, W/8]
        if d2.shape[2:] != x2.shape[2:]:
            x2_resized = F.interpolate(x2, size=d2.shape[2:], mode='bilinear', align_corners=False)
        else:
            x2_resized = x2
        d2 = torch.cat([d2, x2_resized], dim=1)  # [B, 256, H/8, W/8]
        
        d3 = self.decoder3(d2)  # [B, 64, H/4, W/4]
        if d3.shape[2:] != x1.shape[2:]:
            x1_resized = F.interpolate(x1, size=d3.shape[2:], mode='bilinear', align_corners=False)
        else:
            x1_resized = x1
        d3 = torch.cat([d3, x1_resized], dim=1)  # [B, 128, H/4, W/4]
        
        d4 = self.decoder4(d3)  # [B, 32, H/2, W/2]
        
        # Final layers and upscale to input resolution
        output = self.final(d4)  # [B, 1, H/2, W/2]
        output = F.interpolate(output, size=(input_h, input_w), mode='bilinear', align_corners=False)
        
        return torch.sigmoid(output)

# Collaborative Ensemble model
class CollaborativeEnsemble(DimensionAwareModule):
    """Simple weighted ensemble of UNet, FastCNN/FasterCNN, and SegFormer"""
    
    def __init__(self, config=None, unet=None, fastcnn=None, segformer=None):
        super().__init__()
        self.config = config
        
        # Create or use provided UNet model
        if unet is None:
            self.unet = ImprovedUNet(config)
        else:
            self.unet = unet
            
        # Create or use provided FastCNN/FasterCNN model based on config
        if fastcnn is None:
            if hasattr(config, 'use_fastercnn') and config.use_fastercnn:
                logger.info("Using NucleiFasterCNN in Ensemble")
                self.fastcnn = NucleiFasterCNN(config)
            else:
                logger.info("Using ImprovedFastCNN in Ensemble")
                self.fastcnn = ImprovedFastCNN(config)
        else:
            self.fastcnn = fastcnn
            
        # Create or use provided SegFormer model
        if segformer is None:
            self.segformer = ImprovedSegFormer(config)
        else:
            self.segformer = segformer
    
    def forward(self, x):
        # Get UNet output - our primary model
        unet_output = self.unet(x)
        
        # Get outputs from supporting models
        with torch.no_grad():  # Don't compute gradients for supporting models
            fastcnn_output = self.fastcnn(x)
            segformer_output = self.segformer(x)
        
        # Ensure all outputs have the same shape
        target_shape = unet_output.shape[-2:] 
        
        if fastcnn_output.shape[-2:] != target_shape:
            fastcnn_output = F.interpolate(fastcnn_output, size=target_shape, mode='bilinear', align_corners=False)
            
        if segformer_output.shape[-2:] != target_shape:
            segformer_output = F.interpolate(segformer_output, size=target_shape, mode='bilinear', align_corners=False)
        
        # Ensure all outputs are 4D (B, C, H, W)
        if len(unet_output.shape) > 4:
            unet_output = unet_output.squeeze(2)
        
        if len(fastcnn_output.shape) > 4:
            fastcnn_output = fastcnn_output.squeeze(2)
        
        if len(segformer_output.shape) > 4:
            segformer_output = segformer_output.squeeze(2)
        
        # Simple weighted ensemble - heavily favor UNet
        # Use a 6:3:1 weighting for UNet:FasterCNN:SegFormer
        ensemble_output = (unet_output * 0.6 + 
                          fastcnn_output * 0.3 + 
                          segformer_output * 0.1)
        
        return ensemble_output
    
    def get_model_weights(self):
        """Return the fixed weights used for each model"""
        return np.array([0.6, 0.3, 0.1])

def create_models(config):
    """Create all the models based on config"""
    models = {
        'unet': ImprovedUNet(config)
    }
    
    # Add the appropriate CNN model based on config
    if hasattr(config, 'use_fastercnn') and config.use_fastercnn:
        models['fastercnn'] = NucleiFasterCNN(config)
    else:
        models['fastcnn'] = ImprovedFastCNN(config)
        
    models['segformer'] = ImprovedSegFormer(config)
    
    # Create the ensemble with the right components
    ensemble = CollaborativeEnsemble(config)
    models['ensemble'] = ensemble
    
    return {name: model.to(config.device) for name, model in models.items()}

def visualize_all_models_results(images, masks, all_outputs, labels, output_dir, prefix=""):
    """
    Visualize segmentation results for all models and save to disk
    
    Args:
        images: List of original images (numpy arrays)
        masks: List of ground truth masks
        all_outputs: Dictionary of model outputs for each model
        labels: List of class labels (0=benign, 1=malignant)
        output_dir: Directory to save visualizations
        prefix: Prefix for saved files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (image, mask, label) in enumerate(zip(images, masks, labels)):
        class_name = "malignant" if label == 1 else "benign"
        
        # Create comparison visualization for all models
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(image)
        plt.title(f"Original ({class_name})")
        plt.axis('off')
        
        # Ground truth mask
        plt.subplot(2, 3, 2)
        plt.imshow(mask.squeeze(), cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')
        
        # Model outputs
        for idx, (model_name, outputs) in enumerate(all_outputs.items()):
            plt.subplot(2, 3, idx+3)
            plt.imshow(outputs[i].squeeze(), cmap='gray')
            plt.title(f"{model_name.capitalize()} Prediction")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_sample_{i}_{class_name}_comparison.png"))
        plt.close()
        
        # Save overlay visualization for ensemble model
        overlay = image.copy()
        # Add red channel for ground truth and green channel for prediction
        overlay_mask = np.zeros_like(overlay)
        overlay_mask[:,:,0] = mask.squeeze() * 255 * 0.5  # Red for ground truth
        overlay_mask[:,:,1] = all_outputs['ensemble'][i].squeeze() * 255 * 0.5  # Green for prediction
        # Combine
        overlay = cv2.addWeighted(overlay, 1.0, overlay_mask, 0.5, 0)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay)
        plt.title(f"Overlay: Ground Truth (red) vs Ensemble Prediction (green)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_overlay_{i}_{class_name}.png"))
        plt.close()

def generate_comparative_report(config, all_metrics):
    """
    Generate a comprehensive report comparing all models
    
    Args:
        config: Configuration object
        all_metrics: Dictionary with metrics for each model
    """
    report = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "use_nuclei_features": config.use_nuclei_features,
        "metrics": all_metrics
    }
    
    # Save as JSON
    report_path = os.path.join(config.output_dir, "reports", f"comparative_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Print summary
    print("\n" + "="*80)
    print(f"Nuclei Segmentation Comparative Report")
    print("="*80)
    
    for model_name, metrics in all_metrics.items():
        print(f"\n--- {model_name.upper()} MODEL ---")
        print("\nOverall Metrics:")
        for metric, value in metrics['overall'].items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nClass-specific Metrics:")
        for class_name, class_metrics in metrics['by_class'].items():
            print(f"\n  {class_name}:")
            for metric, value in class_metrics.items():
                print(f"    {metric}: {value:.4f}")
    
    print(f"\nDetailed comparative report saved to: {report_path}")
    return report

def evaluate_all_models(models, dataloader, config):
    """
    Evaluate all models on the given dataloader and produce visualizations and metrics
    
    Args:
        models: Dictionary of models
        dataloader: DataLoader with test data
        config: Configuration object
    
    Returns:
        Dictionary of metrics for all models
    """
    device = config.device
    all_metrics = {}
    
    # For visualization
    sample_images = []
    sample_masks = []
    sample_labels = []
    sample_outputs = {model_name: [] for model_name in models.keys()}
    
    max_samples = 5  # Limit the number of samples for visualization
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name.upper()} model...")
        model.eval()
        
        # Metrics
        dice_scores = []
        iou_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        # Class-specific metrics
        benign_metrics = {
            'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f1': []
        }
        malignant_metrics = {
            'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f1': []
        }
        
        # Process batches
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {model_name}")):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                labels = batch['label'].to(device)
                original_images = batch['original_image'].numpy()
                
                # Prepare input (handle nuclei features if present)
                if config.use_nuclei_features and 'nuclei_features' in batch:
                    nuclei_features = batch['nuclei_features'].to(device)
                    inputs = {'image': images, 'nuclei_features': nuclei_features}
                else:
                    inputs = images
                
                # Forward pass - special handling for ensemble model
                outputs = model(inputs)
                
                # Ensure outputs are in sigmoid range [0, 1] for metric calculation
                if model_name == 'ensemble' and model.training == False:
                    # No need to apply sigmoid again - ensemble already returns sigmoid in eval mode
                    pass
                
                # Calculate metrics for each sample in batch
                for i in range(outputs.shape[0]):
                    output = outputs[i].detach().cpu().numpy()
                    mask = masks[i].detach().cpu().numpy()
                    label = labels[i].item()
                    
                    # Binarize prediction
                    pred = (output > 0.5).astype(np.float32)
                    
                    # Calculate metrics
                    dice = dice_coefficient(pred, mask)
                    iou = iou_score(pred, mask)
                    prec = precision_score(pred, mask)
                    rec = recall_score(pred, mask)
                    f1 = f1_score(prec, rec)
                    
                    dice_scores.append(dice)
                    iou_scores.append(iou)
                    precision_scores.append(prec)
                    recall_scores.append(rec)
                    f1_scores.append(f1)
                    
                    # Store by class
                    if label == 0:  # Benign
                        benign_metrics['dice'].append(dice)
                        benign_metrics['iou'].append(iou)
                        benign_metrics['precision'].append(prec)
                        benign_metrics['recall'].append(rec)
                        benign_metrics['f1'].append(f1)
                    else:  # Malignant
                        malignant_metrics['dice'].append(dice)
                        malignant_metrics['iou'].append(iou)
                        malignant_metrics['precision'].append(prec)
                        malignant_metrics['recall'].append(rec)
                        malignant_metrics['f1'].append(f1)
                    
                    # Store samples for visualization
                    if len(sample_images) < max_samples:
                        sample_images.append(original_images[i])
                        sample_masks.append(mask)
                        sample_labels.append(label)
                        sample_outputs[model_name].append(pred)
        # Calculate overall metrics
        overall_dice = np.mean(dice_scores) if dice_scores else 0
        overall_iou = np.mean(iou_scores) if iou_scores else 0
        overall_precision = np.mean(precision_scores) if precision_scores else 0
        overall_recall = np.mean(recall_scores) if recall_scores else 0
        overall_f1 = np.mean(f1_scores) if f1_scores else 0
        
        # Calculate class-specific metrics
        benign_dice = np.mean(benign_metrics['dice']) if benign_metrics['dice'] else 0
        benign_iou = np.mean(benign_metrics['iou']) if benign_metrics['iou'] else 0
        benign_precision = np.mean(benign_metrics['precision']) if benign_metrics['precision'] else 0
        benign_recall = np.mean(benign_metrics['recall']) if benign_metrics['recall'] else 0
        benign_f1 = np.mean(benign_metrics['f1']) if benign_metrics['f1'] else 0
        
        malignant_dice = np.mean(malignant_metrics['dice']) if malignant_metrics['dice'] else 0
        malignant_iou = np.mean(malignant_metrics['iou']) if malignant_metrics['iou'] else 0
        malignant_precision = np.mean(malignant_metrics['precision']) if malignant_metrics['precision'] else 0
        malignant_recall = np.mean(malignant_metrics['recall']) if malignant_metrics['recall'] else 0
        malignant_f1 = np.mean(malignant_metrics['f1']) if malignant_metrics['f1'] else 0
        
        # Store metrics in all_metrics
        all_metrics[model_name] = {
            'overall': {
                'dice': overall_dice,
                'iou': overall_iou,
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1
            },
            'by_class': {
                'benign': {
                    'dice': benign_dice,
                    'iou': benign_iou,
                    'precision': benign_precision,
                    'recall': benign_recall,
                    'f1': benign_f1
                },
                'malignant': {
                    'dice': malignant_dice,
                    'iou': malignant_iou,
                    'precision': malignant_precision,
                    'recall': malignant_recall,
                    'f1': malignant_f1
                }
            }
        }
        
        # Print metrics
        print(f"\n--- {model_name.upper()} MODEL ---")
        print("\nOverall Metrics:")
        for metric, value in all_metrics[model_name]['overall'].items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nClass-specific Metrics:")
        for class_name, class_metrics in all_metrics[model_name]['by_class'].items():
            print(f"\n  {class_name}:")
            for metric, value in class_metrics.items():
                print(f"    {metric}: {value:.4f}")
    
    # Visualize results for all models
    if len(sample_images) > 0:
        # Organize outputs in the right format for visualization
        all_model_outputs = {}
        for model_name in models.keys():
            all_model_outputs[model_name] = sample_outputs[model_name]
        
        # Visualize
        vis_dir = os.path.join(config.output_dir, "visualizations")
        visualize_all_models_results(
            sample_images, 
            sample_masks, 
            all_model_outputs,
            sample_labels, 
            vis_dir, 
            prefix="test"
        )
    
    return all_metrics

def load_pretrained_models(models, config):
    """
    Load pre-trained weights for all models if available
    
    Args:
        models: Dictionary of models
        config: Configuration object
    
    Returns:
        Dictionary of loaded models
    """
    for model_name, model in models.items():
        model_path = os.path.join(config.output_dir, "checkpoints", f"{model_name}_model.pth")
        if os.path.exists(model_path):
            logger.info(f"Loading pre-trained {model_name} model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=config.device))
        else:
            logger.warning(f"No pre-trained model found for {model_name} at {model_path}. Using untrained model.")
    
    return models

def main(skip_training=True):
    """
    Main function to run the nuclei segmentation pipeline
    
    Args:
        skip_training: If True, skip training and just do evaluation
    """
    # Initialize configuration
    config = Config()
    
    # Set smaller batch size to avoid memory issues
    config.batch_size = 2
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Disable nuclei features for testing to reduce complexity
    config.use_nuclei_features = False
    
    # Initialize mask generator with improved settings
    logger.info("Initializing AdvancedNucleiMaskGenerator with improved settings...")
    mask_generator = AdvancedNucleiMaskGenerator(method=config.mask_method)
    
    # Create test dataset with a small number of samples for quicker testing
    logger.info("Creating test dataset with samples from both benign and malignant classes...")
    test_dataset = NucleiDataset(
        data_dir=config.test_dir,
        mode='test',
        mask_generator=mask_generator,
        use_nuclei_features=config.use_nuclei_features,
        force_balanced=True  # Force including samples from both classes
    )
    
    # Limit to a small number of samples for quicker testing
    max_samples = 10
    if len(test_dataset) > max_samples:
        logger.info(f"Limiting test dataset to {max_samples} samples for quicker testing")
        indices = np.random.choice(len(test_dataset), max_samples, replace=False)
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, indices)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=1,  # Reduce for debugging
        pin_memory=True
    )
    
    logger.info(f"Test dataset: {len(test_dataset)} images")
    
    # Create output directories
    os.makedirs(os.path.join(config.output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)
    
    # Create only UNet model for testing
    logger.info("Creating UNet model only for testing...")
    model = ImprovedUNet(config).to(config.device)
    
    # Test UNet on dataset
    logger.info("Evaluating UNet model...")
    model.eval()
    
    # Create visualization directory
    vis_dir = os.path.join(config.output_dir, "visualizations", "unet")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Initialize metrics
    benign_metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f1': []}
    malignant_metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f1': []}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing UNet")):
            # Handle Subset wrapper for dataset
            if isinstance(batch, tuple):
                batch = batch[0]
                
            images = batch['image'].to(config.device)
            masks = batch['mask'].to(device)
            labels = batch['label']
            class_names = batch['class']
            paths = batch['path']
            
            # Forward pass
            outputs = model(images)
            
            # Convert to binary masks for visualization and metrics
            preds = (outputs > 0.5).float()
            
            # Process each image in the batch
            for i in range(images.size(0)):
                img = images[i].cpu()
                mask = masks[i].cpu()
                pred = preds[i].cpu()
                label = labels[i].item() if isinstance(labels, torch.Tensor) else labels[i]
                class_name = class_names[i] if isinstance(class_names, list) else class_names
                img_path = paths[i] if isinstance(paths, list) else paths
                
                # Denormalize image for visualization
                img_np = img.numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1)
                img_vis = (img_np * 255).astype(np.uint8)
                
                # Convert mask and prediction for visualization
                mask_vis = (mask.numpy()[0] * 255).astype(np.uint8)
                pred_vis = (pred.numpy()[0] * 255).astype(np.uint8)
                
                # Calculate metrics
                intersection = torch.sum(pred * mask).item()
                union = torch.sum(pred) + torch.sum(mask) - intersection
                union = union.item()
                
                dice = 2 * intersection / (torch.sum(pred) + torch.sum(mask)).item() if (torch.sum(pred) + torch.sum(mask)).item() > 0 else 0.0
                iou = intersection / union if union > 0 else 0.0
                
                # Calculate precision, recall, and F1
                tp = intersection
                fp = torch.sum(pred * (1 - mask)).item()
                fn = torch.sum((1 - pred) * mask).item()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                # Store metrics by class
                if label == 0:  # Benign
                    benign_metrics['dice'].append(dice)
                    benign_metrics['iou'].append(iou)
                    benign_metrics['precision'].append(precision)
                    benign_metrics['recall'].append(recall)
                    benign_metrics['f1'].append(f1)
                else:  # Malignant
                    malignant_metrics['dice'].append(dice)
                    malignant_metrics['iou'].append(iou)
                    malignant_metrics['precision'].append(precision)
                    malignant_metrics['recall'].append(recall)
                    malignant_metrics['f1'].append(f1)
                
                # Log metrics
                logger.info(f"Sample {batch_idx * config.batch_size + i + 1} ({class_name}) - "
                          f"Dice: {dice:.4f}, IoU: {iou:.4f}, F1: {f1:.4f}")
                
                # Create overlay (red: ground truth, green: prediction)
                overlay = np.zeros_like(img_vis)
                overlay[:, :, 0] = pred_vis  # Red for predictions
                overlay[:, :, 1] = mask_vis  # Green for ground truth
                
                # Save visualizations
                filename_base = os.path.basename(img_path).split('.')[0]
                
                # Save original image
                Image.fromarray(img_vis).save(os.path.join(vis_dir, f"{class_name}_{filename_base}_original.png"))
                
                # Save ground truth mask
                Image.fromarray(mask_vis).save(os.path.join(vis_dir, f"{class_name}_{filename_base}_ground_truth.png"))
                
                # Save prediction mask
                Image.fromarray(pred_vis).save(os.path.join(vis_dir, f"{class_name}_{filename_base}_prediction.png"))
                
                # Save overlay
                Image.fromarray(overlay).save(os.path.join(vis_dir, f"{class_name}_{filename_base}_overlay.png"))
    
    # Calculate and log average metrics for each class
    logger.info("\n===== AVERAGE METRICS =====")
    
    # Benign metrics
    if benign_metrics['dice']:
        avg_dice = sum(benign_metrics['dice']) / len(benign_metrics['dice'])
        avg_iou = sum(benign_metrics['iou']) / len(benign_metrics['iou'])
        avg_precision = sum(benign_metrics['precision']) / len(benign_metrics['precision'])
        avg_recall = sum(benign_metrics['recall']) / len(benign_metrics['recall'])
        avg_f1 = sum(benign_metrics['f1']) / len(benign_metrics['f1'])
        
        logger.info(f"BENIGN: Dice={avg_dice:.4f}, IoU={avg_iou:.4f}, Precision={avg_precision:.4f}, "
                    f"Recall={avg_recall:.4f}, F1={avg_f1:.4f}")
    else:
        logger.warning("No benign samples were processed")
    
    # Malignant metrics
    if malignant_metrics['dice']:
        avg_dice = sum(malignant_metrics['dice']) / len(malignant_metrics['dice'])
        avg_iou = sum(malignant_metrics['iou']) / len(malignant_metrics['iou'])
        avg_precision = sum(malignant_metrics['precision']) / len(malignant_metrics['precision'])
        avg_recall = sum(malignant_metrics['recall']) / len(malignant_metrics['recall'])
        avg_f1 = sum(malignant_metrics['f1']) / len(malignant_metrics['f1'])
        
        logger.info(f"MALIGNANT: Dice={avg_dice:.4f}, IoU={avg_iou:.4f}, Precision={avg_precision:.4f}, "
                    f"Recall={avg_recall:.4f}, F1={avg_f1:.4f}")
    else:
        logger.warning("No malignant samples were processed")
    
    logger.info(f"Testing complete. Results saved to {config.output_dir}")

def train_all_models(models_to_train=None, epochs=15, batch_size=8, num_samples=None, save_checkpoint_freq=5, quick_test=False):
    """Train all specified models"""
    # Create configuration
    config = Config()
    config.epochs = epochs
    config.batch_size = batch_size
    
    # Enable quick test mode if specified
    if quick_test:
        config.quick_test_samples = 32 if num_samples is None else num_samples
        logger.info(f"Quick test enabled, using {config.quick_test_samples} samples")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        config, 
        batch_size=batch_size,
        num_samples=num_samples
    )
    
    # Configure models to train
    if models_to_train is None:
        models_to_train = ['unet', 'fastcnn', 'segformer', 'ensemble']
    
    if config.use_fastercnn and 'fastcnn' in models_to_train:
        # Replace FastCNN with FasterCNN if enabled
        idx = models_to_train.index('fastcnn')
        models_to_train[idx] = 'fastercnn'
    
    logger.info(f"Beginning training of {len(models_to_train)} models at {datetime.now().strftime('%H:%M:%S')}")
    
    # Create models
    models = create_models(config)
    
    # Train each model
    for i, model_name in enumerate(models_to_train):
        if model_name not in models:
            logger.warning(f"Model {model_name} not found, skipping")
            continue
                
        logger.info(f"===== Training {model_name.upper()} model ({i+1}/{len(models_to_train)}) =====")
        
        model = models[model_name]
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Create directories for checkpoints
        checkpoint_dir = os.path.join(config.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create directory for this run's output
        run_dir = os.path.join(config.output_dir)
        os.makedirs(run_dir, exist_ok=True)
        
        # Train the model
        device = config.device
        logger.info(f"Starting training {model_name} for {epochs} epochs on {device}")
        start_time = time.time()
        logger.info(f"{model_name} training started at {datetime.now().strftime('%H:%M:%S')}")
        
        best_val_loss = float('inf')
        best_dice = 0.0
        
        for epoch in range(1, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs}")
            
            # Training
            model.train()
            train_loss = 0.0
            batch_count = 0
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training {model_name} Epoch {epoch}")):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                batch_count += 1
                
                # Log progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # Calculate average training loss
            avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_batch_count = 0
            dice_scores = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validating {model_name} Epoch {epoch}"):
                    images = batch['image'].to(device)
                    masks = batch['mask'].to(device)
                    
                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                    val_loss += loss.item()
                    val_batch_count += 1
                    
                    # Calculate Dice score
                    preds = (outputs > 0.5).float()
                    batch_dice = dice_coefficient(preds.cpu().numpy(), masks.cpu().numpy())
                    dice_scores.append(batch_dice)
            
            # Calculate average validation metrics
            avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
            avg_dice_score = np.mean(dice_scores) if dice_scores else 0
            
            # Log epoch metrics
            logger.info(f"  {model_name} Epoch {epoch} - Training Loss: {avg_train_loss:.4f}, "
                      f"Validation Loss: {avg_val_loss:.4f}, Dice Score: {avg_dice_score:.4f}")
            
            # Save model if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_dice = avg_dice_score
                
                # Save model
                model_path = os.path.join(checkpoint_dir, f"{model_name}_model.pth")
                torch.save(model.state_dict(), model_path)
                logger.info(f"  Saved {model_name} model to {model_path} (Best validation loss: {best_val_loss:.4f})")
        
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Log learning rate
            logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Log best metrics
            logger.info(f"  Best Dice Score: {best_dice:.4f}")
            logger.info(f"  Best Validation Loss: {best_val_loss:.4f}")
            
            # Log epoch duration
            epoch_duration = time.time() - start_time
            logger.info(f"  Epoch Duration: {epoch_duration:.2f} seconds")
            
            # Log training time
            logger.info(f"  Total Training Time: {time.time() - start_time:.2f} seconds")
            
            # Log training history
            history = {
                'train_loss': [avg_train_loss],
                'val_loss': [avg_val_loss],
                'val_dice': [avg_dice_score],
                'training_time': [time.time() - start_time],
                'device': str(device),
                'batch_size': batch_size,
                'best_val_loss': best_val_loss,
                'best_val_dice': best_dice
            }
            
            with open(os.path.join(run_dir, f"{model_name}_training_history.json"), 'w') as f:
                json.dump(history, f, indent=4)
        
        # Log final metrics
        logger.info(f"Training complete! {model_name} - Best Validation Loss: {best_val_loss:.4f}, Best Dice Score: {best_dice:.4f}")
        logger.info(f"Total Training Time: {time.time() - start_time:.2f} seconds")
        
        # Store trained model in dictionary
        trained_models = {model_name: model}
        
        # Store metrics for overall comparison
        all_training_metrics = {
            model_name: {
            'best_val_loss': best_val_loss,
                'best_val_dice': best_dice,
                'training_time': time.time() - start_time
            }
        }
        
        # Reset batch size to original value if it was changed for SegFormer
        if model_name == 'segformer' and config.batch_size != batch_size:
            logger.info(f"Resetting batch size from {config.batch_size} to original {batch_size}")
            config.batch_size = batch_size
            # Recreate dataloaders with original batch size for next models
            train_loader, val_loader = create_dataloaders(batch_size)
    
    # Calculate overall training time
    overall_time = time.time() - start_time
    logger.info(f"===== ALL MODELS TRAINING COMPLETE =====")
    logger.info(f"Total training time: {overall_time:.2f} seconds")
    
    # Save overall training summary
    overall_summary = {
        'total_training_time': str(overall_time),
        'device': str(config.device),
        'timestamp': config.run_dir.split('_')[-1],
        'models_trained': list(all_training_metrics.keys()),
        'metrics': all_training_metrics
    }
    
    with open(os.path.join(config.output_dir, "training_logs", "overall_training_summary.json"), 'w') as f:
        json.dump(overall_summary, f, indent=4)
    
    # Print comparative summary
    logger.info("\n===== COMPARATIVE VALIDATION PERFORMANCE =====")
    for model_name, metrics in all_training_metrics.items():
        logger.info(f"{model_name.upper()}: Val Loss={metrics['best_val_loss']:.4f}, "
                    f"Dice={metrics['best_val_dice']:.4f}, Time={metrics['training_time']:.2f} seconds")
    
    # Return trained models and the config with updated paths
    return trained_models, config

def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, config, epochs=10):
    """Train a model for nuclei segmentation"""
    device = config.device  # Ensure we're using config.device consistently
    logger.info(f"Training {model_name} for {epochs} epochs")
    
    # Create results directory
    results_dir = os.path.join(config.output_dir, 'training_results', model_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup metrics tracking
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'epochs': list(range(1, epochs + 1))
    }
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training {model_name} Epoch {epoch}")):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            
            # Log progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Calculate average training loss
        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        dice_scores = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating {model_name} Epoch {epoch}"):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_batch_count += 1
                
                # Calculate Dice score
                preds = (outputs > 0.5).float()
                batch_dice = dice_coefficient(preds.cpu().numpy(), masks.cpu().numpy())
                dice_scores.append(batch_dice)
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
        avg_dice_score = np.mean(dice_scores) if dice_scores else 0
        
        # Log epoch metrics
        logger.info(f"  {model_name} Epoch {epoch} - Training Loss: {avg_train_loss:.4f}, "
                  f"Validation Loss: {avg_val_loss:.4f}, Dice Score: {avg_dice_score:.4f}")
        
        # Save model checkpoint
        model_path = os.path.join(results_dir, f"{model_name}_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': {
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_dice': avg_dice_score
            }
        }, model_path)
        
        # Update metrics
        metrics['train_loss'].append(avg_train_loss)
        metrics['val_loss'].append(avg_val_loss)
        metrics['val_dice'].append(avg_dice_score)
    
    # Log final metrics
    logger.info(f"Training complete! {model_name} - Best Validation Loss: {min(metrics['val_loss']):.4f}, Best Dice Score: {max(metrics['val_dice']):.4f}")
    logger.info(f"Total Training Time: {time.time() - start_time:.2f} seconds")
    
    return model, metrics

def test_models(models_to_test=None, batch_size=4, num_samples=None, save_visualizations=True, quick_test=False):
    """
    Test models on the nuclei segmentation test dataset.
    
    Args:
        models_to_test (list): List of model names to test. If None, test all available models.
        batch_size (int): Batch size for testing.
        num_samples (int): Number of samples to test. If None, test all samples.
        save_visualizations (bool): Whether to save visualizations of model predictions.
        quick_test (bool): If True, use a reduced number of samples for quick testing.
    
    Returns:
        dict: Dictionary of test results for each model.
    """
    # Initialize configuration
    config = Config()
    config.batch_size = batch_size
    
    # Set device reference for consistency
    device = config.device
    
    # Get dataset size based on quick test flag
    if quick_test:
        num_samples = min(config.quick_test_samples, num_samples) if num_samples else config.quick_test_samples
        logger.info(f"Quick test enabled, using {num_samples} samples")
    
    # Create test dataset and loader
    test_loader = create_test_dataloader(config, batch_size, num_samples)
    
    # Log test dataset info
    logger.info(f"Testing on {len(test_loader.dataset)} samples with batch size {batch_size}")
    
    # Define the default models to test
    available_models = ['unet', 'fastcnn', 'fastercnn', 'segformer', 'ensemble']
    
    # If no models specified, test all available
    if models_to_test is None:
        if config.use_fastercnn:
            # Default to FasterCNN instead of FastCNN
            models_to_test = ['unet', 'fastercnn', 'segformer', 'ensemble']
        else:
            models_to_test = ['unet', 'fastcnn', 'segformer', 'ensemble']
    
    # Validate model names
    for model_name in models_to_test:
        if model_name not in available_models:
            logger.warning(f"Unknown model: {model_name}. Skipping.")
            models_to_test.remove(model_name)
    
    # Load or create models
    logger.info(f"Testing models: {', '.join(models_to_test)}")
    models = {}
    
    # Define checkpoint directory - use central output/checkpoints directory
    checkpoint_dir = os.path.join('output', 'checkpoints')
    
    # Check if models are saved and load them, or create new ones
    for model_name in models_to_test:
        model_path = os.path.join(checkpoint_dir, f"{model_name}_model.pth")
        
        if os.path.exists(model_path):
            logger.info(f"Loading {model_name} from {model_path}")
            if model_name == 'unet':
                models[model_name] = ImprovedUNet(config).to(config.device)
            elif model_name == 'fastcnn':
                models[model_name] = ImprovedFastCNN(config).to(config.device)
            elif model_name == 'fastercnn':
                models[model_name] = NucleiFasterCNN(config).to(config.device)
            elif model_name == 'segformer':
                models[model_name] = ImprovedSegFormer(config).to(config.device)
            elif model_name == 'ensemble':
                models[model_name] = CollaborativeEnsemble(config).to(config.device)
            
            # Load model weights
            try:
                checkpoint = torch.load(model_path, map_location=config.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    models[model_name].load_state_dict(checkpoint['model_state_dict'])
                else:
                    models[model_name].load_state_dict(checkpoint)
                logger.info(f"Successfully loaded {model_name} weights")
            except Exception as e:
                logger.error(f"Failed to load {model_name} weights: {str(e)}")
                
                # If ensemble failed to load but we have the component models, create it from scratch
                if model_name == 'ensemble' and all(m in models for m in ['unet', 'fastcnn' if not config.use_fastercnn else 'fastercnn', 'segformer']):
                    logger.info("Creating ensemble from loaded component models")
                    
                    # Use the appropriate CNN model based on config
                    cnn_model = models['fastercnn'] if config.use_fastercnn else models['fastcnn']
                    
                    models['ensemble'] = CollaborativeEnsemble(
                        config=config,
                        unet=models['unet'],
                        fastcnn=cnn_model,
                        segformer=models['segformer']
                    ).to(config.device)
        else:
            logger.warning(f"Could not find saved model for {model_name} at {model_path}")
            
            # Create new model if not found
            if model_name == 'unet':
                models[model_name] = ImprovedUNet(config).to(config.device)
            elif model_name == 'fastcnn':
                models[model_name] = ImprovedFastCNN(config).to(config.device)
            elif model_name == 'fastercnn':
                models[model_name] = NucleiFasterCNN(config).to(config.device)
            elif model_name == 'segformer':
                models[model_name] = ImprovedSegFormer(config).to(config.device)
            elif model_name == 'ensemble':
                # Skip ensemble if component models are not available
                required_models = ['unet', 'segformer']
                if config.use_fastercnn:
                    required_models.append('fastercnn')
                else:
                    required_models.append('fastcnn')
                
                if not all(m in models for m in required_models):
                    logger.warning(f"Cannot create ensemble without component models: {', '.join(required_models)}")
                    continue
                
                # Create ensemble from component models
                logger.info("Creating ensemble from available component models")
                
                # Use the appropriate CNN model based on config
                cnn_model = models['fastercnn'] if config.use_fastercnn else models['fastcnn']
                
                models['ensemble'] = CollaborativeEnsemble(
                    config=config,
                    unet=models['unet'],
                    fastcnn=cnn_model,
                    segformer=models['segformer']
                ).to(config.device)
    
    # Evaluate each model
    test_results = {}
    
    for model_name, model in models.items():
        logger.info(f"===== Testing {model_name.upper()} model =====")
        model.eval()
        
        # Metrics
        dice_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        iou_scores = []
        binary_accuracy_scores = []
        hausdorff_distances = []
        inference_times = []
        
        # Process the test set
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Testing {model_name}")):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # Measure inference time
                start_time = time.time()
                outputs = model(images)
                end_time = time.time()
                inference_time = (end_time - start_time) / images.shape[0]  # Per-image inference time
                inference_times.append(inference_time)
                
                # Binarize outputs for evaluation
                binary_outputs = (outputs > 0.5).float()
                
                # Calculate metrics (handling empty masks)
                for i in range(images.shape[0]):
                    gt = masks[i].cpu().numpy().squeeze()
                    pred = binary_outputs[i].cpu().numpy().squeeze()
                    
                    # Skip if either gt or pred is all zeros/ones
                    if np.all(gt == 0) and np.all(pred == 0):
                        # Both empty - perfect match
                        dice_scores.append(1.0)
                        precision_scores.append(1.0)
                        recall_scores.append(1.0)
                        f1_scores.append(1.0)
                        iou_scores.append(1.0)
                        binary_accuracy_scores.append(1.0)
                        hausdorff_distances.append(0.0)
                        continue
                    
                    # Calculate Dice score
                    dice = calculate_dice_score(pred, gt)
                    dice_scores.append(dice)
                    
                    # Calculate precision and recall with manual handling of zero division
                    try:
                        precision = precision_score(gt.flatten(), pred.flatten())
                    except Exception:
                        # If all predictions are negative (no positive class), precision is 1.0
                        precision = 1.0 if np.all(pred == 0) else 0.0
                    
                    try:
                        recall = recall_score(gt.flatten(), pred.flatten())
                    except Exception:
                        # If all actual values are negative (no positive class), recall is 1.0
                        recall = 1.0 if np.all(gt == 0) else 0.0
                    
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    
                    # Calculate F1 score with manual handling of zero division
                    try:
                        f1 = f1_score(gt.flatten(), pred.flatten())
                    except Exception:
                        # If precision and recall are both 0, f1 is 0
                        f1 = 0.0 if (precision == 0 or recall == 0) else (2 * precision * recall) / (precision + recall)
                    
                    f1_scores.append(f1)
                    
                    # Calculate IoU (Jaccard index) with manual handling of zero division
                    try:
                        iou = jaccard_score(gt.flatten(), pred.flatten())
                    except Exception:
                        # If union is empty, IoU is 1.0
                        iou = 1.0 if np.all(gt == 0) and np.all(pred == 0) else 0.0
                    
                    iou_scores.append(iou)
                    
                    # Calculate binary accuracy
                    accuracy = accuracy_score(gt.flatten(), pred.flatten())
                    binary_accuracy_scores.append(accuracy)
                    
                    # Calculate Hausdorff distance if both masks have content
                    if np.any(gt > 0) and np.any(pred > 0):
                        try:
                            hd = directed_hausdorff(np.argwhere(gt > 0), np.argwhere(pred > 0))[0]
                            hausdorff_distances.append(hd)
                        except Exception as e:
                            logger.warning(f"Failed to calculate Hausdorff distance: {str(e)}")
                else:
                        # One is empty, the other is not - maximum distance
                        hausdorff_distances.append(max(gt.shape[0], gt.shape[1]))
                
                # Save visualizations for the first batch
                if save_visualizations and batch_idx == 0:
                    save_batch_visualizations(images, masks, outputs, model_name, os.path.join('output', 'visualizations'))
        
        # Calculate average metrics
        avg_dice = np.mean(dice_scores)
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_f1 = np.mean(f1_scores)
        avg_iou = np.mean(iou_scores)
        avg_accuracy = np.mean(binary_accuracy_scores)
        avg_hausdorff = np.mean(hausdorff_distances) if hausdorff_distances else float('inf')
        avg_inference_time = np.mean(inference_times)
        
        # Log results
        logger.info(f"{model_name.upper()} Results:")
        logger.info(f"  - Dice Score: {avg_dice:.4f}")
        logger.info(f"  - Precision: {avg_precision:.4f}")
        logger.info(f"  - Recall: {avg_recall:.4f}")
        logger.info(f"  - F1 Score: {avg_f1:.4f}")
        logger.info(f"  - IoU: {avg_iou:.4f}")
        logger.info(f"  - Binary Accuracy: {avg_accuracy:.4f}")
        logger.info(f"  - Average Hausdorff Distance: {avg_hausdorff:.4f}")
        logger.info(f"  - Average Inference Time: {avg_inference_time*1000:.2f} ms per image")
        
        # Store results
        test_results[model_name] = {
            'dice_score': float(avg_dice),
            'precision': float(avg_precision),
            'recall': float(avg_recall),
            'f1_score': float(avg_f1),
            'iou': float(avg_iou),
            'accuracy': float(avg_accuracy),
            'hausdorff_distance': float(avg_hausdorff),
            'inference_time_ms': float(avg_inference_time * 1000)
        }
    
    # Create comparative report
    comparative_report = {
        'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(config.device),
        'num_samples': len(test_loader.dataset),
        'batch_size': batch_size,
        'results': test_results
    }
    
    # Save comparative report
    report_path = os.path.join('output', 'comparative_report.json')
    with open(report_path, 'w') as f:
        json.dump(comparative_report, f, indent=4)
    
    logger.info(f"Comparative report saved to {report_path}")
    
    return test_results

def create_test_dataloader(config, batch_size=None, num_samples=None):
    """
    Create a DataLoader for the test dataset.
    
    Args:
        config (Config): Configuration object with dataset paths
        batch_size (int, optional): Batch size for testing. Defaults to config.batch_size.
        num_samples (int, optional): Number of samples to use. If None, use all samples.
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for the test dataset
    """
    # Use batch size from config if not provided
    if batch_size is None:
        batch_size = config.batch_size
    
    # Initialize mask generator with adaptive threshold method
    mask_generator = AdvancedNucleiMaskGenerator(method=config.mask_method)
    
    # Create test dataset using the original NucleiDataset class
    test_dataset = NucleiDataset(
        data_dir=config.test_dir,
        mode='test',
        mask_generator=mask_generator,
        use_nuclei_features=config.use_nuclei_features
    )
    
    # Limit dataset size if specified
    if num_samples and len(test_dataset) > num_samples:
        logger.info(f"Limiting test dataset to {num_samples} samples")
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, indices)
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    return test_loader

def create_dataloaders(config, batch_size=None, num_samples=None):
    """
    Create DataLoaders for training and validation datasets.
    
    Args:
        config (Config): Configuration object with dataset paths
        batch_size (int, optional): Batch size for training and validation. Defaults to config.batch_size.
        num_samples (int, optional): Number of samples to use. If None, use all samples.
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Use batch size from config if not provided
    if batch_size is None:
        batch_size = config.batch_size
    
    # Initialize mask generator with adaptive threshold method
    mask_generator = AdvancedNucleiMaskGenerator(method=config.mask_method)
    
    # Create training dataset
    train_dataset = NucleiDataset(
        data_dir=config.train_dir,
        mode='train',
        mask_generator=mask_generator,
        use_nuclei_features=config.use_nuclei_features
    )
    
    # Create validation dataset
    val_dataset = NucleiDataset(
        data_dir=config.val_dir,
        mode='val',
        mask_generator=mask_generator,
        use_nuclei_features=config.use_nuclei_features
    )
    
    # Limit dataset size if specified
    if num_samples and len(train_dataset) > num_samples:
        logger.info(f"Limiting training dataset to {num_samples} samples")
        indices = np.random.choice(len(train_dataset), num_samples, replace=False)
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, indices)
    
    if num_samples and len(val_dataset) > max(1, num_samples // 10):
        val_samples = max(1, num_samples // 10)
        logger.info(f"Limiting validation dataset to {val_samples} samples")
        indices = np.random.choice(len(val_dataset), val_samples, replace=False)
        from torch.utils.data import Subset
        val_dataset = Subset(val_dataset, indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader

def save_batch_visualizations(images, masks, outputs, model_name, output_dir):
    """
    Save visualizations of model predictions for a batch of images.
    
    Args:
        images (torch.Tensor): Batch of input images [B, C, H, W]
        masks (torch.Tensor): Batch of ground truth masks [B, 1, H, W]
        outputs (torch.Tensor): Batch of model predictions [B, 1, H, W]
        model_name (str): Name of the model
        output_dir (str): Directory to save visualizations
    """
    vis_dir = os.path.join(output_dir, 'visualizations', model_name)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Save up to 4 images from the batch
    num_images = min(images.shape[0], 4)
    
    for i in range(num_images):
        # Convert tensors to numpy arrays
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        mask = masks[i].cpu().numpy()
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        output = outputs[i].cpu().numpy()
        if len(output.shape) > 2:
            output = output.squeeze()
        
        # Normalize image for visualization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Convert to uint8 for saving
        img = (img * 255).astype(np.uint8)
        mask = (mask * 255).astype(np.uint8)
        output = (output * 255).astype(np.uint8)
        
        # Create overlay (red: ground truth, green: prediction)
        overlay = img.copy()
        overlay[..., 0] = np.maximum(overlay[..., 0], mask // 2)  # Add red for ground truth
        overlay[..., 1] = np.maximum(overlay[..., 1], output // 2)  # Add green for prediction
        
        # Save images
        cv2.imwrite(os.path.join(vis_dir, f'sample_{i}_input.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(vis_dir, f'sample_{i}_ground_truth.png'), mask)
        cv2.imwrite(os.path.join(vis_dir, f'sample_{i}_prediction.png'), output)
        cv2.imwrite(os.path.join(vis_dir, f'sample_{i}_overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
class AdvancedNucleiMaskGenerator:
    """Enhanced mask generator for nuclei segmentation with multiple methods"""
    
    def __init__(self, method='adaptive_threshold'):
        """
        Initialize the mask generator
        
        Args:
            method (str): Mask generation method ('adaptive_threshold', 'watershed', 'ensemble')
        """
        self.method = method
        logger.info(f"Initialized AdvancedNucleiMaskGenerator with method: {method}")
    
    def generate_mask(self, image):
        """
        Generate a mask for the given image
        
        Args:
            image (numpy.ndarray): Input image (BGR format)
            
        Returns:
            numpy.ndarray: Binary mask
        """
        # Convert BGR to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply method based on selection
        if self.method == 'adaptive_threshold':
            return self._adaptive_threshold_method(gray)
        elif self.method == 'watershed':
            return self._watershed_method(image, gray)
        elif self.method == 'ensemble':
            # Combine multiple methods
            mask1 = self._adaptive_threshold_method(gray)
            mask2 = self._watershed_method(image, gray)
            # Combine masks (intersection)
            return cv2.bitwise_and(mask1, mask2)
        else:
            logger.warning(f"Unknown method: {self.method}. Using adaptive threshold.")
            return self._adaptive_threshold_method(gray)
    
    def _adaptive_threshold_method(self, gray):
        """Adaptive threshold based segmentation"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        return opening
    
    def _watershed_method(self, image, gray):
        """Watershed based segmentation"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        
        # Finding unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        if len(image.shape) == 2:
            # Convert grayscale to BGR for watershed
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_color = image.copy()
        
        markers = cv2.watershed(image_color, markers)
        
        # Create mask
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[markers > 1] = 255
        
        return mask

def calculate_dice_score(pred, target, smooth=1e-5):
    """
    Calculate Dice coefficient between predicted and target masks
    
    Args:
        pred: predicted binary mask (after thresholding)
        target: ground truth binary mask
        smooth: smoothing factor to prevent division by zero
    
    Returns:
        float: Dice coefficient (0-1)
    """
    if not isinstance(pred, np.ndarray):
        pred = pred.detach().cpu().numpy()
    if not isinstance(target, np.ndarray):
        target = target.detach().cpu().numpy()
    
    # Ensure binary masks
    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)
    
    # Calculate Dice
    intersection = np.sum(pred * target)
    return (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Comprehensive Nuclei Segmentation Pipeline')
    parser.add_argument('--train', action='store_true', help='Train the models')
    parser.add_argument('--test', action='store_true', help='Test the models')
    parser.add_argument('--visualize', action='store_true', help='Save visualization results')
    parser.add_argument('--quick', action='store_true', help='Run with minimal samples for quick testing')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and testing')
    parser.add_argument('--samples', type=int, default=None, help='Number of samples to use (for quick testing)')
    parser.add_argument('--models', type=str, default=None, help='Comma-separated list of models to train/test')
    parser.add_argument('--use_fastercnn', action='store_true', help='Use FasterCNN instead of FastCNN')
    parser.add_argument('--method', type=str, default='adaptive_threshold', choices=['adaptive_threshold', 'watershed', 'ensemble'], 
                      help='Nuclei mask generation method')
    args = parser.parse_args()
    
    # Convert models string to list if provided
    models_list = None
    if args.models:
        models_list = args.models.split(',')
    
    # Create configuration with command line parameters
    config = Config()
    
    # Set use_fastercnn flag in config
    if args.use_fastercnn:
        config.use_fastercnn = True
        logger.info("Running with config: use_fastercnn=True")
    
    # Set mask generation method
    config.mask_method = args.method
    logger.info(f"Using mask generation method: {config.mask_method}")
    
    # Print which models will be used
    if models_list:
        logger.info(f"Models: {', '.join(models_list)}")
    
    # Train models if requested
    if args.train:
        logger.info("=== Training Mode ===")
        
        # Create directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "training_logs"), exist_ok=True)
        
        # Train models with provided parameters
        train_all_models(
            models_to_train=models_list,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_samples=args.samples,
            quick_test=args.quick
        )
    
    # Test models if requested
    if args.test:
        logger.info("=== Testing Mode ===")
        
        # Test with provided parameters
        test_models(
            models_to_test=models_list,
            batch_size=args.batch_size,
            num_samples=args.samples,
            save_visualizations=args.visualize,
            quick_test=args.quick
        )