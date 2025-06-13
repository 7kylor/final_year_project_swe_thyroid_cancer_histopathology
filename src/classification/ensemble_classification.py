"""
PyTorch Implementation of Optimized Ensemble Classification Model for Histopathology Image Diagnosis

This script implements an ensemble model using three state-of-the-art CNN architectures:
  - ResNet152 (deeper and more powerful than ResNet50)
  - DenseNet121
  - EfficientNetV2_L (more recent and powerful than EfficientNetB4)

Each model is fine-tuned for binary classification (benign vs malignant) and their predictions
are fused using one of several strategies:
  1. Weighted fusion: Learnable scalar weights (softmax normalized) are used to combine the logits.
  2. Concat fusion: Logits are concatenated and further processed with a fully-connected layer.
  3. Attentive fusion: Attention-based dynamic weighting of model outputs
  4. Max fusion: The model with highest confidence for a prediction is selected.

The dataset is assumed to be organized in the following structure under the 'data' directory:
    data/train/images/benign
    data/train/images/malignant
    data/val/images/benign
    data/val/images/malignant
    data/test/images/benign
    data/test/images/malignant

Supported image formats: TIFF (".tif", ".tiff"), JPG, JPEG, and PNG.

Usage:
    python ensemble_classification.py --mode train --data_root data --epochs 50 --batch_size 16
    python ensemble_classification.py --mode val --epochs 5 --batch_size 16
    python ensemble_classification.py --mode test --batch_size 16
    python ensemble_classification.py --mode test --batch_size 16 --with_insights
"""

import os
import sys
import time
import json
import random
import logging
import argparse
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union

# Set up logging with a more comprehensive approach
# Define function to suppress NumExpr thread info messages
def suppress_numexpr_messages():
    """Suppress NumExpr's verbose thread allocation messages."""
    # Set NumExpr environment variable to control threading messages
    os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())
    
    # Configure the numexpr logger to only show warnings
    numexpr_logger = logging.getLogger('numexpr')
    if numexpr_logger:
        numexpr_logger.setLevel(logging.WARNING)
    
    # Also try to intercept other NumExpr messages
    try:
        import numexpr
        numexpr.set_num_threads(os.cpu_count())
    except (ImportError, AttributeError):
        pass

# Call suppression function immediately
suppress_numexpr_messages()

def setup_logger():
    """Set up and configure the logger to prevent duplicate logs using a singleton pattern."""
    # Use a module-level variable to track whether logger has been configured
    if hasattr(setup_logger, "initialized") and setup_logger.initialized:
        return logging.getLogger(__name__)
    
    # Get the root logger and configure it
    root_logger = logging.getLogger()
    
    # Clear all existing handlers to prevent duplication
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    # Configure the root logger with an improved format showing date, level, and message
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Add a single console handler to the root logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Configure library loggers to appropriate levels to reduce noise
    for logger_name in ['torch', 'torchvision', 'PIL', 'matplotlib', 'numexpr']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Special handling for NumExpr which tends to be very verbose
    # Set the NumExpr environment variable to control threading messages
    os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())
    
    # Get our module logger - it will inherit settings from root
    logger = logging.getLogger(__name__)
    
    # Mark logger as initialized to prevent duplicate setup
    setup_logger.initialized = True
    
    # Store handler as a module attribute for reference
    setup_logger.console_handler = console_handler
    
    # Prevent additional basicConfig calls from having any effect
    logging.basicConfig = lambda **kwargs: None
    
    return logger

# Ensure attributes exist even if setup_logger is not called
setup_logger.initialized = False
setup_logger.console_handler = None

# Initialize global logger with our singleton setup
logger = setup_logger()

# Prevent additional basicConfig calls from having any effect
logging.basicConfig = lambda **kwargs: None

# Check for required packages
required_packages = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'torch': 'torch',
    'torchvision': 'torchvision',
    'PIL': 'pillow',
    'tqdm': 'tqdm',
    'matplotlib': 'matplotlib',  # Optional for insights
    'seaborn': 'seaborn',        # Optional for insights
    'sklearn': 'scikit-learn'    # Optional for insights
}

missing_packages = []
optional_missing = []

for module_name, package_name in required_packages.items():
    try:
        __import__(module_name)
    except ImportError:
        if module_name in ['matplotlib', 'seaborn', 'sklearn']:
            optional_missing.append(package_name)
        else:
            missing_packages.append(package_name)

if missing_packages:
    logger.error(f"Required packages missing: {', '.join(missing_packages)}")
    logger.error(f"Please install them using: pip install {' '.join(missing_packages)}")
    sys.exit(1)

if optional_missing:
    logger.warning(f"Optional packages for insights missing: {', '.join(optional_missing)}")
    logger.warning(f"For full insights functionality, install: pip install {' '.join(optional_missing)}")

# Now import required packages
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchvision import transforms, models
from torchvision.transforms import ToTensor, Normalize, Resize, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter

# Additional imports for model insights and visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, roc_auc_score
    HAS_VISUALIZATION_LIBS = True
except ImportError:
    HAS_VISUALIZATION_LIBS = False
    logger.warning("Visualization libraries not available. Install matplotlib, seaborn, and scikit-learn for full insights capabilities.")

# ---------------------------------------------------

# Function to ensure directories exist
def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")
    return directory

# ---------------------------------------------------
# Device Selection and Utility Functions
# ---------------------------------------------------

def select_device():
    """Select the best available device: CUDA (NVIDIA), MPS (Apple), or CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        # Enable cuDNN benchmark for better performance
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        logger.info("GPU not available, using CPU")
    return device


def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS doesn't support direct seed setting like CUDA
        torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed}")


# ---------------------------------------------------
# Configuration and Argument Parsing
# ---------------------------------------------------

class ModelConfig:
    """
    Configuration for optimized ensemble model training and evaluation.
    
    Features:
    - Advanced training optimizations
    - Efficient memory usage settings
    - Mixed precision training support
    - Improved scheduler options
    - Performance-tuned defaults
    """
    def __init__(self, **kwargs):
        # Default configuration
        # General settings
        self.model_name = "ensemble"
        self.data_root = kwargs.get('data_root', 'data')
        self.num_classes = kwargs.get('num_classes', 2)
        self.image_size = kwargs.get('image_size', 224)
        self.num_workers = kwargs.get('num_workers', None)  # Auto-determined in create_dataloaders
        self.use_weighted_sampling = kwargs.get('use_weighted_sampling', True)
        
        # Model architecture
        self.active_models = kwargs.get('active_models', ['resnet', 'densenet', 'efficientnet'])
        self.fusion_type = kwargs.get('fusion_type', 'attentive' )  # 'weighted', 'concat', 'attentive', 'max' , 'cross_attention'
        self.trainable_layers = kwargs.get('trainable_layers', {'resnet': -1, 'densenet': -1, 'efficientnet': -1})
        self.dropout_rate = kwargs.get('dropout_rate', 0.2)
        self.model_weights = kwargs.get('model_weights', {'resnet': 0.25, 'densenet': 0.50, 'efficientnet': 0.25})
        self.attention_heads = kwargs.get('attention_heads', 4)  # For attentive fusion
        
        # Training parameters
        self.mode = kwargs.get('mode', 'train')
        self.batch_size = kwargs.get('batch_size', 16)
        self.num_epochs = kwargs.get('num_epochs', 50)
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.weight_decay = kwargs.get('weight_decay', 1e-5)
        self.label_smoothing = kwargs.get('label_smoothing', 0.1)
        
        # Memory optimization and mixed precision
        self.use_amp = kwargs.get('use_amp', True)  # Automatic mixed precision
        self.use_grad_scaler = kwargs.get('use_grad_scaler', True)
        self.grad_accum_steps = kwargs.get('grad_accum_steps', 1)
        self.use_fp16 = kwargs.get('use_fp16', True)
        self.cache_data = kwargs.get('cache_data', False)
        self.gradient_checkpointing = kwargs.get('gradient_checkpointing', False)  # Trade compute for memory
        
        # Learning rate scheduler
        self.use_scheduler = kwargs.get('use_scheduler', True)
        self.scheduler_type = kwargs.get('scheduler_type', 'cosine')  # 'cosine', 'reduce_on_plateau', 'warmup_cosine', 'one_cycle'
        self.warmup_epochs = kwargs.get('warmup_epochs', 3)
        self.warmup_factor = kwargs.get('warmup_factor', 10.0)
        self.min_lr = kwargs.get('min_lr', 1e-6)
        
        # Checkpointing and early stopping
        self.checkpoint_dir = kwargs.get('checkpoint_dir', 'checkpoints_classification')
        self.save_best_only = kwargs.get('save_best_only', True)
        self.checkpoint_freq = kwargs.get('checkpoint_freq', 1)
        self.max_checkpoints = kwargs.get('max_checkpoints', 5)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 10)
        self.early_stopping_delta = kwargs.get('early_stopping_delta', 1e-4)
        
        # Evaluation and insights
        self.with_insights = kwargs.get('with_insights', False)
        self.insights_dir = kwargs.get('insights_dir', "insights/classification")
        self.eval_interval = kwargs.get('eval_interval', 1)
        self.log_interval = kwargs.get('log_interval', 10)
        
        # Logging
        self.no_tensorboard = kwargs.get('no_tensorboard', False)
        self.log_dir = kwargs.get('log_dir', 'logs')
        
        # Advanced options
        self.seed = kwargs.get('seed', 42)
        self.compile_model = kwargs.get('compile_model', False)  # PyTorch 2.0+ model compilation
        
        # Update with any other provided arguments
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)
        
        # Ensure directories exist
        ensure_dir(self.checkpoint_dir)
        ensure_dir(self.log_dir)
        if self.with_insights:
            ensure_dir(self.insights_dir)
            
    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Ensemble Classification Training Script')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'], 
                      help='Mode: train, val or test')
    parser.add_argument('--data_root', type=str, default='data', 
                      help='Root directory for dataset')
    parser.add_argument('--image_size', type=int, default=224, 
                      help='Input image size')
    parser.add_argument('--epochs', type=int, default=50, 
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, 
                      help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, 
                      help='Weight decay for optimizer')
    parser.add_argument('--fusion_type', type=str, default='weighted', 
                      choices=['weighted', 'concat', 'max', 'attentive', 'cross_attention'], 
                      help='Type of ensemble fusion')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_classification', 
                      help='Directory to save checkpoints')
    parser.add_argument('--save_best_only', action='store_true', 
                      help='Only save checkpoint if model improves')
    parser.add_argument('--seed', type=int, default=42, 
                      help='Random seed')
    parser.add_argument('--disable_amp', action='store_true', 
                      help='Disable automatic mixed precision')
    parser.add_argument('--no_cache_data', action='store_true', 
                      help='Disable dataset caching')
    parser.add_argument('--early_stopping', action='store_true', 
                      help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=10, 
                      help='Patience for early stopping')
    parser.add_argument('--grad_accum_steps', type=int, default=1, 
                      help='Number of gradient accumulation steps')
    parser.add_argument('--active_models', type=str, default='resnet,densenet,efficientnet', 
                      help='Comma-separated list of models to use in ensemble')
    parser.add_argument('--dropout_rate', type=float, default=0.2, 
                      help='Dropout rate for regularization')
    parser.add_argument('--label_smoothing', type=float, default=0.1, 
                      help='Label smoothing factor for loss function')
    parser.add_argument('--unfreeze_layers', type=str, default='', 
                      help='Comma separated list of layers to unfreeze (format: model:layers, e.g., resnet:10,densenet:5,efficientnet:10)')
    parser.add_argument('--no_scheduler', action='store_true', 
                      help='Disable learning rate scheduler')
    parser.add_argument('--scheduler_type', type=str, default='cosine', 
                      choices=['cosine', 'reduce_on_plateau'], 
                      help='Type of learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=3, 
                      help='Number of warmup epochs')
    parser.add_argument('--warmup_factor', type=float, default=10.0, 
                      help='Factor to divide learning rate by during warmup')
    parser.add_argument('--no_tensorboard', action='store_true', 
                      help='Disable TensorBoard logging')
    parser.add_argument('--checkpoint_freq', type=int, default=1, 
                      help='Save checkpoint every N epochs')
    parser.add_argument('--max_checkpoints', type=int, default=20, 
                      help='Maximum number of checkpoints to keep')
    parser.add_argument('--resume', action='store_true', default=False,
                      help='Resume training from latest checkpoint')
    parser.add_argument('--with_insights', action='store_true', default=True,
                      help='Generate and save detailed model insights')
    parser.add_argument('--insights_dir', type=str, default='insights/classification',
                      help='Directory to save model insights')
    return parser.parse_args()


def parse_unfreeze_layers(unfreeze_str):
    """Parse the unfreeze layers string to a dictionary."""
    if not unfreeze_str:
        return {'resnet': -1, 'densenet': -1, 'efficientnet': -1}
    
    result = {'resnet': -1, 'densenet': -1, 'efficientnet': -1}
    parts = unfreeze_str.split(',')
    
    for part in parts:
        if ':' in part:
            model, layers = part.split(':')
            if model in result and layers.isdigit():
                result[model] = int(layers)
    
    return result


# ---------------------------------------------------
# Dataset and DataLoader
# ---------------------------------------------------

class HistopathologyDataset(Dataset):
    """
    Dataset for histopathology image classification with performance optimizations.
    
    Features:
    - Image cache to reduce I/O operations
    - Efficient error handling
    - Memory-efficient data loading
    - Support for dataset statistics calculation
    """
    def __init__(self, root_dir, transform=None, mode='train', cache_size=100):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.cache_size = cache_size
        self.image_cache = {}  # Simple LRU cache for frequently accessed images
        
        # Define image directories
        mode_dir = os.path.join(root_dir, mode)
        self.images_dir = os.path.join(mode_dir, 'images')
        
        if not os.path.exists(self.images_dir):
            raise ValueError(f"Images directory does not exist: {self.images_dir}")
        
        # Get class folders
        self.class_dirs = [d for d in os.listdir(self.images_dir) 
                         if os.path.isdir(os.path.join(self.images_dir, d))]
        
        if not self.class_dirs:
            raise ValueError(f"No class directories found in {self.images_dir}")
        
        logger.info(f"Found classes: {self.class_dirs}")
        
        # Map class names to indices
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(self.class_dirs))}
        
        # Get all image paths and labels
        self.img_paths = []
        self.labels = []
        
        # Track class distribution for weighted sampling
        self.class_counts = {cls: 0 for cls in self.class_dirs}
        
        # Valid image extensions
        valid_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        
        # Load dataset with progress bar for large datasets
        for class_name in self.class_dirs:
            class_dir = os.path.join(self.images_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files for this class
            files = [
                f for f in os.listdir(class_dir) 
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]
            
            # Add to dataset
            for file_name in files:
                self.img_paths.append(os.path.join(class_dir, file_name))
                self.labels.append(class_idx)
                self.class_counts[class_name] += 1
        
        # Calculate class weights for potential weighted sampling
        total_samples = len(self.img_paths)
        self.class_weights = {
            cls: total_samples / (len(self.class_dirs) * count) if count > 0 else 0
            for cls, count in self.class_counts.items()
        }
        
        # Convert to sample weights for weighted sampling
        self.sample_weights = [
            self.class_weights[os.path.basename(os.path.dirname(path))] 
            for path in self.img_paths
        ]
        
        logger.info(f"Loaded {len(self.img_paths)} images for '{mode}' mode")
        logger.info(f"Class distribution: {self.class_counts}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        
        # Check cache first
        if idx in self.image_cache:
            img = self.image_cache[idx]
        else:
            # Load image with error handling
            try:
                img = Image.open(img_path).convert('RGB')
                
                # Update cache (simple LRU implementation)
                if len(self.image_cache) >= self.cache_size:
                    # Remove random item if cache is full
                    self.image_cache.pop(next(iter(self.image_cache)))
                self.image_cache[idx] = img
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                # Return a placeholder if image loading fails
                img = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms if any
        if self.transform:
            img = self.transform(img)
        
        # For binary classification, return a single label for BCE loss or one-hot for CE loss
        if len(self.class_to_idx) == 2:  # Binary classification
            if self.class_to_idx.get('malignant', -1) == 1:  # Ensure 'malignant' is labeled as 1
                return img, torch.tensor([label], dtype=torch.float32)
            else:
                return img, torch.tensor([label], dtype=torch.float32)
        else:  # Multi-class classification
            one_hot = torch.zeros(len(self.class_to_idx))
            one_hot[label] = 1.0
            return img, one_hot
    
    def get_weighted_sampler(self):
        """
        Returns a weighted sampler to handle class imbalance.
        
        Returns:
            torch.utils.data.WeightedRandomSampler: Sampler that handles class imbalance
        """
        weights = torch.DoubleTensor(self.sample_weights)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, len(weights), replacement=True
        )
        return sampler
    
    def calculate_statistics(self):
        """
        Calculate dataset statistics for normalization.
        
        Returns:
            tuple: (mean, std) tensors for RGB channels
        """
        # Use a subset of data to compute statistics for large datasets
        MAX_IMAGES = 1000
        indices = torch.randperm(len(self))[:min(len(self), MAX_IMAGES)]
        
        # Collect normalized tensors
        tensors = []
        for idx in indices:
            img_path = self.img_paths[idx]
            try:
                img = Image.open(img_path).convert('RGB')
                tensor = transforms.ToTensor()(img)
                tensors.append(tensor)
            except Exception:
                continue
        
        if not tensors:
            return ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet defaults
        
        # Stack tensors and compute statistics
        tensors = torch.stack(tensors)
        mean = tensors.mean(dim=[0, 2, 3])
        std = tensors.std(dim=[0, 2, 3])
        
        return mean.tolist(), std.tolist()


def get_transforms(image_size, mode='train'):
    """
    Get optimized image transforms based on mode.
    
    For training: Applies a comprehensive set of augmentations suitable for histopathology images
    For validation/testing: Applies only normalization and resizing for consistent evaluation
    
    Args:
        image_size: Target image size (scalar value)
        mode: 'train', 'val', or 'test'
        
    Returns:
        torchvision.transforms.Compose: Composed transformation pipeline
    """
    # Standard ImageNet normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if mode == 'train':
        return transforms.Compose([
            # Spatial transforms
            Resize((image_size, image_size)),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            
            # Color and intensity transforms - histopathology specific
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.RandomAutocontrast(p=0.2),
            
            # Normalization
            ToTensor(),
            Normalize(mean=mean, std=std),
            
            # Additional regularization
            transforms.RandomApply([transforms.RandomErasing(p=0.5, scale=(0.02, 0.1))], p=0.1)
        ])
    else:  # val or test
        return transforms.Compose([
            Resize((image_size, image_size)),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])


def create_dataloaders(config, log_info=True):
    """
    Create optimized train, validation, and test dataloaders.
    
    Features:
    - Configurable batch size based on available memory and device
    - Proper prefetching with adequate workers
    - Weighted sampling for handling class imbalance
    - Memory pinning for faster data transfer to GPU
    - Optional persistent workers to reduce overhead
    
    Args:
        config: Configuration object containing dataset and training parameters
        log_info: Whether to log information about the dataloaders
        
    Returns:
        dict: Dictionary of DataLoader objects for different splits
    """
    dataloaders = {}
    
    # Determine optimal worker count based on system resources
    num_workers = min(8, os.cpu_count() or 1)
    if config.num_workers is not None:
        num_workers = config.num_workers
    
    # Check if persistent workers can be enabled (PyTorch 1.7.0+)
    persistent_workers = hasattr(DataLoader, 'persistent_workers') and num_workers > 0
    
    # Create transform pipelines
    transform_train = get_transforms(config.image_size, mode='train')
    transform_val = get_transforms(config.image_size, mode='val')
    
    # Auto-detect pin_memory based on device
    pin_memory = config.device != 'cpu'
    
    # Common dataloader kwargs
    dataloader_kwargs = {
        'pin_memory': pin_memory,
        'num_workers': num_workers,
        'persistent_workers': persistent_workers if persistent_workers else False,
    }
    
    # Calculate optimal prefetch factor based on batch size and image size
    # Higher values use more memory but may improve throughput
    prefetch_factor = min(2, max(1, 4096 // (config.image_size * config.image_size * 3 // 1024)))
    if num_workers > 0 and hasattr(DataLoader, 'prefetch_factor'):
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
    
    # Create datasets based on mode
    if config.mode in ['train', 'all']:
        train_dataset = HistopathologyDataset(
            root_dir=config.data_root,
            transform=transform_train,
            mode='train',
            cache_size=min(200, config.batch_size * 4)  # Cache size relative to batch size
        )
        
        # Enable weighted sampling for imbalanced datasets
        if config.use_weighted_sampling and len(set(train_dataset.labels)) > 1:
            if log_info:
                logger.info("Using weighted sampling to handle class imbalance")
            sampler = train_dataset.get_weighted_sampler()
            shuffle = False  # Disable shuffle when using a sampler
        else:
            sampler = None
            shuffle = True
            
        dataloaders['train'] = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=True,  # Drop last batch if incomplete (helps with batch norm)
            **dataloader_kwargs
        )
    
    if config.mode in ['train', 'val', 'all']:
        val_dataset = HistopathologyDataset(
            root_dir=config.data_root,
            transform=transform_val,
            mode='val'
        )
        dataloaders['val'] = DataLoader(
            val_dataset,
            batch_size=config.batch_size * 2,  # Can often use larger batch size for validation
            shuffle=False,
            **dataloader_kwargs
        )
    
    if config.mode in ['test', 'all']:
        test_dataset = HistopathologyDataset(
            root_dir=config.data_root,
            transform=transform_val,
            mode='test'
        )
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=config.batch_size * 2,  # Can often use larger batch size for testing
            shuffle=False,
            **dataloader_kwargs
        )
    
    # Log dataloader information if requested
    if log_info:
        for split, loader in dataloaders.items():
            logger.info(f"{split} dataloader: {len(loader.dataset)} images, {len(loader)} batches")
    
    return dataloaders


# ---------------------------------------------------
# Model Architecture
# ---------------------------------------------------

class BaseModel(nn.Module):
    """Base class for classification models."""
    def __init__(self, name, num_classes=2, trainable_layers=-1, dropout_rate=0.2):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.trainable_layers = trainable_layers
        self.dropout_rate = dropout_rate
        self.backbone = None
        self.classifier = None
        
    def _freeze_layers(self):
        """Freeze backbone layers based on trainable_layers parameter."""
        if self.trainable_layers < 0:
            # No freezing, make all parameters trainable
            return
        
        # Freeze all layers first
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze the last trainable_layers if needed
        if self.trainable_layers > 0:
            # Get all parameters that can be trained
            trainable_params = [p for p in self.backbone.parameters()]
            
            # Unfreeze the last trainable_layers
            for param in trainable_params[-self.trainable_layers:]:
                param.requires_grad = True
                
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

class ResNetModel(BaseModel):
    """ResNet model for classification."""
    def __init__(self, num_classes=2, trainable_layers=-1, dropout_rate=0.2):
        super().__init__('resnet', num_classes, trainable_layers, dropout_rate)
        
        # Load pre-trained ResNet152
        self.backbone = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        
        # Replace final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove existing fc layer
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Freeze layers if needed
        self._freeze_layers()
        
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits


class DenseNetModel(BaseModel):
    """DenseNet model for classification."""
    def __init__(self, num_classes=2, trainable_layers=-1, dropout_rate=0.2):
        super().__init__('densenet', num_classes, trainable_layers, dropout_rate)
        
        # Load pre-trained DenseNet121
        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        
        # Replace final classifier
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()  # Remove existing classifier
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Freeze layers if needed
        self._freeze_layers()
        
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits


class EfficientNetModel(BaseModel):
    """EfficientNet model for classification."""
    def __init__(self, num_classes=2, trainable_layers=-1, dropout_rate=0.2):
        super().__init__('efficientnet', num_classes, trainable_layers, dropout_rate)
        
        # Load pre-trained EfficientNet-V2-L
        self.backbone = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
        
        # Replace final classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()  # Remove existing classifier
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Freeze layers if needed
        self._freeze_layers()
        
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits


class AttentionFusion(nn.Module):
    """
    Advanced attention-based fusion module for ensemble models.
    
    Features:
    - Self-attention mechanism to capture model interactions
    - Multi-head attention for more expressive fusion
    - Adaptive layer normalization for stable training
    - Residual connections to preserve individual model strengths
    
    Args:
        num_models (int): Number of models to fuse
        num_classes (int): Number of output classes
        hidden_dim (int, optional): Dimension of hidden layers. Defaults to 128.
        num_heads (int, optional): Number of attention heads. Defaults to 4.
        dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.1.
    """
    def __init__(self, num_models, num_classes, hidden_dim=128, num_heads=4, dropout_rate=0.1):
        super().__init__()
        
        self.num_models = num_models
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Input feature extraction for each model
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_classes, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),  # More stable than ReLU for attention architectures
                nn.Dropout(dropout_rate)
            ) for _ in range(num_models)
        ])
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Post-attention processing
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Layer normalization for pre-norm formulation (more stable)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Output generation
        self.output_weights = nn.Sequential(
            nn.Linear(hidden_dim, num_models),
            nn.Softmax(dim=1)
        )

    def forward(self, logits_list):
        """
        Forward pass of the attention fusion module.
        
        Args:
            logits_list (list): List of logits tensors from individual models
                                Each tensor has shape [batch_size, num_classes]
        
        Returns:
            torch.Tensor: Fused output tensor of shape [batch_size, num_classes]
        """
        batch_size = logits_list[0].size(0)
        
        # Extract features from each model's logits
        features = []
        for i, logits in enumerate(logits_list):
            features.append(self.feature_extractors[i](logits))
        
        # Stack features for attention [batch_size, num_models, hidden_dim]
        stacked_features = torch.stack(features, dim=1)
        
        # Apply self-attention with pre-norm formulation
        normed_features = self.norm1(stacked_features)
        attn_output, _ = self.self_attention(
            normed_features, normed_features, normed_features
        )
        
        # First residual connection
        attn_output = attn_output + stacked_features
        
        # Feed-forward with pre-norm formulation
        normed_attn = self.norm2(attn_output)
        ff_output = self.feed_forward(normed_attn)
        
        # Second residual connection
        output = ff_output + attn_output
        
        # Mean pooling across the model dimension to get [batch_size, hidden_dim]
        pooled = output.mean(dim=1)
        
        # Generate fusion weights for each model [batch_size, num_models]
        weights = self.output_weights(pooled)
        
        # Apply weights to original logits and sum
        weighted_sum = torch.zeros_like(logits_list[0])
        for i, logits in enumerate(logits_list):
            weighted_sum += logits * weights[:, i].unsqueeze(1)
            
        return weighted_sum


class CrossAttentionFusion(nn.Module):
    """
    Advanced cross-attention fusion module that operates on feature-level representations.
    
    Features:
    - Cross-attention mechanism to capture inter-model feature interactions
    - Multi-head attention for parallel processing of different representation subspaces
    - Feature-level fusion before classification
    - Residual connections to preserve important information
    
    Args:
        feature_dims (dict): Dictionary mapping model names to their feature dimensions
        num_classes (int): Number of output classes
        hidden_dim (int): Dimension for the projected feature space
        num_heads (int): Number of attention heads
        dropout_rate (float): Dropout rate for regularization
    """
    def __init__(self, feature_dims, num_classes, hidden_dim=512, num_heads=8, dropout_rate=0.1):
        super().__init__()
        
        self.feature_dims = feature_dims
        self.model_names = list(feature_dims.keys())
        self.num_models = len(self.model_names)
        self.hidden_dim = hidden_dim
        
        # Feature projectors to map different backbone features to common dimension
        self.feature_projectors = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ) for name, dim in feature_dims.items()
        })
        
        # Cross-attention layers (one for each model as query)
        self.cross_attention_layers = nn.ModuleDict()
        for query_model in self.model_names:
            self.cross_attention_layers[query_model] = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout_rate,
                batch_first=True
            )
        
        # Layer normalization for pre-norm formulation
        self.norm_layers = nn.ModuleDict({
            name: nn.LayerNorm(hidden_dim) for name in self.model_names
        })
        
        # Feed-forward networks after attention
        self.ffn_layers = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim * 4, hidden_dim)
            ) for name in self.model_names
        })
        
        # Final classifier from fused features
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * self.num_models, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, features_dict):
        """
        Forward pass of the cross-attention fusion module.
        
        Args:
            features_dict (dict): Dictionary mapping model names to their features
                                 Each tensor has shape [batch_size, feature_dim]
        
        Returns:
            torch.Tensor: Fused output tensor of shape [batch_size, num_classes]
        """
        batch_size = next(iter(features_dict.values())).size(0)
        
        # Project features to common dimension
        projected_features = {
            name: self.feature_projectors[name](features) 
            for name, features in features_dict.items()
        }
        
        # Apply cross-attention where each model attends to others
        attended_features = {}
        for query_model in self.model_names:
            # Prepare query, key, value tensors
            query = projected_features[query_model].unsqueeze(1)  # [B, 1, H]
            
            # Create key and value by stacking all other models' features
            other_models = [m for m in self.model_names if m != query_model]
            if not other_models:  # Handle case with only one model
                attended_features[query_model] = projected_features[query_model]
                continue
                
            keys = torch.stack([projected_features[m] for m in other_models], dim=1)  # [B, M-1, H]
            values = keys  # Use same tensor for keys and values
            
            # Apply attention
            normed_query = self.norm_layers[query_model](query)
            attn_output, _ = self.cross_attention_layers[query_model](
                normed_query, keys, values
            )
            
            # Residual connection and squeeze unnecessary dimension
            attn_output = (attn_output + query).squeeze(1)
            
            # Feed-forward network with residual
            attended_features[query_model] = self.ffn_layers[query_model](attn_output) + attn_output
        
        # Concatenate attended features from all models
        concat_features = torch.cat([attended_features[name] for name in self.model_names], dim=1)
        
        # Final classification
        output = self.classifier(concat_features)
        
        return output


class DynamicConfidenceWeighting(nn.Module):
    """
    Dynamic confidence weighting module that predicts model weights based on input characteristics.
    
    Features:
    - Input-adaptive weighting of models
    - Image characteristic analysis through convolutional layers
    - Temperature-scaled softmax for dynamic weight assignment
    
    Args:
        num_models (int): Number of models to weight
        temperature (float): Temperature for softmax scaling
    """
    def __init__(self, num_models, temperature=1.0):
        super().__init__()
        
        self.num_models = num_models
        self.temperature = temperature
        
        # Image characteristic analyzer (lightweight CNN)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Weight predictor
        self.weight_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_models)
        )
        
    def forward(self, x):
        """
        Predict weights for each model based on input image characteristics.
        
        Args:
            x (torch.Tensor): Input image tensor [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Predicted weights [batch_size, num_models]
        """
        features = self.feature_extractor(x)
        logits = self.weight_predictor(features)
        
        # Apply temperature scaling for smoother weight distribution
        weights = F.softmax(logits / self.temperature, dim=1)
        
        return weights


class EnhancedEnsembleModel(nn.Module):
    """
    Enhanced ensemble model with advanced fusion techniques:
    1. Feature-level fusion
    2. Cross-attention between model features
    3. Dynamic confidence weighting
    
    Args:
        config: ModelConfig object containing model configuration
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.fusion_type = config.fusion_type
        self.dropout_rate = config.dropout_rate
        
        # Parse trainable layers
        self.trainable_layers = config.trainable_layers
        
        # Create dictionary of base models
        self.models = nn.ModuleDict()
        
        # Define feature dimensions for each model
        self.feature_dims = {}
        
        # Check which models to include
        if 'resnet' in config.active_models:
            self.models['resnet'] = ResNetModel(
                num_classes=self.num_classes,
                trainable_layers=self.trainable_layers.get('resnet', -1),
                dropout_rate=self.dropout_rate
            )
            self.feature_dims['resnet'] = 2048  # ResNet152 feature dimension
            
        if 'densenet' in config.active_models:
            self.models['densenet'] = DenseNetModel(
                num_classes=self.num_classes,
                trainable_layers=self.trainable_layers.get('densenet', -1),
                dropout_rate=self.dropout_rate
            )
            self.feature_dims['densenet'] = 1024  # DenseNet121 feature dimension
            
        if 'efficientnet' in config.active_models:
            self.models['efficientnet'] = EfficientNetModel(
                num_classes=self.num_classes,
                trainable_layers=self.trainable_layers.get('efficientnet', -1),
                dropout_rate=self.dropout_rate
            )
            self.feature_dims['efficientnet'] = 1280  # EfficientNet-V2-L feature dimension
        
        # Add dynamic confidence weighting module
        self.dynamic_weighting = DynamicConfidenceWeighting(
            num_models=len(self.models),
            temperature=getattr(config, 'weight_temperature', 2.0)
        )
        
        # Set up fusion mechanisms
        if self.fusion_type == 'weighted':
            # Create learnable weights for each model
            init_weights = torch.tensor([config.model_weights.get(name, 1.0) for name in self.models.keys()])
            # Apply softmax to ensure they sum to 1
            init_weights = F.softmax(init_weights, dim=0)
            self.weights = nn.Parameter(init_weights)
            
        elif self.fusion_type == 'concat':
            # Concatenation followed by FC layer with more expressive architecture
            self.fusion_layer = nn.Sequential(
                nn.Linear(len(self.models) * self.num_classes, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(128, self.num_classes)
            )
            
        elif self.fusion_type == 'attentive':
            # Enhanced attention-based fusion
            self.attention_fusion = AttentionFusion(
                num_models=len(self.models),
                num_classes=self.num_classes,
                hidden_dim=getattr(config, 'attention_hidden_dim', 128),
                num_heads=getattr(config, 'attention_heads', 4),
                dropout_rate=self.dropout_rate
            )
            
        elif self.fusion_type == 'cross_attention':
            # New cross-attention fusion at feature level
            self.cross_attention_fusion = CrossAttentionFusion(
                feature_dims=self.feature_dims,
                num_classes=self.num_classes,
                hidden_dim=getattr(config, 'cross_attention_dim', 512),
                num_heads=getattr(config, 'cross_attention_heads', 8),
                dropout_rate=self.dropout_rate
            )
        
        # Enable gradient checkpointing for memory efficiency if requested
        self.use_gradient_checkpointing = getattr(config, 'gradient_checkpointing', False)
        if self.use_gradient_checkpointing:
            for name, model in self.models.items():
                if hasattr(model, 'backbone') and hasattr(model.backbone, 'gradient_checkpointing_enable'):
                    model.backbone.gradient_checkpointing_enable()
                    logger.info(f"Gradient checkpointing enabled for {name}")
        
        # Apply model compilation for speed improvement if supported and requested
        if hasattr(torch, 'compile') and getattr(config, 'compile_model', False):
            for name, model in self.models.items():
                self.models[name] = torch.compile(model)
                logger.info(f"Model compilation applied to {name}")
    
    def count_parameters(self):
        """Count trainable parameters for all models and fusion."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_params = {name: sum(p.numel() for p in model.parameters() if p.requires_grad) 
                         for name, model in self.models.items()}
        
        # Return both total and per-model parameters without logging
        return total, model_params
            
    def forward(self, x):
        """
        Forward pass through enhanced ensemble model with feature-level fusion.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            tuple: (ensemble_output, individual_model_outputs)
                - ensemble_output: Final ensemble prediction
                - individual_model_outputs: Dictionary of individual model outputs
        """
        # Get dynamic confidence weights based on input characteristics
        dynamic_weights = self.dynamic_weighting(x)
        
        # Get output from each model
        model_outputs = {}
        model_features = {}
        
        # Forward pass for each model
        for i, (name, model) in enumerate(self.models.items()):
            # Use gradient checkpointing for memory efficiency during training if enabled
            if self.use_gradient_checkpointing and self.training:
                if hasattr(model, 'backbone') and hasattr(model.backbone, 'gradient_checkpointing_enabled'):
                    with torch.set_grad_enabled(True):
                        outputs, features = model(x, return_features=True)
                else:
                    outputs, features = model(x, return_features=True)
            else:
                outputs, features = model(x, return_features=True)
                
            model_outputs[name] = outputs
            model_features[name] = features
            
        # Apply fusion strategy
        if self.fusion_type == 'weighted':
            # Normalize weights using softmax
            normalized_weights = F.softmax(self.weights, dim=0)
            
            # Blend static weights with dynamic weights
            combined_weights = torch.zeros_like(dynamic_weights)
            for i, name in enumerate(self.models.keys()):
                # We use batch-wise dynamic weights and global static weights
                combined_weights[:, i] = 0.7 * dynamic_weights[:, i] + 0.3 * normalized_weights[i]
                
            # Apply weighted fusion
            ensemble_output = torch.zeros_like(list(model_outputs.values())[0])
            for i, (name, output) in enumerate(model_outputs.items()):
                # Use per-sample weights for dynamic weighting
                batch_weights = combined_weights[:, i].unsqueeze(1)  # [B, 1]
                ensemble_output += output * batch_weights
                
        elif self.fusion_type == 'concat':
            # Concatenate all outputs
            concat_output = torch.cat(list(model_outputs.values()), dim=1)
            ensemble_output = self.fusion_layer(concat_output)
            
        elif self.fusion_type == 'attentive':
            # Use enhanced attention mechanism
            ensemble_output = self.attention_fusion(list(model_outputs.values()))
            
        elif self.fusion_type == 'cross_attention':
            # Use cross-attention feature-level fusion
            ensemble_output = self.cross_attention_fusion(model_features)
            
        elif self.fusion_type == 'max':
            # Take the maximum confidence prediction
            probs = [F.softmax(output, dim=1) for output in model_outputs.values()]
            max_conf, _ = torch.max(torch.stack([torch.max(p, dim=1)[0] for p in probs]), dim=0)
            max_idx = torch.stack([torch.max(p, dim=1)[0] for p in probs]).argmax(dim=0)
            
            # Select the output from the model with highest confidence
            ensemble_output = torch.stack(list(model_outputs.values()))[max_idx, torch.arange(x.size(0))]
            
        return ensemble_output, model_outputs


# ---------------------------------------------------
# Loss Functions and Metrics
# ---------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for improved training of imbalanced datasets.
    
    Focal Loss adds a factor (1 - pt)^gamma to the standard cross-entropy
    criterion to down-weight easy examples and focus more on hard examples.
    
    Features:
    - Numerically stable implementation
    - Support for class weights
    - Flexible gamma and alpha parameters
    - Optimized for both binary and multi-class cases
    
    Args:
        alpha (float, optional): Weighting factor for the positive class. Defaults to 0.25.
        gamma (float, optional): Focusing parameter that controls down-weighting of well-classified examples. Defaults to 2.0.
        reduction (str, optional): Specifies the reduction to apply to the output. Defaults to 'mean'.
        eps (float, optional): Small epsilon value for numerical stability. Defaults to 1e-7.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', eps=1e-7):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, inputs, targets):
        """
        Forward pass of the Focal Loss.
        
        Args:
            inputs (torch.Tensor): Model predictions (logits).
            targets (torch.Tensor): Ground truth labels.
            
        Returns:
            torch.Tensor: Calculated loss
        """
        # Handle binary vs multi-class cases
        if inputs.size(1) == 1:  # Binary classification with single output
            # Apply sigmoid to get probability
            probs = torch.sigmoid(inputs)
            probs = probs.view(-1)
            targets = targets.view(-1)
            
            # Compute binary focal loss
            pt = torch.where(targets == 1, probs, 1-probs)
            alpha_t = torch.where(targets == 1, self.alpha, 1-self.alpha)
            
            # Clamp for numerical stability
            pt = torch.clamp(pt, min=self.eps, max=1.0-self.eps)
            
            # Calculate focal loss with stability improvements
            focal_weight = alpha_t * (1 - pt).pow(self.gamma)
            loss = -focal_weight * torch.log(pt)
            
        else:  # Multi-class with multiple outputs
            # Apply softmax to get class probabilities
            log_softmax = F.log_softmax(inputs, dim=1)
            
            # Convert targets to one-hot if needed
            if targets.dim() == 1 or targets.size(1) == 1:
                targets_one_hot = F.one_hot(targets.view(-1), num_classes=inputs.size(1)).float()
            else:
                targets_one_hot = targets
                
            # Compute the focal loss weight
            probs = torch.exp(log_softmax)
            pt = (targets_one_hot * probs).sum(1)
            focal_weight = (1 - pt).pow(self.gamma)
            
            # Apply alpha weighting
            if self.alpha is not None:
                alpha_t = targets_one_hot * self.alpha + (1 - targets_one_hot) * (1 - self.alpha)
                focal_weight = alpha_t.t() * focal_weight
                
            # Calculate the loss
            loss = -torch.sum(focal_weight.view(-1, 1) * targets_one_hot * log_softmax, dim=1)
        
        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else:  # 'sum'
            return loss.sum()


class BinaryClassificationLoss(nn.Module):
    """
    Enhanced binary classification loss with label smoothing and mixed loss options.
    
    Features:
    - Label smoothing for improved generalization
    - Balanced BCE for handling class imbalance
    - Numerically stable implementation
    - Option to combine BCE with Focal Loss
    
    Args:
        label_smoothing (float, optional): Label smoothing factor [0,1]. Defaults to 0.1.
        pos_weight (float, optional): Weight for positive class. Defaults to None.
        focal_weight (float, optional): Weight for focal loss component if enabled. Defaults to 0.0.
        focal_gamma (float, optional): Gamma parameter for focal loss. Defaults to 2.0.
        eps (float, optional): Small epsilon value for numerical stability. Defaults to 1e-7.
    """
    def __init__(self, label_smoothing=0.1, pos_weight=None, focal_weight=0.0, focal_gamma=2.0, eps=1e-7):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight
        self.focal_weight = focal_weight
        self.focal_gamma = focal_gamma
        self.eps = eps
        
    def forward(self, inputs, targets):
        """
        Forward pass of the binary classification loss.
        
        Args:
            inputs (torch.Tensor): Model predictions (logits)
            targets (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Calculated loss
        """
        # Check shapes and handle various cases
        if inputs.dim() > 2:
            # Multi-dimensional input, flatten to [batch_size, num_classes]
            inputs = inputs.view(inputs.size(0), -1)
        
        if targets.dim() > 2:
            # Multi-dimensional target, flatten to [batch_size, num_classes]
            targets = targets.view(targets.size(0), -1)
            
        if inputs.dim() == 2 and inputs.size(1) > 1 and targets.size(1) == 1:
            # Multi-class output but binary target (take appropriate logit)
            inputs = inputs[:, 1:2]  # Take the positive class logit
            
        # Apply label smoothing
        if self.label_smoothing > 0:
            # Label smoothing for binary targets
            smooth_targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            
            # Clamp for numerical stability
            smooth_targets = torch.clamp(smooth_targets, min=self.eps, max=1.0-self.eps)
        else:
            smooth_targets = targets
            
        # Calculate standard BCE loss
        if self.pos_weight is not None:
            # Use BCE with logits and positive class weighting
            pos_weight = torch.tensor([self.pos_weight], device=inputs.device)
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, smooth_targets, pos_weight=pos_weight, reduction='none'
            )
        else:
            # Standard BCE with logits
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, smooth_targets, reduction='none'
            )
            
        # If focal loss component is enabled, mix the losses
        if self.focal_weight > 0:
            # Calculate focal loss component
            probs = torch.sigmoid(inputs)
            probs = torch.clamp(probs, min=self.eps, max=1.0-self.eps)
            pt = torch.where(targets == 1, probs, 1-probs)
            focal_factor = (1 - pt).pow(self.focal_gamma)
            focal_loss = -((targets * torch.log(probs)) + ((1 - targets) * torch.log(1 - probs)))
            focal_loss = focal_loss * focal_factor
            
            # Combine BCE and focal loss
            combined_loss = (1 - self.focal_weight) * bce_loss + self.focal_weight * focal_loss
            return combined_loss.mean()
        else:
            return bce_loss.mean()


def calculate_metrics(outputs, targets, threshold=0.5):
    """Calculate various metrics for binary classification."""
    # Convert logits to probabilities
    if isinstance(outputs, tuple):
        outputs = outputs[0]  # Get ensemble output if tuple
    
    # Handle binary vs multi-class
    if outputs.size(1) == 1:  # Binary with single output
        probs = torch.sigmoid(outputs)
        y_pred = (probs >= threshold).float()
        y_true = targets
    else:  # Multi-class with multiple outputs
        probs = F.softmax(outputs, dim=1)
        y_pred = torch.argmax(probs, dim=1)
        if targets.size(1) > 1:  # One-hot encoded
            y_true = torch.argmax(targets, dim=1)
        else:
            y_true = targets
    
    # Calculate accuracy
    correct = (y_pred.view(-1) == y_true.view(-1)).float().sum()
    acc = correct / targets.size(0)
    
    # For binary classification
    if outputs.size(1) == 1 or outputs.size(1) == 2:
        # Convert to binary predictions
        if outputs.size(1) == 2:  # Two-class output
            binary_preds = probs[:, 1]
            binary_y_pred = (binary_preds >= threshold).float()
            binary_y_true = y_true
        else:
            binary_preds = probs.squeeze()
            binary_y_pred = y_pred.squeeze()
            binary_y_true = y_true.squeeze()
        
        # TP, FP, TN, FN
        TP = ((binary_y_pred == 1) & (binary_y_true == 1)).sum().float()
        FP = ((binary_y_pred == 1) & (binary_y_true == 0)).sum().float()
        TN = ((binary_y_pred == 0) & (binary_y_true == 0)).sum().float()
        FN = ((binary_y_pred == 0) & (binary_y_true == 1)).sum().float()
        
        # Precision, Recall, F1
        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        specificity = TN / (TN + FP + 1e-10)
        
        # ROC AUC calculation (more robust handling)
        auc = 0.5  # Default value when AUC can't be computed
        try:
            from sklearn.metrics import roc_auc_score
            # Convert tensors to numpy arrays for sklearn
            cpu_preds = binary_preds.detach().cpu().numpy()
            cpu_targets = binary_y_true.detach().cpu().numpy()
            
            # Check if both classes are present in the batch
            if np.any(cpu_targets == 0) and np.any(cpu_targets == 1):
                auc = roc_auc_score(cpu_targets, cpu_preds)
            # If only one class is present, AUC is undefined, so keep default 0.5
        except Exception as e:
            # If sklearn is not available or any other error occurs
            pass
            
        return {
            'accuracy': acc.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'specificity': specificity.item(),
            'auc': auc
        }
    else:
        # For multi-class, just return accuracy
        return {'accuracy': acc.item()}


# ---------------------------------------------------
# Training and Validation Functions
# ---------------------------------------------------

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, config, scheduler=None, scaler=None, writer=None):
    """Train model for one epoch."""
    model.train()
    losses = []
    all_metrics = []
    batch_count = 0
    
    # Progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]", unit="batch")
    
    # Initialize gradient accumulation counter
    accum_counter = 0
    
    for idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        targets = targets.to(device)
        
        # Reset gradients only at the start of accumulation cycle or if not using accumulation
        if accum_counter == 0:
            optimizer.zero_grad()
        
        # Forward pass with mixed precision if enabled
        if config.use_amp and scaler is not None:
            # MPS doesn't fully support autocast yet, so only use it for CUDA
            if device.type == 'cuda':
                with autocast(device_type=device.type):
                    outputs, model_outputs = model(images)
                    loss = criterion(outputs, targets)
                    
                    # Scale loss for gradient accumulation if enabled
                    if config.grad_accum_steps > 1:
                        loss = loss / config.grad_accum_steps
                    
                # Backward pass with scaled gradients
                scaler.scale(loss).backward()
            else:  # For MPS or CPU
                outputs, model_outputs = model(images)
                loss = criterion(outputs, targets)
                
                # Scale loss for gradient accumulation if enabled
                if config.grad_accum_steps > 1:
                    loss = loss / config.grad_accum_steps
                
                # Standard backward pass, but still use scaler for consistency
                loss.backward()
            
            # Update weights if accumulation steps reached
            accum_counter += 1
            if accum_counter >= config.grad_accum_steps:
                if device.type == 'cuda':
                    # Unscale for gradient clipping (CUDA only)
                    scaler.unscale_(optimizer)
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # Update weights
                    scaler.step(optimizer)
                    scaler.update()
                else:  # For MPS or CPU
                    # Gradient clipping for MPS/CPU
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # For MPS, use scaler but handle differently
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                accum_counter = 0
                
        else:
            # Forward pass without mixed precision
            outputs, model_outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Scale loss for gradient accumulation if enabled
            if config.grad_accum_steps > 1:
                loss = loss / config.grad_accum_steps
                
            # Backward pass
            loss.backward()
            
            # Update weights if accumulation steps reached
            accum_counter += 1
            if accum_counter >= config.grad_accum_steps:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # Update weights
                optimizer.step()
                accum_counter = 0
                
        # Calculate metrics
        metrics = calculate_metrics(outputs, targets)
        
        # Update progress bar with metrics
        progress_bar.set_postfix({
            'loss': loss.item() * (config.grad_accum_steps if config.grad_accum_steps > 1 else 1),
            'acc': metrics['accuracy']
        })
        
        # Store loss and metrics
        losses.append(loss.item() * (config.grad_accum_steps if config.grad_accum_steps > 1 else 1))
        all_metrics.append(metrics)
        batch_count += 1
            
    # Final backward pass if needed for gradient accumulation
    if accum_counter > 0 and accum_counter < config.grad_accum_steps:
        if config.use_amp and scaler is not None:
            if device.type == 'cuda':
                # Unscale for gradient clipping (CUDA only)
                scaler.unscale_(optimizer)
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            else:
                # Gradient clipping for MPS/CPU
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Update weights
            scaler.step(optimizer)
            scaler.update()
        else:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
    # Calculate average metrics
    avg_loss = np.mean(losses)
    avg_metrics = {key: np.mean([m[key] for m in all_metrics if key in m]) for key in all_metrics[0].keys()}
    
    # Log to tensorboard if enabled
    if writer is not None:
        writer.add_scalar('Loss/train', avg_loss, epoch)
        for key, value in avg_metrics.items():
            writer.add_scalar(f'Metrics/train/{key}', value, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
    return avg_loss, avg_metrics


def validate(model, dataloader, criterion, device, epoch, config, writer=None, phase='val'):
    """Validate model on validation/test set."""
    model.eval()
    losses = []
    all_metrics = []
    
    # Log the start of validation
    logger.info(f"Starting {phase} phase for epoch {epoch+1}")
    
    # Progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs} [{phase.capitalize()}]", unit="batch")
    
    with torch.no_grad():
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs, model_outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Calculate metrics
            metrics = calculate_metrics(outputs, targets)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': metrics['accuracy']
            })
            
            # Store loss and metrics
            losses.append(loss.item())
            all_metrics.append(metrics)
    
    # Calculate average metrics
    avg_loss = np.mean(losses)
    avg_metrics = {key: np.mean([m[key] for m in all_metrics if key in m]) for key in all_metrics[0].keys()}
    
    # Log completion of validation
    logger.info(f"Completed {phase} phase for epoch {epoch+1} - loss={avg_loss:.4f}, accuracy={avg_metrics['accuracy']:.4f}")
    
    # Log to tensorboard if enabled
    if writer is not None:
        writer.add_scalar(f'Loss/{phase}', avg_loss, epoch)
        for key, value in avg_metrics.items():
            writer.add_scalar(f'Metrics/{phase}/{key}', value, epoch)
    
    return avg_loss, avg_metrics


def evaluate_individual_models(model, dataloader, criterion, device, epoch, config, writer=None):
    """Evaluate performance of individual models in the ensemble."""
    model.eval()
    model_metrics = {name: {'loss': [], 'metrics': []} for name in model.models.keys()}
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            # Get outputs from ensemble which includes individual model outputs
            _, model_outputs = model(images)
            
            # Evaluate each model separately
            for name, output in model_outputs.items():
                loss = criterion(output, targets)
                metrics = calculate_metrics(output, targets)
                
                model_metrics[name]['loss'].append(loss.item())
                model_metrics[name]['metrics'].append(metrics)
    
    # Calculate average metrics for each model
    results = {}
    for name in model_metrics:
        avg_loss = np.mean(model_metrics[name]['loss'])
        avg_metrics = {key: np.mean([m[key] for m in model_metrics[name]['metrics'] if key in m]) 
                      for key in model_metrics[name]['metrics'][0].keys()}
        
        results[name] = {'loss': avg_loss, 'metrics': avg_metrics}
        
        # Log to tensorboard if enabled
        if writer is not None:
            writer.add_scalar(f'Loss/val/{name}', avg_loss, epoch)
            for key, value in avg_metrics.items():
                writer.add_scalar(f'Metrics/val/{name}/{key}', value, epoch)
                
    return results


# ---------------------------------------------------
# Learning Rate Scheduler
# ---------------------------------------------------

class WarmupCosineScheduler:
    """Warmup cosine learning rate scheduler."""
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_factor=10.0, eta_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_factor = warmup_factor
        self.eta_min = eta_min
        self.current_epoch = 0
        
        # Store initial learning rate
        self.base_lr = [group['lr'] for group in optimizer.param_groups]
        
    def step(self):
        """Update learning rate based on current epoch."""
        self.current_epoch += 1
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            if self.current_epoch <= self.warmup_epochs:
                # Linear warmup
                lr_scale = (1.0 - (self.warmup_epochs - self.current_epoch) / self.warmup_epochs / self.warmup_factor)
                param_group['lr'] = self.base_lr[i] * lr_scale
            else:
                # Cosine annealing
                progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                param_group['lr'] = self.eta_min + (self.base_lr[i] - self.eta_min) * cosine_decay
                
    def get_last_lr(self):
        """Return last computed learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]


# ---------------------------------------------------
# Checkpoint Management
# ---------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, epoch, config, metrics=None, is_best=False):
    """Save model checkpoint."""
    # Create checkpoint directory if it doesn't exist
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.to_dict()
    }
    
    if scheduler is not None:
        if hasattr(scheduler, 'state_dict'):
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
        
    # Save epoch checkpoint - try with weights_only=True first
    checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    try:
        # Try with weights_only=True (newer PyTorch versions)
        torch.save(checkpoint, checkpoint_path, weights_only=True)
    except TypeError as e:
        # Fall back to older PyTorch versions without weights_only
        if "got an unexpected keyword argument 'weights_only'" in str(e):
            logger.warning("Your PyTorch version doesn't support weights_only parameter, using standard save method")
            torch.save(checkpoint, checkpoint_path)
        else:
            # Re-raise if it's a different error
            raise
    
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Save best model if specified
    if is_best:
        best_path = os.path.join(config.checkpoint_dir, 'model_best.pt')
        try:
            # Try with weights_only=True (newer PyTorch versions)
            torch.save(checkpoint, best_path, weights_only=True)
        except TypeError as e:
            # Fall back to older PyTorch versions without weights_only
            if "got an unexpected keyword argument 'weights_only'" in str(e):
                torch.save(checkpoint, best_path)
            else:
                # Re-raise if it's a different error
                raise
        logger.info(f"Best model saved to {best_path}")
        
    # Manage maximum number of checkpoints
    if config.max_checkpoints > 0:
        checkpoints = sorted([f for f in os.listdir(config.checkpoint_dir) 
                             if f.startswith('checkpoint_epoch_') and f.endswith('.pt')], 
                            key=lambda x: int(x.split('_')[2].split('.')[0]))
        while len(checkpoints) > config.max_checkpoints:
            os.remove(os.path.join(config.checkpoint_dir, checkpoints.pop(0)))


def load_checkpoint(model, optimizer=None, scheduler=None, checkpoint_path=None, config=None):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        checkpoint_path: Path to checkpoint file
        config: Config object with checkpoint_dir
        
    Returns:
        tuple: (checkpoint, start_epoch)
    """
    if checkpoint_path is None and config is not None:
        # Try to find the latest checkpoint
        checkpoints = [f for f in os.listdir(config.checkpoint_dir) 
                     if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
        if not checkpoints:
            logger.warning(f"No checkpoints found in {config.checkpoint_dir}")
            return None, 0
        
        # Get the latest checkpoint
        latest = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
        checkpoint_path = os.path.join(config.checkpoint_dir, latest)
    
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None, 0
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    # First try loading with weights_only=True (more secure)
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except Exception as e:
        # If we encounter any error with weights_only=True, fall back to weights_only=False
        if "weights_only" in str(e) or "UnpicklingError" in str(e):
            logger.warning("Secure loading with weights_only=True failed, falling back to weights_only=False")
            logger.warning("This is less secure but needed for older checkpoints. Future checkpoints will be saved securely.")
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            except Exception as e2:
                logger.error(f"Failed to load checkpoint even with fallback method: {e2}")
                return None, 0
        else:
            logger.error(f"Failed to load checkpoint: {e}")
            return None, 0
    
    # Load model state
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        logger.error(f"Failed to load model state: {e}")
        return None, 0
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            logger.warning(f"Failed to load optimizer state: {e}")
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            if hasattr(scheduler, 'load_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            logger.warning(f"Failed to load scheduler state: {e}")
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    logger.info(f"Loaded checkpoint from epoch {start_epoch-1}")
    
    metrics = checkpoint.get('metrics', None)
    if metrics:
        logger.info(f"Previous best metrics: {metrics}")
    
    return checkpoint, start_epoch


# ---------------------------------------------------
# Main Training Loop
# ---------------------------------------------------

def train(config, dataloaders=None, set_seed_again=True):
    """Main training function."""
    # Set random seed for reproducibility only if needed
    if set_seed_again:
        set_seed(config.seed)
        logger.info(f"Random seed set to {config.seed}")
    
    # Select device
    device = select_device()
    config.device = str(device)
    
    # Create dataloaders if not provided
    if dataloaders is None:
        dataloaders = create_dataloaders(config, log_info=False)  # Don't log again if creating here
    
    # Create ensemble model
    model = EnhancedEnsembleModel(config).to(device)
    
    # Log model parameter counts
    total_params, model_params = model.count_parameters()
    logger.info(f"Total trainable parameters: {total_params:,}")
    for name, params in model_params.items():
        logger.info(f"  - {name}: {params:,} parameters")
    
    # Define loss function
    if config.num_classes == 1 or config.num_classes == 2:
        # Use the custom loss for consistent handling of binary classification
        criterion = BinaryClassificationLoss(label_smoothing=config.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Define learning rate scheduler
    scheduler = None
    if config.use_scheduler:
        if config.scheduler_type == 'cosine':
            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_epochs=config.warmup_epochs,
                max_epochs=config.num_epochs,
                warmup_factor=config.warmup_factor
            )
        elif config.scheduler_type == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                threshold=0.01,
                min_lr=1e-6,
                verbose=True
            )
    
    # Initialize grad scaler for mixed precision training
    scaler = None
    if config.use_amp:
        if device.type == 'cuda':
            logger.info("Using CUDA with automatic mixed precision (AMP)")
            scaler = GradScaler()
        elif device.type == 'mps':
            logger.info("MPS device detected - AMP autocast will be disabled but using GradScaler for training stability")
            # Use GradScaler for stability even though autocast is not used with MPS
            scaler = GradScaler()
        else:
            logger.info("CPU device detected - Limited AMP support")
            scaler = GradScaler()
    
    # TensorBoard writer
    writer = None if config.no_tensorboard else SummaryWriter(
        log_dir=os.path.join('runs', 'classifications', config.mode, datetime.now().strftime('%Y%m%d-%H%M%S'))
    )
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_val_metrics = None
    patience_counter = 0
    start_epoch = 0
    
    # Initialize val_loss and val_metrics in case training is interrupted before first validation
    val_loss = float('inf')
    val_metrics = {'accuracy': 0.0}
    
    # Check for resuming training
    if config.resume and os.path.exists(config.checkpoint_dir):
        checkpoint, start_epoch = load_checkpoint(model, optimizer, scheduler, config=config)
        if checkpoint:
            logger.info(f"Resuming training from epoch {start_epoch}")
            if 'metrics' in checkpoint and checkpoint['metrics']:
                best_val_metrics = checkpoint['metrics']
                best_val_loss = best_val_metrics.get('loss', float('inf'))
            
            # Ensure we still train for the requested number of epochs
            if start_epoch >= config.num_epochs:
                logger.info(f"Loaded checkpoint epoch ({start_epoch}) is >= requested epochs ({config.num_epochs})")
                logger.info(f"Adjusting to train for {config.num_epochs} more epochs from the loaded checkpoint")
                config.num_epochs = start_epoch + config.num_epochs
                logger.info(f"New total epochs: {config.num_epochs}")
    
    # Training loop
    logger.info("Starting training...")
    try:
        for epoch in range(start_epoch, config.num_epochs):
            # Training phase
            train_loss, train_metrics = train_epoch(
                model, dataloaders['train'], optimizer, criterion, device, epoch, config, 
                scheduler=scheduler, scaler=scaler, writer=writer
            )
            
            # Log training metrics
            logger.info(f"Epoch {epoch+1}/{config.num_epochs} - Train: loss={train_loss:.4f}, "
                       f"accuracy={train_metrics['accuracy']:.4f}")
            
            # Validation phase
            val_loss, val_metrics = validate(
                model, dataloaders['val'], criterion, device, epoch, config, writer, phase='val'
            )
            
            # Log validation metrics
            logger.info(f"Epoch {epoch+1}/{config.num_epochs} - Val: loss={val_loss:.4f}, "
                       f"accuracy={val_metrics['accuracy']:.4f}")
            
            # Evaluate individual models periodically
            if (epoch + 1) % 5 == 0 or epoch + 1 == config.num_epochs:
                model_results = evaluate_individual_models(
                    model, dataloaders['val'], criterion, device, epoch, config, writer
                )
                
                # Log individual model metrics
                for name, results in model_results.items():
                    logger.info(f"  - {name}: loss={results['loss']:.4f}, "
                               f"accuracy={results['metrics']['accuracy']:.4f}")
            
            # Update learning rate scheduler
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % config.checkpoint_freq == 0:
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    best_val_metrics = val_metrics
                
                save_checkpoint(
                    model, optimizer, scheduler, epoch, config,
                    metrics={'loss': val_loss, **val_metrics},
                    is_best=is_best
                )
            
            # Early stopping
            if config.early_stopping_patience > 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    # Save final model - only save if we have valid validation metrics
    if val_loss != float('inf'):
        logger.info("Saving final model checkpoint...")
        save_checkpoint(
            model, optimizer, scheduler, config.num_epochs-1, config,
            metrics={'loss': val_loss, **val_metrics},
            is_best=(val_loss < best_val_loss)
        )
    else:
        logger.info("Training was interrupted before validation - no final checkpoint saved")
    
    # Final evaluation on test set if available
    if 'test' in dataloaders:
        logger.info("Evaluating on test set...")
        test_loss, test_metrics = validate(
            model, dataloaders['test'], criterion, device, config.num_epochs, config, 
            writer, phase='test'
        )
        logger.info(f"Test: loss={test_loss:.4f}, accuracy={test_metrics['accuracy']:.4f}")
        
        # Log other metrics if available
        for key, value in test_metrics.items():
            if key != 'accuracy':
                logger.info(f"  - {key}: {value:.4f}")
    
    # Close tensorboard writer
    if writer is not None:
        writer.close()
    
    logger.info("Training completed!")
    
    # Get fusion weights if applicable
    if config.fusion_type == 'weighted':
        weights = F.softmax(model.weights, dim=0).cpu().detach().numpy()
        weight_str = ', '.join([f"{name}: {weight:.4f}" for name, weight in 
                              zip(model.models.keys(), weights)])
        logger.info(f"Final ensemble weights: {weight_str}")
    
    # Generate insights if requested
    if config.with_insights and val_loss != float('inf'):
        # Evaluate individual models to get comprehensive results
        model_results = evaluate_individual_models(
            model, dataloaders['val'], criterion, device, config.num_epochs-1, config
        )
        
        # Combine results for insights generation
        all_results = {
            'ensemble': {'loss': val_loss, 'metrics': val_metrics}
        }
        all_results.update(model_results)
        
        # Generate insights
        insights_dir = generate_model_insights(model, dataloaders, device, config, all_results)
        if insights_dir:
            logger.info(f"Model insights saved to {insights_dir}")
    
    return model, best_val_metrics


# ---------------------------------------------------
# Main Function
# ---------------------------------------------------

def main():
    """Main entry point for model training and evaluation."""
    # Parse command line arguments
    args = parse_args()
    
    # Create configuration object
    config = create_config(args)
    
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Select appropriate device (CUDA, MPS, or CPU)
    device = select_device()
    config.device = device
    
    # Suppress numexpr warnings
    suppress_numexpr_messages()
    
    # Initialize logger
    logger = setup_logger()
    
    # Log configuration settings
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Log device information
    logger.info(f"Using device: {device}")
    
    # Start training
    try:
        # Create dataloaders
        dataloaders = create_dataloaders(config)
        
        # Create loss criterion with label smoothing
        criterion = BinaryClassificationLoss(label_smoothing=config.label_smoothing)
        
        # Use EnhancedEnsembleModel instead of EnsembleModel for improved fusion techniques
        model = EnhancedEnsembleModel(config)
        model = model.to(device)
        
        # Train the model and get results
        results = train(config, dataloaders)
        
        # Generate additional insights if requested
        if config.with_insights:
            logger.info("Generating model insights...")
            generate_model_insights(model, dataloaders, device, config, results)
            
        logger.info("Process completed successfully!")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        raise
        
    return 0


# ---------------------------------------------------
# Model Insights and Analysis
# ---------------------------------------------------

def generate_confusion_matrix(model, dataloader, device, config):
    """Generate confusion matrix for model evaluation."""
    # Check if required libraries are available
    if not HAS_VISUALIZATION_LIBS:
        logger.warning("Visualization libraries not available. Insights will be limited.")
        
    model.eval()
    y_true = []
    y_pred = []
    individual_preds = {name: [] for name in model.models.keys()}
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            # Get outputs from ensemble which includes individual model outputs
            outputs, model_outputs = model(images)
            
            # Process targets to get true labels
            if targets.size(1) == 1:  # Binary with single output
                true_labels = targets.cpu().numpy().flatten()
            else:  # Multi-class with multiple outputs
                true_labels = torch.argmax(targets, dim=1).cpu().numpy()
            
            # Process ensemble outputs
            if outputs.size(1) == 1:  # Binary with single output
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float().cpu().numpy().flatten()
            else:  # Multi-class with multiple outputs
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                
            y_true.extend(true_labels)
            y_pred.extend(preds)
            
            # Process individual model outputs
            for name, output in model_outputs.items():
                if output.size(1) == 1:  # Binary with single output
                    probs = torch.sigmoid(output)
                    preds = (probs >= 0.5).float().cpu().numpy().flatten()
                else:  # Multi-class with multiple outputs
                    probs = F.softmax(output, dim=1)
                    preds = torch.argmax(probs, dim=1).cpu().numpy()
                individual_preds[name].extend(preds)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Get class names
    if hasattr(dataloader.dataset, 'class_dirs'):
        class_names = sorted(dataloader.dataset.class_dirs)
    else:
        class_names = [str(i) for i in range(config.num_classes)]
    
    # Create confusion matrix for ensemble
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create individual confusion matrices
    individual_cms = {}
    for name, preds in individual_preds.items():
        individual_cms[name] = confusion_matrix(y_true, np.array(preds))
    
    return {
        'ensemble_cm': cm,
        'ensemble_cm_norm': cm_norm,
        'individual_cms': individual_cms,
        'class_names': class_names,
        'y_true': y_true,
        'y_pred': y_pred,
        'individual_preds': individual_preds
    }


def calculate_class_metrics(y_true, y_pred, class_idx=1):
    """Calculate detailed metrics for a specific class."""
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    
    # Convert to binary classification problem for the specific class
    y_true_binary = (y_true == class_idx).astype(int)
    y_pred_binary = (y_pred == class_idx).astype(int)
    
    # Calculate true positives, false positives, etc.
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


def analyze_model_errors(cm_data, config):
    """Analyze model errors and identify patterns."""
    cm = cm_data['ensemble_cm']
    y_true = cm_data['y_true']
    y_pred = cm_data['y_pred']
    class_names = cm_data['class_names']
    individual_preds = cm_data['individual_preds']
    
    # Calculate per-class metrics
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        metrics = calculate_class_metrics(y_true, y_pred, class_idx=i)
        class_metrics[class_name] = metrics
    
    # Find which samples are misclassified by the ensemble
    misclassified_indices = np.where(y_true != y_pred)[0]
    
    # Analyze which models got it right when the ensemble got it wrong
    model_correct_on_ensemble_errors = {name: 0 for name in individual_preds.keys()}
    for idx in misclassified_indices:
        for name, preds in individual_preds.items():
            if preds[idx] == y_true[idx]:
                model_correct_on_ensemble_errors[name] += 1
    
    # Calculate percentage of correct predictions on ensemble errors
    if len(misclassified_indices) > 0:
        for name in model_correct_on_ensemble_errors:
            model_correct_on_ensemble_errors[name] = (
                model_correct_on_ensemble_errors[name] / len(misclassified_indices) * 100
            )
    
    # Analyze class imbalance effect
    class_distribution = {}
    for i, class_name in enumerate(class_names):
        class_count = np.sum(y_true == i)
        class_distribution[class_name] = {
            'count': class_count,
            'percentage': class_count / len(y_true) * 100
        }
    
    return {
        'class_metrics': class_metrics,
        'misclassified_count': len(misclassified_indices),
        'model_correct_on_ensemble_errors': model_correct_on_ensemble_errors,
        'class_distribution': class_distribution
    }


def visualize_confusion_matrix(cm, class_names, title, filename):
    """Visualize confusion matrix and save to file."""
    if not HAS_VISUALIZATION_LIBS:
        logger.warning("Visualization libraries not available. Skipping confusion matrix visualization.")
        return
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if np.max(cm) < 2 else 'd',
                cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def visualize_model_comparison(metrics, filename):
    """Visualize model comparison and save to file."""
    if not HAS_VISUALIZATION_LIBS:
        logger.warning("Visualization libraries not available. Skipping model comparison visualization.")
        return
    
    models = list(metrics.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
    
    # Setup plot
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 4*len(metric_names)))
    
    for i, metric_name in enumerate(metric_names):
        values = [metrics[model].get(metric_name, 0) for model in models]
        
        axes[i].bar(models, values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        axes[i].set_title(f'{metric_name.capitalize()} Comparison')
        axes[i].set_ylim(0, 1.0)
        
        # Add values on top of bars
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def visualize_model_weights(model, filename):
    """Visualize the learned weights of ensemble model components."""
    if not HAS_VISUALIZATION_LIBS:
        logger.warning("Visualization libraries not available. Skipping model weights visualization.")
        return
    
    if hasattr(model, 'weights'):
        weights = F.softmax(model.weights, dim=0).cpu().detach().numpy()
        model_names = list(model.models.keys())
        
        plt.figure(figsize=(10, 6))
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
        bars = plt.bar(model_names, weights, color=colors[:len(model_names)])
        
        # Add values on top of bars
        for bar, weight in zip(bars, weights):
            plt.text(bar.get_x() + bar.get_width()/2, weight + 0.01, f"{weight:.4f}", 
                    ha='center', va='bottom', fontsize=12)
        
        plt.ylim(0, 1.0)
        plt.ylabel('Normalized Weight')
        plt.title('Ensemble Model Weights')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        # Also create a pie chart for another perspective
        plt.figure(figsize=(8, 8))
        plt.pie(weights, labels=model_names, autopct='%1.1f%%', colors=colors[:len(model_names)])
        plt.title('Ensemble Weight Distribution')
        plt.tight_layout()
        plt.savefig(filename.replace('.png', '_pie.png'))
        plt.close()


def generate_insights_report(results, config, dataloaders, cm_data, error_analysis, model):
    """Generate comprehensive insights report."""
    try:
        # Handle case where results is a tuple (model, metrics) instead of a dictionary
        if isinstance(results, tuple) and len(results) == 2:
            # Convert to the expected dictionary format
            model_obj, metrics = results
            results = {
                'ensemble': {'loss': 0.0, 'metrics': metrics}  # Use the metrics from the tuple
            }
            # Add individual model results if available
            if hasattr(model, 'models'):
                for name in model.models.keys():
                    results[name] = {'metrics': {'accuracy': 0.0}}  # Placeholder metrics
        
        # Create insights directory with absolute path
        insights_dir = os.path.abspath(os.path.join(config.insights_dir, config.mode))
        logger.info(f"Using insights directory: {insights_dir}")
        os.makedirs(insights_dir, exist_ok=True)
        
        # Create timestamp for the report
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        report_dir = os.path.join(insights_dir, f"report_{timestamp}")
        logger.info(f"Creating report directory: {report_dir}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate visualizations if matplotlib is available
        if HAS_VISUALIZATION_LIBS:
            logger.info("Generating confusion matrix visualizations...")
            try:
                visualize_confusion_matrix(
                    cm_data['ensemble_cm'], 
                    cm_data['class_names'], 
                    f"Ensemble Model Confusion Matrix ({config.mode})",
                    os.path.join(report_dir, "ensemble_confusion_matrix.png")
                )
                
                visualize_confusion_matrix(
                    cm_data['ensemble_cm_norm'], 
                    cm_data['class_names'], 
                    f"Ensemble Model Normalized Confusion Matrix ({config.mode})",
                    os.path.join(report_dir, "ensemble_confusion_matrix_norm.png")
                )
                
                # Save individual model confusion matrices
                for name, cm in cm_data['individual_cms'].items():
                    visualize_confusion_matrix(
                        cm, 
                        cm_data['class_names'], 
                        f"{name.capitalize()} Confusion Matrix ({config.mode})",
                        os.path.join(report_dir, f"{name}_confusion_matrix.png")
                    )
                
                # Generate metrics comparison visualization
                visualize_model_comparison(
                    {name: results.get(name, {}).get('metrics', {}) for name in cm_data['individual_preds'].keys()},
                    os.path.join(report_dir, "model_metrics_comparison.png")
                )
                
                # Visualize ensemble weights if available
                if config.fusion_type == 'weighted' and hasattr(model, 'weights'):
                    visualize_model_weights(model, os.path.join(report_dir, "ensemble_weights.png"))
                
                logger.info("Visualization generation complete")
            except Exception as viz_error:
                logger.error(f"Error during visualization generation: {viz_error}")
        else:
            logger.warning("Visualization libraries not available. Skipping visualizations.")
        
        # Save detailed metrics as CSV if pandas is available
        logger.info("Saving detailed metrics...")
        try:
            metrics_df = pd.DataFrame(columns=['Model', 'Metric', 'Value'])
            
            # Add ensemble metrics
            for metric, value in results.get('ensemble', {}).items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        metrics_df = pd.concat([metrics_df, pd.DataFrame([{'Model': 'ensemble', 'Metric': f"{metric}_{k}", 'Value': v}])], ignore_index=True)
                else:
                    metrics_df = pd.concat([metrics_df, pd.DataFrame([{'Model': 'ensemble', 'Metric': metric, 'Value': value}])], ignore_index=True)
            
            # Add individual model metrics
            for name, model_results in results.items():
                if name != 'ensemble':
                    for metric, value in model_results.items():
                        if isinstance(value, dict):
                            for k, v in value.items():
                                metrics_df = pd.concat([metrics_df, pd.DataFrame([{'Model': name, 'Metric': f"{metric}_{k}", 'Value': v}])], ignore_index=True)
                        else:
                            metrics_df = pd.concat([metrics_df, pd.DataFrame([{'Model': name, 'Metric': metric, 'Value': value}])], ignore_index=True)
            
            metrics_csv_path = os.path.join(report_dir, "detailed_metrics.csv")
            metrics_df.to_csv(metrics_csv_path, index=False)
            logger.info(f"Saved metrics CSV to {metrics_csv_path}")
        except Exception as e:
            logger.warning(f"Could not create metrics CSV: {e}")
            # Save metrics as simple text file instead
            metrics_txt_path = os.path.join(report_dir, "detailed_metrics.txt")
            with open(metrics_txt_path, 'w') as f:
                f.write("Model,Metric,Value\n")
                # Ensure results is a dictionary before iteration
                if isinstance(results, dict):
                    for name, model_results in results.items():
                        for metric, value in model_results.items():
                            if isinstance(value, dict):
                                for k, v in value.items():
                                    f.write(f"{name},{metric}_{k},{v}\n")
                            else:
                                f.write(f"{name},{metric},{value}\n")
                else:
                    # Handle non-dictionary results
                    f.write("ensemble,metrics,N/A\n")
            logger.info(f"Saved metrics as text file to {metrics_txt_path}")
        
        # Save error analysis as markdown
        logger.info("Generating error analysis report...")
        error_analysis_path = os.path.join(report_dir, "error_analysis.md")
        with open(error_analysis_path, 'w') as f:
            f.write(f"# Error Analysis Report - {config.mode.capitalize()} Set\n\n")
            f.write(f"## Dataset Information\n")
            f.write(f"- Total samples: {len(cm_data['y_true'])}\n")
            
            f.write(f"\n## Class Distribution\n")
            for class_name, data in error_analysis['class_distribution'].items():
                f.write(f"- {class_name}: {data['count']} samples ({data['percentage']:.2f}%)\n")
            
            f.write(f"\n## Misclassification Analysis\n")
            f.write(f"- Total misclassified samples: {error_analysis['misclassified_count']}\n")
            f.write(f"- Misclassification rate: {error_analysis['misclassified_count'] / len(cm_data['y_true']) * 100:.2f}%\n")
            
            f.write(f"\n## Per-Class Metrics\n")
            for class_name, metrics in error_analysis['class_metrics'].items():
                f.write(f"\n### Class: {class_name}\n")
                f.write(f"- Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"- Precision: {metrics['precision']:.4f}\n")
                f.write(f"- Recall: {metrics['recall']:.4f}\n")
                f.write(f"- Specificity: {metrics['specificity']:.4f}\n")
                f.write(f"- F1 Score: {metrics['f1']:.4f}\n")
                f.write(f"- True Positives: {metrics['tp']}\n")
                f.write(f"- False Positives: {metrics['fp']}\n")
                f.write(f"- True Negatives: {metrics['tn']}\n")
                f.write(f"- False Negatives: {metrics['fn']}\n")
            
            f.write(f"\n## Individual Model Performance on Ensemble Errors\n")
            f.write("Percentage of cases where individual models correctly classified samples that the ensemble misclassified:\n\n")
            for name, percentage in error_analysis['model_correct_on_ensemble_errors'].items():
                f.write(f"- {name.capitalize()}: {percentage:.2f}%\n")
            
            # Additional insights based on weights
            if config.fusion_type == 'weighted':
                f.write(f"\n## Ensemble Weight Analysis\n")
                f.write(f"- Default model weights: {config.model_weights}\n")
                if hasattr(model, 'weights'):
                    weights = F.softmax(model.weights, dim=0).cpu().detach().numpy()
                    weight_str = ', '.join([f"{name}: {weight:.4f}" for name, weight in 
                                          zip(model.models.keys(), weights)])
                    f.write(f"- Learned ensemble weights: {weight_str}\n")
                    
                f.write("\nBased on the performance metrics and the weight distribution, ")
                # Safely determine the best individual model, handling non-dictionary results
                try:
                    if isinstance(results, dict) and len(results) > 1:
                        best_model = max([item for item in results.items() if item[0] != 'ensemble'], 
                                        key=lambda x: x[1].get('metrics', {}).get('accuracy', 0))[0]
                    else:
                        best_model = next(iter(model.models.keys())) if hasattr(model, 'models') else "unknown"
                except (ValueError, AttributeError, StopIteration):
                    best_model = "unknown"
                
                f.write(f"the {best_model} appears to be the strongest individual model.\n")
        logger.info(f"Saved error analysis to {error_analysis_path}")
        
        # Create a summary report
        logger.info("Generating summary report...")
        summary_path = os.path.join(report_dir, "summary.md")
        with open(summary_path, 'w') as f:
            f.write(f"# Model Insights Summary - {config.mode.capitalize()} Set\n\n")
            f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Configuration\n")
            for key, value in config.to_dict().items():
                f.write(f"- {key}: {value}\n")
            
            f.write(f"\n## Performance Summary\n")
            
            # Ensemble performance
            ensemble_metrics = results.get('ensemble', {}).get('metrics', {}) if isinstance(results, dict) else {}
            if isinstance(results, tuple) and len(results) == 2:
                # If results is a tuple (model, metrics), use the metrics
                _, ensemble_metrics = results
                
            # Ensure ensemble_metrics is a dictionary
            if ensemble_metrics is None:
                ensemble_metrics = {}
            
            f.write("### Ensemble Model Performance\n")
            for metric, value in ensemble_metrics.items():
                f.write(f"- {metric.capitalize()}: {value:.4f}\n")
                
            # Individual model performance comparison
            f.write("\n### Individual Model Performance\n")
            f.write("| Model | Accuracy | Precision | Recall | F1 | Specificity | AUC |\n")
            f.write("|-------|----------|-----------|--------|----|-----------|---------|\n")
            
            # Sort models by accuracy for ranked comparison
            if isinstance(results, dict):
                sorted_models = sorted(
                    [(name, results.get(name, {}).get('metrics', {}).get('accuracy', 0)) 
                    for name in results.keys() if name != 'ensemble'],
                    key=lambda x: x[1],
                    reverse=True
                )
            else:
                # If results is not a dictionary, use the model names with placeholder metrics
                sorted_models = [(name, 0.0) for name in model.models.keys()] if hasattr(model, 'models') else []
            
            for name, _ in sorted_models:
                if isinstance(results, dict):
                    model_metrics = results.get(name, {}).get('metrics', {})
                else:
                    model_metrics = {}  # Empty metrics if results is not a dictionary
                f.write(f"| {name.capitalize()} | {model_metrics.get('accuracy', 0):.4f} | {model_metrics.get('precision', 0):.4f} | {model_metrics.get('recall', 0):.4f} | {model_metrics.get('f1', 0):.4f} | {model_metrics.get('specificity', 0):.4f} | {model_metrics.get('auc', 0):.4f} |\n")
            
            # Add ensemble model at the end for comparison
            f.write(f"| **Ensemble** | {ensemble_metrics.get('accuracy', 0):.4f} | {ensemble_metrics.get('precision', 0):.4f} | {ensemble_metrics.get('recall', 0):.4f} | {ensemble_metrics.get('f1', 0):.4f} | {ensemble_metrics.get('specificity', 0):.4f} | {ensemble_metrics.get('auc', 0):.4f} |\n")
            
            # Per-model detailed analysis
            f.write("\n## Per-Model Analysis\n")
            
            # Only do per-model analysis if results is a dictionary
            if isinstance(results, dict):
                for name in results.keys():
                    if name == 'ensemble':
                        continue
                        
                    model_metrics = results.get(name, {}).get('metrics', {})
                    model_loss = results.get(name, {}).get('loss', 0)
                    
                    f.write(f"\n### {name.capitalize()} Model\n")
                    f.write(f"- Loss: {model_loss:.4f}\n")
                    for metric, value in model_metrics.items():
                        f.write(f"- {metric.capitalize()}: {value:.4f}\n")
            else:
                # Simplified per-model section when no detailed results are available
                f.write("\nDetailed per-model metrics are not available in this report.\n")
        
        logger.info(f"Saved summary report to {summary_path}")
        
        return report_dir
    except Exception as e:
        logger.error(f"Error generating insights report: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def generate_model_insights(model, dataloaders, device, config, results):
    """Main function to generate model insights."""
    if not config.with_insights:
        return
        
    logger.info("Generating model insights...")
    
    try:
        # Use the test dataloader by default
        dataloader = dataloaders[config.mode]
        
        # Generate confusion matrix data
        logger.info("Generating confusion matrix data...")
        cm_data = generate_confusion_matrix(model, dataloader, device, config)
        
        # Perform error analysis
        logger.info("Analyzing model errors...")
        error_analysis = analyze_model_errors(cm_data, config)
        
        # Generate comprehensive report
        logger.info("Generating comprehensive insights report...")
        report_dir = generate_insights_report(results, config, dataloaders, cm_data, error_analysis, model)
        
        logger.info(f"Model insights generated and saved to {report_dir}")
        return report_dir
    except Exception as e:
        logger.error(f"Error generating model insights: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def create_config(args):
    """Create a ModelConfig object from command line arguments."""
    # Parse unfreeze layers argument
    trainable_layers = parse_unfreeze_layers(args.unfreeze_layers)
    
    # Parse active models
    active_models = [m.strip() for m in args.active_models.split(',')]
    
    # Create configuration
    config = ModelConfig(
        mode=args.mode,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        fusion_type=args.fusion_type,
        checkpoint_dir=args.checkpoint_dir,
        save_best_only=args.save_best_only,
        checkpoint_freq=args.checkpoint_freq,
        max_checkpoints=args.max_checkpoints,
        use_amp=not args.disable_amp,
        cache_data=not args.no_cache_data,
        early_stopping_patience=args.patience if args.early_stopping else 0,
        grad_accum_steps=args.grad_accum_steps,
        dropout_rate=args.dropout_rate,
        label_smoothing=args.label_smoothing,
        trainable_layers=trainable_layers,
        use_scheduler=not args.no_scheduler,
        scheduler_type=args.scheduler_type,
        warmup_epochs=args.warmup_epochs,
        warmup_factor=args.warmup_factor,
        resume=args.resume,
        seed=args.seed,
        active_models=active_models,
        no_tensorboard=args.no_tensorboard,
        with_insights=args.with_insights,
        insights_dir=args.insights_dir
    )
    
    return config

if __name__ == "__main__":
    main()
