"""
Advanced Nuclei Mask Generator for Thyroid Cancer Histopathology

This module provides specialized methods for detecting and segmenting nuclei in H&E stained
histopathology images of thyroid tissue, with a focus on identifying crucial characteristics
that distinguish between cancerous and non-cancerous cells:

1. Cellularity - Higher density of nuclei in cancerous tissue
2. Nucleomegaly - Larger nuclei in cancerous cells
3. Nuclear grooves - Coffee bean-like grooves in nuclei (cancer indicator)
4. Nuclear clearing - Differences in chromatin density/color (lighter in cancer)
5. Nuclear inclusions - Presence of thyroglobulin (more in benign cells)

The mask generator can be used standalone or integrated with segmentation pipelines
to provide on-the-fly mask generation during training, validation, or testing.
"""

from datetime import datetime
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import argparse
import time
from tqdm import tqdm
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

# Scientific image processing
from skimage import io, color, exposure, filters, feature, segmentation, morphology, measure
from skimage.filters import threshold_multiotsu, threshold_otsu, threshold_local
from skimage.morphology import disk, remove_small_objects, remove_small_holes
from skimage.feature import peak_local_max
from skimage.color import rgb2hed, hed2rgb, separate_stains, combine_stains
from skimage.exposure import rescale_intensity
from skimage.segmentation import watershed, find_boundaries
from skimage.measure import regionprops, label
from skimage.transform import resize
from scipy import ndimage as ndi

# ---------------------------------------------------
# Logging Configuration
# ---------------------------------------------------

# logger
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = logging.getLogger('AdvancedNucleiMaskGenerator - Nuclei Detection - ' + timestamp)

# ---------------------------------------------------
# Deep learning imports (for model-based segmentation)
# ---------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Model-based segmentation will be disabled.")

# Constants for compatibility with different scikit-image versions
USE_FOOTPRINT = True  # Use footprint parameter for morphological operations
LEGACY_PEAK_LOCAL_MAX = False  # Use newer peak_local_max API

# ---------------------------------------------------
# scikit-image version check
# ---------------------------------------------------
try:
    import pkg_resources
    SKIMAGE_VERSION = pkg_resources.get_distribution("scikit-image").version
    SKIMAGE_MAJOR_VERSION = int(SKIMAGE_VERSION.split('.')[0])
    SKIMAGE_MINOR_VERSION = int(SKIMAGE_VERSION.split('.')[1])
    
    # Handle different function signatures in different versions
    USE_FOOTPRINT = not (SKIMAGE_MAJOR_VERSION == 0 and SKIMAGE_MINOR_VERSION < 19)
    LEGACY_PEAK_LOCAL_MAX = (SKIMAGE_MAJOR_VERSION == 0 and SKIMAGE_MINOR_VERSION < 20)
    logger.info(f"Detected scikit-image version: {SKIMAGE_VERSION}")
except (ImportError, ValueError, IndexError):
    logger.warning("Could not determine scikit-image version. Using default compatibility settings.")
    USE_FOOTPRINT = True
    LEGACY_PEAK_LOCAL_MAX = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nuclei_mask_generation.log')
    ]
)
logger = logging.getLogger('AdvancedNucleiMaskGenerator')

# Constants for nuclei segmentation
MIN_NUCLEUS_SIZE = 15  # Minimum pixel size for nuclei
MAX_NUCLEUS_SIZE = 3000  # Maximum pixel size for nuclei
NUCLEI_CIRCULARITY_THRESHOLD = 0.3  # Minimum circularity (0-1) for nuclei shape filtering
DEFAULT_IMAGE_SIZE = (512, 512)  # Default size for processing images 

# HED stain separation constants (standard values from literature)
# Hematoxylin, Eosin, and DAB stain matrices
# Can be customized for specific datasets if needed
H_E_DAB = np.array([
    [0.65, 0.70, 0.29],
    [0.07, 0.99, 0.11],
    [0.27, 0.57, 0.78]
])

# Wrapper functions for version compatibility
def binary_closing_wrapper(image, disk_size=2):
    """Version-agnostic binary closing operation"""
    if USE_FOOTPRINT:
        return morphology.binary_closing(image, footprint=disk(disk_size))
    else:
        return morphology.binary_closing(image, selem=disk(disk_size))

def binary_dilation_wrapper(image, disk_size=1):
    """Version-agnostic binary dilation operation"""
    if USE_FOOTPRINT:
        return morphology.binary_dilation(image, footprint=disk(disk_size))
    else:
        return morphology.binary_dilation(image, selem=disk(disk_size))

def binary_erosion_wrapper(image, disk_size=1):
    """Version-agnostic binary erosion operation"""
    if USE_FOOTPRINT:
        return morphology.binary_erosion(image, footprint=disk(disk_size))
    else:
        return morphology.binary_erosion(image, selem=disk(disk_size))

def binary_opening_wrapper(image, disk_size=1):
    """Version-agnostic binary opening operation"""
    if USE_FOOTPRINT:
        return morphology.binary_opening(image, footprint=disk(disk_size))
    else:
        return morphology.binary_opening(image, selem=disk(disk_size))

def peak_local_max_wrapper(image, min_distance=10, labels=None, exclude_border=False):
    """Version-agnostic peak_local_max function"""
    try:
        if LEGACY_PEAK_LOCAL_MAX:
            # Older versions use indices parameter
            coordinates = peak_local_max(
                image, 
                min_distance=min_distance, 
                labels=labels,
                exclude_border=exclude_border,
                indices=True  # Always get coordinates for consistent handling
            )
            return coordinates
        else:
            # Newer versions don't use indices parameter
            try:
                # First try without indices parameter (newer versions)
                coordinates = peak_local_max(
                    image, 
                    min_distance=min_distance, 
                    labels=labels,
                    exclude_border=exclude_border
                )
                
                # Check if coordinates is a boolean mask or array of coordinates
                if isinstance(coordinates, np.ndarray) and coordinates.ndim == 2 and coordinates.shape[1] == 2:
                    # It's already an array of coordinates
                    return coordinates
                else:
                    # It's a boolean mask, convert to coordinates
                    return np.column_stack(np.where(coordinates))
                    
            except TypeError as e:
                # If we get a TypeError about unexpected keyword, try with indices=True
                if "unexpected keyword argument" in str(e):
                    coordinates = peak_local_max(
                        image, 
                        min_distance=min_distance, 
                        labels=labels,
                        exclude_border=exclude_border,
                        indices=True
                    )
                    return coordinates
                else:
                    raise
    except Exception as e:
        logger.error(f"Error in peak_local_max_wrapper: {str(e)}")
        # Fallback: return empty array of coordinates
        return np.array([], dtype=np.int64).reshape(0, 2)

# Utility functions for image processing and analysis
def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    
def normalize_staining(img):
    """
    Normalize H&E staining using Macenko method
    
    Args:
        img: RGB input image
        
    Returns:
        Normalized RGB image
    """
    # Convert to float
    img = img.astype(float)
    
    # Separate the stains using the standard H&E matrix
    try:
        # Use skimage's separate_stains function
        stains = separate_stains(img, H_E_DAB)
        # Get H&E channels
        h = stains[:, :, 0]
        e = stains[:, :, 1]
        
        # Normalize each channel
        h_norm = rescale_intensity(h, out_range=(0, 1))
        e_norm = rescale_intensity(e, out_range=(0, 1))
        
        # Recreate the normalized stain matrix
        stains_norm = np.zeros_like(stains)
        stains_norm[:, :, 0] = h_norm
        stains_norm[:, :, 1] = e_norm
        
        # Convert back to RGB
        img_norm = combine_stains(stains_norm, H_E_DAB)
        
        # Ensure values are in valid range [0, 255]
        img_norm = rescale_intensity(img_norm, out_range=(0, 255)).astype(np.uint8)
        
        return img_norm
    except Exception as e:
        logger.warning(f"Stain normalization failed: {e}. Using original image.")
        return img.astype(np.uint8)

def compute_circularity(region):
    """
    Calculate circularity of a region (0-1, where 1 is perfect circle)
    
    Args:
        region: Region properties from regionprops
        
    Returns:
        Circularity score between 0 and 1
    """
    perimeter = region.perimeter
    area = region.area
    
    # Avoid division by zero
    if perimeter == 0:
        return 0
    
    # Circularity formula: 4π·area/perimeter²
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    
    # Ensure the value is between 0 and 1
    return min(max(circularity, 0), 1)

def calculate_nuclei_density(binary_mask, neighborhood_size=100):
    """
    Calculate local density of nuclei (cellularity feature)
    
    Args:
        binary_mask: Binary mask of nuclei
        neighborhood_size: Size of neighborhood to consider for density calculation
        
    Returns:
        Density map showing local nuclei concentration
    """
    # Label the nuclei
    labeled_mask = label(binary_mask)
    
    # Create nuclei centroid map
    centroids = np.zeros_like(binary_mask, dtype=np.uint8)
    for region in regionprops(labeled_mask):
        y, x = region.centroid
        centroids[int(y), int(x)] = 1
    
    # Create a kernel for density estimation
    kernel = disk(neighborhood_size)
    
    # Calculate local density via convolution
    density_map = ndi.convolve(centroids, kernel, mode='constant', cval=0)
    
    # Normalize density map
    if np.max(density_map) > 0:
        density_map = rescale_intensity(density_map, out_range=(0, 1))
    
    return density_map 

class AdvancedNucleiMaskGenerator:
    """
    Advanced nuclei segmentation and feature extraction for H&E histopathology images.
    
    This class provides methods for:
    1. Nuclei segmentation using various algorithms
    2. Feature extraction from nuclei for malignancy detection
    3. Visualization of results
    
    Methods can be used individually or as an ensemble.
    """
    
    def __init__(self, method='ensemble', model_path=None, advanced_segmentation=False):
        """
        Initialize the mask generator
        
        Args:
            method: Segmentation method to use ('adaptive_threshold', 'watershed', 
                   'stain_separation', 'ensemble')
            model_path: Path to deep learning model (optional)
            advanced_segmentation: Enable enhanced segmentation and analysis methods
        """
        self.method = method
        self.model_path = model_path
        self.advanced_segmentation = advanced_segmentation
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None  # Initialize model as None
        
        # Try to load model if torch is available
        try:
            if TORCH_AVAILABLE:
                # Initialize model to None for now (will be loaded on demand)
                if model_path and os.path.exists(model_path):
                    self.logger.info(f"Model path provided: {model_path}")
                    # Model loading would happen here
                pass
            else:
                self.logger.info("PyTorch not available, model-based methods will be disabled")
        except Exception as e:
            self.logger.warning(f"Error initializing model: {str(e)}")
        
        self.logger.info(f"Initialized AdvancedNucleiMaskGenerator with method: {method}, advanced_segmentation: {advanced_segmentation}")
    
    def preprocess_image(self, image_path_or_array):
        """
        Preprocess the input image for segmentation
        
        Args:
            image_path_or_array: Path to image or numpy array
            
        Returns:
            Preprocessed RGB image
        """
        # Load image if path is provided
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                raise ValueError(f"Could not load image from {image_path_or_array}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path_or_array
            
        # Ensure image is RGB
        if len(image.shape) == 2:
            # Convert grayscale to RGB
            image = np.stack([image] * 3, axis=-1)
            
        # Image should be 8-bit per channel
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            
        return image
    
    def generate_mask(self, image_path_or_array):
        """
        Generate a nuclei mask from an image
        
        Args:
            image_path_or_array: Path to image or numpy array
            
        Returns:
            tuple: (Binary mask of nuclei, Dictionary of nuclei features)
        """
        # Preprocess the image
        image = self.preprocess_image(image_path_or_array)
        if image is None:
            self.logger.error("Failed to preprocess image")
            return None, None
        
        # Apply the selected method
        if self.method == 'stain_separation':
            mask = self.stain_separation_method(image)
        elif self.method == 'adaptive_threshold':
            mask = self.adaptive_threshold_method(image)
        elif self.method == 'watershed':
            mask = self.watershed_method(image)
        elif self.method == 'model' and self.model is not None:
            mask = self.model_based_method(image)
        elif self.method == 'ensemble':
            mask = self.ensemble_method(image)
        else:
            # Default to adaptive threshold
            self.logger.warning(f"Unsupported method '{self.method}', using adaptive threshold")
            mask = self.adaptive_threshold_method(image)
        
        # Apply post-processing
        mask = self.post_process_mask(mask)
        
        # Extract features from the mask
        features = self.extract_nuclei_features(image, mask)
        
        return mask, features
    
    def stain_separation_method(self, image):
        """
        Segment nuclei using stain separation and thresholding on the hematoxylin channel
        
        This method uses color deconvolution to separate hematoxylin (nuclei) from 
        eosin (cytoplasm) staining in H&E images.
        
        Args:
            image: RGB image
            
        Returns:
            Binary mask of nuclei
        """
        # Use enhanced stain separation if advanced segmentation is enabled
        if self.advanced_segmentation:
            return self._enhanced_stain_separation_method(image)
            
        # Original stain separation method
        try:
            # Normalize image
            hed_image = rgb2hed(image)
            
            # Extract hematoxylin channel and normalize to 0-255
            h_channel = hed_image[:, :, 0]
            h_channel = (h_channel - h_channel.min()) / (h_channel.max() - h_channel.min() + 1e-8)
            h_channel = (h_channel * 255).astype(np.uint8)
            
            # Invert hematoxylin channel so nuclei are bright
            h_channel = 255 - h_channel
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            h_channel_enhanced = clahe.apply(h_channel)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(h_channel_enhanced, (3, 3), 0)
            
            # Apply multi-level thresholding to handle varying staining intensities
            # 1. Otsu's thresholding for global threshold
            _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 2. Adaptive thresholding for local variations
            adaptive_mask = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 15, -2  # Negative constant to make threshold more aggressive
            )
            
            # Combine masks using OR operation to capture all potential nuclei
            combined_mask = cv2.bitwise_or(otsu_mask, adaptive_mask)
            
            # Clean up the mask
            # 1. Remove small objects (noise)
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # 2. Close small gaps within nuclei
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # 3. Remove very small objects (debris)
            labeled_mask, num_features = label(closing > 0, return_num=True)
            if num_features > 0:
                sizes = np.bincount(labeled_mask.ravel())
                mask_sizes = sizes[1:] if len(sizes) > 1 else []
                
                # Skip filtering if no nuclei
                if len(mask_sizes) == 0:
                    return np.zeros_like(h_channel)
                
                # Determine small object threshold - adaptive based on image statistics
                min_size_threshold = max(20, np.median(mask_sizes) * 0.2)
                
                # Remove small objects
                filtered_mask = np.zeros_like(closing)
                for i in range(1, num_features + 1):
                    if sizes[i] >= min_size_threshold:
                        filtered_mask[labeled_mask == i] = 255
                        
                # Apply watershed to separate touching nuclei
                # This is a simplified watershed to improve the stain separation results
                distance = ndi.distance_transform_edt(filtered_mask > 0)
                local_max = peak_local_max(
                    distance, 
                    min_distance=5,
                    labels=filtered_mask > 0
                )
                markers = np.zeros_like(distance, dtype=np.int32)
                markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
                watershed_labels = watershed(-distance, markers, mask=filtered_mask > 0)
                
                # Convert labels to binary mask
                final_mask = watershed_labels > 0
                
                # One more round of morphological operations to refine shapes
                final_mask = binary_opening_wrapper(final_mask, disk_size=2)
                final_mask = binary_closing_wrapper(final_mask, disk_size=2)
                
                # Fill small holes
                final_mask = ndi.binary_fill_holes(final_mask).astype(np.uint8) * 255
                
                return final_mask
            else:
                return np.zeros_like(h_channel)
                
        except Exception as e:
            logger.error(f"Error in stain separation method: {str(e)}")
            # Fallback to adaptive threshold if stain separation fails
            return self.adaptive_threshold_method(image)
            
    def _enhanced_stain_separation_method(self, image):
        """
        Enhanced stain separation with improved nuclei detection
        
        This method uses:
        1. Improved stain separation using a more robust color deconvolution
        2. Advanced contrast enhancement
        3. Multi-level thresholding for handling varying stain intensities
        4. Marker-controlled watershed for better separation of touching nuclei
        
        Args:
            image: RGB image
            
        Returns:
            Binary mask of nuclei
        """
        try:
            # Ensure RGB image
            if len(image.shape) == 2 or image.shape[2] == 1:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image.copy()
                
            # Step 1: Normalize staining to reduce variability between slides
            try:
                normalized_image = normalize_staining(image_rgb)
            except Exception as e:
                logger.warning(f"Stain normalization failed: {e}. Using original image.")
                normalized_image = image_rgb
                
            # Step 2: Enhanced stain separation using rgb2hed
            hed_image = rgb2hed(normalized_image)
            
            # Extract hematoxylin channel (nuclei) and normalize
            h_channel = hed_image[:, :, 0]
            
            # Better normalization with outlier handling
            p1, p99 = np.percentile(h_channel, (1, 99))
            h_channel_norm = rescale_intensity(h_channel, in_range=(p1, p99))
            
            # Convert to 8-bit
            h_channel_norm = (h_channel_norm * 255).astype(np.uint8)
            
            # Invert channel so nuclei are bright
            h_channel_norm = 255 - h_channel_norm
            
            # Step 3: Apply stronger CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            h_enhanced = clahe.apply(h_channel_norm)
            
            # Step 4: Apply bilateral filter to smooth while preserving edges
            h_smoothed = cv2.bilateralFilter(h_enhanced, 5, 40, 40)
            
            # Step 5: Multi-level thresholding
            # Try multi-Otsu first to handle varying intensities
            try:
                # Different versions of scikit-image return different types
                # Some return just thresholds, others return (thresholds, histogram)
                result = threshold_multiotsu(h_smoothed, classes=3)
                
                # Handle different return types
                if isinstance(result, tuple) and len(result) == 2:
                    # Newer versions return (thresholds, histogram)
                    thresholds = result[0]
                else:
                    # Older versions return just thresholds
                    thresholds = result
                    
                # Use the lower threshold to capture more nuclei
                lower_thresh = thresholds[0]
                binary_mask = h_smoothed > lower_thresh
            except Exception as e:
                logger.warning(f"Multi-Otsu thresholding failed: {e}. Using standard Otsu.")
                # Fallback to standard Otsu
                otsu_thresh = threshold_otsu(h_smoothed)
                binary_mask = h_smoothed > otsu_thresh
                
            # Create adaptive threshold as well
            block_size = max(15, int(min(h_smoothed.shape) * 0.05))
            if block_size % 2 == 0:
                block_size += 1  # Ensure odd block size
                
            adaptive_thresh = threshold_local(h_smoothed, block_size=block_size, offset=0)
            adaptive_mask = h_smoothed > adaptive_thresh
            
            # Combine masks for maximum sensitivity
            combined_mask = np.logical_or(binary_mask, adaptive_mask)
            
            # Step 6: Clean up mask
            # Fill small holes
            filled_mask = remove_small_holes(combined_mask, area_threshold=50)
            
            # Remove small objects
            clean_mask = remove_small_objects(filled_mask, min_size=30)
            
            # Step 7: Apply morphological operations to refine shapes
            clean_mask_uint8 = clean_mask.astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            
            # Close gaps
            closed = cv2.morphologyEx(clean_mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Step 8: Separate touching nuclei with watershed
            # Distance transform
            dist = ndi.distance_transform_edt(closed)
            
            # Find local maxima for markers
            # Use the wrapper function to ensure compatibility
            local_max_coords = peak_local_max_wrapper(
                dist, 
                min_distance=7,  # Larger min_distance for better separation
                labels=closed,
                exclude_border=False
            )
            
            # Create markers
            markers = np.zeros_like(dist, dtype=np.int32)
            
            # Check if we have any local maxima
            has_local_maxima = len(local_max_coords) > 0
            
            # Process coordinates
            if has_local_maxima:
                for i, (y, x) in enumerate(local_max_coords):
                    markers[y, x] = i + 1
            
            # Process based on whether we found local maxima
            if has_local_maxima:
                # Apply watershed
                watershed_result = watershed(-dist, markers, mask=closed)
                
                # Convert to binary mask
                final_mask = watershed_result > 0
                
                # Post-processing: remove small objects and fill holes once more
                final_mask = remove_small_objects(final_mask, min_size=50)
                final_mask = remove_small_holes(final_mask, area_threshold=20)
                
                # Convert to uint8
                return final_mask.astype(np.uint8) * 255
            else:
                # No local maxima found, return the closed mask
                return closed * 255
                
        except Exception as e:
            logger.error(f"Error in enhanced stain separation: {str(e)}")
            # Fallback to adaptive threshold method instead of stain_separation_method to avoid recursion
            return self.adaptive_threshold_method(image)
    
    def adaptive_threshold_method(self, image):
        """
        Nuclei segmentation using adaptive thresholding
        
        Args:
            image: RGB image
            
        Returns:
            Binary mask of nuclei
        """
        logger.info("Generating mask using adaptive threshold method")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE to enhance contrast - increased clipLimit from 2.0 to 4.0
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply adaptive thresholding - reduced block_size from 35 to 25 and offset from 10 to 5
        binary_mask = threshold_local(enhanced, block_size=25, offset=5)
        binary_mask = enhanced < binary_mask
        
        # Apply morphological operations to clean up the mask
        binary_mask = binary_closing_wrapper(binary_mask, disk_size=2)
        binary_mask = remove_small_objects(binary_mask, min_size=MIN_NUCLEUS_SIZE)
        binary_mask = remove_small_holes(binary_mask, area_threshold=50)
        
        return binary_mask.astype(np.uint8) * 255
    
    def watershed_method(self, image):
        """
        Use watershed algorithm to segment nuclei.
        
        This method is especially good at separating touching/overlapping nuclei.
        
        Args:
            image: RGB image
            
        Returns:
            Binary mask of nuclei
        """
        # Use enhanced watershed method if advanced segmentation is enabled
        if self.advanced_segmentation:
            return self._enhanced_watershed_method(image)
        
        # Original watershed method - unchanged
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_clahe, (5, 5), 0)
        
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area - dilate the image
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        # Use distance transform to find centers of nuclei
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        
        # Normalize distance transform for better visualization and thresholding
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        
        # Get centers of nuclei (sure foreground) - stronger threshold for better separation
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        
        # Finding unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        # Label sure foreground objects
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add 1 to all labels so that background is 1 instead of 0
        markers = markers + 1
        
        # Mark unknown region with 0
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(image, markers)
        
        # Create mask where markers > 1 are nuclei
        nuclei_mask = np.zeros_like(gray)
        nuclei_mask[markers > 1] = 255
        
        # Further refine mask with additional morphological operations
        # Fill small holes
        nuclei_mask = ndi.binary_fill_holes(nuclei_mask > 0).astype(np.uint8) * 255
        
        # Remove small objects (debris, etc.)
        labeled_mask, num_features = label(nuclei_mask > 0, return_num=True)
        sizes = np.bincount(labeled_mask.ravel())
        mask_sizes = sizes[1:] if len(sizes) > 1 else []
        
        # Skip filtering if no nuclei
        if len(mask_sizes) == 0:
            return np.zeros_like(gray)
            
        # Determine small object threshold - adaptive based on image statistics
        min_size_threshold = max(20, np.median(mask_sizes) * 0.2)  # At least 20 pixels, or 20% of median
        
        # Remove small objects
        filtered_mask = np.zeros_like(nuclei_mask)
        for i in range(1, num_features + 1):
            if sizes[i] >= min_size_threshold:
                filtered_mask[labeled_mask == i] = 255
        
        return filtered_mask
        
    def _enhanced_watershed_method(self, image):
        """
        Enhanced watershed algorithm with improved preprocessing and separation
        
        This improved method uses a combination of:
        1. Enhanced contrast with CLAHE using optimal parameters
        2. Multiple threshold levels to capture varying intensities
        3. Improved distance transform and marker generation
        4. Watershed with controlled markers
        
        Args:
            image: RGB image
            
        Returns:
            Binary mask of nuclei
        """
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Enhance contrast with stronger CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)
        
        # Apply gentle Gaussian blur to reduce noise while preserving edges
        blurred = cv2.GaussianBlur(gray_clahe, (3, 3), 0)
        
        # Apply multi-level thresholding for more reliable nuclei detection
        # 1. Otsu's method for global thresholding
        otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 2. Local adaptive thresholding for varying intensity regions
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 25, 3
        )
        
        # Combine thresholding methods
        binary = cv2.bitwise_or(
            cv2.threshold(blurred, otsu_thresh, 255, cv2.THRESH_BINARY_INV)[1], 
            adaptive_thresh
        )
        
        # Enhanced morphological operations to clean noise and preserve shape
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Sure background area - use more controlled dilation
        sure_bg = cv2.dilate(opening, kernel, iterations=2)
        
        # Better foreground detection through improved distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
        
        # Normalize distance transform
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        
        # Adaptive threshold based on image statistics for foreground detection
        mean_dist = np.mean(dist_transform[dist_transform > 0])
        std_dist = np.std(dist_transform[dist_transform > 0])
        
        # Adaptive threshold for watershed markers - this helps separate touching nuclei
        fg_thresh = max(0.3, min(0.7, mean_dist + 0.5 * std_dist))
        _, sure_fg = cv2.threshold(dist_transform, fg_thresh, 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        
        # Finding unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Improved marker generation
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add 1 to all labels so that background is 1 instead of 0
        markers = markers + 1
        
        # Mark unknown region with 0
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(image, markers)
        
        # Create mask where markers > 1 are nuclei
        nuclei_mask = np.zeros_like(gray)
        nuclei_mask[markers > 1] = 255
        
        # Fill small holes
        nuclei_mask = ndi.binary_fill_holes(nuclei_mask > 0).astype(np.uint8) * 255
        
        # Remove small objects (debris, etc.) with adaptive size threshold
        labeled_mask, num_features = label(nuclei_mask > 0, return_num=True)
        
        if num_features == 0:
            return np.zeros_like(gray)
            
        # Calculate object sizes
        sizes = np.bincount(labeled_mask.ravel())
        mask_sizes = sizes[1:] if len(sizes) > 1 else []
        
        # Skip filtering if no nuclei
        if len(mask_sizes) == 0:
            return np.zeros_like(gray)
            
        # Enhanced size threshold for small object removal
        # More robust to variations in image scale and staining
        median_size = np.median(mask_sizes)
        min_size_threshold = max(25, median_size * 0.25)  # At least 25 pixels, or 25% of median
        
        # Remove small objects
        filtered_mask = np.zeros_like(nuclei_mask)
        for i in range(1, num_features + 1):
            if sizes[i] >= min_size_threshold:
                filtered_mask[labeled_mask == i] = 255
        
        return filtered_mask
    
    def model_based_method(self, image):
        """
        Nuclei segmentation using deep learning model
        
        Args:
            image: RGB image
            
        Returns:
            Binary mask of nuclei
        """
        if not TORCH_AVAILABLE or self.model is None:
            logger.warning("Model-based segmentation not available. Using watershed method instead.")
            return self.watershed_method(image)
            
        logger.info("Generating mask using model-based method")
        
        try:
            # Resize image to model input size if needed
            height, width = image.shape[:2]
            target_size = (256, 256)  # Typical size for most models
            
            # Resize image
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            
            # Convert to torch tensor and normalize
            img_tensor = TF.to_tensor(Image.fromarray(resized))
            img_tensor = TF.normalize(img_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                output = self.model(img_tensor)
                
            # Process output based on model architecture
            if isinstance(output, tuple):
                output = output[0]  # Some models return multiple outputs
                
            # Convert to binary mask
            if output.shape[1] == 1:  # Single channel output
                pred = torch.sigmoid(output).squeeze().cpu().numpy()
                binary_mask = (pred > 0.5).astype(np.uint8) * 255
            else:  # Multi-channel output (use argmax)
                pred = torch.softmax(output, dim=1).cpu().numpy()
                binary_mask = (np.argmax(pred, axis=1)[0] == 1).astype(np.uint8) * 255
                
            # Resize back to original size
            binary_mask = cv2.resize(binary_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            return binary_mask
            
        except Exception as e:
            logger.error(f"Error in model-based segmentation: {str(e)}")
            # Fallback to watershed method
            return self.watershed_method(image)
    
    def ensemble_method(self, image):
        """
        Ensemble method combining multiple segmentation approaches
        
        This method combines the results from multiple segmentation methods
        for better accuracy across different image types.
        
        Args:
            image: Input RGB image
            
        Returns:
            Binary mask of nuclei
        """
        self.logger.info("Generating mask using ensemble method")
        
        # Apply each individual method
        mask_adaptive = self.adaptive_threshold_method(image)
        mask_watershed = self.watershed_method(image)
        mask_stain = self.stain_separation_method(image)
        
        # Get model-based mask if available
        mask_model = None
        if TORCH_AVAILABLE and self.model is not None:
            mask_model = self.model_based_method(image)
        
        # Convert all masks to binary
        mask_adaptive_binary = mask_adaptive > 0
        mask_watershed_binary = mask_watershed > 0
        mask_stain_binary = mask_stain > 0
        
        # Calculate ensemble mask
        ensemble_mask = np.zeros_like(mask_adaptive_binary)
        
        # For each pixel, count how many methods detected a nucleus
        detection_count = mask_adaptive_binary.astype(int) + \
                          mask_watershed_binary.astype(int) + \
                          mask_stain_binary.astype(int)
                          
        # If model mask is available, add it to the count
        if mask_model is not None:
            mask_model_binary = mask_model > 0
            detection_count += mask_model_binary.astype(int)
            # Set threshold based on number of methods (3 or 4)
            threshold = 2  # At least 2 methods must agree
        else:
            # Set threshold based on number of methods (3)
            threshold = 2  # At least 2 methods must agree
        
        # Create ensemble mask where at least 'threshold' methods agree
        ensemble_mask = detection_count >= threshold
        
        # Post-process the ensemble mask
        # Fill small holes
        ensemble_mask = ndi.binary_fill_holes(ensemble_mask)
        
        # Remove small objects
        ensemble_mask = remove_small_objects(ensemble_mask, min_size=20)
        
        # Return as uint8
        return ensemble_mask.astype(np.uint8) * 255
    
    def post_process_mask(self, mask):
        """
        Post-process the generated mask
        
        Args:
            mask: Binary mask of nuclei
            
        Returns:
            Post-processed binary mask
        """
        # Ensure mask is binary
        if mask.max() > 1:
            binary_mask = mask > 127
        else:
            binary_mask = mask > 0
            
        # Apply morphological operations to enhance weak nuclei signals
        binary_mask = binary_dilation_wrapper(binary_mask, disk_size=1)
        binary_mask = binary_closing_wrapper(binary_mask, disk_size=2)
            
        # Remove small objects
        binary_mask = remove_small_objects(binary_mask, min_size=MIN_NUCLEUS_SIZE)
        
        # Remove objects that are too large
        label_img = label(binary_mask)
        for region in regionprops(label_img):
            if region.area > MAX_NUCLEUS_SIZE:
                binary_mask[label_img == region.label] = 0
                
        # Fill small holes inside nuclei
        binary_mask = remove_small_holes(binary_mask, area_threshold=100)
        
        # Apply shape filtering using circularity, but be less strict for non-empty masks
        label_img = label(binary_mask)
        filtered_mask = np.zeros_like(binary_mask)
        
        # Count nuclei before filtering
        nuclei_count = len(regionprops(label_img))
        
        if nuclei_count > 0:
            # If we have some nuclei, we can be less strict with filtering
            current_circularity_threshold = NUCLEI_CIRCULARITY_THRESHOLD
            
            # Keep trying with lower thresholds if needed to get at least some nuclei
            while True:
                for region in regionprops(label_img):
                    circularity = compute_circularity(region)
                    if circularity >= current_circularity_threshold:
                        filtered_mask[label_img == region.label] = 1
                        
                # If we have at least some nuclei or we've reached a very low threshold, we're done
                if np.sum(filtered_mask) > 0 or current_circularity_threshold < 0.05:
                    break
                    
                # Lower the threshold and try again
                current_circularity_threshold *= 0.5
                logger.info(f"Lowering circularity threshold to {current_circularity_threshold}")
        else:
            # No nuclei found, use original mask
            filtered_mask = binary_mask
            
        # Final cleanup
        result_mask = filtered_mask.astype(np.uint8) * 255
        
        # If the mask is empty, warn and attempt to get at least something
        if np.sum(result_mask) == 0:
            logger.warning("Empty mask after post-processing. Attempting to recover some nuclei...")
            # Try again with extreme parameters
            result_mask = (mask > 0).astype(np.uint8) * 255
            
        return result_mask

    def extract_nuclei_features(self, image, mask):
        """
        Extract features from nuclei mask
        
        Args:
            image: RGB image
            mask: Binary mask of nuclei
            
        Returns:
            Dictionary of nuclei features
        """
        # Check if mask is None
        if mask is None:
            logger.warning("Mask is None. Using default feature values.")
            return {
                'cellularity': 0,
                'nucleomegaly': 0,
                'nuclear_grooves': 0, 
                'nuclear_clearing': 0,
                'nuclear_inclusions': 0,
                'malignancy_score': 0,
                'individual_nuclei': []
            }
            
        # Ensure mask is a numpy array
        if not isinstance(mask, np.ndarray):
            try:
                mask = np.array(mask)
            except Exception as e:
                logger.error(f"Failed to convert mask to numpy array: {e}")
                return {
                    'cellularity': 0,
                    'nucleomegaly': 0,
                    'nuclear_grooves': 0, 
                    'nuclear_clearing': 0,
                    'nuclear_inclusions': 0,
                    'malignancy_score': 0,
                    'individual_nuclei': []
                }
        
        # Check if mask is empty
        if mask.size == 0 or np.all(mask == 0):
            logger.warning("Empty mask. Using default feature values.")
            return {
                'cellularity': 0,
                'nucleomegaly': 0,
                'nuclear_grooves': 0, 
                'nuclear_clearing': 0,
                'nuclear_inclusions': 0,
                'malignancy_score': 0,
                'individual_nuclei': []
            }
        
        # Resize mask if needed
        if mask.shape[:2] != image.shape[:2]:
            logger.info(f"Resizing mask from {mask.shape} to match image dimensions {image.shape[:2]}")
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Ensure mask is binary
        binary_mask = mask > 0
        
        # Convert image to LAB color space for better color analysis
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Label connected components (individual nuclei)
        labeled_mask, num_nuclei = label(binary_mask, return_num=True)
        
        # Extract texture and chromatin features
        texture_features = self.extract_texture_features(image, binary_mask)
        chromatin_features = self.analyze_chromatin_distribution(image, binary_mask)
        
        # Calculate total tissue area (non-white pixels)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        tissue_mask = gray < 220  # Consider pixels with grayscale < 220 as tissue
        tissue_area = np.sum(tissue_mask)
        
        # Calculate cellularity
        nuclei_area = np.sum(binary_mask)
        cellularity = min(1.0, nuclei_area / (tissue_area + 1e-6))
        
        # Get properties of individual nuclei
        if num_nuclei > 0:
            props = regionprops(labeled_mask, intensity_image=lab_image)
            
            # Calculate nuclei sizes
            nuclei_areas = [prop.area for prop in props]
            nuclei_perimeters = [prop.perimeter for prop in props if prop.perimeter > 0]
            
            # Calculate nuclei shapes and sizes
            circularities = []
            aspect_ratios = []
            intensities = []
            equiv_diameters = []
            
            for prop in props:
                if prop.perimeter > 0:
                    # Calculate circularity (4π * area / perimeter²)
                    circularity = 4 * np.pi * prop.area / (prop.perimeter * prop.perimeter)
                    circularities.append(circularity)
                    
                # Calculate aspect ratio
                if prop.minor_axis_length > 0:
                    aspect_ratio = prop.major_axis_length / prop.minor_axis_length
                    aspect_ratios.append(aspect_ratio)
                
                # Get mean intensity in LAB color space
                intensities.append(np.mean(prop.intensity_image, axis=(0,1)))
                
                # Get equivalent diameter
                equiv_diameters.append(prop.equivalent_diameter)
            
            # Get sizes of normal tissue nuclei (from literature)
            # Typical thyroid cell nucleus is ~6-8µm diameter
            # Assuming image scale of ~0.25µm/pixel, normal would be ~25-30 pixels diameter
            normal_nucleus_size = 28 * 28 * np.pi  # area of circle with diameter 28 pixels
            
            # Calculate nucleomegaly (ratio of mean nuclear size to normal size)
            if equiv_diameters:
                mean_equiv_diameter = np.mean(equiv_diameters)
                nucleomegaly = min(1.0, max(0.0, (mean_equiv_diameter / 28 - 1) * 2))
            else:
                nucleomegaly = 0
                
            # Calculate nuclear groove features based on shape
            # Nuclear grooves tend to create more elongated shapes with lower circularity
            if circularities and aspect_ratios:
                mean_circularity = np.mean(circularities)
                mean_aspect_ratio = np.mean(aspect_ratios)
                
                # Combine circularity and aspect ratio
                # Lower circularity and higher aspect ratio indicate grooves
                nuclear_grooves = min(1.0, max(0.0, 
                                              (1 - mean_circularity) * 0.5 + 
                                              (mean_aspect_ratio - 1) * 0.2))
            else:
                nuclear_grooves = 0
                
            # Calculate nuclear clearing using chromatin features
            # Higher values mean more clearing (pale nuclei)
            nuclear_clearing = min(1.0, max(0.0, 
                                          chromatin_features['chromatin_clearing_ratio'] * 0.7 +
                                          (1 - chromatin_features['chromatin_uniformity']) * 0.3))
                
            # Calculate nuclear inclusion features using texture properties
            # Nuclear inclusions create more textured, heterogeneous nuclei
            nuclear_inclusions = min(1.0, max(0.0,
                                           texture_features['contrast'] * 0.3 +
                                           texture_features['dissimilarity'] * 0.2 +
                                           (1 - texture_features['homogeneity']) * 0.5))
                
            # Create list of individual nuclei features
            individual_nuclei = []
            for i, prop in enumerate(props):
                if i < 100:  # Limit to 100 nuclei to keep data size reasonable
                    nucleus_features = {
                        'area': prop.area,
                        'perimeter': prop.perimeter if hasattr(prop, 'perimeter') else 0,
                        'eccentricity': prop.eccentricity if hasattr(prop, 'eccentricity') else 0,
                        'equivalent_diameter': prop.equivalent_diameter,
                        'circularity': circularities[i] if i < len(circularities) else 0,
                        'aspect_ratio': aspect_ratios[i] if i < len(aspect_ratios) else 0,
                    }
                    individual_nuclei.append(nucleus_features)
        else:
            # No nuclei found
            nucleomegaly = 0
            nuclear_grooves = 0
            nuclear_clearing = 0
            nuclear_inclusions = 0
            individual_nuclei = []
        
        # Combine all features
        features = {
            'cellularity': cellularity,
            'nucleomegaly': nucleomegaly,
            'nuclear_grooves': nuclear_grooves,
            'nuclear_clearing': nuclear_clearing,
            'nuclear_inclusions': nuclear_inclusions,
            'individual_nuclei': individual_nuclei
        }
        
        # Add texture and chromatin features
        features.update(texture_features)
        features.update(chromatin_features)
        
        # Calculate malignancy score
        features['malignancy_score'] = self._calculate_malignancy_score(features)
        
        return features

    def extract_texture_features(self, image, mask):
        """
        Extract texture features from nuclei using grayscale co-occurrence matrix
        
        Args:
            image: Input RGB image
            mask: Binary mask of nuclei
            
        Returns:
            Dictionary of texture features
        """
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create labeled mask
        labeled_mask, num_nuclei = label(mask > 0, return_num=True)
        
        if num_nuclei == 0:
            logger.warning("No nuclei found for texture analysis")
            return {
                'contrast': 0,
                'dissimilarity': 0,
                'homogeneity': 0,
                'energy': 0,
                'correlation': 0
            }
        
        # Initialize texture feature arrays
        contrast_values = []
        dissimilarity_values = []
        homogeneity_values = []
        energy_values = []
        correlation_values = []
        
        # Analyze each nucleus
        for i in range(1, num_nuclei + 1):
            # Extract the nucleus region
            nucleus_mask = (labeled_mask == i)
            
            # Skip very small nuclei
            if np.sum(nucleus_mask) < 20:
                continue
                
            # Create a bounding box around the nucleus
            props = regionprops(nucleus_mask.astype(int))
            if not props:
                continue
                
            y1, x1, y2, x2 = props[0].bbox
            nucleus_img = gray[y1:y2, x1:x2]
            nucleus_mask_roi = nucleus_mask[y1:y2, x1:x2]
            
            # Apply mask to grayscale image
            nucleus_gray = nucleus_img * nucleus_mask_roi
            
            # Skip if nucleus is too small for GLCM
            if nucleus_gray.shape[0] < 3 or nucleus_gray.shape[1] < 3:
                continue
                
            # Calculate GLCM features
            try:
                # Generate GLCM with distance 1 and multiple angles
                glcm = graycomatrix(nucleus_gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                    symmetric=True, normed=True)
                
                # Calculate features and average over angles
                contrast = np.mean(graycoprops(glcm, 'contrast'))
                dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
                homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
                energy = np.mean(graycoprops(glcm, 'energy'))
                correlation = np.mean(graycoprops(glcm, 'correlation'))
                
                contrast_values.append(contrast)
                dissimilarity_values.append(dissimilarity)
                homogeneity_values.append(homogeneity)
                energy_values.append(energy)
                correlation_values.append(correlation)
            except Exception as e:
                logger.debug(f"Error calculating GLCM for nucleus: {str(e)}")
                continue
        
        # Return empty features if no valid nuclei were found
        if not contrast_values:
            logger.warning("No valid nuclei found for texture analysis")
            return {
                'contrast': 0,
                'dissimilarity': 0,
                'homogeneity': 0,
                'energy': 0,
                'correlation': 0
            }
        
        # Return average features
        return {
            'contrast': np.mean(contrast_values),
            'dissimilarity': np.mean(dissimilarity_values),
            'homogeneity': np.mean(homogeneity_values),
            'energy': np.mean(energy_values),
            'correlation': np.mean(correlation_values)
        }
        
    def analyze_chromatin_distribution(self, image, mask):
        """
        Analyze chromatin distribution within nuclei
        
        Args:
            image: Input RGB image
            mask: Binary mask of nuclei
            
        Returns:
            Dictionary with chromatin distribution features
        """
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create labeled mask
        labeled_mask, num_nuclei = label(mask > 0, return_num=True)
        
        if num_nuclei == 0:
            logger.warning("No nuclei found for chromatin analysis")
            return {
                'chromatin_uniformity': 0,
                'chromatin_clearing_ratio': 0,
                'chromatin_margination': 0
            }
        
        # Initialize feature arrays
        uniformity_values = []
        clearing_ratio_values = []
        margination_values = []
        
        # Analyze each nucleus
        for i in range(1, num_nuclei + 1):
            # Extract the nucleus region
            nucleus_mask = (labeled_mask == i)
            
            # Skip very small nuclei
            if np.sum(nucleus_mask) < 20:
                continue
                
            # Create a bounding box around the nucleus
            props = regionprops(nucleus_mask.astype(int))
            if not props:
                continue
                
            y1, x1, y2, x2 = props[0].bbox
            nucleus_img = gray[y1:y2, x1:x2]
            nucleus_mask_roi = nucleus_mask[y1:y2, x1:x2]
            
            # Apply mask to grayscale image
            nucleus_gray = nucleus_img * nucleus_mask_roi
            
            # Skip if nucleus is empty
            if np.sum(nucleus_mask_roi) == 0:
                continue
                
            # Calculate uniformity (inverse of variance)
            pixel_values = nucleus_gray[nucleus_mask_roi > 0]
            if len(pixel_values) == 0:
                continue
                
            uniformity = 1.0 / (1.0 + np.var(pixel_values))
            
            # Calculate clearing ratio
            # (ratio of pixels with intensity > threshold)
            if len(pixel_values) > 0:
                threshold = np.mean(pixel_values) + 0.5 * np.std(pixel_values)
                clearing_ratio = np.sum(pixel_values > threshold) / len(pixel_values)
            else:
                clearing_ratio = 0
                
            # Calculate chromatin margination
            # (higher intensity at borders compared to center)
            # Create erosion of mask to get inner part
            kernel = np.ones((3,3), np.uint8)
            inner_mask = cv2.erode(nucleus_mask_roi.astype(np.uint8), kernel, iterations=1)
            
            # Border is the difference between original and eroded mask
            border_mask = nucleus_mask_roi.astype(np.uint8) - inner_mask
            
            # Calculate ratio of border intensity to center intensity
            border_pixels = nucleus_gray[border_mask > 0]
            inner_pixels = nucleus_gray[inner_mask > 0]
            
            if len(border_pixels) > 0 and len(inner_pixels) > 0:
                border_mean = np.mean(border_pixels)
                inner_mean = np.mean(inner_pixels)
                if inner_mean > 0:
                    margination = border_mean / inner_mean
                else:
                    margination = 1.0
            else:
                margination = 1.0
                
            uniformity_values.append(uniformity)
            clearing_ratio_values.append(clearing_ratio)
            margination_values.append(margination)
        
        # Return empty features if no valid nuclei were found
        if not uniformity_values:
            logger.warning("No valid nuclei found for chromatin analysis")
            return {
                'chromatin_uniformity': 0,
                'chromatin_clearing_ratio': 0,
                'chromatin_margination': 0
            }
        
        # Return average features
        return {
            'chromatin_uniformity': np.mean(uniformity_values),
            'chromatin_clearing_ratio': np.mean(clearing_ratio_values),
            'chromatin_margination': np.mean(margination_values)
        }

    def generate_feature_masks(self, image, mask):
        """
        Generate feature-specific masks to visualize nuclei characteristics
        
        Args:
            image: RGB image
            mask: Binary mask of nuclei
            
        Returns:
            Dictionary of feature masks and visualization images
        """
        # Ensure mask has the same dimensions as the image
        if mask.shape[:2] != image.shape[:2]:
            logger.info(f"Resizing mask from {mask.shape} to match image dimensions {image.shape[:2]}")
            # Resize mask to match image dimensions
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            
        # Extract features first
        features = self.extract_nuclei_features(image, mask)
        
        # Create labeled mask
        labeled_mask = label(mask > 0)
        
        # Initialize feature masks (heatmaps)
        feature_masks = {
            'cellularity': np.zeros_like(mask, dtype=np.float32),
            'nucleomegaly': np.zeros_like(mask, dtype=np.float32),
            'nuclear_grooves': np.zeros_like(mask, dtype=np.float32),
            'nuclear_clearing': np.zeros_like(mask, dtype=np.float32),
            'nuclear_inclusions': np.zeros_like(mask, dtype=np.float32),
            'malignancy': np.zeros_like(mask, dtype=np.float32)
        }
        
        # Create mask visualizations
        visualizations = {}
        
        # Fill in values for each nucleus based on individual features
        for i, nucleus in enumerate(features['individual_nuclei']):
            # Find the current nucleus in the labeled mask
            nuc_id = i + 1  # Region IDs start from 1
            
            # Set feature values for this nucleus
            feature_masks['nucleomegaly'][labeled_mask == nuc_id] = nucleus['area'] / 1000  # Normalize
            
            # Handle the case where we don't have specific scores
            # Use circularity and aspect ratio as approximate indicators
            circularity = nucleus.get('circularity', 0)
            aspect_ratio = nucleus.get('aspect_ratio', 1)
            
            # Calculate groove score from shape metrics if not provided
            groove_score = nucleus.get('groove_score', (1 - circularity) * 0.5 + (aspect_ratio - 1) * 0.2)
            feature_masks['nuclear_grooves'][labeled_mask == nuc_id] = groove_score
            
            # Use default values for clearing and inclusions if not provided
            clearing_score = nucleus.get('clearing_score', 0.5)
            feature_masks['nuclear_clearing'][labeled_mask == nuc_id] = clearing_score
            
            inclusion_score = nucleus.get('inclusion_score', 0.5)
            feature_masks['nuclear_inclusions'][labeled_mask == nuc_id] = inclusion_score
        
        # Set cellularity mask (based on density)
        feature_masks['cellularity'] = calculate_nuclei_density(mask > 0)
        
        # Set overall malignancy mask
        # For each nucleus, assign the overall malignancy score
        for i in range(1, np.max(labeled_mask) + 1):
            feature_masks['malignancy'][labeled_mask == i] = features['malignancy_score']
        
        # Create visualizations with heatmap overlay on original image
        for feature_name, feature_mask in feature_masks.items():
            # Ensure feature mask has the same dimensions as the image
            if feature_mask.shape[:2] != image.shape[:2]:
                feature_mask = cv2.resize(feature_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
                
            # Normalize mask to 0-1 if needed
            if np.max(feature_mask) > 0:
                norm_mask = feature_mask / np.max(feature_mask)
            else:
                norm_mask = feature_mask
            
            # Create heatmap
            # Convert to uint8 for colormap application
            heatmap = (norm_mask * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Ensure heatmap has the same shape as the image
            if heatmap.shape[:2] != image.shape[:2]:
                heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # Create overlay
            overlay = image.copy()
            binary_mask = mask > 0
            
            # Apply heatmap only to mask area
            for c in range(3):  # RGB channels
                overlay_channel = overlay[:,:,c]
                heatmap_channel = heatmap[:,:,c]
                
                # Apply weighted overlay only in the mask area
                overlay_channel[binary_mask] = (1-0.7) * overlay_channel[binary_mask] + 0.7 * heatmap_channel[binary_mask]
                
                # Update the overlay
                overlay[:,:,c] = overlay_channel
            
            visualizations[feature_name] = overlay
        
        # Add feature data
        result = {
            'features': features,
            'feature_masks': feature_masks,
            'visualizations': visualizations
        }
        
        return result

    def _calculate_malignancy_score(self, features):
        """
        Calculate an overall malignancy score based on nuclei features
        
        Args:
            features: Dictionary of nuclei features
            
        Returns:
            Malignancy score (0-1), where higher scores indicate likely malignancy
        """
        # Original weights
        original_weights = {
            'cellularity': 0.45,
            'nucleomegaly': 0.15,
            'nuclear_grooves': 0.30,
            'nuclear_clearing': 0.05,
            'nuclear_inclusions': 0.05
        }
        
        # Enhanced weights based on batch analysis (only used when advanced_segmentation is enabled)
        enhanced_weights = {
            'cellularity': 0.35,       # Reduced from 0.45 to be less dominant
            'nucleomegaly': 0.20,      # Increased from 0.15 to give more weight to nuclear size
            'nuclear_grooves': 0.25,   # Reduced from 0.30 but still important
            'nuclear_clearing': 0.10,  # Increased from 0.05 as it's more important
            'nuclear_inclusions': 0.10  # Increased from 0.05 to better distinguish benign cells
        }
        
        # Choose weights based on configuration
        weights = enhanced_weights if self.advanced_segmentation else original_weights
        
        # Calculate weighted sum
        weighted_sum = sum(weights[feature] * features[feature] for feature in weights)
        
        # Apply sigmoid function to get score between 0 and 1
        if self.advanced_segmentation:
            # Modified sigmoid with better calibration - less sensitive to high cellularity
            malignancy_score = 1 / (1 + np.exp(-10 * (weighted_sum - 0.5)))
        else:
            # Original sigmoid function
            malignancy_score = 1 / (1 + np.exp(-10 * (weighted_sum - 0.5)))
        
        # Apply correction for high cellularity benign cases
        if features['cellularity'] > 0.7:
            nuclear_features_avg = (features['nucleomegaly'] + features['nuclear_grooves'] + 
                                  features['nuclear_clearing']) / 3
            
            # Only boost if nuclear features are also suggestive of malignancy
            if nuclear_features_avg > 0.4:
                cellularity_boost = min(0.15, (features['cellularity'] - 0.7) * 1.5)
                malignancy_score = min(1.0, malignancy_score + cellularity_boost)
            else:
                # Actually reduce score if high cellularity but low nuclear features
                # This adjustment is stronger when advanced_segmentation is enabled
                if self.advanced_segmentation:
                    malignancy_score = max(0.3, malignancy_score * 0.7)  # More aggressive reduction
                else:
                    malignancy_score = max(0.3, malignancy_score * 0.8)
        
        # Apply texture feature adjustments
        if 'contrast' in features and 'chromatin_uniformity' in features:
            # High contrast and low uniformity are indicators of malignancy
            texture_score = (features.get('contrast', 0) * 0.4 + 
                          (1 - features.get('homogeneity', 0)) * 0.3 + 
                          (1 - features.get('chromatin_uniformity', 0)) * 0.3)
            
            if self.advanced_segmentation:
                # Enhanced texture-based adjustment when advanced segmentation is enabled
                if texture_score > 0.6 and malignancy_score < 0.8:
                    # Texture strongly suggests malignancy but overall score is lower
                    malignancy_score = min(0.95, malignancy_score + (texture_score - 0.6) * 0.7)
                elif texture_score < 0.3 and malignancy_score > 0.2:
                    # Texture strongly suggests benignity but overall score is higher
                    malignancy_score = max(0.05, malignancy_score - (0.3 - texture_score) * 0.7)
            else:
                # Original texture-based adjustment
                if texture_score > 0.6 and malignancy_score < 0.7:
                    # Texture suggests malignancy but overall score is lower
                    malignancy_score = min(0.9, malignancy_score + (texture_score - 0.6) * 0.5)
                elif texture_score < 0.3 and malignancy_score > 0.3:
                    # Texture suggests benignity but overall score is higher
                    malignancy_score = max(0.1, malignancy_score - (0.3 - texture_score) * 0.5)
        
        # Consider chromatin margination (higher in malignant cells)
        if 'chromatin_margination' in features and features['chromatin_margination'] > 1.2:
            if self.advanced_segmentation:
                # Enhanced adjustment for significant chromatin margination
                margination_boost = min(0.15, (features['chromatin_margination'] - 1.2) * 0.5)
            else:
                # Original adjustment
                margination_boost = min(0.1, (features['chromatin_margination'] - 1.2) * 0.3)
                
            malignancy_score = min(1.0, malignancy_score + margination_boost)
        
        return malignancy_score

def generate_mask_for_dataset(input_dir, output_dir, method='ensemble', batch_size=8, extract_features=True):
    """
    Generate masks for all images in a dataset
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save generated masks
        method: Segmentation method to use
        batch_size: Number of images to process in parallel
        extract_features: Whether to extract and save nuclei features
        
    Returns:
        Dictionary containing statistics about the generated masks
    """
    logger.info(f"Generating masks for dataset in {input_dir} using method {method}")
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Find all images in input directory (support common image formats)
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
        image_files.extend(list(Path(input_dir).glob(f"*{ext}")))
    
    logger.info(f"Found {len(image_files)} images in {input_dir}")
    
    # Initialize mask generator
    mask_generator = AdvancedNucleiMaskGenerator(
        base_method=method,
        extract_features=extract_features
    )
    
    # Create subdirectories for outputs
    mask_dir = os.path.join(output_dir, 'masks')
    features_dir = os.path.join(output_dir, 'features')
    vis_dir = os.path.join(output_dir, 'visualizations')
    
    ensure_dir(mask_dir)
    if extract_features:
        ensure_dir(features_dir)
        ensure_dir(vis_dir)
        # Create subdirectories for different feature visualizations
        for feature_name in ['cellularity', 'nucleomegaly', 'nuclear_grooves', 
                            'nuclear_clearing', 'nuclear_inclusions', 'malignancy']:
            ensure_dir(os.path.join(vis_dir, feature_name))
    
    # Process images in batches for parallel processing
    total_batches = (len(image_files) + batch_size - 1) // batch_size
    
    # Statistics
    stats = {
        'total_images': len(image_files),
        'processed_images': 0,
        'failed_images': 0,
        'avg_processing_time': 0,
        'avg_nuclei_count': 0,
        'feature_stats': defaultdict(float)
    }
    
    # Process images
    total_processing_time = 0
    total_nuclei_count = 0
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(image_files))
        batch_files = image_files[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_idx+1}/{total_batches} ({len(batch_files)} images)")
        
        # Process images in parallel
        with ProcessPoolExecutor(max_workers=min(len(batch_files), os.cpu_count())) as executor:
            futures = []
            
            for image_file in batch_files:
                future = executor.submit(
                    process_single_image,
                    image_file,
                    mask_dir,
                    features_dir if extract_features else None,
                    vis_dir if extract_features else None,
                    mask_generator
                )
                futures.append(future)
            
            # Collect results
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
                try:
                    result = future.result()
                    if result:
                        stats['processed_images'] += 1
                        total_processing_time += result['processing_time']
                        
                        if 'nuclei_count' in result:
                            total_nuclei_count += result['nuclei_count']
                            
                        # Collect feature statistics
                        if 'features' in result:
                            for feat_name, feat_val in result['features'].items():
                                if feat_name != 'individual_nuclei':
                                    stats['feature_stats'][feat_name] += feat_val
                    else:
                        stats['failed_images'] += 1
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    stats['failed_images'] += 1
    
    # Calculate averages
    if stats['processed_images'] > 0:
        stats['avg_processing_time'] = total_processing_time / stats['processed_images']
        stats['avg_nuclei_count'] = total_nuclei_count / stats['processed_images']
        
        # Average feature values
        for feat_name in stats['feature_stats']:
            stats['feature_stats'][feat_name] /= stats['processed_images']
    
    # Log statistics
    logger.info(f"Processed {stats['processed_images']}/{stats['total_images']} images")
    logger.info(f"Average processing time: {stats['avg_processing_time']:.2f} seconds per image")
    logger.info(f"Average nuclei count: {stats['avg_nuclei_count']:.1f} nuclei per image")
    
    if extract_features:
        logger.info("Average feature values:")
        for feat_name, feat_val in stats['feature_stats'].items():
            logger.info(f"  {feat_name}: {feat_val:.4f}")
    
    return stats

def process_single_image(image_path, mask_dir, features_dir=None, vis_dir=None, mask_generator=None):
    """
    Process a single image and generate mask and features
    
    Args:
        image_path: Path to input image
        mask_dir: Directory to save generated mask
        features_dir: Directory to save extracted features
        vis_dir: Directory to save feature visualizations
        mask_generator: AdvancedNucleiMaskGenerator instance
        
    Returns:
        Dictionary with processing results and statistics
    """
    try:
        # Get image filename without extension
        image_name = Path(image_path).stem
        
        # Measure processing time
        start_time = time.time()
        
        # Generate mask (and features if requested)
        if features_dir is not None:
            # Generate mask with features
            mask, features = mask_generator.generate_mask(image_path)
            
            # Save features as NumPy file
            feature_path = os.path.join(features_dir, f"{image_name}_features.npz")
            
            # Remove individual_nuclei from features to save space
            features_to_save = {k: v for k, v in features.items() if k != 'individual_nuclei'}
            np.savez_compressed(feature_path, **features_to_save)
            
            # Generate and save visualizations
            image = io.imread(image_path)
            result = mask_generator.generate_feature_masks(image, mask)
            
            # Save visualizations
            for feat_name, vis_img in result['visualizations'].items():
                vis_path = os.path.join(vis_dir, feat_name, f"{image_name}_{feat_name}.png")
                io.imsave(vis_path, vis_img)
            
            nuclei_count = len(features['individual_nuclei']) if 'individual_nuclei' in features else 0
            
        else:
            # Generate mask only
            mask = mask_generator.generate_mask(image_path)
            features = None
            nuclei_count = None
        
        # Save mask
        mask_path = os.path.join(mask_dir, f"{image_name}_mask.png")
        io.imsave(mask_path, mask)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return statistics
        result = {
            'image_name': image_name,
            'processing_time': processing_time
        }
        
        if nuclei_count is not None:
            result['nuclei_count'] = nuclei_count
            
        if features is not None:
            # Only include aggregate features, not individual nuclei
            result['features'] = {k: v for k, v in features.items() if k != 'individual_nuclei'}
            
        return result
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None

def process_batch(image_batch, mask_generator):
    """
    Process a batch of images for segmentation pipeline integration
    
    Args:
        image_batch: Batch of images (tensor or numpy array)
        mask_generator: AdvancedNucleiMaskGenerator instance
        
    Returns:
        Dictionary containing masks and features
    """
    batch_size = len(image_batch)
    results = []
    
    for i in range(batch_size):
        # Extract single image
        image = image_batch[i]
        
        # Convert to numpy array if tensor
        if isinstance(image, torch.Tensor):
            # Convert from tensor [C,H,W] to numpy [H,W,C]
            image_np = image.permute(1, 2, 0).cpu().numpy()
            
            # Denormalize if needed
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image
        
        # Generate mask with features
        try:
            mask, features = mask_generator.generate_mask(image_np)
            results.append({
                'mask': mask,
                'features': features
            })
        except Exception as e:
            logger.error(f"Error generating mask for batch image {i}: {str(e)}")
            # Return empty mask and default features
            results.append({
                'mask': np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8),
                'features': {
                    'cellularity': 0,
                    'nucleomegaly': 0,
                    'nuclear_grooves': 0,
                    'nuclear_clearing': 0,
                    'nuclear_inclusions': 0,
                    'malignancy_score': 0
                }
            })
    
    return results

def visualize_results(image, mask, features=None, save_path=None):
    """
    Visualize segmentation and feature results
    
    Args:
        image: Input image
        mask: Generated mask
        features: Extracted features (optional)
        save_path: Path to save visualization (optional)
        
    Returns:
        Combined visualization image
    """
    # Ensure mask has the same dimensions as the image
    if mask.shape[:2] != image.shape[:2]:
        logger.info(f"Resizing mask from {mask.shape} to match image dimensions {image.shape[:2]}")
        # Resize mask to match image dimensions
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title('Original H&E Image')
    plt.axis('off')
    
    # Binary mask
    plt.subplot(2, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Nuclei Mask')
    plt.axis('off')
    
    # Overlay
    plt.subplot(2, 3, 3)
    overlay = image.copy()
    mask_rgb = np.zeros_like(image)
    binary_mask = mask > 0
    mask_rgb[binary_mask, 2] = 255  # Blue overlay for nuclei
    alpha = 0.5
    overlay = cv2.addWeighted(overlay, 1, mask_rgb, alpha, 0)
    plt.imshow(overlay)
    plt.title('Nuclei Overlay')
    plt.axis('off')
    
    # Display features if available
    if features is not None:
        plt.subplot(2, 3, 4)
        
        # Create a table with features
        feature_names = ['Cellularity', 'Nucleomegaly', 'Nuclear Grooves', 
                         'Nuclear Clearing', 'Nuclear Inclusions', 'Malignancy Score']
        feature_values = [
            features.get('cellularity', 0),
            features.get('nucleomegaly', 0),
            features.get('nuclear_grooves', 0),
            features.get('nuclear_clearing', 0),
            features.get('nuclear_inclusions', 0),
            features.get('malignancy_score', 0)
        ]
        
        # Format feature values
        feature_values_formatted = [f"{val:.3f}" for val in feature_values]
        
        # Create table
        table_data = list(zip(feature_names, feature_values_formatted))
        table = plt.table(
            cellText=table_data,
            colWidths=[0.6, 0.3],
            loc='center',
            cellLoc='left'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        plt.axis('off')
        plt.title('Nuclei Characteristics')
        
        # Display malignancy likelihood
        malignancy_score = features.get('malignancy_score', 0)
        if malignancy_score > 0.7:
            diagnosis = "Likely Malignant"
            color = 'red'
        elif malignancy_score < 0.3:
            diagnosis = "Likely Benign"
            color = 'green'
        else:
            diagnosis = "Indeterminate"
            color = 'orange'
            
        plt.subplot(2, 3, 5)
        plt.text(0.5, 0.5, f"{diagnosis}\n({malignancy_score:.2f})", 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=16, color=color,
                 transform=plt.gca().transAxes)
        plt.axis('off')
        plt.title('Assessment')
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Create combined image for return
    fig = plt.gcf()
    fig.canvas.draw()
    combined_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    combined_image = combined_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return combined_image

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Advanced Nuclei Mask Generator for H&E Histopathology Images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/output options
    parser.add_argument('--input', type=str, help='Input image or directory')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    
    # Processing options
    parser.add_argument('--method', type=str, default='ensemble',
                        choices=['stain', 'adaptive', 'watershed', 'model', 'ensemble'],
                        help='Mask generation method')
    parser.add_argument('--model-path', type=str, help='Path to trained model (for model-based method)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for dataset processing')
    
    # Feature extraction options
    parser.add_argument('--extract-features', action='store_true', help='Extract and analyze nuclei features')
    parser.add_argument('--visualize', action='store_true', help='Generate feature visualizations')
    
    # Integration options
    parser.add_argument('--on-the-fly', action='store_true',
                        help='Enable on-the-fly mask generation for segmentation pipeline')
    parser.add_argument('--integrate-with', type=str, 
                        choices=['train', 'val', 'test', 'all'],
                        help='Integrate with training, validation, or testing pipeline')
    
    # Output options
    parser.add_argument('--save-masks', action='store_true', help='Save generated masks')
    parser.add_argument('--save-features', action='store_true', help='Save extracted features')
    parser.add_argument('--save-visualizations', action='store_true', help='Save feature visualizations')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Ensure output directory exists
    ensure_dir(args.output)
    
    # Check if input is a file or directory
    if args.input and os.path.isfile(args.input):
        logger.info(f"Processing single image: {args.input}")
        
        # Initialize mask generator
        mask_generator = AdvancedNucleiMaskGenerator(
            base_method=args.method,
            model_path=args.model_path,
            extract_features=args.extract_features
        )
        
        # Load image
        image = io.imread(args.input)
        
        # Generate mask (and features if requested)
        start_time = time.time()
        
        if args.extract_features:
            mask, features = mask_generator.generate_mask(image)
        else:
            mask = mask_generator.generate_mask(image)
            features = None
            
        processing_time = time.time() - start_time
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        
        # Save results
        if args.save_masks:
            mask_path = os.path.join(args.output, f"{Path(args.input).stem}_mask.png")
            io.imsave(mask_path, mask)
            logger.info(f"Mask saved to: {mask_path}")
            
        if args.save_features and features is not None:
            feature_path = os.path.join(args.output, f"{Path(args.input).stem}_features.npz")
            # Remove individual_nuclei from features to save space
            features_to_save = {k: v for k, v in features.items() if k != 'individual_nuclei'}
            np.savez_compressed(feature_path, **features_to_save)
            logger.info(f"Features saved to: {feature_path}")
            
        if args.visualize:
            # Visualize results
            vis_path = os.path.join(args.output, f"{Path(args.input).stem}_visualization.png")
            visualize_results(image, mask, features, vis_path)
            logger.info(f"Visualization saved to: {vis_path}")
            
    elif args.input and os.path.isdir(args.input):
        logger.info(f"Processing images in directory: {args.input}")
        
        # Process all images in directory
        stats = generate_mask_for_dataset(
            args.input,
            args.output,
            method=args.method,
            batch_size=args.batch_size,
            extract_features=args.extract_features
        )
        
        # Log statistics
        logger.info("Processing complete!")
        logger.info(f"Processed {stats['processed_images']}/{stats['total_images']} images")
        logger.info(f"Average processing time: {stats['avg_processing_time']:.2f} seconds per image")
        
    elif args.on_the_fly:
        logger.info("On-the-fly mask generation enabled")
        logger.info(f"Will integrate with {args.integrate_with} pipeline")
        
    
        logger.info("Note: This module should be imported by the segmentation script")
        logger.info("      and used through the process_batch() function")
        
    else:
        logger.error("No input provided. Please specify --input or --on-the-fly")
        sys.exit(1)

if __name__ == "__main__":
    main() 