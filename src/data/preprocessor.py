import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Image preprocessing pipeline for retinopathy detection"""
    
    def __init__(self, image_size: int = 512, normalize: bool = True):
        self.image_size = image_size
        self.normalize = normalize
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def preprocess(self, image_path: str) -> np.ndarray:
        """Complete preprocessing pipeline"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self._resize(image)
        image = self._apply_clahe(image)
        
        if self.normalize:
            image = self._normalize(image)
        
        return image.astype(np.float32)
    
    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image maintaining aspect ratio"""
        h, w = image.shape[:2]
        aspect = w / h
        
        if aspect > 1:
            new_w = self.image_size
            new_h = int(self.image_size / aspect)
        else:
            new_h = self.image_size
            new_w = int(self.image_size * aspect)
        
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        top = (self.image_size - new_h) // 2
        bottom = self.image_size - new_h - top
        left = (self.image_size - new_w) // 2
        right = self.image_size - new_w - left
        
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        return image
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to enhance contrast"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return image
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize using ImageNet statistics"""
        image = image / 255.0
        image = (image - self.mean) / self.std
        return image

class DataLoader:
    """Load and prepare data for training"""
    
    def __init__(self, config):
        self.config = config
        self.preprocessor = ImagePreprocessor(
            image_size=config.data.image_size,
            normalize=config.data.normalize
        )
    
    def load_dataset(self, data_dir: str) -> Tuple[List, List]:
        """Load dataset from directory structure"""
        data_dir = Path(data_dir)
        images = []
        labels = []
        
        grade_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        
        for grade, grade_dir in enumerate(grade_dirs):
            image_files = list(grade_dir.glob("*.jpg")) + list(grade_dir.glob("*.png"))
            logger.info(f"Found {len(image_files)} images in grade {grade}")
            
            for img_path in image_files:
                images.append(str(img_path))
                labels.append(grade)
        
        return images, labels