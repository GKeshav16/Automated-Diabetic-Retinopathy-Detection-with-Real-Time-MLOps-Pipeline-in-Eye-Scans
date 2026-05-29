import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class AugmentationPipeline:
    """Data augmentation for retinopathy images"""
    
    @staticmethod
    def get_train_transforms(image_size: int = 512):
        """Training augmentation pipeline"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.RandomBrightnessContrast(p=0.2),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3))
    
    @staticmethod
    def get_val_transforms(image_size: int = 512):
        """Validation/test augmentation pipeline"""
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])