#!/usr/bin/env python3
"""
Training script for Diabetic Retinopathy Detection Model
"""

import argparse
import logging
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import DataLoader as DataLoaderClass
from src.data.augmentor import AugmentationPipeline
from src.models.architecture import get_model, FocalLoss
from src.models.trainer import RetinopathyDataset, ModelTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration class"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __dict__(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.__dict__()
            else:
                result[key] = value
        return result


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)


def setup_directories(config: Config):
    """Create necessary directories"""
    Path(config.mlops.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.mlops.logs_dir).mkdir(parents=True, exist_ok=True)
    Path(config.data.data_dir).mkdir(parents=True, exist_ok=True)
    logger.info("✓ Directories created")


def create_sample_dataset(config: Config):
    """Create sample dataset structure for testing"""
    data_dir = Path(config.data.data_dir)
    
    # Create grade directories
    for grade in range(config.model.num_classes):
        grade_dir = data_dir / f"grade_{grade}"
        grade_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy images for testing
        import cv2
        import numpy as np
        
        for i in range(5):  # 5 images per grade
            img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            img_path = grade_dir / f"image_{i}.jpg"
            if not img_path.exists():
                cv2.imwrite(str(img_path), img)
    
    logger.info("✓ Sample dataset created")


def load_dataset(config: Config):
    """Load dataset"""
    data_dir = Path(config.data.data_dir)
    
    # Check if data exists
    image_files = list(data_dir.rglob("*.jpg")) + list(data_dir.rglob("*.png"))
    
    if len(image_files) == 0:
        logger.warning("No images found. Creating sample dataset...")
        create_sample_dataset(config)
        image_files = list(data_dir.rglob("*.jpg")) + list(data_dir.rglob("*.png"))
    
    logger.info(f"Found {len(image_files)} images")
    
    # Load images and labels
    images = []
    labels = []
    
    for grade in range(config.model.num_classes):
        grade_dir = data_dir / f"grade_{grade}"
        if grade_dir.exists():
            grade_images = list(grade_dir.glob("*.jpg")) + list(grade_dir.glob("*.png"))
            for img_path in grade_images:
                images.append(str(img_path))
                labels.append(grade)
    
    logger.info(f"Loaded {len(images)} images with {config.model.num_classes} classes")
    
    return images, labels


def main(args):
    """Main training function"""
    logger.info("=" * 60)
    logger.info("Diabetic Retinopathy Detection - Training Pipeline")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Configuration loaded from {args.config}")
    
    # Setup directories
    setup_directories(config)
    
    # Set random seed
    torch.manual_seed(config.training.seed)
    logger.info(f"Random seed set to {config.training.seed}")
    
    # Determine device
    device = config.training.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load dataset
    images, labels = load_dataset(config)
    
    # Create train/val/test splits
    total_samples = len(images)
    train_size = int(total_samples * config.data.train_split)
    val_size = int(total_samples * config.data.val_split)
    test_size = total_samples - train_size - val_size
    
    indices = list(range(total_samples))
    torch.manual_seed(config.training.seed)
    train_indices = torch.randperm(total_samples)
    
    train_images = [images[i] for i in train_indices[:train_size]]
    train_labels = [labels[i] for i in train_indices[:train_size]]
    
    val_images = [images[i] for i in train_indices[train_size:train_size + val_size]]
    val_labels = [labels[i] for i in train_indices[train_size:train_size + val_size]]
    
    logger.info(f"Train samples: {len(train_images)}")
    logger.info(f"Validation samples: {len(val_images)}")
    logger.info(f"Test samples: {test_size}")
    
    # Create datasets
    train_transforms = AugmentationPipeline.get_train_transforms(config.data.image_size)
    val_transforms = AugmentationPipeline.get_val_transforms(config.data.image_size)
    
    train_dataset = RetinopathyDataset(train_images, train_labels, transform=train_transforms)
    val_dataset = RetinopathyDataset(val_images, val_labels, transform=val_transforms)
    
    logger.info("✓ Datasets created")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    logger.info("✓ DataLoaders created")
    
    # Build model
    model = get_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = ModelTrainer(model, config, device=device)
    logger.info("✓ Model trainer initialized")
    
    # Training
    logger.info("\n" + "=" * 60)
    logger.info("Starting Training...")
    logger.info("=" * 60)
    
    try:
        history = trainer.train(train_loader, val_loader)
        
        logger.info("\n" + "=" * 60)
        logger.info("Training Completed Successfully!")
        logger.info("=" * 60)
        logger.info(f"Best validation accuracy: {trainer.best_val_accuracy:.4f}")
        logger.info(f"Model saved to: {config.mlops.checkpoint_dir}/best_model.pt")
        
        # Plot training history
        if args.plot:
            plot_training_history(history, config)
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise


def plot_training_history(history, config):
    """Plot training history"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(history['train_acc'], label='Train Accuracy')
        axes[1].plot(history['val_acc'], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plot_path = Path(config.mlops.logs_dir) / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path)
        logger.info(f"Training history plot saved to {plot_path}")
        
    except ImportError:
        logger.warning("Matplotlib not available for plotting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diabetic Retinopathy Detection Model")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--plot", action="store_true", help="Plot training history")
    
    args = parser.parse_args()
    
    main(args)
