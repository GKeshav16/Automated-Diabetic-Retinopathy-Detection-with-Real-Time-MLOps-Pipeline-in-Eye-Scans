import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class RetinopathyDataset(Dataset):
    """PyTorch dataset for retinopathy images"""
    
    def __init__(self, image_paths: list, labels: list, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from src.data.preprocessor import ImagePreprocessor
        
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        preprocessor = ImagePreprocessor()
        image = preprocessor.preprocess(image_path)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, torch.tensor(label, dtype=torch.long)

class ModelTrainer:
    """Train and validate retinopathy detection model"""
    
    def __init__(self, model: nn.Module, config, device: str = "cuda"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.model.learning_rate,
            weight_decay=config.model.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        Path(config.mlops.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }
    
    def validate(self, val_loader: DataLoader) -> Dict:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Full training loop"""
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        for epoch in range(self.config.model.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.model.epochs}")
            
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            
            val_metrics = self.validate(val_loader)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            if val_metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['accuracy']
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics)
                logger.info(f"✓ Best model saved with accuracy: {val_metrics['accuracy']:.4f}")
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config.model.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            self.scheduler.step()
        
        return history
    
    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__
        }
        
        path = Path(self.config.mlops.checkpoint_dir) / f"best_model.pt"
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")