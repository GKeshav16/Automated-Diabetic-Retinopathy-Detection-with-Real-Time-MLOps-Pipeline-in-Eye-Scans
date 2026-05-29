import torch
import torch.nn as nn
import timm
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class RetinopathyDetectionModel(nn.Module):
    """EfficientNet-based model for diabetic retinopathy detection"""
    
    def __init__(
        self,
        model_name: str = "efficientnet_b4",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone parameters frozen")
        
        in_features = self.backbone.num_features
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.backbone(x)
        x = self.head(x)
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head"""
        return self.backbone(x)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        ce_loss = self.ce(inputs, targets)
        p = torch.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = self.alpha * ((1 - p_t) ** self.gamma) * ce_loss
        return loss.mean()

def get_model(config) -> nn.Module:
    """Factory function to get model"""
    model = RetinopathyDetectionModel(
        model_name=config.model.model_name,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout
    )
    logger.info(f"Created model: {config.model.model_name}")
    return model