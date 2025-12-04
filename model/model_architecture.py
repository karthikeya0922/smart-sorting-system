"""
Model Architecture Definition
Defines the WasteClassifier using transfer learning with MobileNetV2.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class WasteClassifier(nn.Module):
    """
    Waste classification model based on MobileNetV2.
    Uses transfer learning with pretrained ImageNet weights.
    """
    
    def __init__(self, num_classes=6, pretrained=True, dropout=0.2):
        """
        Initialize the model.
        
        Args:
            num_classes (int): Number of output classes (default: 6)
            pretrained (bool): Use ImageNet pretrained weights (default: True)
            dropout (float): Dropout rate (default: 0.2)
        """
        super(WasteClassifier, self).__init__()
        
        # Load pretrained MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # Replace the classifier head
        in_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)


class WasteClassifierResNet(nn.Module):
    """
    Alternative: Waste classification model based on ResNet18.
    """
    
    def __init__(self, num_classes=6, pretrained=True, dropout=0.2):
        super(WasteClassifierResNet, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def create_model(architecture='mobilenet_v2', num_classes=6, pretrained=True, dropout=0.2):
    """
    Factory function to create a model.
    
    Args:
        architecture (str): Model architecture ('mobilenet_v2' or 'resnet18')
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained weights
        dropout (float): Dropout rate
    
    Returns:
        nn.Module: Initialized model
    """
    if architecture == 'mobilenet_v2':
        return WasteClassifier(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    elif architecture == 'resnet18':
        return WasteClassifierResNet(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model):
    """
    Print a summary of the model architecture.
    
    Args:
        model (nn.Module): PyTorch model
    """
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Model architecture: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    print()
    print("Model structure:")
    print(model)
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    print("Creating MobileNetV2 model...")
    model = create_model('mobilenet_v2', num_classes=6, pretrained=True)
    print_model_summary(model)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected shape: (1, 6)")
