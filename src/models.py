"""
models.py - Neural Network Models for Cell Classification

This module contains the CNN architectures for binary tumor cell classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union


class CellClassifierCNN(nn.Module):
    """
    Convolutional Neural Network for binary cell classification.
    
    This model takes multichannel cell images and classifies them as:
    - 0: Non-tumor cells
    - 1: Tumor cells
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 num_classes: int = 2,
                 dropout_rate: float = 0.3):
        """
        Initialize the CNN model.
        
        Args:
            input_channels: Number of input channels (protein markers)
            num_classes: Number of output classes (2 for binary classification)
            dropout_rate: Dropout rate for regularization
        """
        super(CellClassifierCNN, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fourth convolutional block
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using proper initialization schemes."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Convolutional feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Global pooling and classification
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities using softmax.
        
        Args:
            x: Input tensor
            
        Returns:
            Class probabilities
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class labels.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class labels
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        
        return out


class CellClassifierResNet(nn.Module):
    """
    ResNet-based architecture for cell classification.
    More robust for deeper networks and better gradient flow.
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 num_classes: int = 2,
                 dropout_rate: float = 0.3):
        super(CellClassifierResNet, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                   num_blocks: int, stride: int) -> nn.Sequential:
        """Create a layer with multiple residual blocks."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x


class CellClassifierResNetFeatureExtractor(nn.Module):
    """
    Feature extractor version of CellClassifierResNet.
    Allows extraction of features from different layers and provides flexible output options.
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 num_classes: int = 2,
                 dropout_rate: float = 0.3,
                 feature_layer: str = 'avgpool',
                 feature_dim: Optional[int] = None,
                 return_all_features: bool = False):
        """
        Initialize the feature extractor.
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes (used if loading pretrained weights)
            dropout_rate: Dropout rate for classifier (if needed for weight loading)
            feature_layer: Layer to extract features from. Options:
                - 'conv1': After initial convolution (64 channels)
                - 'layer1': After first residual layer (64 channels)
                - 'layer2': After second residual layer (128 channels)
                - 'layer3': After third residual layer (256 channels)
                - 'layer4': After fourth residual layer (512 channels)
                - 'avgpool': After global average pooling (512 features)
                - 'classifier_input': Before final classification layers (512 features)
            feature_dim: If specified, add a projection layer to this dimension
            return_all_features: If True, return features from multiple layers
        """
        super(CellClassifierResNetFeatureExtractor, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.feature_layer = feature_layer
        self.feature_dim = feature_dim
        self.return_all_features = return_all_features
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Keep classifier for potential weight loading compatibility
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Feature dimensions for each layer
        self.feature_dims = {
            'conv1': 64,
            'layer1': 64,
            'layer2': 128,
            'layer3': 256,
            'layer4': 512,
            'avgpool': 512,
            'classifier_input': 512
        }
        
        # Optional projection layer
        if feature_dim is not None:
            output_dim = self.feature_dims.get(feature_layer, 512)
            self.projection = nn.Linear(output_dim, feature_dim)
        else:
            self.projection = nn.Identity()
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                   num_blocks: int, stride: int) -> nn.Sequential:
        """Create a layer with multiple residual blocks."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through ResNet feature extractor.
        
        Returns:
            If return_all_features=False: Feature tensor from specified layer
            If return_all_features=True: Dictionary with features from all layers
        """
        features = {}
        
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        features['conv1'] = x
        if self.feature_layer == 'conv1' and not self.return_all_features:
            return self._process_output(x, 'conv1')
        
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        features['layer1'] = x
        if self.feature_layer == 'layer1' and not self.return_all_features:
            return self._process_output(x, 'layer1')
        
        x = self.layer2(x)
        features['layer2'] = x
        if self.feature_layer == 'layer2' and not self.return_all_features:
            return self._process_output(x, 'layer2')
        
        x = self.layer3(x)
        features['layer3'] = x
        if self.feature_layer == 'layer3' and not self.return_all_features:
            return self._process_output(x, 'layer3')
        
        x = self.layer4(x)
        features['layer4'] = x
        if self.feature_layer == 'layer4' and not self.return_all_features:
            return self._process_output(x, 'layer4')
        
        # Global average pooling
        x = self.avgpool(x)
        x_pooled = torch.flatten(x, 1)
        features['avgpool'] = x_pooled
        features['classifier_input'] = x_pooled
        
        if self.return_all_features:
            # Process all features and return dictionary
            processed_features = {}
            for layer_name, feat in features.items():
                processed_features[layer_name] = self._process_output(feat, layer_name)
            return processed_features
        else:
            # Return features from specified layer
            if self.feature_layer in ['avgpool', 'classifier_input']:
                return self._process_output(x_pooled, self.feature_layer)
            else:
                # This shouldn't happen if the above conditions are correct
                return self._process_output(x_pooled, 'avgpool')
    
    def _process_output(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Process output features (flatten if needed, apply projection)."""
        # Flatten spatial features if they're not already flattened
        if len(x.shape) > 2:
            if layer_name in ['avgpool', 'classifier_input']:
                x = torch.flatten(x, 1)
            else:
                # For spatial features, apply global average pooling
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = torch.flatten(x, 1)
        
        # Apply projection if specified
        if hasattr(self, 'projection') and self.projection is not None:
            x = self.projection(x)
        
        return x
    

# Utility function to create feature extractor from existing model
def convert_to_feature_extractor(model: nn.Module, 
                                feature_layer: str = 'avgpool',
                                feature_dim: Optional[int] = None) -> CellClassifierResNetFeatureExtractor:
    """
    Convert a trained CellClassifierResNet model to a feature extractor.
    
    Args:
        model: Trained CellClassifierResNet model
        feature_layer: Layer to extract features from
        feature_dim: Optional projection dimension
    
    Returns:
        Feature extractor with loaded weights
    """
    # Get model parameters
    input_channels = model.input_channels if hasattr(model, 'input_channels') else 3
    num_classes = model.num_classes if hasattr(model, 'num_classes') else 2
    
    # Create feature extractor
    feature_extractor = CellClassifierResNetFeatureExtractor(
        input_channels=input_channels,
        num_classes=num_classes,
        feature_layer=feature_layer,
        feature_dim=feature_dim
    )
    
    # Copy weights (excluding projection layer if it's new)
    model_dict = model.state_dict()
    extractor_dict = feature_extractor.state_dict()
    
    # Filter out keys that don't match (like projection layer)
    filtered_dict = {k: v for k, v in model_dict.items() if k in extractor_dict and v.shape == extractor_dict[k].shape}
    
    extractor_dict.update(filtered_dict)
    feature_extractor.load_state_dict(extractor_dict)
    
    return feature_extractor


def create_model(model_type: str = 'cnn', **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('cnn' or 'resnet')
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized model
    """
    if model_type.lower() == 'cnn':
        return CellClassifierCNN(**kwargs)
    elif model_type.lower() == 'resnet':
        return CellClassifierResNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'architecture': model.__class__.__name__
    }