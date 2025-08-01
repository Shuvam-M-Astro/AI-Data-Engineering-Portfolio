#!/usr/bin/env python3
"""
Additional Model Architectures for MNIST A/B Testing

This module contains various model architectures to test in the A/B testing framework:
- ResNet18
- SimpleCNN
- WideResNet
- EfficientNet
- Custom architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SimpleCNN(nn.Module):
    """Simple CNN architecture without residual connections."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class ResNet18(nn.Module):
    """ResNet18 architecture adapted for MNIST."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(self._make_block(in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(self._make_block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _make_block(self, in_channels, out_channels, stride):
        return ResBlock(in_channels, out_channels, stride)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResBlock(nn.Module):
    """Residual block for ResNet."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

class WideResNet(nn.Module):
    """Wide ResNet architecture with configurable width factor."""
    
    def __init__(self, num_classes: int = 10, width_factor: int = 2):
        super().__init__()
        self.width_factor = width_factor
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 16 * width_factor, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16 * width_factor)
        
        # Wide ResNet blocks
        self.layer1 = self._make_layer(16 * width_factor, 32 * width_factor, 2, stride=2)
        self.layer2 = self._make_layer(32 * width_factor, 64 * width_factor, 2, stride=2)
        
        # Classifier
        self.bn2 = nn.BatchNorm2d(64 * width_factor)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * width_factor, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(WideResBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(WideResBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.relu(self.bn2(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class WideResBlock(nn.Module):
    """Wide ResNet block."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

class EfficientNet(nn.Module):
    """Simplified EfficientNet architecture for MNIST."""
    
    def __init__(self, num_classes: int = 10, compound_coef: int = 0):
        super().__init__()
        # Compound coefficient scaling
        self.depth_coef = 1.2 ** compound_coef
        self.width_coef = 1.1 ** compound_coef
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, int(32 * self.width_coef), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(32 * self.width_coef))
        
        # MBConv blocks
        self.mbconv1 = self._make_mbconv(int(32 * self.width_coef), int(16 * self.width_coef), 1, 1)
        self.mbconv2 = self._make_mbconv(int(16 * self.width_coef), int(24 * self.width_coef), 6, 2)
        self.mbconv3 = self._make_mbconv(int(24 * self.width_coef), int(40 * self.width_coef), 6, 2)
        
        # Final convolution
        self.conv2 = nn.Conv2d(int(40 * self.width_coef), int(80 * self.width_coef), kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(80 * self.width_coef))
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(int(80 * self.width_coef), num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_mbconv(self, in_channels, out_channels, expansion_factor, stride):
        return MBConvBlock(in_channels, out_channels, expansion_factor, stride)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck block for EfficientNet."""
    
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        
        expanded_channels = int(in_channels * expansion_factor)
        
        # Expansion
        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(expanded_channels)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, 
                                       stride=stride, padding=1, groups=expanded_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)
        
        # SE block (simplified)
        self.se = SEBlock(expanded_channels)
        
        # Projection
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        residual = x
        
        # Expansion
        x = F.relu(self.expand_bn(self.expand_conv(x)))
        
        # Depthwise convolution
        x = F.relu(self.depthwise_bn(self.depthwise_conv(x)))
        
        # SE block
        x = self.se(x)
        
        # Projection
        x = self.project_bn(self.project_conv(x))
        
        # Residual connection
        if self.use_residual:
            x += residual
        
        return x

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AttentionNet(nn.Module):
    """CNN with attention mechanism."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Attention layers
        self.attention1 = SelfAttention(32)
        self.attention2 = SelfAttention(64)
        self.attention3 = SelfAttention(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention1(x)
        x = self.pool(F.relu(x))
        
        x = self.conv2(x)
        x = self.attention2(x)
        x = self.pool(F.relu(x))
        
        x = self.conv3(x)
        x = self.attention3(x)
        x = self.pool(F.relu(x))
        
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class SelfAttention(nn.Module):
    """Self-attention mechanism."""
    
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate Q, K, V
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)
        
        # Compute attention
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x

# Model factory function
def create_model(model_name: str, **kwargs) -> nn.Module:
    """Factory function to create models by name."""
    model_classes = {
        'ResNet18': ResNet18,
        'SimpleCNN': SimpleCNN,
        'WideResNet': WideResNet,
        'EfficientNet': EfficientNet,
        'AttentionNet': AttentionNet
    }
    
    if model_name not in model_classes:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model_classes[model_name](**kwargs) 