#!/usr/bin/env python3
"""
A/B Testing Configuration for MNIST Classification

This module defines the configuration for comprehensive A/B testing with multiple factors:
- Model Architectures
- Optimizers
- Learning Rate Schedules
- Data Augmentation Strategies
- Training Configurations
- Regularization Techniques
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, CosineAnnealingLR, StepLR
import torchvision.transforms as transforms
import json
from pathlib import Path

@dataclass
class ABTestConfig:
    """Configuration for A/B testing parameters."""
    experiment_name: str = "mnist_ab_test"
    num_runs_per_config: int = 3
    random_seed: int = 42
    output_dir: str = "ab_test_results"
    save_models: bool = True
    early_stopping: bool = True
    max_epochs: int = 30
    patience: int = 5
    use_gpu: bool = True
    mixed_precision: bool = True

@dataclass
class ModelArchitecture:
    """Model architecture configuration."""
    name: str
    model_class: str
    parameters: Dict[str, Any]
    description: str

@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    name: str
    optimizer_class: str
    parameters: Dict[str, Any]
    description: str

@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    name: str
    scheduler_class: str
    parameters: Dict[str, Any]
    description: str

@dataclass
class DataAugmentationConfig:
    """Data augmentation configuration."""
    name: str
    transforms: List[Dict[str, Any]]
    description: str

@dataclass
class TrainingConfig:
    """Training configuration."""
    name: str
    batch_size: int
    learning_rate: float
    weight_decay: float
    dropout_rate: float
    description: str

# Define Model Architectures
MODEL_ARCHITECTURES = [
    ModelArchitecture(
        name="ResNet18",
        model_class="ResNet18",
        parameters={"num_classes": 10},
        description="ResNet18 architecture with residual connections"
    ),
    ModelArchitecture(
        name="AdvancedNet",
        model_class="AdvancedNet", 
        parameters={"num_classes": 10},
        description="Custom advanced network with attention mechanism"
    ),
    ModelArchitecture(
        name="SimpleCNN",
        model_class="SimpleCNN",
        parameters={"num_classes": 10},
        description="Simple CNN without residual connections"
    ),
    ModelArchitecture(
        name="WideResNet",
        model_class="WideResNet",
        parameters={"num_classes": 10, "width_factor": 2},
        description="Wide ResNet with increased channel width"
    ),
    ModelArchitecture(
        name="EfficientNet",
        model_class="EfficientNet",
        parameters={"num_classes": 10, "compound_coef": 0},
        description="EfficientNet-B0 architecture"
    )
]

# Define Optimizers
OPTIMIZERS = [
    OptimizerConfig(
        name="Adam",
        optimizer_class="Adam",
        parameters={"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8},
        description="Adam optimizer with default parameters"
    ),
    OptimizerConfig(
        name="AdamW",
        optimizer_class="AdamW",
        parameters={"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01},
        description="AdamW optimizer with weight decay"
    ),
    OptimizerConfig(
        name="SGD",
        optimizer_class="SGD",
        parameters={"lr": 0.01, "momentum": 0.9, "nesterov": True},
        description="SGD with Nesterov momentum"
    ),
    OptimizerConfig(
        name="RMSprop",
        optimizer_class="RMSprop",
        parameters={"lr": 0.001, "alpha": 0.99, "eps": 1e-8},
        description="RMSprop optimizer"
    ),
    OptimizerConfig(
        name="AdaBelief",
        optimizer_class="AdaBelief",
        parameters={"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8},
        description="AdaBelief optimizer (if available)"
    )
]

# Define Learning Rate Schedulers
SCHEDULERS = [
    SchedulerConfig(
        name="OneCycleLR",
        scheduler_class="OneCycleLR",
        parameters={"max_lr": 0.01, "epochs": 30, "steps_per_epoch": 1, "pct_start": 0.3},
        description="OneCycleLR scheduler for super convergence"
    ),
    SchedulerConfig(
        name="CosineAnnealingLR",
        scheduler_class="CosineAnnealingLR",
        parameters={"T_max": 30, "eta_min": 1e-6},
        description="Cosine annealing scheduler"
    ),
    SchedulerConfig(
        name="ReduceLROnPlateau",
        scheduler_class="ReduceLROnPlateau",
        parameters={"mode": "max", "factor": 0.5, "patience": 3, "verbose": True},
        description="Reduce LR on plateau"
    ),
    SchedulerConfig(
        name="StepLR",
        scheduler_class="StepLR",
        parameters={"step_size": 10, "gamma": 0.5},
        description="Step LR scheduler"
    ),
    SchedulerConfig(
        name="ExponentialLR",
        scheduler_class="ExponentialLR",
        parameters={"gamma": 0.95},
        description="Exponential LR decay"
    )
]

# Define Data Augmentation Strategies
DATA_AUGMENTATIONS = [
    DataAugmentationConfig(
        name="Basic",
        transforms=[
            {"name": "ToTensor", "parameters": {}},
            {"name": "Normalize", "parameters": {"mean": [0.1307], "std": [0.3081]}}
        ],
        description="Basic normalization only"
    ),
    DataAugmentationConfig(
        name="Standard",
        transforms=[
            {"name": "RandomRotation", "parameters": {"degrees": 10}},
            {"name": "RandomAffine", "parameters": {"degrees": 0, "translate": (0.1, 0.1)}},
            {"name": "ToTensor", "parameters": {}},
            {"name": "Normalize", "parameters": {"mean": [0.1307], "std": [0.3081]}}
        ],
        description="Standard augmentation with rotation and translation"
    ),
    DataAugmentationConfig(
        name="Aggressive",
        transforms=[
            {"name": "RandomRotation", "parameters": {"degrees": 15}},
            {"name": "RandomAffine", "parameters": {"degrees": 0, "translate": (0.15, 0.15), "scale": (0.9, 1.1)}},
            {"name": "RandomErasing", "parameters": {"p": 0.2, "scale": (0.02, 0.33)}},
            {"name": "ToTensor", "parameters": {}},
            {"name": "Normalize", "parameters": {"mean": [0.1307], "std": [0.3081]}}
        ],
        description="Aggressive augmentation with erasing"
    ),
    DataAugmentationConfig(
        name="Minimal",
        transforms=[
            {"name": "RandomHorizontalFlip", "parameters": {"p": 0.5}},
            {"name": "ToTensor", "parameters": {}},
            {"name": "Normalize", "parameters": {"mean": [0.1307], "std": [0.3081]}}
        ],
        description="Minimal augmentation with horizontal flip"
    ),
    DataAugmentationConfig(
        name="Advanced",
        transforms=[
            {"name": "RandomRotation", "parameters": {"degrees": 10}},
            {"name": "RandomAffine", "parameters": {"degrees": 0, "translate": (0.1, 0.1), "shear": 10}},
            {"name": "ColorJitter", "parameters": {"brightness": 0.2, "contrast": 0.2}},
            {"name": "ToTensor", "parameters": {}},
            {"name": "Normalize", "parameters": {"mean": [0.1307], "std": [0.3081]}}
        ],
        description="Advanced augmentation with color jittering"
    )
]

# Define Training Configurations
TRAINING_CONFIGS = [
    TrainingConfig(
        name="Standard",
        batch_size=128,
        learning_rate=0.001,
        weight_decay=1e-4,
        dropout_rate=0.5,
        description="Standard training configuration"
    ),
    TrainingConfig(
        name="LargeBatch",
        batch_size=256,
        learning_rate=0.002,
        weight_decay=1e-4,
        dropout_rate=0.5,
        description="Large batch size with scaled learning rate"
    ),
    TrainingConfig(
        name="SmallBatch",
        batch_size=64,
        learning_rate=0.0005,
        weight_decay=1e-4,
        dropout_rate=0.3,
        description="Small batch size with reduced learning rate"
    ),
    TrainingConfig(
        name="HighLR",
        batch_size=128,
        learning_rate=0.01,
        weight_decay=1e-3,
        dropout_rate=0.5,
        description="High learning rate with increased weight decay"
    ),
    TrainingConfig(
        name="Conservative",
        batch_size=128,
        learning_rate=0.0001,
        weight_decay=1e-5,
        dropout_rate=0.2,
        description="Conservative training with low learning rate"
    )
]

# Define Regularization Techniques
REGULARIZATION_CONFIGS = [
    {
        "name": "Dropout",
        "parameters": {"dropout_rate": 0.5},
        "description": "Standard dropout regularization"
    },
    {
        "name": "BatchNorm",
        "parameters": {"use_batch_norm": True},
        "description": "Batch normalization"
    },
    {
        "name": "WeightDecay",
        "parameters": {"weight_decay": 1e-4},
        "description": "L2 weight decay"
    },
    {
        "name": "LabelSmoothing",
        "parameters": {"label_smoothing": 0.1},
        "description": "Label smoothing for better generalization"
    },
    {
        "name": "Mixup",
        "parameters": {"mixup_alpha": 0.2},
        "description": "Mixup data augmentation"
    }
]

# Define A/B Test Combinations
AB_TEST_COMBINATIONS = [
    # Baseline configuration
    {
        "name": "Baseline",
        "model": "AdvancedNet",
        "optimizer": "Adam",
        "scheduler": "OneCycleLR",
        "augmentation": "Standard",
        "training": "Standard",
        "regularization": "Dropout"
    },
    # Model architecture variations
    {
        "name": "ResNet18_Test",
        "model": "ResNet18",
        "optimizer": "Adam",
        "scheduler": "OneCycleLR",
        "augmentation": "Standard",
        "training": "Standard",
        "regularization": "Dropout"
    },
    {
        "name": "SimpleCNN_Test",
        "model": "SimpleCNN",
        "optimizer": "Adam",
        "scheduler": "OneCycleLR",
        "augmentation": "Standard",
        "training": "Standard",
        "regularization": "Dropout"
    },
    # Optimizer variations
    {
        "name": "AdamW_Test",
        "model": "AdvancedNet",
        "optimizer": "AdamW",
        "scheduler": "OneCycleLR",
        "augmentation": "Standard",
        "training": "Standard",
        "regularization": "Dropout"
    },
    {
        "name": "SGD_Test",
        "model": "AdvancedNet",
        "optimizer": "SGD",
        "scheduler": "OneCycleLR",
        "augmentation": "Standard",
        "training": "Standard",
        "regularization": "Dropout"
    },
    # Scheduler variations
    {
        "name": "CosineAnnealing_Test",
        "model": "AdvancedNet",
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "augmentation": "Standard",
        "training": "Standard",
        "regularization": "Dropout"
    },
    {
        "name": "ReduceLROnPlateau_Test",
        "model": "AdvancedNet",
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "augmentation": "Standard",
        "training": "Standard",
        "regularization": "Dropout"
    },
    # Augmentation variations
    {
        "name": "AggressiveAug_Test",
        "model": "AdvancedNet",
        "optimizer": "Adam",
        "scheduler": "OneCycleLR",
        "augmentation": "Aggressive",
        "training": "Standard",
        "regularization": "Dropout"
    },
    {
        "name": "MinimalAug_Test",
        "model": "AdvancedNet",
        "optimizer": "Adam",
        "scheduler": "OneCycleLR",
        "augmentation": "Minimal",
        "training": "Standard",
        "regularization": "Dropout"
    },
    # Training configuration variations
    {
        "name": "LargeBatch_Test",
        "model": "AdvancedNet",
        "optimizer": "Adam",
        "scheduler": "OneCycleLR",
        "augmentation": "Standard",
        "training": "LargeBatch",
        "regularization": "Dropout"
    },
    {
        "name": "Conservative_Test",
        "model": "AdvancedNet",
        "optimizer": "Adam",
        "scheduler": "OneCycleLR",
        "augmentation": "Standard",
        "training": "Conservative",
        "regularization": "Dropout"
    },
    # Regularization variations
    {
        "name": "BatchNorm_Test",
        "model": "AdvancedNet",
        "optimizer": "Adam",
        "scheduler": "OneCycleLR",
        "augmentation": "Standard",
        "training": "Standard",
        "regularization": "BatchNorm"
    },
    {
        "name": "LabelSmoothing_Test",
        "model": "AdvancedNet",
        "optimizer": "Adam",
        "scheduler": "OneCycleLR",
        "augmentation": "Standard",
        "training": "Standard",
        "regularization": "LabelSmoothing"
    },
    # Combined variations
    {
        "name": "BestCombination_Test",
        "model": "ResNet18",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "augmentation": "Aggressive",
        "training": "LargeBatch",
        "regularization": "BatchNorm"
    },
    {
        "name": "ConservativeCombination_Test",
        "model": "SimpleCNN",
        "optimizer": "SGD",
        "scheduler": "ReduceLROnPlateau",
        "augmentation": "Minimal",
        "training": "Conservative",
        "regularization": "Dropout"
    }
]

def get_config_by_name(config_list, name):
    """Get configuration by name from a list of configurations."""
    for config in config_list:
        if config.name == name:
            return config
    raise ValueError(f"Configuration '{name}' not found")

def create_transform_from_config(aug_config):
    """Create transform from augmentation configuration."""
    transform_list = []
    
    for transform_dict in aug_config.transforms:
        transform_name = transform_dict["name"]
        parameters = transform_dict["parameters"]
        
        if transform_name == "ToTensor":
            transform_list.append(transforms.ToTensor())
        elif transform_name == "Normalize":
            transform_list.append(transforms.Normalize(**parameters))
        elif transform_name == "RandomRotation":
            transform_list.append(transforms.RandomRotation(**parameters))
        elif transform_name == "RandomAffine":
            transform_list.append(transforms.RandomAffine(**parameters))
        elif transform_name == "RandomHorizontalFlip":
            transform_list.append(transforms.RandomHorizontalFlip(**parameters))
        elif transform_name == "ColorJitter":
            transform_list.append(transforms.ColorJitter(**parameters))
        elif transform_name == "RandomErasing":
            # RandomErasing is applied after ToTensor
            continue
        else:
            raise ValueError(f"Unknown transform: {transform_name}")
    
    return transforms.Compose(transform_list)

def save_config_to_json(config, filename):
    """Save configuration to JSON file."""
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4, default=str)

def load_config_from_json(filename):
    """Load configuration from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f) 