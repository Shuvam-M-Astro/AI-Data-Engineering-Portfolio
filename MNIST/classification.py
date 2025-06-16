import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from torch.quantization import quantize_dynamic
from torch.ao.pruning import BasePruner, L1Unstructured
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Dict, Optional, List, Union
import time
import numpy as np
import pandas as pd
import argparse
import os
import logging
from datetime import datetime
from pathlib import Path
import json
from dataclasses import dataclass
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    mixed_precision: bool = True
    num_workers: int = 2
    pin_memory: bool = True

class MNISTDataset(torch.utils.data.Dataset):
    """Custom dataset class for MNIST with advanced preprocessing."""
    def __init__(self, root: str, train: bool = True, transform: Optional[transforms.Compose] = None):
        self.dataset = torchvision.datasets.MNIST(root=root, train=train, download=True, transform=transform)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

class ResidualBlock(nn.Module):
    """Residual block for improved gradient flow."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = torch.relu(out)
        return out

class AdvancedNet(nn.Module):
    """Advanced CNN architecture with residual connections and attention."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Apply attention
        attention = self.attention(x)
        x = x * attention
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ModelTrainer:
    """Advanced model trainer with comprehensive training features."""
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
        writer: Optional[SummaryWriter] = None
    ):
        self.model = model
        self.config = config
        self.device = device
        self.writer = writer
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=1,
            pct_start=0.3
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.best_acc = 0
        self.patience_counter = 0

    def train_epoch(self, trainloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Train for one epoch with progress bar and metrics tracking."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(trainloader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.config.mixed_precision:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return running_loss / len(trainloader), 100. * correct / total

    def validate(self, testloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Validate the model with detailed metrics."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(testloader, desc='Validation'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                if self.config.mixed_precision:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100. * correct / total
        avg_loss = running_loss / len(testloader)
        
        # Log metrics
        if self.writer:
            self.writer.add_scalar('Validation/Loss', avg_loss)
            self.writer.add_scalar('Validation/Accuracy', accuracy)
        
        return avg_loss, accuracy

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint with comprehensive metadata."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'config': self.config.__dict__,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pth')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, 'checkpoints/best_model.pth')
            
            # Save model metadata
            metadata = {
                'architecture': self.model.__class__.__name__,
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'best_accuracy': self.best_acc,
                'training_config': self.config.__dict__,
                'timestamp': datetime.now().isoformat()
            }
            
            with open('checkpoints/model_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description='MNIST Classification with Advanced Features')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference', 'benchmark'],
                      help='Mode to run the script in')
    parser.add_argument('--quantize', action='store_true', help='Enable quantization')
    parser.add_argument('--prune', action='store_true', help='Enable pruning')
    parser.add_argument('--prune_amount', type=float, default=0.3, help='Amount of pruning (0-1)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training/inference')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    return parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(42)

# Decide whether to use multiple cores or a single core
use_multiple_cores = False  # Set to False to use a single core

# Device configuration with proper CUDA support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
else:
    num_threads = 1 if not use_multiple_cores else torch.get_num_threads()
    torch.set_num_threads(num_threads)
    print(f'Using CPU with {num_threads} threads')

# Enhanced transform pipeline with data augmentation
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load datasets with improved transforms and pin memory for faster GPU transfer
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, 
                                        pin_memory=True, num_workers=2 if device.type == 'cuda' else 0)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False,
                                       pin_memory=True, num_workers=2 if device.type == 'cuda' else 0)

def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        torch.set_num_threads(1)
        print('Using CPU with single thread')
    return device

def get_data_loaders(batch_size, device):
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                            pin_memory=True, num_workers=2 if device.type == 'cuda' else 0)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                           pin_memory=True, num_workers=2 if device.type == 'cuda' else 0)
    
    return trainloader, testloader

def quantize_model(model: nn.Module) -> nn.Module:
    """Quantize the model to reduce its size and improve inference speed."""
    logger.info("Quantizing model...")
    try:
        # Prepare model for quantization
        model.eval()
        
        # Quantize the model
        quantized_model = quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},  # Specify layers to quantize
            dtype=torch.qint8
        )
        
        logger.info("Model quantization completed successfully")
        return quantized_model
    except Exception as e:
        logger.error(f"Error during model quantization: {str(e)}")
        return model

def prune_model(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """Prune the model to reduce its size while maintaining performance."""
    logger.info(f"Pruning model with amount: {amount}")
    try:
        # Create pruner
        pruner = L1Unstructured(
            amount=amount,
            module_type=nn.Conv2d
        )
        
        # Prepare model for pruning
        model.eval()
        
        # Apply pruning
        pruner.prepare(model)
        pruner.step()
        pruner.squash_mask()
        
        logger.info("Model pruning completed successfully")
        return model
    except Exception as e:
        logger.error(f"Error during model pruning: {str(e)}")
        return model

def inference(model: nn.Module, image: torch.Tensor, device: torch.device) -> Tuple[int, torch.Tensor]:
    """Perform inference on a single image."""
    model.eval()
    with torch.no_grad():
        # Move image to device and add batch dimension
        image = image.unsqueeze(0).to(device)
        
        # Get prediction
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted = outputs.argmax(dim=1)
        
        return predicted.item(), probabilities[0]

def batch_inference(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[List[int], List[int], float]:
    """Perform inference on a batch of images."""
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Inference'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    return all_preds, all_labels, accuracy

def main():
    """Main training loop with comprehensive error handling and logging."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Setup device
        device = setup_device()
        
        # Create checkpoint directory
        Path('checkpoints').mkdir(exist_ok=True)
        
        # Initialize tensorboard
        writer = SummaryWriter('runs/mnist_experiment')
        
        # Load data
        trainloader, testloader = get_data_loaders(args.batch_size, device)
        
        # Initialize model
        model = AdvancedNet().to(device)
        
        # Initialize trainer
        config = TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs
        )
        trainer = ModelTrainer(model, config, device, writer)
        
        # Training loop
        for epoch in range(args.epochs):
            logger.info(f'\nEpoch {epoch+1}/{args.epochs}')
        if args.mode == 'train':
            # Training loop
            for epoch in range(args.epochs):
                logger.info(f'\nEpoch {epoch+1}/{args.epochs}')
                
                # Train
                train_loss, train_acc = trainer.train_epoch(trainloader)
                logger.info(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
                
                # Validate
                val_loss, val_acc = trainer.validate(testloader)
                logger.info(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
                
                # Update learning rate
                trainer.scheduler.step()
                
                # Save checkpoint
                is_best = val_acc > trainer.best_acc
                if is_best:
                    trainer.best_acc = val_acc
                    trainer.patience_counter = 0
                else:
                    trainer.patience_counter += 1
                
                trainer.save_checkpoint(epoch, is_best)
                
                # Early stopping
                if trainer.patience_counter >= config.early_stopping_patience:
                    logger.info('Early stopping triggered!')
                    break
            
            # Train
            train_loss, train_acc = trainer.train_epoch(trainloader)
            logger.info(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            # Apply quantization if requested
            if args.quantize:
                logger.info("Applying model quantization...")
                model = quantize_model(model)
                trainer.model = model
                trainer.save_checkpoint(args.epochs - 1, is_best=True)
            
            # Apply pruning if requested
            if args.prune:
                logger.info("Applying model pruning...")
                model = prune_model(model, args.prune_amount)
                trainer.model = model
                trainer.save_checkpoint(args.epochs - 1, is_best=True)
            
            # Save final model
            trainer.save_checkpoint(args.epochs - 1)
            
        elif args.mode == 'inference':
            # Load the best model
            checkpoint = torch.load('checkpoints/best_model.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Apply quantization if requested
            if args.quantize:
                model = quantize_model(model)
            
            # Early stopping
            if trainer.patience_counter >= config.early_stopping_patience:
                logger.info('Early stopping triggered!')
                break
        
        # Save final model
        trainer.save_checkpoint(args.epochs - 1)
            # Apply pruning if requested
            if args.prune:
                model = prune_model(model, args.prune_amount)
            
            logger.info("Model loaded and ready for inference")
            
            # Perform batch inference on test set
            predictions, labels, accuracy = batch_inference(model, testloader, device)
            logger.info(f'Test Accuracy: {accuracy:.2f}%')
            
            # Log detailed metrics
            if writer:
                writer.add_scalar('Inference/Accuracy', accuracy)
                writer.add_pr_curve('Inference/PR_Curve', torch.tensor(predictions), torch.tensor(labels))
            
            # Save inference results
            results = {
                'predictions': predictions,
                'labels': labels,
                'accuracy': accuracy,
                'timestamp': datetime.now().isoformat()
            }
            
            with open('inference_results.json', 'w') as f:
                json.dump(results, f, indent=4)
            
            logger.info("Inference results saved to inference_results.json")
        
        # Close tensorboard writer
        writer.close()
        
        logger.info('Process completed successfully!')
        
    except Exception as e:
        logger.error(f'An error occurred: {str(e)}', exc_info=True)
        raise

if __name__ == '__main__':
    main()

