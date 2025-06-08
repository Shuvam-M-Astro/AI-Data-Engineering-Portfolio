import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import time
import numpy as np

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

# Improved CNN architecture with batch normalization and dropout
class ImprovedNet(nn.Module):
    def __init__(self):
        super(ImprovedNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

net = ImprovedNet()
net.to(device)

# Initialize gradient scaler for mixed precision training
scaler = GradScaler()

# Loss function and optimizer with improved settings
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# Training with early stopping and mixed precision
def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # Scale loss and perform backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

# Training loop with early stopping
best_acc = 0
patience = 5
patience_counter = 0
epochs = 20

print(f"\nStarting training on {device}...")
start_time = time.time()

for epoch in range(epochs):
    epoch_start = time.time()
    train_loss, train_acc = train_epoch(net, trainloader, criterion, optimizer, scaler)
    val_loss, val_acc = validate(net, testloader, criterion)
    epoch_time = time.time() - epoch_start
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    print(f'Epoch: {epoch+1}/{epochs} | Time: {epoch_time:.2f}s')
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }, 'mnist_best.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= patience:
        print("\nEarly stopping triggered!")
        break

total_time = time.time() - start_time
print(f'\nTraining completed in {total_time:.2f} seconds')

# Load best model and evaluate
checkpoint = torch.load('mnist_best.pth')
net.load_state_dict(checkpoint['model_state_dict'])
_, final_acc = validate(net, testloader, criterion)
print(f'Best model accuracy on test set: {final_acc:.2f}%')

# Print hardware utilization summary
if device.type == 'cuda':
    print(f'\nGPU Memory Usage:')
    print(f'Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB')
    print(f'Cached: {torch.cuda.memory_reserved(0) / 1024**2:.1f}MB')

