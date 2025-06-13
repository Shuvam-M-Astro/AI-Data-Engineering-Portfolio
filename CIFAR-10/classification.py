import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.quantization import quantize_dynamic
from torch.ao.pruning import BasePruner, L1Unstructured
import time
import numpy as np
import pandas as pd
import argparse
import os
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 Classification with Advanced Features')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference', 'benchmark'],
                      help='Mode to run the script in')
    parser.add_argument('--quantize', action='store_true', help='Enable quantization')
    parser.add_argument('--prune', action='store_true', help='Enable pruning')
    parser.add_argument('--prune_amount', type=float, default=0.3, help='Amount of pruning (0-1)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training/inference')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    return parser.parse_args()

class ImprovedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
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

class ModelBenchmark:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.results = []
        
    def log_metric(self, metric_name, value, metadata=None):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        entry = {
            'timestamp': timestamp,
            'metric': metric_name,
            'value': value,
            'device': self.device.type,
            'model_size': self.get_model_size(),
            'metadata': metadata or {}
        }
        self.results.append(entry)
        
    def get_model_size(self):
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1024  # Size in KB
        
    def save_results(self, filename='cifar10_benchmark_results.csv'):
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

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
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                            pin_memory=True, num_workers=2 if device.type == 'cuda' else 0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                           pin_memory=True, num_workers=2 if device.type == 'cuda' else 0)
    
    return trainloader, testloader

def train_model(model, trainloader, testloader, device, args, benchmark):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    scaler = GradScaler()
    
    best_acc = 0
    patience = 5
    patience_counter = 0
    
    print(f"\nStarting training on {device}...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(trainloader)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        val_loss, val_acc = validate(model, testloader, criterion, device)
        epoch_time = time.time() - epoch_start
        
        scheduler.step(val_loss)
        
        print(f'Epoch: {epoch+1}/{args.epochs} | Time: {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        benchmark.log_metric('epoch_time', epoch_time, {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc)
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("\nEarly stopping triggered!")
            break
    
    total_time = time.time() - start_time
    print(f'\nTraining completed in {total_time:.2f} seconds')
    benchmark.log_metric('total_training_time', total_time)
    
    return model

def validate(model, loader, criterion, device):
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

def save_checkpoint(model, optimizer, scheduler, epoch, best_acc):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
    }, 'cifar10_best.pth')

def load_checkpoint(model, device):
    checkpoint = torch.load('cifar10_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def quantize_model(model):
    print("\nQuantizing model...")
    quantized_model = quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    return quantized_model

def prune_model(model, amount):
    print(f"\nPruning model by {amount*100}%...")
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            parameters_to_prune.append((module, 'weight'))
    
    pruner = L1Unstructured(amount=amount)
    pruner.apply(model, parameters_to_prune)
    return model

def benchmark_inference(model, testloader, device, benchmark):
    print("\nRunning inference benchmark...")
    model.eval()
    
    # Warmup
    for _ in range(10):
        inputs, _ = next(iter(testloader))
        inputs = inputs.to(device)
        with torch.no_grad():
            _ = model(inputs)
    
    # Benchmark
    total_time = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, _ in testloader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            start_time = time.time()
            _ = model(inputs)
            batch_time = time.time() - start_time
            
            total_time += batch_time
            total_samples += batch_size
            
            # Log per-batch metrics
            benchmark.log_metric('inference_time_per_batch', batch_time, {
                'batch_size': batch_size,
                'samples_per_second': batch_size / batch_time
            })
    
    avg_time_per_sample = total_time / total_samples
    samples_per_second = total_samples / total_time
    
    print(f"Average inference time per sample: {avg_time_per_sample*1000:.2f}ms")
    print(f"Samples per second: {samples_per_second:.2f}")
    
    benchmark.log_metric('inference_throughput', samples_per_second)
    benchmark.log_metric('avg_inference_time', avg_time_per_sample)

def main():
    args = parse_args()
    device = setup_device()
    
    # Create model and benchmark instance
    model = ImprovedNet().to(device)
    benchmark = ModelBenchmark(model, device)
    
    if args.mode == 'train':
        trainloader, testloader = get_data_loaders(args.batch_size, device)
        model = train_model(model, trainloader, testloader, device, args, benchmark)
        
        if args.quantize:
            model = quantize_model(model)
            benchmark.log_metric('model_size_after_quantization', benchmark.get_model_size())
        
        if args.prune:
            model = prune_model(model, args.prune_amount)
            benchmark.log_metric('model_size_after_pruning', benchmark.get_model_size())
    
    elif args.mode == 'inference':
        model = load_checkpoint(model, device)
        _, testloader = get_data_loaders(args.batch_size, device)
        
        if args.quantize:
            model = quantize_model(model)
        if args.prune:
            model = prune_model(model, args.prune_amount)
        
        benchmark_inference(model, testloader, device, benchmark)
    
    elif args.mode == 'benchmark':
        model = load_checkpoint(model, device)
        _, testloader = get_data_loaders(args.batch_size, device)
        
        # Benchmark original model
        print("\nBenchmarking original model...")
        benchmark_inference(model, testloader, device, benchmark)
        
        # Benchmark quantized model
        quantized_model = quantize_model(model)
        print("\nBenchmarking quantized model...")
        benchmark_inference(quantized_model, testloader, device, benchmark)
        
        # Benchmark pruned model
        pruned_model = prune_model(model, args.prune_amount)
        print("\nBenchmarking pruned model...")
        benchmark_inference(pruned_model, testloader, device, benchmark)
    
    # Save benchmark results
    benchmark.save_results()

if __name__ == '__main__':
    main()
