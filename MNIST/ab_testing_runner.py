#!/usr/bin/env python3
"""
A/B Testing Runner for MNIST Classification

This module runs comprehensive A/B testing with multiple factors and collects results
for statistical analysis and comparison.
"""

import os
import sys
import time
import json
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, CosineAnnealingLR, StepLR, ExponentialLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import asdict
import argparse
from tqdm import tqdm
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Import local modules
from ab_testing_config import (
    ABTestConfig, MODEL_ARCHITECTURES, OPTIMIZERS, SCHEDULERS, 
    DATA_AUGMENTATIONS, TRAINING_CONFIGS, REGULARIZATION_CONFIGS,
    AB_TEST_COMBINATIONS, get_config_by_name, create_transform_from_config
)
from model_architectures import create_model
from classification import AdvancedNet, ModelTrainer, TrainingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ab_testing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ABTestRunner:
    """Main A/B testing runner class."""
    
    def __init__(self, config: ABTestConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu')
        self.results = []
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(config.random_seed)
            torch.cuda.manual_seed_all(config.random_seed)
        
        logger.info(f"Initialized AB Test Runner on device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def create_optimizer(self, optimizer_config: Dict, model: nn.Module, lr: float) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_name = optimizer_config["optimizer_class"]
        params = optimizer_config["parameters"].copy()
        params["lr"] = lr
        
        if optimizer_name == "Adam":
            return optim.Adam(model.parameters(), **params)
        elif optimizer_name == "AdamW":
            return optim.AdamW(model.parameters(), **params)
        elif optimizer_name == "SGD":
            return optim.SGD(model.parameters(), **params)
        elif optimizer_name == "RMSprop":
            return optim.RMSprop(model.parameters(), **params)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def create_scheduler(self, scheduler_config: Dict, optimizer: optim.Optimizer, 
                        train_loader: DataLoader) -> Any:
        """Create learning rate scheduler based on configuration."""
        scheduler_name = scheduler_config["scheduler_class"]
        params = scheduler_config["parameters"].copy()
        
        if scheduler_name == "OneCycleLR":
            params["steps_per_epoch"] = len(train_loader)
            return OneCycleLR(optimizer, **params)
        elif scheduler_name == "CosineAnnealingLR":
            return CosineAnnealingLR(optimizer, **params)
        elif scheduler_name == "ReduceLROnPlateau":
            return ReduceLROnPlateau(optimizer, **params)
        elif scheduler_name == "StepLR":
            return StepLR(optimizer, **params)
        elif scheduler_name == "ExponentialLR":
            return ExponentialLR(optimizer, **params)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def create_criterion(self, regularization_config: Dict) -> nn.Module:
        """Create loss function with regularization."""
        reg_name = regularization_config["name"]
        
        if reg_name == "LabelSmoothing":
            smoothing = regularization_config["parameters"]["label_smoothing"]
            return nn.CrossEntropyLoss(label_smoothing=smoothing)
        else:
            return nn.CrossEntropyLoss()
    
    def get_data_loaders(self, train_transform, test_transform, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders with specified transforms."""
        trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=train_transform
        )
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            pin_memory=True, num_workers=2 if self.device.type == 'cuda' else 0
        )
        
        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=test_transform
        )
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            pin_memory=True, num_workers=2 if self.device.type == 'cuda' else 0
        )
        
        return trainloader, testloader
    
    def train_model(self, model: nn.Module, trainloader: DataLoader, testloader: DataLoader,
                   optimizer: optim.Optimizer, scheduler: Any, criterion: nn.Module,
                   config_name: str, run_id: int) -> Dict[str, Any]:
        """Train a single model and return results."""
        model = model.to(self.device)
        scaler = GradScaler() if self.config.mixed_precision else None
        
        best_acc = 0.0
        patience_counter = 0
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        logger.info(f"Training {config_name} - Run {run_id + 1}")
        
        for epoch in range(self.config.max_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in tqdm(trainloader, desc=f'Epoch {epoch+1}/{self.config.max_epochs}'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                if self.config.mixed_precision:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_loss = running_loss / len(trainloader)
            train_acc = 100. * correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            # Validation phase
            model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    if self.config.mixed_precision:
                        with autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            val_loss = running_loss / len(testloader)
            val_acc = 100. * correct / total
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # Update scheduler
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
            
            # Early stopping
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                if self.config.save_models:
                    self.save_model(model, config_name, run_id, epoch, best_acc)
            else:
                patience_counter += 1
            
            if self.config.early_stopping and patience_counter >= self.config.patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            logger.info(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                       f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        return {
            "config_name": config_name,
            "run_id": run_id,
            "best_accuracy": best_acc,
            "final_epoch": epoch + 1,
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "early_stopped": patience_counter >= self.config.patience
        }
    
    def save_model(self, model: nn.Module, config_name: str, run_id: int, 
                  epoch: int, accuracy: float):
        """Save model checkpoint."""
        model_dir = self.output_dir / "models" / config_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'accuracy': accuracy,
            'config_name': config_name,
            'run_id': run_id
        }
        
        torch.save(checkpoint, model_dir / f"run_{run_id}_best.pth")
    
    def run_single_config(self, config: Dict) -> List[Dict[str, Any]]:
        """Run a single configuration multiple times."""
        config_name = config["name"]
        logger.info(f"Running configuration: {config_name}")
        
        results = []
        
        for run_id in range(self.config.num_runs_per_config):
            logger.info(f"Starting run {run_id + 1}/{self.config.num_runs_per_config}")
            
            # Get configurations
            model_config = get_config_by_name(MODEL_ARCHITECTURES, config["model"])
            optimizer_config = get_config_by_name(OPTIMIZERS, config["optimizer"])
            scheduler_config = get_config_by_name(SCHEDULERS, config["scheduler"])
            aug_config = get_config_by_name(DATA_AUGMENTATIONS, config["augmentation"])
            train_config = get_config_by_name(TRAINING_CONFIGS, config["training"])
            reg_config = next(r for r in REGULARIZATION_CONFIGS if r["name"] == config["regularization"])
            
            # Create transforms
            train_transform = create_transform_from_config(aug_config)
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            # Create data loaders
            trainloader, testloader = self.get_data_loaders(
                train_transform, test_transform, train_config.batch_size
            )
            
            # Create model
            if model_config.model_class == "AdvancedNet":
                model = AdvancedNet(**model_config.parameters)
            else:
                model = create_model(model_config.model_class, **model_config.parameters)
            
            # Create optimizer
            optimizer = self.create_optimizer(optimizer_config, model, train_config.learning_rate)
            
            # Create scheduler
            scheduler = self.create_scheduler(scheduler_config, optimizer, trainloader)
            
            # Create criterion
            criterion = self.create_criterion(reg_config)
            
            # Train model
            result = self.train_model(
                model, trainloader, testloader, optimizer, scheduler, criterion,
                config_name, run_id
            )
            
            # Add configuration details to result
            result.update({
                "model": config["model"],
                "optimizer": config["optimizer"],
                "scheduler": config["scheduler"],
                "augmentation": config["augmentation"],
                "training": config["training"],
                "regularization": config["regularization"],
                "batch_size": train_config.batch_size,
                "learning_rate": train_config.learning_rate,
                "weight_decay": train_config.weight_decay,
                "dropout_rate": train_config.dropout_rate
            })
            
            results.append(result)
            
            # Clear GPU memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return results
    
    def run_all_configs(self) -> List[Dict[str, Any]]:
        """Run all configurations and collect results."""
        all_results = []
        
        for config in AB_TEST_COMBINATIONS:
            try:
                results = self.run_single_config(config)
                all_results.extend(results)
                
                # Save intermediate results
                self.save_results(all_results, "intermediate_results.json")
                
            except Exception as e:
                logger.error(f"Error running configuration {config['name']}: {e}")
                continue
        
        return all_results
    
    def save_results(self, results: List[Dict[str, Any]], filename: str):
        """Save results to JSON file."""
        output_file = self.output_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for result in results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, torch.Tensor):
                    serializable_result[key] = value.cpu().numpy().tolist()
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        logger.info(f"Results saved to {output_file}")
    
    def generate_summary_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate a summary report of all results."""
        df = pd.DataFrame(results)
        
        # Calculate statistical significance
        stats_analysis = self.calculate_statistical_significance(results)
        
        # Group by configuration and calculate statistics
        summary = df.groupby('config_name').agg({
            'best_accuracy': ['mean', 'std', 'min', 'max'],
            'final_epoch': 'mean',
            'early_stopped': 'sum'
        }).round(4)
        
        # Sort by mean accuracy
        summary = summary.sort_values(('best_accuracy', 'mean'), ascending=False)
        
        report = f"""
A/B Testing Summary Report
==========================

Total Configurations: {len(df['config_name'].unique())}
Total Runs: {len(df)}
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

Top 5 Configurations by Mean Accuracy (with 95% Confidence Intervals):
"""
        
        for i, (config_name, row) in enumerate(summary.head().iterrows(), 1):
            mean_acc = row[('best_accuracy', 'mean')]
            std_acc = row[('best_accuracy', 'std')]
            min_acc = row[('best_accuracy', 'min')]
            max_acc = row[('best_accuracy', 'max')]
            avg_epochs = row[('final_epoch', 'mean')]
            early_stops = row[('early_stopped', 'sum')]
            
            # Get confidence interval
            ci_data = stats_analysis['confidence_intervals'][config_name]
            ci_lower = ci_data['ci_lower']
            ci_upper = ci_data['ci_upper']
            power = ci_data['statistical_power']
            sufficient_power = ci_data['sufficient_power']
            
            report += f"""
{i}. {config_name}
   Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}
   Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]
   Range: {min_acc:.4f} - {max_acc:.4f}
   Avg Epochs: {avg_epochs:.1f}
   Early Stops: {early_stops}/{self.config.num_runs_per_config}
   Statistical Power: {power:.3f} {'✓' if sufficient_power else '✗'}
"""
        
        # Factor analysis
        report += "\nFactor Analysis:\n"
        
        # Model analysis
        model_analysis = df.groupby('model')['best_accuracy'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        report += "\nModel Architectures:\n"
        for model, (mean_acc, std_acc) in model_analysis.iterrows():
            report += f"  {model}: {mean_acc:.4f} ± {std_acc:.4f}\n"
        
        # Optimizer analysis
        optimizer_analysis = df.groupby('optimizer')['best_accuracy'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        report += "\nOptimizers:\n"
        for optimizer, (mean_acc, std_acc) in optimizer_analysis.iterrows():
            report += f"  {optimizer}: {mean_acc:.4f} ± {std_acc:.4f}\n"
        
        # Scheduler analysis
        scheduler_analysis = df.groupby('scheduler')['best_accuracy'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        report += "\nSchedulers:\n"
        for scheduler, (mean_acc, std_acc) in scheduler_analysis.iterrows():
            report += f"  {scheduler}: {mean_acc:.4f} ± {std_acc:.4f}\n"
        
        # Statistical significance summary
        significant_pairs = [k for k, v in stats_analysis['pairwise_tests'].items() if v['significant']]
        report += f"\nStatistical Significance Summary:\n"
        report += f"Significant differences found: {len(significant_pairs)}/{len(stats_analysis['pairwise_tests'])} pairwise comparisons\n"
        
        if significant_pairs:
            report += "Significant pairs (p < 0.05):\n"
            for pair in significant_pairs[:5]:  # Show top 5
                test_data = stats_analysis['pairwise_tests'][pair]
                report += f"  {pair}: p={test_data['p_value']:.4f}, effect_size={test_data['effect_size']:.3f}\n"
        
        return report

    def calculate_statistical_significance(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistical significance and confidence intervals for all configurations."""
        df = pd.DataFrame(results)
        configs = df['config_name'].unique()
        
        significance_results = {}
        
        # Calculate confidence intervals for each configuration
        for config in configs:
            config_data = df[df['config_name'] == config]['best_accuracy'].values
            mean_acc = np.mean(config_data)
            std_acc = np.std(config_data, ddof=1)
            n_samples = len(config_data)
            
            # Calculate 95% confidence interval
            t_value = stats.t.ppf(0.975, n_samples - 1)
            margin_of_error = t_value * (std_acc / np.sqrt(n_samples))
            ci_lower = mean_acc - margin_of_error
            ci_upper = mean_acc + margin_of_error
            
            # Calculate statistical power (assuming 80% power threshold)
            effect_size = 0.5  # Medium effect size
            alpha = 0.05
            power = stats.power.ttest_power(effect_size, n_samples, alpha)
            
            significance_results[config] = {
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'n_samples': n_samples,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'margin_of_error': margin_of_error,
                'statistical_power': power,
                'sufficient_power': power >= 0.8
            }
        
        # Perform pairwise significance tests
        pairwise_tests = {}
        for i, config1 in enumerate(configs):
            for config2 in configs[i+1:]:
                acc1 = df[df['config_name'] == config1]['best_accuracy'].values
                acc2 = df[df['config_name'] == config2]['best_accuracy'].values
                
                t_stat, p_value = stats.ttest_ind(acc1, acc2)
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(((len(acc1) - 1) * np.var(acc1, ddof=1) + 
                                     (len(acc2) - 1) * np.var(acc2, ddof=1)) / (len(acc1) + len(acc2) - 2))
                cohens_d = (np.mean(acc1) - np.mean(acc2)) / pooled_std
                
                pairwise_tests[f"{config1}_vs_{config2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': cohens_d,
                    'mean_difference': np.mean(acc1) - np.mean(acc2)
                }
        
        return {
            'confidence_intervals': significance_results,
            'pairwise_tests': pairwise_tests
        }

def main():
    """Main function to run A/B testing."""
    parser = argparse.ArgumentParser(description="MNIST A/B Testing Runner")
    parser.add_argument("--config", type=str, help="Path to A/B testing configuration file")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs per configuration")
    parser.add_argument("--max-epochs", type=int, default=30, help="Maximum epochs per run")
    parser.add_argument("--output-dir", type=str, default="ab_test_results", help="Output directory")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision training")
    
    args = parser.parse_args()
    
    # Create configuration
    config = ABTestConfig(
        num_runs_per_config=args.num_runs,
        max_epochs=args.max_epochs,
        output_dir=args.output_dir,
        use_gpu=args.gpu,
        mixed_precision=args.mixed_precision
    )
    
    # Create runner
    runner = ABTestRunner(config)
    
    try:
        # Run all configurations
        logger.info("Starting A/B testing...")
        results = runner.run_all_configs()
        
        # Save final results
        runner.save_results(results, "final_results.json")
        
        # Save statistical analysis
        stats_analysis = runner.calculate_statistical_significance(results)
        stats_file = runner.output_dir / "statistical_analysis.json"
        with open(stats_file, 'w') as f:
            json.dump(stats_analysis, f, indent=4, default=str)
        logger.info(f"Statistical analysis saved to {stats_file}")
        
        # Generate and save summary report
        report = runner.generate_summary_report(results)
        report_file = runner.output_dir / "summary_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(report)
        logger.info(f"Summary report saved to {report_file}")
        
        # Create results DataFrame and save as CSV
        df = pd.DataFrame(results)
        csv_file = runner.output_dir / "results.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Results saved to CSV: {csv_file}")
        
    except KeyboardInterrupt:
        logger.info("A/B testing interrupted by user")
    except Exception as e:
        logger.error(f"A/B testing failed: {e}")
        raise

if __name__ == "__main__":
    main() 