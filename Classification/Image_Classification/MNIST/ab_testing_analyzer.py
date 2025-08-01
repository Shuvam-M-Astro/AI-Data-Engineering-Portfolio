#!/usr/bin/env python3
"""
A/B Testing Results Analyzer and Visualizer

This module analyzes and visualizes the results from MNIST A/B testing experiments.
It provides statistical analysis, visualization, and insights into the performance
of different configurations.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import argparse
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ABTestAnalyzer:
    """Analyzer for A/B testing results."""
    
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.df = None
        self.load_results()
        
    def load_results(self):
        """Load results from JSON file."""
        with open(self.results_file, 'r') as f:
            results = json.load(f)
        self.df = pd.DataFrame(results)
        print(f"Loaded {len(self.df)} results from {len(self.df['config_name'].unique())} configurations")
    
    def basic_statistics(self) -> pd.DataFrame:
        """Calculate basic statistics for each configuration."""
        stats_df = self.df.groupby('config_name').agg({
            'best_accuracy': ['mean', 'std', 'min', 'max', 'count'],
            'final_epoch': 'mean',
            'early_stopped': 'sum'
        }).round(4)
        
        # Flatten column names
        stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns]
        stats_df = stats_df.sort_values('best_accuracy_mean', ascending=False)
        
        return stats_df
    
    def factor_analysis(self) -> Dict[str, pd.DataFrame]:
        """Analyze the impact of individual factors."""
        factors = ['model', 'optimizer', 'scheduler', 'augmentation', 'training', 'regularization']
        factor_analysis = {}
        
        for factor in factors:
            if factor in self.df.columns:
                analysis = self.df.groupby(factor)['best_accuracy'].agg([
                    'mean', 'std', 'count', 'min', 'max'
                ]).round(4)
                analysis = analysis.sort_values('mean', ascending=False)
                factor_analysis[factor] = analysis
        
        return factor_analysis
    
    def statistical_significance_test(self, config1: str, config2: str) -> Dict[str, float]:
        """Perform statistical significance test between two configurations."""
        acc1 = self.df[self.df['config_name'] == config1]['best_accuracy'].values
        acc2 = self.df[self.df['config_name'] == config2]['best_accuracy'].values
        
        if len(acc1) == 0 or len(acc2) == 0:
            return {"error": "Configuration not found"}
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(acc1, acc2)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(acc1) - 1) * np.var(acc1, ddof=1) + 
                             (len(acc2) - 1) * np.var(acc2, ddof=1)) / (len(acc1) + len(acc2) - 2))
        cohens_d = (np.mean(acc1) - np.mean(acc2)) / pooled_std
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "significant": p_value < 0.05,
            "mean_diff": np.mean(acc1) - np.mean(acc2)
        }
    
    def create_accuracy_boxplot(self, save_path: str = None):
        """Create boxplot of accuracy distributions."""
        plt.figure(figsize=(15, 8))
        
        # Get top 10 configurations
        top_configs = self.df.groupby('config_name')['best_accuracy'].mean().nlargest(10).index
        
        data_to_plot = []
        labels = []
        
        for config in top_configs:
            data = self.df[self.df['config_name'] == config]['best_accuracy'].values
            data_to_plot.append(data)
            labels.append(config)
        
        plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Best Accuracy (%)')
        plt.title('Accuracy Distribution by Configuration (Top 10)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_factor_impact_plot(self, save_path: str = None):
        """Create plot showing impact of different factors."""
        factors = ['model', 'optimizer', 'scheduler', 'augmentation', 'training', 'regularization']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, factor in enumerate(factors):
            if factor in self.df.columns:
                factor_data = self.df.groupby(factor)['best_accuracy'].agg(['mean', 'std']).sort_values('mean', ascending=False)
                
                bars = axes[i].bar(range(len(factor_data)), factor_data['mean'], 
                                 yerr=factor_data['std'], capsize=5, alpha=0.7)
                axes[i].set_title(f'{factor.title()} Impact')
                axes[i].set_ylabel('Mean Accuracy (%)')
                axes[i].set_xticks(range(len(factor_data)))
                axes[i].set_xticklabels(factor_data.index, rotation=45, ha='right')
                axes[i].grid(True, alpha=0.3)
                
                # Color bars based on performance
                colors = plt.cm.RdYlGn(np.linspace(0, 1, len(bars)))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_learning_curves(self, config_name: str, save_path: str = None):
        """Create learning curves for a specific configuration."""
        config_data = self.df[self.df['config_name'] == config_name]
        
        if len(config_data) == 0:
            print(f"Configuration '{config_name}' not found")
            return
        
        plt.figure(figsize=(12, 5))
        
        # Training accuracy
        plt.subplot(1, 2, 1)
        for _, row in config_data.iterrows():
            train_acc = row['train_accuracies']
            plt.plot(train_acc, alpha=0.7, label=f"Run {row['run_id'] + 1}")
        
        plt.title(f'Training Accuracy - {config_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Validation accuracy
        plt.subplot(1, 2, 2)
        for _, row in config_data.iterrows():
            val_acc = row['val_accuracies']
            plt.plot(val_acc, alpha=0.7, label=f"Run {row['run_id'] + 1}")
        
        plt.title(f'Validation Accuracy - {config_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_correlation_matrix(self, save_path: str = None):
        """Create correlation matrix of numerical features."""
        # Select numerical columns
        numerical_cols = ['best_accuracy', 'final_epoch', 'batch_size', 'learning_rate', 
                         'weight_decay', 'dropout_rate']
        numerical_df = self.df[numerical_cols].copy()
        
        # Add one-hot encoded categorical variables
        categorical_cols = ['model', 'optimizer', 'scheduler', 'augmentation', 'training', 'regularization']
        for col in categorical_cols:
            if col in self.df.columns:
                dummies = pd.get_dummies(self.df[col], prefix=col)
                numerical_df = pd.concat([numerical_df, dummies], axis=1)
        
        # Calculate correlation matrix
        corr_matrix = numerical_df.corr()
        
        plt.figure(figsize=(20, 16))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Features')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_performance_heatmap(self, save_path: str = None):
        """Create heatmap showing performance across different factor combinations."""
        # Create pivot table for model vs optimizer
        pivot_data = self.df.pivot_table(
            values='best_accuracy', 
            index='model', 
            columns='optimizer', 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', center=pivot_data.mean().mean())
        plt.title('Model vs Optimizer Performance Heatmap')
        plt.ylabel('Model Architecture')
        plt.xlabel('Optimizer')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_confidence_intervals(self, save_path: str = None):
        """Create confidence intervals for top configurations."""
        # Get top 10 configurations
        top_configs = self.df.groupby('config_name')['best_accuracy'].mean().nlargest(10).index
        
        plt.figure(figsize=(12, 8))
        
        for i, config in enumerate(top_configs):
            data = self.df[self.df['config_name'] == config]['best_accuracy'].values
            mean_acc = np.mean(data)
            std_acc = np.std(data)
            ci_95 = 1.96 * std_acc / np.sqrt(len(data))  # 95% confidence interval
            
            plt.errorbar(i, mean_acc, yerr=ci_95, fmt='o', capsize=5, capthick=2, 
                        markersize=8, label=config)
        
        plt.xlabel('Configuration Rank')
        plt.ylabel('Accuracy (%)')
        plt.title('Top 10 Configurations with 95% Confidence Intervals')
        plt.xticks(range(len(top_configs)), [f"#{i+1}" for i in range(len(top_configs))])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self, output_dir: str = "analysis_results"):
        """Generate comprehensive analysis report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("Generating comprehensive analysis report...")
        
        # Basic statistics
        stats_df = self.basic_statistics()
        stats_df.to_csv(output_path / "basic_statistics.csv")
        print("✓ Basic statistics saved")
        
        # Factor analysis
        factor_analysis = self.factor_analysis()
        for factor, analysis in factor_analysis.items():
            analysis.to_csv(output_path / f"{factor}_analysis.csv")
        print("✓ Factor analysis saved")
        
        # Statistical significance tests
        top_configs = stats_df.head(5).index
        significance_results = {}
        
        for i, config1 in enumerate(top_configs):
            for config2 in top_configs[i+1:]:
                test_result = self.statistical_significance_test(config1, config2)
                significance_results[f"{config1}_vs_{config2}"] = test_result
        
        with open(output_path / "significance_tests.json", 'w') as f:
            json.dump(significance_results, f, indent=4)
        print("✓ Statistical significance tests saved")
        
        # Create visualizations
        self.create_accuracy_boxplot(output_path / "accuracy_boxplot.png")
        self.create_factor_impact_plot(output_path / "factor_impact.png")
        self.create_correlation_matrix(output_path / "correlation_matrix.png")
        self.create_performance_heatmap(output_path / "performance_heatmap.png")
        self.create_confidence_intervals(output_path / "confidence_intervals.png")
        
        # Learning curves for top configuration
        if len(top_configs) > 0:
            self.create_learning_curves(top_configs[0], output_path / "top_config_learning_curves.png")
        
        print("✓ All visualizations saved")
        
        # Generate text report
        report = self.generate_text_report()
        with open(output_path / "analysis_report.txt", 'w') as f:
            f.write(report)
        print("✓ Text report saved")
        
        print(f"\nAnalysis complete! Results saved to: {output_path}")
    
    def generate_text_report(self) -> str:
        """Generate comprehensive text report."""
        stats_df = self.basic_statistics()
        factor_analysis = self.factor_analysis()
        
        report = f"""
A/B Testing Analysis Report
===========================

Dataset Overview:
- Total configurations tested: {len(self.df['config_name'].unique())}
- Total runs: {len(self.df)}
- Average runs per configuration: {len(self.df) / len(self.df['config_name'].unique()):.1f}

Top 5 Performing Configurations:
"""
        
        for i, (config_name, row) in enumerate(stats_df.head().iterrows(), 1):
            report += f"""
{i}. {config_name}
   Mean Accuracy: {row['best_accuracy_mean']:.4f} ± {row['best_accuracy_std']:.4f}
   Range: {row['best_accuracy_min']:.4f} - {row['best_accuracy_max']:.4f}
   Runs: {row['best_accuracy_count']}
   Avg Epochs: {row['final_epoch_mean']:.1f}
"""
        
        report += "\nFactor Analysis:\n"
        
        for factor, analysis in factor_analysis.items():
            report += f"\n{factor.upper()}:\n"
            for level, row in analysis.iterrows():
                report += f"  {level}: {row['mean']:.4f} ± {row['std']:.4f} (n={row['count']})\n"
        
        # Statistical significance
        top_configs = stats_df.head(5).index
        report += "\nStatistical Significance Tests (Top 5 vs Top 5):\n"
        
        for i, config1 in enumerate(top_configs):
            for config2 in top_configs[i+1:]:
                test_result = self.statistical_significance_test(config1, config2)
                if 'error' not in test_result:
                    significance = "SIGNIFICANT" if test_result['significant'] else "NOT SIGNIFICANT"
                    report += f"  {config1} vs {config2}: p={test_result['p_value']:.4f} ({significance})\n"
        
        report += "\nKey Insights:\n"
        
        # Best performing factors
        for factor, analysis in factor_analysis.items():
            best_level = analysis.index[0]
            best_mean = analysis.iloc[0]['mean']
            report += f"- Best {factor}: {best_level} ({best_mean:.4f}%)\n"
        
        # Configuration recommendations
        report += "\nRecommendations:\n"
        report += f"- Best overall configuration: {top_configs[0]}\n"
        report += f"- Most consistent configuration: {stats_df.loc[stats_df['best_accuracy_std'].idxmin(), 'config_name']}\n"
        
        return report

def main():
    """Main function to run analysis."""
    parser = argparse.ArgumentParser(description="A/B Testing Results Analyzer")
    parser.add_argument("--results", type=str, required=True, help="Path to results JSON file")
    parser.add_argument("--output-dir", type=str, default="analysis_results", help="Output directory for analysis")
    parser.add_argument("--config", type=str, help="Specific configuration to analyze")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ABTestAnalyzer(args.results)
    
    if args.config:
        # Analyze specific configuration
        print(f"Analyzing configuration: {args.config}")
        analyzer.create_learning_curves(args.config)
    else:
        # Generate comprehensive analysis
        analyzer.generate_comprehensive_report(args.output_dir)

if __name__ == "__main__":
    main() 