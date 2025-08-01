# MNIST A/B Testing Framework

A comprehensive A/B testing framework for MNIST classification with multiple factors and statistical analysis.

## Overview

This framework allows you to systematically test different combinations of:
- **Model Architectures**: ResNet18, AdvancedNet, SimpleCNN, WideResNet, EfficientNet
- **Optimizers**: Adam, AdamW, SGD, RMSprop
- **Learning Rate Schedulers**: OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau, StepLR, ExponentialLR
- **Data Augmentation**: Basic, Standard, Aggressive, Minimal, Advanced
- **Training Configurations**: Different batch sizes, learning rates, weight decay, dropout rates
- **Regularization Techniques**: Dropout, BatchNorm, WeightDecay, LabelSmoothing, Mixup

## Features

- **Multi-factor A/B Testing**: Test combinations of multiple factors simultaneously
- **Statistical Analysis**: Comprehensive statistical testing and significance analysis
- **Reproducible Results**: Fixed random seeds and detailed logging
- **Visualization**: Rich set of plots and charts for result analysis
- **Automated Reporting**: Generate comprehensive reports automatically
- **Model Persistence**: Save best models for each configuration
- **Early Stopping**: Prevent overfitting with configurable early stopping
- **Mixed Precision Training**: Optional mixed precision for faster training

## File Structure

```
MNIST/
├── ab_testing_config.py      # Configuration definitions and test combinations
├── model_architectures.py    # Additional model architectures
├── ab_testing_runner.py      # Main A/B testing runner
├── ab_testing_analyzer.py    # Results analysis and visualization
├── classification.py         # Original MNIST classification (AdvancedNet)
├── requirements_ab_testing.txt # Dependencies for A/B testing
└── README_AB_TESTING.md      # This file
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_ab_testing.txt
```

2. Ensure you have sufficient disk space for results and models.

## Usage

### 1. Running A/B Tests

#### Basic Usage
```bash
python ab_testing_runner.py --num-runs 3 --max-epochs 30 --gpu --mixed-precision
```

#### Advanced Usage
```bash
python ab_testing_runner.py \
    --num-runs 5 \
    --max-epochs 50 \
    --output-dir "my_experiment_results" \
    --gpu \
    --mixed-precision
```

#### Parameters
- `--num-runs`: Number of runs per configuration (default: 3)
- `--max-epochs`: Maximum epochs per run (default: 30)
- `--output-dir`: Output directory for results (default: "ab_test_results")
- `--gpu`: Use GPU if available
- `--mixed-precision`: Use mixed precision training

### 2. Analyzing Results

#### Generate Comprehensive Analysis
```bash
python ab_testing_analyzer.py --results ab_test_results/final_results.json --output-dir analysis_results
```

#### Analyze Specific Configuration
```bash
python ab_testing_analyzer.py --results ab_test_results/final_results.json --config "BestCombination_Test"
```

### 3. Understanding the Results

The framework generates several output files:

#### Results Files
- `final_results.json`: Complete results from all runs
- `results.csv`: Results in CSV format for easy analysis
- `summary_report.txt`: Text summary of key findings

#### Analysis Files
- `basic_statistics.csv`: Statistical summary for each configuration
- `*_analysis.csv`: Factor-specific analysis (model_analysis.csv, optimizer_analysis.csv, etc.)
- `significance_tests.json`: Statistical significance tests between configurations

#### Visualizations
- `accuracy_boxplot.png`: Box plots of accuracy distributions
- `factor_impact.png`: Impact of different factors
- `correlation_matrix.png`: Feature correlations
- `performance_heatmap.png`: Model vs optimizer performance
- `confidence_intervals.png`: Confidence intervals for top configurations
- `top_config_learning_curves.png`: Learning curves for best configuration

## Test Configurations

The framework includes 15 predefined test configurations:

### Baseline Configuration
- **Baseline**: AdvancedNet + Adam + OneCycleLR + Standard augmentation + Standard training + Dropout

### Model Architecture Tests
- **ResNet18_Test**: ResNet18 architecture
- **SimpleCNN_Test**: Simple CNN architecture

### Optimizer Tests
- **AdamW_Test**: AdamW optimizer
- **SGD_Test**: SGD with Nesterov momentum

### Scheduler Tests
- **CosineAnnealing_Test**: Cosine annealing scheduler
- **ReduceLROnPlateau_Test**: Reduce LR on plateau

### Augmentation Tests
- **AggressiveAug_Test**: Aggressive data augmentation
- **MinimalAug_Test**: Minimal data augmentation

### Training Configuration Tests
- **LargeBatch_Test**: Large batch size (256)
- **Conservative_Test**: Conservative training parameters

### Regularization Tests
- **BatchNorm_Test**: Batch normalization
- **LabelSmoothing_Test**: Label smoothing

### Combined Tests
- **BestCombination_Test**: Optimized combination of factors
- **ConservativeCombination_Test**: Conservative combination

## Customizing Tests

### Adding New Model Architectures

1. Add the model class to `model_architectures.py`
2. Update the `create_model` function
3. Add configuration to `MODEL_ARCHITECTURES` in `ab_testing_config.py`

### Adding New Test Combinations

Edit `AB_TEST_COMBINATIONS` in `ab_testing_config.py`:

```python
{
    "name": "MyCustomTest",
    "model": "ResNet18",
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
    "augmentation": "Aggressive",
    "training": "LargeBatch",
    "regularization": "BatchNorm"
}
```

### Modifying Factor Levels

Edit the configuration lists in `ab_testing_config.py`:
- `OPTIMIZERS`: Add new optimizers
- `SCHEDULERS`: Add new schedulers
- `DATA_AUGMENTATIONS`: Add new augmentation strategies
- `TRAINING_CONFIGS`: Add new training configurations
- `REGULARIZATION_CONFIGS`: Add new regularization techniques

## Statistical Analysis

The framework provides comprehensive statistical analysis:

### Basic Statistics
- Mean, standard deviation, min, max for each configuration
- Confidence intervals
- Effect sizes (Cohen's d)

### Factor Analysis
- Individual factor impact analysis
- Interaction effects
- Performance rankings

### Statistical Significance
- T-tests between configurations
- P-values and significance levels
- Multiple comparison corrections

## Performance Considerations

### Memory Usage
- Models are saved for each configuration
- Consider disk space for large experiments
- Use `--save-models false` to disable model saving

### Time Requirements
- Full experiment: ~6-12 hours (depending on hardware)
- Single configuration: ~30-60 minutes
- Use GPU for faster training

### Scaling
- Increase `--num-runs` for more statistical power
- Reduce `--max-epochs` for faster experiments
- Use early stopping to prevent overfitting

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use CPU
2. **Slow Training**: Enable GPU and mixed precision
3. **Inconsistent Results**: Check random seed settings
4. **Import Errors**: Install all dependencies from requirements file

### Debug Mode

For debugging, modify configurations in `ab_testing_config.py`:
- Reduce number of configurations
- Use smaller models
- Reduce epochs

## Example Output

### Summary Report
```
A/B Testing Summary Report
==========================

Total Configurations: 15
Total Runs: 45
Date: 2024-01-15 14:30:00

Top 5 Configurations by Mean Accuracy:

1. BestCombination_Test
   Mean Accuracy: 99.45 ± 0.12
   Range: 99.32 - 99.58
   Avg Epochs: 28.3
   Early Stops: 0/3

2. ResNet18_Test
   Mean Accuracy: 99.38 ± 0.15
   Range: 99.23 - 99.53
   Avg Epochs: 25.7
   Early Stops: 1/3

Factor Analysis:

MODEL:
  ResNet18: 99.38 ± 0.15 (n=9)
  AdvancedNet: 99.25 ± 0.18 (n=9)
  SimpleCNN: 98.95 ± 0.22 (n=9)

OPTIMIZER:
  AdamW: 99.42 ± 0.14 (n=9)
  Adam: 99.28 ± 0.17 (n=9)
  SGD: 99.15 ± 0.20 (n=9)
```

## Contributing

To extend the framework:

1. Add new model architectures to `model_architectures.py`
2. Add new configurations to `ab_testing_config.py`
3. Extend analysis capabilities in `ab_testing_analyzer.py`
4. Update documentation

## License

This framework is part of the Simple ML Portfolio project.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{mnist_ab_testing,
  title={MNIST A/B Testing Framework},
  author={ML Portfolio},
  year={2024},
  url={https://github.com/username/Simple-ML-Portfolio}
}
``` 