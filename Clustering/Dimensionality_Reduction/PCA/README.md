# Principal Component Analysis (PCA)

This directory contains a comprehensive implementation of PCA for dimensionality reduction and data visualization.

## Overview

PCA is a linear dimensionality reduction technique that transforms data into a new coordinate system where the greatest variance lies on the first coordinate, the second greatest variance on the second coordinate, and so on.

## Features

- **Automatic component selection**: Find optimal number of components to preserve variance
- **Explained variance analysis**: Visualize cumulative and individual explained variance
- **Feature importance**: Analyze loadings of principal components
- **Data reconstruction**: Transform data back to original space
- **Comprehensive visualization**: Multiple plotting options for analysis

## Files

- `pca_analysis.py`: Main implementation with comprehensive analysis tools
- `requirements.txt`: Required Python packages

## Usage

```python
from pca_analysis import PCAAnalysis

# Initialize PCA
pca = PCAAnalysis()

# Fit PCA to data
pca.fit(X, feature_names)

# Get variance information
var_info = pca.get_explained_variance_info()
print(f"Total variance explained: {var_info['total_variance_explained']:.3f}")

# Find optimal components
optimal_components = pca.find_optimal_components(threshold=0.95)

# Transform data
X_transformed = pca.transform(X)

# Plot explained variance
pca.plot_explained_variance()

# Plot first two components
pca.plot_components_2d(X)

# Plot feature importance
pca.plot_feature_importance(n_components=3)
```

## Parameters

- `n_components`: Number of components to keep (None for all, float for variance ratio)
- `random_state`: Random state for reproducibility

## Demo

Run the demo to see PCA in action:

```bash
python pca_analysis.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Key Advantages

1. **Linear transformation**: Preserves linear relationships in data
2. **Variance preservation**: Maximizes variance in first components
3. **Feature reduction**: Reduces dimensionality while preserving information
4. **Interpretability**: Principal components can be interpreted as linear combinations of original features
5. **Reconstruction capability**: Can transform data back to original space 