# t-Distributed Stochastic Neighbor Embedding (t-SNE)

This directory contains a comprehensive implementation of t-SNE for dimensionality reduction and visualization of high-dimensional data.

## Overview

t-SNE is a non-linear dimensionality reduction technique that is particularly well-suited for embedding high-dimensional data in a low-dimensional space for visualization.

## Features

- **Perplexity optimization**: Find optimal perplexity parameter using KL divergence
- **Visualization**: Plot embeddings with customizable parameters
- **Comparison with other methods**: Compare t-SNE with PCA and other techniques
- **3D visualization**: Support for 3D embeddings
- **Quality metrics**: Calculate embedding quality metrics

## Files

- `tsne_analysis.py`: Main implementation with comprehensive analysis tools
- `requirements.txt`: Required Python packages

## Usage

```python
from tsne_analysis import TSNEAnalysis

# Initialize t-SNE
tsne = TSNEAnalysis(perplexity=30, n_iter=1000)

# Find optimal perplexity
optimal_perplexity = tsne.find_optimal_perplexity(X)

# Fit and transform data
embedding = tsne.fit_transform(X)

# Plot embedding
tsne.plot_embedding(X, labels)

# Compare with other methods
tsne.compare_with_pca(X, labels)

# Get quality metrics
metrics = tsne.get_embedding_quality_metrics(X)
print(f"KL divergence: {metrics['kl_divergence']:.4f}")
```

## Parameters

- `n_components`: Dimension of the embedded space (default: 2)
- `perplexity`: The perplexity is related to the number of nearest neighbors (default: 30.0)
- `learning_rate`: The learning rate for t-SNE (default: 'auto')
- `n_iter`: Maximum number of iterations for the optimization (default: 1000)

## Demo

Run the demo to see t-SNE in action on various datasets:

```bash
python tsne_analysis.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Key Advantages

1. **Non-linear mapping**: Preserves local structure and non-linear relationships
2. **Visualization focus**: Optimized for data visualization
3. **Local structure preservation**: Maintains local neighborhoods in the embedding
4. **Perplexity tuning**: Automatic optimization of perplexity parameter
5. **Multiple datasets**: Works well on various types of high-dimensional data

## Important Notes

- t-SNE is primarily for visualization and doesn't support transform for new data
- The algorithm is stochastic and results may vary between runs
- Perplexity should be tuned based on the dataset size and structure 