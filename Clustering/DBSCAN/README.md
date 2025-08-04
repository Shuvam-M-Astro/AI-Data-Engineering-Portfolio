# DBSCAN Clustering

This directory contains a comprehensive implementation of DBSCAN (Density-Based Spatial Clustering of Applications with Noise) clustering algorithm.

## Overview

DBSCAN is a density-based clustering algorithm that groups together points that are closely packed, marking as outliers points that lie alone in low-density regions.

## Features

- **Automatic parameter selection**: Find optimal eps parameter using k-nearest neighbors
- **Visualization**: Plot clusters and noise points with different colors
- **Performance metrics**: Calculate silhouette score and Calinski-Harabasz score
- **Multiple datasets**: Test on various synthetic datasets
- **Noise detection**: Identify and visualize noise points

## Files

- `dbscan_clustering.py`: Main implementation with comprehensive analysis tools
- `requirements.txt`: Required Python packages

## Usage

```python
from dbscan_clustering import DBSCANClustering

# Initialize DBSCAN
dbscan = DBSCANClustering()

# Find optimal eps parameter
optimal_eps = dbscan.find_optimal_eps(X)

# Fit the model
dbscan.fit(X, eps=optimal_eps)

# Get statistics
stats = dbscan.get_cluster_statistics()
print(f"Number of clusters: {stats['n_clusters']}")
print(f"Number of noise points: {stats['n_noise']}")

# Plot results
dbscan.plot_clusters(X)

# Evaluate clustering
metrics = dbscan.evaluate_clustering(X)
```

## Parameters

- `eps`: The maximum distance between two samples for one to be considered as in the neighborhood of the other
- `min_samples`: The number of samples in a neighborhood for a point to be considered as a core point
- `metric`: The metric to use when calculating distance between instances

## Demo

Run the demo to see DBSCAN in action on various datasets:

```bash
python dbscan_clustering.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Key Advantages

1. **Noise handling**: Automatically identifies and handles noise points
2. **Shape flexibility**: Can find clusters of arbitrary shapes
3. **Parameter optimization**: Automatic selection of optimal eps parameter
4. **Comprehensive evaluation**: Multiple metrics for clustering quality assessment 