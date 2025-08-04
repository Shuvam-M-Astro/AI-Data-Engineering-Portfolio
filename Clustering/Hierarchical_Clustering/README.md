# Hierarchical Clustering

This directory contains a comprehensive implementation of hierarchical clustering with dendrogram visualization and analysis capabilities.

## Overview

Hierarchical clustering is a method of cluster analysis that builds a hierarchy of clusters. It can be either agglomerative (bottom-up) or divisive (top-down).

## Features

- **Dendrogram visualization**: Visualize the hierarchical structure of clusters
- **Optimal cluster selection**: Find optimal number of clusters using various metrics
- **Linkage method comparison**: Compare different linkage methods (ward, complete, average, single)
- **Performance metrics**: Calculate silhouette score and Calinski-Harabasz score
- **Comprehensive analysis**: Multiple visualization and analysis tools

## Files

- `hierarchical_clustering.py`: Main implementation with comprehensive analysis tools
- `requirements.txt`: Required Python packages

## Usage

```python
from hierarchical_clustering import HierarchicalClustering

# Initialize hierarchical clustering
hc = HierarchicalClustering(linkage='ward')

# Find optimal number of clusters
optimal_clusters = hc.find_optimal_clusters(X)

# Fit with optimal clusters
hc.n_clusters = optimal_clusters
labels = hc.fit_predict(X)

# Plot dendrogram
hc.plot_dendrogram(X)

# Plot clusters
hc.plot_clusters(X)

# Compare linkage methods
hc.compare_linkage_methods(X)

# Get statistics
stats = hc.get_cluster_statistics()
print(f"Number of clusters: {stats['n_clusters']}")
```

## Parameters

- `n_clusters`: Number of clusters to form (optional)
- `distance_threshold`: The linkage distance threshold above which clusters will not be merged
- `linkage`: Linkage criterion to use ('ward', 'complete', 'average', 'single')
- `metric`: Metric used for distance computation (default: 'euclidean')

## Demo

Run the demo to see hierarchical clustering in action on various datasets:

```bash
python hierarchical_clustering.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Key Advantages

1. **Hierarchical structure**: Provides a tree-like structure of clusters
2. **No assumptions**: Doesn't assume specific cluster shapes
3. **Visualization**: Dendrogram provides intuitive visualization
4. **Flexible linkage**: Multiple linkage methods for different scenarios
5. **Optimal selection**: Automatic selection of optimal number of clusters

## Linkage Methods

- **Ward**: Minimizes the variance of the clusters being merged
- **Complete**: Uses the maximum distance between points in clusters
- **Average**: Uses the average distance between points in clusters
- **Single**: Uses the minimum distance between points in clusters

## Important Notes

- Ward linkage works best with Euclidean distance
- Dendrogram visualization helps in understanding the clustering structure
- Optimal cluster selection uses silhouette score as primary metric
- The algorithm can be computationally expensive for large datasets 