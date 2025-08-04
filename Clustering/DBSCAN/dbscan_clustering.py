"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) Implementation

This module provides a comprehensive implementation of DBSCAN clustering algorithm
with visualization, parameter tuning, and analysis capabilities.

Features:
- DBSCAN clustering with customizable parameters
- Automatic parameter selection using elbow method
- Visualization of clusters and noise points
- Performance metrics calculation
- Comparison with other clustering algorithms
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs, make_moons, make_circles
import warnings
warnings.filterwarnings('ignore')

class DBSCANClustering:
    """
    A comprehensive DBSCAN clustering implementation with analysis tools.
    """
    
    def __init__(self, eps=None, min_samples=5, metric='euclidean'):
        """
        Initialize DBSCAN clustering.
        
        Parameters:
        -----------
        eps : float, optional
            The maximum distance between two samples for one to be considered
            as in the neighborhood of the other.
        min_samples : int, default=5
            The number of samples in a neighborhood for a point to be considered
            as a core point.
        metric : str, default='euclidean'
            The metric to use when calculating distance between instances.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.model = None
        self.labels = None
        self.n_clusters = 0
        self.n_noise = 0
        
    def find_optimal_eps(self, X, k=5, plot=True):
        """
        Find optimal eps parameter using k-nearest neighbors.
        
        Parameters:
        -----------
        X : array-like
            Input data
        k : int, default=5
            Number of neighbors to consider
        plot : bool, default=True
            Whether to plot the k-distance graph
            
        Returns:
        --------
        float : Optimal eps value
        """
        # Calculate k-nearest neighbors distances
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Sort distances to k-th neighbor
        k_distances = np.sort(distances[:, -1])
        
        # Find the "elbow" point
        # Use the point where the slope changes significantly
        diffs = np.diff(k_distances)
        elbow_idx = np.argmax(diffs) + 1
        optimal_eps = k_distances[elbow_idx]
        
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(k_distances)), k_distances, 'b-', label='k-distance')
            plt.axhline(y=optimal_eps, color='r', linestyle='--', 
                       label=f'Optimal eps: {optimal_eps:.3f}')
            plt.axvline(x=elbow_idx, color='g', linestyle='--', 
                       label=f'Elbow point: {elbow_idx}')
            plt.xlabel('Points')
            plt.ylabel(f'{k}-Distance')
            plt.title('K-Distance Graph for Optimal Eps Selection')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
        return optimal_eps
    
    def fit(self, X, eps=None, min_samples=None):
        """
        Fit DBSCAN clustering to the data.
        
        Parameters:
        -----------
        X : array-like
            Training data
        eps : float, optional
            The maximum distance between two samples
        min_samples : int, optional
            The number of samples in a neighborhood
        """
        if eps is None:
            eps = self.eps
        if min_samples is None:
            min_samples = self.min_samples
            
        if eps is None:
            eps = self.find_optimal_eps(X, plot=False)
            
        self.model = DBSCAN(eps=eps, min_samples=min_samples, metric=self.metric)
        self.labels = self.model.fit_predict(X)
        
        # Calculate statistics
        self.n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        self.n_noise = list(self.labels).count(-1)
        
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for new data.
        
        Parameters:
        -----------
        X : array-like
            Data to predict
            
        Returns:
        --------
        array : Cluster labels
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.fit_predict(X)
    
    def get_cluster_statistics(self):
        """
        Get statistics about the clustering results.
        
        Returns:
        --------
        dict : Statistics about the clustering
        """
        if self.labels is None:
            raise ValueError("Model must be fitted before getting statistics")
            
        stats = {
            'n_clusters': self.n_clusters,
            'n_noise': self.n_noise,
            'n_total_points': len(self.labels),
            'noise_percentage': (self.n_noise / len(self.labels)) * 100,
            'cluster_sizes': {}
        }
        
        unique_labels = set(self.labels)
        for label in unique_labels:
            if label != -1:  # Not noise
                stats['cluster_sizes'][f'Cluster_{label}'] = np.sum(self.labels == label)
                
        return stats
    
    def plot_clusters(self, X, figsize=(12, 8)):
        """
        Visualize the clustering results.
        
        Parameters:
        -----------
        X : array-like
            Input data
        figsize : tuple, default=(12, 8)
            Figure size
        """
        if self.labels is None:
            raise ValueError("Model must be fitted before plotting")
            
        plt.figure(figsize=figsize)
        
        # Create scatter plot
        unique_labels = set(self.labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Black used for noise
                col = 'black'
                label_name = 'Noise'
            else:
                col = color
                label_name = f'Cluster {label}'
                
            class_member_mask = (self.labels == label)
            xy = X[class_member_mask]
            plt.scatter(xy[:, 0], xy[:, 1], c=[col], s=50, 
                       label=label_name, alpha=0.7)
        
        plt.title(f'DBSCAN Clustering Results\n'
                 f'Clusters: {self.n_clusters}, Noise: {self.n_noise}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def evaluate_clustering(self, X):
        """
        Evaluate clustering quality using various metrics.
        
        Parameters:
        -----------
        X : array-like
            Input data
            
        Returns:
        --------
        dict : Evaluation metrics
        """
        if self.labels is None:
            raise ValueError("Model must be fitted before evaluation")
            
        # Remove noise points for evaluation
        non_noise_mask = self.labels != -1
        X_clean = X[non_noise_mask]
        labels_clean = self.labels[non_noise_mask]
        
        if len(set(labels_clean)) < 2:
            return {'error': 'Not enough clusters for evaluation'}
            
        metrics = {}
        
        try:
            metrics['silhouette_score'] = silhouette_score(X_clean, labels_clean)
        except:
            metrics['silhouette_score'] = None
            
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_clean, labels_clean)
        except:
            metrics['calinski_harabasz_score'] = None
            
        return metrics

def generate_sample_datasets():
    """
    Generate sample datasets for testing DBSCAN.
    
    Returns:
    --------
    dict : Dictionary containing sample datasets
    """
    np.random.seed(42)
    
    # Dataset 1: Blobs with noise
    X1, y1 = make_blobs(n_samples=300, centers=4, cluster_std=0.60, 
                        random_state=42, center_box=(-10, 10))
    X1 = np.vstack([X1, np.random.uniform(-10, 10, (50, 2))])
    
    # Dataset 2: Moons
    X2, y2 = make_moons(n_samples=200, noise=0.1, random_state=42)
    
    # Dataset 3: Circles
    X3, y3 = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
    
    # Dataset 4: Complex structure
    X4 = np.random.randn(300, 2)
    X4 = np.vstack([X4, np.random.randn(100, 2) * 0.3 + np.array([3, 3])])
    X4 = np.vstack([X4, np.random.randn(100, 2) * 0.3 + np.array([-3, -3])])
    
    return {
        'blobs_with_noise': (X1, 'Blobs with Noise'),
        'moons': (X2, 'Moons'),
        'circles': (X3, 'Circles'),
        'complex': (X4, 'Complex Structure')
    }

def demo_dbscan():
    """
    Demonstrate DBSCAN clustering on various datasets.
    """
    print("DBSCAN Clustering Demo")
    print("=" * 50)
    
    # Generate sample datasets
    datasets = generate_sample_datasets()
    
    for name, (X, title) in datasets.items():
        print(f"\nProcessing dataset: {title}")
        print("-" * 30)
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Initialize DBSCAN
        dbscan = DBSCANClustering()
        
        # Find optimal eps
        optimal_eps = dbscan.find_optimal_eps(X_scaled)
        print(f"Optimal eps: {optimal_eps:.3f}")
        
        # Fit the model
        dbscan.fit(X_scaled, eps=optimal_eps)
        
        # Get statistics
        stats = dbscan.get_cluster_statistics()
        print(f"Number of clusters: {stats['n_clusters']}")
        print(f"Number of noise points: {stats['n_noise']}")
        print(f"Noise percentage: {stats['noise_percentage']:.2f}%")
        
        # Evaluate clustering
        metrics = dbscan.evaluate_clustering(X_scaled)
        if 'silhouette_score' in metrics and metrics['silhouette_score'] is not None:
            print(f"Silhouette score: {metrics['silhouette_score']:.3f}")
        if 'calinski_harabasz_score' in metrics and metrics['calinski_harabasz_score'] is not None:
            print(f"Calinski-Harabasz score: {metrics['calinski_harabasz_score']:.3f}")
        
        # Plot results
        dbscan.plot_clusters(X_scaled)
        
        print("\n" + "="*50)

if __name__ == "__main__":
    demo_dbscan() 