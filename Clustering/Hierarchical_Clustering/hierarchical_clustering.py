"""
Hierarchical Clustering Implementation

This module provides a comprehensive implementation of hierarchical clustering
with visualization, dendrogram analysis, and comparison capabilities.

Features:
- Agglomerative hierarchical clustering
- Dendrogram visualization
- Optimal cluster number selection
- Distance metric comparison
- Linkage method comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.datasets import make_blobs, make_moons, make_circles
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

class HierarchicalClustering:
    """
    A comprehensive hierarchical clustering implementation with analysis tools.
    """
    
    def __init__(self, n_clusters=None, distance_threshold=None, 
                 linkage='ward', metric='euclidean', random_state=42):
        """
        Initialize hierarchical clustering.
        
        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters to form
        distance_threshold : float, optional
            The linkage distance threshold above which clusters will not be merged
        linkage : str, default='ward'
            Linkage criterion to use
        metric : str, default='euclidean'
            Metric used for distance computation
        random_state : int, default=42
            Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.linkage = linkage
        self.metric = metric
        self.random_state = random_state
        self.model = None
        self.labels = None
        self.linkage_matrix = None
        self.scaler = StandardScaler()
        
    def fit(self, X):
        """
        Fit hierarchical clustering to the data.
        
        Parameters:
        -----------
        X : array-like
            Training data
        """
        # Standardize the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Create linkage matrix
        self.linkage_matrix = linkage(X_scaled, method=self.linkage, metric=self.metric)
        
        # Fit clustering model
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            distance_threshold=self.distance_threshold,
            linkage=self.linkage,
            metric=self.metric
        )
        
        self.labels = self.model.fit_predict(X_scaled)
        
        return self
    
    def fit_predict(self, X):
        """
        Fit hierarchical clustering and predict cluster labels.
        
        Parameters:
        -----------
        X : array-like
            Training data
            
        Returns:
        --------
        array : Cluster labels
        """
        self.fit(X)
        return self.labels
    
    def plot_dendrogram(self, X, max_d=None, figsize=(12, 8), title=None):
        """
        Plot the dendrogram.
        
        Parameters:
        -----------
        X : array-like
            Input data
        max_d : float, optional
            Maximum distance to show in dendrogram
        figsize : tuple, default=(12, 8)
            Figure size
        title : str, optional
            Plot title
        """
        if self.linkage_matrix is None:
            raise ValueError("Model must be fitted before plotting dendrogram")
        
        plt.figure(figsize=figsize)
        
        # Create dendrogram
        dendrogram(
            self.linkage_matrix,
            truncate_mode='lastp',
            p=30,  # Show last 30 merged clusters
            leaf_rotation=90,
            leaf_font_size=10,
            show_contracted=True
        )
        
        if title is None:
            title = f'Hierarchical Clustering Dendrogram\nLinkage: {self.linkage}, Metric: {self.metric}'
        
        plt.title(title)
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        
        if max_d is not None:
            plt.axhline(y=max_d, color='r', linestyle='--', label=f'Distance threshold: {max_d}')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    
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
            raise ValueError("Model must be fitted before plotting clusters")
        
        X_scaled = self.scaler.transform(X)
        
        plt.figure(figsize=figsize)
        
        # Create scatter plot
        unique_labels = set(self.labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            class_member_mask = (self.labels == label)
            xy = X_scaled[class_member_mask]
            plt.scatter(xy[:, 0], xy[:, 1], c=[color], s=50, 
                       label=f'Cluster {label}', alpha=0.7)
        
        plt.title(f'Hierarchical Clustering Results\n'
                 f'Clusters: {len(unique_labels)}, Linkage: {self.linkage}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def find_optimal_clusters(self, X, max_clusters=10, plot=True):
        """
        Find optimal number of clusters using various metrics.
        
        Parameters:
        -----------
        X : array-like
            Input data
        max_clusters : int, default=10
            Maximum number of clusters to test
        plot : bool, default=True
            Whether to plot the results
            
        Returns:
        --------
        int : Optimal number of clusters
        """
        X_scaled = self.scaler.fit_transform(X)
        
        n_clusters_range = range(2, min(max_clusters + 1, len(X) // 2))
        silhouette_scores = []
        calinski_scores = []
        
        for n_clusters in n_clusters_range:
            # Fit clustering
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=self.linkage,
                metric=self.metric
            )
            labels = model.fit_predict(X_scaled)
            
            # Calculate metrics
            try:
                silhouette_scores.append(silhouette_score(X_scaled, labels))
            except:
                silhouette_scores.append(0)
            
            try:
                calinski_scores.append(calinski_harabasz_score(X_scaled, labels))
            except:
                calinski_scores.append(0)
        
        # Find optimal number of clusters
        optimal_silhouette = n_clusters_range[np.argmax(silhouette_scores)]
        optimal_calinski = n_clusters_range[np.argmax(calinski_scores)]
        
        # Use silhouette score as primary metric
        optimal_clusters = optimal_silhouette
        
        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot silhouette scores
            ax1.plot(n_clusters_range, silhouette_scores, 'bo-', linewidth=2)
            ax1.axvline(x=optimal_silhouette, color='r', linestyle='--',
                       label=f'Optimal: {optimal_silhouette}')
            ax1.set_xlabel('Number of Clusters')
            ax1.set_ylabel('Silhouette Score')
            ax1.set_title('Silhouette Score vs Number of Clusters')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot Calinski-Harabasz scores
            ax2.plot(n_clusters_range, calinski_scores, 'go-', linewidth=2)
            ax2.axvline(x=optimal_calinski, color='r', linestyle='--',
                       label=f'Optimal: {optimal_calinski}')
            ax2.set_xlabel('Number of Clusters')
            ax2.set_ylabel('Calinski-Harabasz Score')
            ax2.set_title('Calinski-Harabasz Score vs Number of Clusters')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        return optimal_clusters
    
    def compare_linkage_methods(self, X, linkage_methods=None, figsize=(15, 10)):
        """
        Compare different linkage methods.
        
        Parameters:
        -----------
        X : array-like
            Input data
        linkage_methods : list, optional
            List of linkage methods to compare
        figsize : tuple, default=(15, 10)
            Figure size
        """
        if linkage_methods is None:
            linkage_methods = ['ward', 'complete', 'average', 'single']
        
        X_scaled = self.scaler.fit_transform(X)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, linkage_method in enumerate(linkage_methods):
            if i >= len(axes):
                break
                
            # Fit clustering with current linkage method
            model = AgglomerativeClustering(
                n_clusters=3,  # Fixed number for comparison
                linkage=linkage_method,
                metric=self.metric
            )
            labels = model.fit_predict(X_scaled)
            
            # Plot clusters
            unique_labels = set(labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                class_member_mask = (labels == label)
                xy = X_scaled[class_member_mask]
                axes[i].scatter(xy[:, 0], xy[:, 1], c=[color], s=50, 
                               label=f'Cluster {label}', alpha=0.7)
            
            axes[i].set_title(f'Linkage: {linkage_method}')
            axes[i].set_xlabel('Feature 1')
            axes[i].set_ylabel('Feature 2')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(linkage_methods), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
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
            'n_clusters': len(set(self.labels)),
            'n_total_points': len(self.labels),
            'cluster_sizes': {},
            'linkage_method': self.linkage,
            'metric': self.metric
        }
        
        unique_labels = set(self.labels)
        for label in unique_labels:
            stats['cluster_sizes'][f'Cluster_{label}'] = np.sum(self.labels == label)
        
        return stats
    
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
        
        X_scaled = self.scaler.transform(X)
        
        metrics = {}
        
        try:
            metrics['silhouette_score'] = silhouette_score(X_scaled, self.labels)
        except:
            metrics['silhouette_score'] = None
        
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_scaled, self.labels)
        except:
            metrics['calinski_harabasz_score'] = None
        
        return metrics

def generate_sample_datasets():
    """
    Generate sample datasets for hierarchical clustering analysis.
    
    Returns:
    --------
    dict : Dictionary containing sample datasets
    """
    np.random.seed(42)
    
    # Dataset 1: Blobs with different sizes
    X1, y1 = make_blobs(n_samples=300, centers=4, cluster_std=[0.5, 1.0, 0.8, 1.2],
                        random_state=42)
    
    # Dataset 2: Moons
    X2, y2 = make_moons(n_samples=200, noise=0.1, random_state=42)
    
    # Dataset 3: Circles
    X3, y3 = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
    
    # Dataset 4: Hierarchical structure
    n_samples = 400
    
    # Create hierarchical clusters
    cluster1 = np.random.randn(100, 2) + np.array([0, 0])
    cluster2 = np.random.randn(100, 2) + np.array([4, 0])
    cluster3 = np.random.randn(100, 2) + np.array([0, 4])
    cluster4 = np.random.randn(100, 2) + np.array([4, 4])
    
    X4 = np.vstack([cluster1, cluster2, cluster3, cluster4])
    y4 = np.concatenate([np.zeros(100), np.ones(100), np.ones(100) * 2, np.ones(100) * 3])
    
    return {
        'blobs': (X1, y1, 'Blobs'),
        'moons': (X2, y2, 'Moons'),
        'circles': (X3, y3, 'Circles'),
        'hierarchical': (X4, y4, 'Hierarchical Structure')
    }

def demo_hierarchical_clustering():
    """
    Demonstrate hierarchical clustering on various datasets.
    """
    print("Hierarchical Clustering Demo")
    print("=" * 50)
    
    # Generate sample datasets
    datasets = generate_sample_datasets()
    
    for name, (X, y, title) in datasets.items():
        print(f"\nProcessing dataset: {title}")
        print("-" * 30)
        print(f"Data shape: {X.shape}")
        
        # Initialize hierarchical clustering
        hc = HierarchicalClustering(linkage='ward')
        
        # Find optimal number of clusters
        optimal_clusters = hc.find_optimal_clusters(X, plot=False)
        print(f"Optimal number of clusters: {optimal_clusters}")
        
        # Fit with optimal clusters
        hc.n_clusters = optimal_clusters
        labels = hc.fit_predict(X)
        
        # Get statistics
        stats = hc.get_cluster_statistics()
        print(f"Number of clusters: {stats['n_clusters']}")
        print(f"Linkage method: {stats['linkage_method']}")
        
        # Evaluate clustering
        metrics = hc.evaluate_clustering(X)
        if metrics['silhouette_score'] is not None:
            print(f"Silhouette score: {metrics['silhouette_score']:.3f}")
        if metrics['calinski_harabasz_score'] is not None:
            print(f"Calinski-Harabasz score: {metrics['calinski_harabasz_score']:.3f}")
        
        # Plot dendrogram
        hc.plot_dendrogram(X)
        
        # Plot clusters
        hc.plot_clusters(X)
        
        # Compare linkage methods
        hc.compare_linkage_methods(X)
        
        print("\n" + "="*50)

if __name__ == "__main__":
    demo_hierarchical_clustering() 