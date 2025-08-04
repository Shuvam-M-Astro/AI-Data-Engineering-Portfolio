"""
Uniform Manifold Approximation and Projection (UMAP) Implementation

This module provides a comprehensive implementation of UMAP for dimensionality
reduction and visualization of high-dimensional data.

Features:
- UMAP with customizable parameters
- Parameter optimization
- Visualization of embeddings
- Comparison with other dimensionality reduction techniques
- Interactive plotting capabilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, load_iris, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

class UMAPAnalysis:
    """
    A comprehensive UMAP implementation with analysis and visualization tools.
    """
    
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1,
                 metric='euclidean', random_state=42):
        """
        Initialize UMAP analysis.
        
        Parameters:
        -----------
        n_components : int, default=2
            Dimension of the embedded space
        n_neighbors : int, default=15
            Number of neighbors to consider for each point
        min_dist : float, default=0.1
            Minimum distance between points in the embedded space
        metric : str, default='euclidean'
            Distance metric to use
        random_state : int, default=42
            Random state for reproducibility
        """
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.umap_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def fit(self, X, feature_names=None):
        """
        Fit UMAP to the data.
        
        Parameters:
        -----------
        X : array-like
            Training data
        feature_names : list, optional
            Names of the features
        """
        self.feature_names = feature_names
        
        # Standardize the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit UMAP
        self.umap_model = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state
        )
        
        self.umap_model.fit(X_scaled)
        return self
    
    def fit_transform(self, X, feature_names=None):
        """
        Fit UMAP and transform the data.
        
        Parameters:
        -----------
        X : array-like
            Training data
        feature_names : list, optional
            Names of the features
            
        Returns:
        --------
        array : Transformed data
        """
        self.fit(X, feature_names)
        return self.umap_model.embedding_
    
    def transform(self, X):
        """
        Apply UMAP transformation to new data.
        
        Parameters:
        -----------
        X : array-like
            Data to transform
            
        Returns:
        --------
        array : Transformed data
        """
        if self.umap_model is None:
            raise ValueError("UMAP must be fitted before transformation")
        
        X_scaled = self.scaler.transform(X)
        return self.umap_model.transform(X_scaled)
    
    def find_optimal_parameters(self, X, n_neighbors_range=None, min_dist_range=None, plot=True):
        """
        Find optimal UMAP parameters using reconstruction error.
        
        Parameters:
        -----------
        X : array-like
            Input data
        n_neighbors_range : list, optional
            Range of n_neighbors values to test
        min_dist_range : list, optional
            Range of min_dist values to test
        plot : bool, default=True
            Whether to plot the results
            
        Returns:
        --------
        tuple : Optimal (n_neighbors, min_dist) values
        """
        if n_neighbors_range is None:
            n_neighbors_range = [5, 10, 15, 20, 25, 30]
        if min_dist_range is None:
            min_dist_range = [0.01, 0.05, 0.1, 0.2, 0.5]
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Test different parameter combinations
        results = []
        
        for n_neighbors in n_neighbors_range:
            for min_dist in min_dist_range:
                try:
                    umap_temp = umap.UMAP(
                        n_components=self.n_components,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        metric=self.metric,
                        random_state=self.random_state
                    )
                    
                    embedding = umap_temp.fit_transform(X_scaled)
                    
                    # Calculate reconstruction error (approximate)
                    # We'll use the inverse transform if available, otherwise use a simple metric
                    try:
                        reconstructed = umap_temp.inverse_transform(embedding)
                        error = np.mean((X_scaled - reconstructed) ** 2)
                    except:
                        # If inverse_transform is not available, use embedding quality
                        error = -np.mean(np.linalg.norm(embedding, axis=1))
                    
                    results.append({
                        'n_neighbors': n_neighbors,
                        'min_dist': min_dist,
                        'error': error
                    })
                except:
                    continue
        
        # Find optimal parameters
        best_result = min(results, key=lambda x: x['error'])
        optimal_n_neighbors = best_result['n_neighbors']
        optimal_min_dist = best_result['min_dist']
        
        if plot:
            # Create parameter grid plot
            n_neighbors_vals = list(set([r['n_neighbors'] for r in results]))
            min_dist_vals = list(set([r['min_dist'] for r in results]))
            
            error_matrix = np.zeros((len(min_dist_vals), len(n_neighbors_vals)))
            
            for result in results:
                i = min_dist_vals.index(result['min_dist'])
                j = n_neighbors_vals.index(result['n_neighbors'])
                error_matrix[i, j] = result['error']
            
            plt.figure(figsize=(10, 6))
            plt.imshow(error_matrix, cmap='viridis', aspect='auto')
            plt.colorbar(label='Reconstruction Error')
            plt.xticks(range(len(n_neighbors_vals)), n_neighbors_vals)
            plt.yticks(range(len(min_dist_vals)), min_dist_vals)
            plt.xlabel('n_neighbors')
            plt.ylabel('min_dist')
            plt.title('Parameter Optimization Results')
            
            # Mark optimal point
            opt_i = min_dist_vals.index(optimal_min_dist)
            opt_j = n_neighbors_vals.index(optimal_n_neighbors)
            plt.plot(opt_j, opt_i, 'r*', markersize=15, label=f'Optimal: ({optimal_n_neighbors}, {optimal_min_dist})')
            plt.legend()
            plt.show()
        
        return optimal_n_neighbors, optimal_min_dist
    
    def plot_embedding(self, X, labels=None, figsize=(12, 8), title=None):
        """
        Plot the UMAP embedding.
        
        Parameters:
        -----------
        X : array-like
            Original data
        labels : array-like, optional
            Labels for coloring the points
        figsize : tuple, default=(12, 8)
            Figure size
        title : str, optional
            Plot title
        """
        if self.umap_model is None:
            raise ValueError("UMAP must be fitted before plotting")
        
        embedding = self.umap_model.embedding_
        
        plt.figure(figsize=figsize)
        
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                mask = labels == label
                plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                           c=[color], label=f'Class {label}', alpha=0.7)
        else:
            plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7)
        
        if title is None:
            title = f'UMAP Embedding (n_neighbors: {self.n_neighbors}, min_dist: {self.min_dist})'
        
        plt.title(title)
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        
        if labels is not None:
            plt.legend()
        
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_embedding_3d(self, X, labels=None, figsize=(12, 8), title=None):
        """
        Plot the UMAP embedding in 3D (if n_components=3).
        
        Parameters:
        -----------
        X : array-like
            Original data
        labels : array-like, optional
            Labels for coloring the points
        figsize : tuple, default=(12, 8)
            Figure size
        title : str, optional
            Plot title
        """
        if self.n_components != 3:
            raise ValueError("3D plotting requires n_components=3")
        
        if self.umap_model is None:
            raise ValueError("UMAP must be fitted before plotting")
        
        embedding = self.umap_model.embedding_
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                mask = labels == label
                ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                          c=[color], label=f'Class {label}', alpha=0.7)
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], alpha=0.7)
        
        if title is None:
            title = f'UMAP 3D Embedding (n_neighbors: {self.n_neighbors}, min_dist: {self.min_dist})'
        
        ax.set_title(title)
        ax.set_xlabel('UMAP Component 1')
        ax.set_ylabel('UMAP Component 2')
        ax.set_zlabel('UMAP Component 3')
        
        if labels is not None:
            ax.legend()
        
        plt.show()
    
    def compare_with_other_methods(self, X, labels=None, figsize=(15, 10)):
        """
        Compare UMAP embedding with PCA and t-SNE.
        
        Parameters:
        -----------
        X : array-like
            Original data
        labels : array-like, optional
            Labels for coloring the points
        figsize : tuple, default=(15, 10)
            Figure size
        """
        # Fit PCA
        pca = PCA(n_components=2, random_state=self.random_state)
        X_scaled = self.scaler.fit_transform(X)
        pca_embedding = pca.fit_transform(X_scaled)
        
        # Fit t-SNE
        tsne = TSNE(n_components=2, random_state=self.random_state)
        tsne_embedding = tsne.fit_transform(X_scaled)
        
        # Get UMAP embedding
        umap_embedding = self.umap_model.embedding_
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        embeddings = [pca_embedding, tsne_embedding, umap_embedding]
        titles = ['PCA', 't-SNE', 'UMAP']
        
        for i, (embedding, title) in enumerate(zip(embeddings, titles)):
            row = i // 2
            col = i % 2
            
            if labels is not None:
                unique_labels = np.unique(labels)
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
                
                for label, color in zip(unique_labels, colors):
                    mask = labels == label
                    axes[row, col].scatter(embedding[mask, 0], embedding[mask, 1],
                                         c=[color], label=f'Class {label}', alpha=0.7)
            else:
                axes[row, col].scatter(embedding[:, 0], embedding[:, 1], alpha=0.7)
            
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel(f'{title} Component 1')
            axes[row, col].set_ylabel(f'{title} Component 2')
            axes[row, col].grid(True, alpha=0.3)
            
            if labels is not None:
                axes[row, col].legend()
        
        # Add a fourth subplot for parameter info
        axes[1, 1].text(0.5, 0.5, f'UMAP Parameters:\nn_neighbors: {self.n_neighbors}\nmin_dist: {self.min_dist}',
                        ha='center', va='center', transform=axes[1, 1].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 1].set_title('Parameters')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_embedding_quality_metrics(self, X):
        """
        Calculate quality metrics for the embedding.
        
        Parameters:
        -----------
        X : array-like
            Original data
            
        Returns:
        --------
        dict : Quality metrics
        """
        if self.umap_model is None:
            raise ValueError("UMAP must be fitted before calculating metrics")
        
        embedding = self.umap_model.embedding_
        
        # Calculate various metrics
        metrics = {
            'embedding_shape': embedding.shape,
            'embedding_range': {
                'min': np.min(embedding, axis=0).tolist(),
                'max': np.max(embedding, axis=0).tolist()
            },
            'embedding_std': np.std(embedding, axis=0).tolist(),
            'n_neighbors': self.n_neighbors,
            'min_dist': self.min_dist
        }
        
        return metrics

def generate_sample_datasets():
    """
    Generate sample datasets for UMAP analysis.
    
    Returns:
    --------
    dict : Dictionary containing sample datasets
    """
    np.random.seed(42)
    
    # Dataset 1: Swiss roll
    X1, y1 = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)
    X1 = X1[:, [0, 2]]  # Take only 2D projection
    
    # Dataset 2: Blobs with different densities
    X2, y2 = make_blobs(n_samples=1000, centers=5, cluster_std=[0.5, 1.0, 0.8, 1.2, 0.6],
                        random_state=42)
    
    # Dataset 3: Iris dataset
    iris = load_iris()
    X3, y3 = iris.data, iris.target
    
    # Dataset 4: High-dimensional data with clusters
    n_samples = 500
    n_features = 50
    
    X4 = np.random.randn(n_samples, n_features)
    
    # Create 3 clusters in high-dimensional space
    cluster1 = np.random.randn(200, n_features) + np.array([2] * n_features)
    cluster2 = np.random.randn(200, n_features) + np.array([-2] * n_features)
    cluster3 = np.random.randn(100, n_features) + np.array([0] * n_features)
    
    X4 = np.vstack([cluster1, cluster2, cluster3])
    y4 = np.concatenate([np.zeros(200), np.ones(200), np.ones(100) * 2])
    
    return {
        'swiss_roll': (X1, y1, 'Swiss Roll'),
        'blobs': (X2, y2, 'Blobs'),
        'iris': (X3, y3, 'Iris Dataset'),
        'high_dim': (X4, y4, 'High-Dimensional Clusters')
    }

def demo_umap():
    """
    Demonstrate UMAP analysis on various datasets.
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available. Install with: pip install umap-learn")
        return
    
    print("UMAP Analysis Demo")
    print("=" * 50)
    
    # Generate sample datasets
    datasets = generate_sample_datasets()
    
    for name, (X, y, title) in datasets.items():
        print(f"\nProcessing dataset: {title}")
        print("-" * 30)
        print(f"Data shape: {X.shape}")
        
        # Initialize UMAP
        umap_analysis = UMAPAnalysis(n_neighbors=15, min_dist=0.1)
        
        # Find optimal parameters
        optimal_n_neighbors, optimal_min_dist = umap_analysis.find_optimal_parameters(X, plot=False)
        print(f"Optimal n_neighbors: {optimal_n_neighbors}")
        print(f"Optimal min_dist: {optimal_min_dist}")
        
        # Fit UMAP with optimal parameters
        umap_analysis.n_neighbors = optimal_n_neighbors
        umap_analysis.min_dist = optimal_min_dist
        embedding = umap_analysis.fit_transform(X)
        print(f"Embedding shape: {embedding.shape}")
        
        # Get quality metrics
        metrics = umap_analysis.get_embedding_quality_metrics(X)
        print(f"Embedding range: {metrics['embedding_range']}")
        
        # Plot embedding
        umap_analysis.plot_embedding(X, y, title=f'UMAP: {title}')
        
        # Compare with other methods
        umap_analysis.compare_with_other_methods(X, y)
        
        print("\n" + "="*50)

if __name__ == "__main__":
    demo_umap() 