"""
t-Distributed Stochastic Neighbor Embedding (t-SNE) Implementation

This module provides a comprehensive implementation of t-SNE for dimensionality
reduction and visualization of high-dimensional data.

Features:
- t-SNE with customizable parameters
- Perplexity optimization
- Visualization of embeddings
- Comparison with other dimensionality reduction techniques
- Interactive plotting capabilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, load_iris, load_breast_cancer, make_swiss_roll
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class TSNEAnalysis:
    """
    A comprehensive t-SNE implementation with analysis and visualization tools.
    """
    
    def __init__(self, n_components=2, perplexity=30.0, learning_rate='auto',
                 n_iter=1000, random_state=42):
        """
        Initialize t-SNE analysis.
        
        Parameters:
        -----------
        n_components : int, default=2
            Dimension of the embedded space
        perplexity : float, default=30.0
            The perplexity is related to the number of nearest neighbors
        learning_rate : float or str, default='auto'
            The learning rate for t-SNE
        n_iter : int, default=1000
            Maximum number of iterations for the optimization
        random_state : int, default=42
            Random state for reproducibility
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.tsne = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def fit(self, X, feature_names=None):
        """
        Fit t-SNE to the data.
        
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
        
        # Fit t-SNE
        self.tsne = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        self.tsne.fit(X_scaled)
        return self
    
    def fit_transform(self, X, feature_names=None):
        """
        Fit t-SNE and transform the data.
        
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
        return self.tsne.embedding_
    
    def transform(self, X):
        """
        Apply t-SNE transformation to new data.
        Note: t-SNE doesn't support transform for new data.
        This method will refit the model.
        
        Parameters:
        -----------
        X : array-like
            Data to transform
            
        Returns:
        --------
        array : Transformed data
        """
        if self.tsne is None:
            raise ValueError("t-SNE must be fitted before transformation")
        
        # t-SNE doesn't support transform, so we refit
        X_scaled = self.scaler.transform(X)
        return self.tsne.fit_transform(X_scaled)
    
    def find_optimal_perplexity(self, X, perplexities=None, plot=True):
        """
        Find optimal perplexity value using KL divergence.
        
        Parameters:
        -----------
        X : array-like
            Input data
        perplexities : list, optional
            List of perplexity values to test
        plot : bool, default=True
            Whether to plot the results
            
        Returns:
        --------
        float : Optimal perplexity value
        """
        if perplexities is None:
            perplexities = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        
        X_scaled = self.scaler.fit_transform(X)
        kl_divergences = []
        
        for perplexity in perplexities:
            tsne_temp = TSNE(
                n_components=self.n_components,
                perplexity=perplexity,
                learning_rate=self.learning_rate,
                n_iter=self.n_iter,
                random_state=self.random_state
            )
            
            embedding = tsne_temp.fit_transform(X_scaled)
            kl_divergences.append(tsne_temp.kl_divergence_)
        
        optimal_perplexity = perplexities[np.argmin(kl_divergences)]
        
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(perplexities, kl_divergences, 'bo-', linewidth=2)
            plt.axvline(x=optimal_perplexity, color='r', linestyle='--',
                       label=f'Optimal perplexity: {optimal_perplexity}')
            plt.xlabel('Perplexity')
            plt.ylabel('KL Divergence')
            plt.title('KL Divergence vs Perplexity')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        
        return optimal_perplexity
    
    def plot_embedding(self, X, labels=None, figsize=(12, 8), title=None):
        """
        Plot the t-SNE embedding.
        
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
        if self.tsne is None:
            raise ValueError("t-SNE must be fitted before plotting")
        
        embedding = self.tsne.embedding_
        
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
            title = f't-SNE Embedding (Perplexity: {self.perplexity})'
        
        plt.title(title)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        
        if labels is not None:
            plt.legend()
        
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_embedding_3d(self, X, labels=None, figsize=(12, 8), title=None):
        """
        Plot the t-SNE embedding in 3D (if n_components=3).
        
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
        
        if self.tsne is None:
            raise ValueError("t-SNE must be fitted before plotting")
        
        embedding = self.tsne.embedding_
        
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
            title = f't-SNE 3D Embedding (Perplexity: {self.perplexity})'
        
        ax.set_title(title)
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_zlabel('t-SNE Component 3')
        
        if labels is not None:
            ax.legend()
        
        plt.show()
    
    def compare_with_pca(self, X, labels=None, figsize=(15, 6)):
        """
        Compare t-SNE embedding with PCA.
        
        Parameters:
        -----------
        X : array-like
            Original data
        labels : array-like, optional
            Labels for coloring the points
        figsize : tuple, default=(15, 6)
            Figure size
        """
        # Fit PCA
        pca = PCA(n_components=2, random_state=self.random_state)
        X_scaled = self.scaler.fit_transform(X)
        pca_embedding = pca.fit_transform(X_scaled)
        
        # Get t-SNE embedding
        tsne_embedding = self.tsne.embedding_
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot PCA
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                mask = labels == label
                ax1.scatter(pca_embedding[mask, 0], pca_embedding[mask, 1],
                           c=[color], label=f'Class {label}', alpha=0.7)
        else:
            ax1.scatter(pca_embedding[:, 0], pca_embedding[:, 1], alpha=0.7)
        
        ax1.set_title('PCA Embedding')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.grid(True, alpha=0.3)
        
        # Plot t-SNE
        if labels is not None:
            for label, color in zip(unique_labels, colors):
                mask = labels == label
                ax2.scatter(tsne_embedding[mask, 0], tsne_embedding[mask, 1],
                           c=[color], label=f'Class {label}', alpha=0.7)
        else:
            ax2.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], alpha=0.7)
        
        ax2.set_title(f't-SNE Embedding (Perplexity: {self.perplexity})')
        ax2.set_xlabel('t-SNE Component 1')
        ax2.set_ylabel('t-SNE Component 2')
        ax2.grid(True, alpha=0.3)
        
        if labels is not None:
            ax1.legend()
            ax2.legend()
        
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
        if self.tsne is None:
            raise ValueError("t-SNE must be fitted before calculating metrics")
        
        metrics = {
            'kl_divergence': self.tsne.kl_divergence_,
            'n_iter': self.tsne.n_iter_,
            'embedding_shape': self.tsne.embedding_.shape
        }
        
        return metrics

def generate_sample_datasets():
    """
    Generate sample datasets for t-SNE analysis.
    
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

def demo_tsne():
    """
    Demonstrate t-SNE analysis on various datasets.
    """
    print("t-SNE Analysis Demo")
    print("=" * 50)
    
    # Generate sample datasets
    datasets = generate_sample_datasets()
    
    for name, (X, y, title) in datasets.items():
        print(f"\nProcessing dataset: {title}")
        print("-" * 30)
        print(f"Data shape: {X.shape}")
        
        # Initialize t-SNE
        tsne = TSNEAnalysis(perplexity=30, n_iter=1000)
        
        # Find optimal perplexity
        optimal_perplexity = tsne.find_optimal_perplexity(X, plot=False)
        print(f"Optimal perplexity: {optimal_perplexity}")
        
        # Fit t-SNE with optimal perplexity
        tsne.perplexity = optimal_perplexity
        embedding = tsne.fit_transform(X)
        print(f"Embedding shape: {embedding.shape}")
        
        # Get quality metrics
        metrics = tsne.get_embedding_quality_metrics(X)
        print(f"KL divergence: {metrics['kl_divergence']:.4f}")
        print(f"Number of iterations: {metrics['n_iter']}")
        
        # Plot embedding
        tsne.plot_embedding(X, y, title=f't-SNE: {title}')
        
        # Compare with PCA
        tsne.compare_with_pca(X, y)
        
        print("\n" + "="*50)

if __name__ == "__main__":
    demo_tsne() 