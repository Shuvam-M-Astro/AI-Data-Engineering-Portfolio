"""
Principal Component Analysis (PCA) Implementation

This module provides a comprehensive implementation of PCA with visualization,
analysis, and practical applications for dimensionality reduction.

Features:
- PCA with automatic component selection
- Explained variance analysis
- Visualization of principal components
- Feature importance analysis
- Data reconstruction capabilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, load_iris, load_breast_cancer
import warnings
warnings.filterwarnings('ignore')

class PCAAnalysis:
    """
    A comprehensive PCA implementation with analysis and visualization tools.
    """
    
    def __init__(self, n_components=None, random_state=42):
        """
        Initialize PCA analysis.
        
        Parameters:
        -----------
        n_components : int, float, or str, optional
            Number of components to keep. If None, all components are kept.
            If float, it represents the fraction of variance to preserve.
        random_state : int, default=42
            Random state for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.pca = None
        self.scaler = StandardScaler()
        self.explained_variance_ratio = None
        self.cumulative_variance_ratio = None
        self.feature_names = None
        
    def fit(self, X, feature_names=None):
        """
        Fit PCA to the data.
        
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
        
        # Fit PCA
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca.fit(X_scaled)
        
        # Calculate explained variance ratios
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.cumulative_variance_ratio = np.cumsum(self.explained_variance_ratio)
        
        return self
    
    def transform(self, X):
        """
        Apply PCA transformation to the data.
        
        Parameters:
        -----------
        X : array-like
            Data to transform
            
        Returns:
        --------
        array : Transformed data
        """
        if self.pca is None:
            raise ValueError("PCA must be fitted before transformation")
        
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
    
    def inverse_transform(self, X_transformed):
        """
        Transform data back to original space.
        
        Parameters:
        -----------
        X_transformed : array-like
            Transformed data
            
        Returns:
        --------
        array : Data in original space
        """
        if self.pca is None:
            raise ValueError("PCA must be fitted before inverse transformation")
        
        X_scaled = self.pca.inverse_transform(X_transformed)
        return self.scaler.inverse_transform(X_scaled)
    
    def get_explained_variance_info(self):
        """
        Get information about explained variance.
        
        Returns:
        --------
        dict : Information about explained variance
        """
        if self.pca is None:
            raise ValueError("PCA must be fitted before getting variance info")
        
        info = {
            'n_components': self.pca.n_components_,
            'explained_variance_ratio': self.explained_variance_ratio,
            'cumulative_variance_ratio': self.cumulative_variance_ratio,
            'total_variance_explained': self.cumulative_variance_ratio[-1],
            'components_info': []
        }
        
        for i, (var_ratio, cum_var_ratio) in enumerate(zip(
            self.explained_variance_ratio, self.cumulative_variance_ratio)):
            info['components_info'].append({
                'component': i + 1,
                'explained_variance_ratio': var_ratio,
                'cumulative_variance_ratio': cum_var_ratio
            })
        
        return info
    
    def plot_explained_variance(self, figsize=(12, 8)):
        """
        Plot explained variance and cumulative explained variance.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 8)
            Figure size
        """
        if self.pca is None:
            raise ValueError("PCA must be fitted before plotting")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot individual explained variance
        components = range(1, len(self.explained_variance_ratio) + 1)
        ax1.bar(components, self.explained_variance_ratio, alpha=0.7)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Explained Variance by Component')
        ax1.grid(True, alpha=0.3)
        
        # Plot cumulative explained variance
        ax2.plot(components, self.cumulative_variance_ratio, 'bo-', linewidth=2)
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
        ax2.axhline(y=0.99, color='g', linestyle='--', label='99% Variance')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance Ratio')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_components_2d(self, X, figsize=(15, 10)):
        """
        Plot first two principal components.
        
        Parameters:
        -----------
        X : array-like
            Original data
        figsize : tuple, default=(15, 10)
            Figure size
        """
        if self.pca is None:
            raise ValueError("PCA must be fitted before plotting")
        
        # Transform data
        X_transformed = self.transform(X)
        
        plt.figure(figsize=figsize)
        plt.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.7)
        plt.xlabel(f'PC1 ({self.explained_variance_ratio[0]:.3f})')
        plt.ylabel(f'PC2 ({self.explained_variance_ratio[1]:.3f})')
        plt.title('First Two Principal Components')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_feature_importance(self, n_components=5, figsize=(12, 8)):
        """
        Plot feature importance (loadings) for principal components.
        
        Parameters:
        -----------
        n_components : int, default=5
            Number of components to plot
        figsize : tuple, default=(12, 8)
            Figure size
        """
        if self.pca is None:
            raise ValueError("PCA must be fitted before plotting")
        
        n_components = min(n_components, self.pca.n_components_)
        
        # Get feature names or create generic ones
        if self.feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(self.pca.components_.shape[1])]
        else:
            feature_names = self.feature_names
        
        fig, axes = plt.subplots(n_components, 1, figsize=figsize)
        if n_components == 1:
            axes = [axes]
        
        for i in range(n_components):
            component = self.pca.components_[i]
            explained_var = self.explained_variance_ratio[i]
            
            # Sort features by absolute loading value
            feature_importance = np.abs(component)
            sorted_indices = np.argsort(feature_importance)[::-1]
            
            # Plot top features
            top_n = min(10, len(feature_names))
            top_features = [feature_names[j] for j in sorted_indices[:top_n]]
            top_loadings = component[sorted_indices[:top_n]]
            
            axes[i].barh(range(len(top_features)), top_loadings)
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features)
            axes[i].set_xlabel('Loading')
            axes[i].set_title(f'PC{i+1} Loadings (Explained Variance: {explained_var:.3f})')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def find_optimal_components(self, threshold=0.95):
        """
        Find optimal number of components to preserve specified variance.
        
        Parameters:
        -----------
        threshold : float, default=0.95
            Minimum variance to preserve
            
        Returns:
        --------
        int : Optimal number of components
        """
        if self.pca is None:
            raise ValueError("PCA must be fitted before finding optimal components")
        
        for i, cum_var in enumerate(self.cumulative_variance_ratio):
            if cum_var >= threshold:
                return i + 1
        
        return len(self.cumulative_variance_ratio)
    
    def get_reconstruction_error(self, X):
        """
        Calculate reconstruction error after PCA transformation.
        
        Parameters:
        -----------
        X : array-like
            Original data
            
        Returns:
        --------
        float : Mean squared reconstruction error
        """
        if self.pca is None:
            raise ValueError("PCA must be fitted before calculating reconstruction error")
        
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        
        mse = np.mean((X - X_reconstructed) ** 2)
        return mse

def generate_sample_data():
    """
    Generate sample datasets for PCA analysis.
    
    Returns:
    --------
    dict : Dictionary containing sample datasets
    """
    # High-dimensional data with known structure
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Create correlated features
    X = np.random.randn(n_samples, n_features)
    
    # Add correlations between features
    X[:, 1] = X[:, 0] * 0.8 + np.random.randn(n_samples) * 0.2
    X[:, 2] = X[:, 0] * 0.6 + X[:, 1] * 0.4 + np.random.randn(n_samples) * 0.1
    X[:, 3] = X[:, 2] * 0.7 + np.random.randn(n_samples) * 0.3
    
    # Add some noise features
    X[:, 4:8] = np.random.randn(n_samples, 4) * 0.5
    
    # Create feature names
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    return X, feature_names

def demo_pca():
    """
    Demonstrate PCA analysis on sample data.
    """
    print("PCA Analysis Demo")
    print("=" * 50)
    
    # Generate sample data
    X, feature_names = generate_sample_data()
    print(f"Original data shape: {X.shape}")
    
    # Initialize PCA
    pca = PCAAnalysis()
    
    # Fit PCA
    pca.fit(X, feature_names)
    
    # Get variance information
    var_info = pca.get_explained_variance_info()
    print(f"\nNumber of components: {var_info['n_components']}")
    print(f"Total variance explained: {var_info['total_variance_explained']:.3f}")
    
    # Find optimal components
    optimal_components = pca.find_optimal_components(threshold=0.95)
    print(f"Components needed for 95% variance: {optimal_components}")
    
    # Plot explained variance
    pca.plot_explained_variance()
    
    # Plot first two components
    pca.plot_components_2d(X)
    
    # Plot feature importance
    pca.plot_feature_importance(n_components=3)
    
    # Calculate reconstruction error
    reconstruction_error = pca.get_reconstruction_error(X)
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    
    # Demonstrate dimensionality reduction
    X_transformed = pca.transform(X)
    print(f"Transformed data shape: {X_transformed.shape}")
    
    # Show variance explained by first few components
    print("\nVariance explained by components:")
    for i, (var_ratio, cum_var_ratio) in enumerate(zip(
        var_info['explained_variance_ratio'][:5], 
        var_info['cumulative_variance_ratio'][:5])):
        print(f"PC{i+1}: {var_ratio:.3f} (Cumulative: {cum_var_ratio:.3f})")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    demo_pca() 