"""
PCA Analysis Module

Comprehensive PCA implementation with interpretation and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAAnalyzer:
    """
    PCA analysis with interpretation tools.
    
    Parameters
    ----------
    n_components : int or float
        Number of components or variance threshold (0-1)
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.pca = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
    
    def fit_transform(self, X):
        """
        Fit PCA and transform data.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Input data
        
        Returns
        -------
        X_transformed : array
            Transformed data
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA
        self.pca = PCA(n_components=self.n_components)
        X_transformed = self.pca.fit_transform(X_scaled)
        
        self.is_fitted = True
        
        return X_transformed
    
    def transform(self, X):
        """Transform new data."""
        if not self.is_fitted:
            raise ValueError("Must fit first")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
    
    def inverse_transform(self, X_transformed):
        """Reconstruct original data."""
        X_scaled = self.pca.inverse_transform(X_transformed)
        return self.scaler.inverse_transform(X_scaled)
    
    def get_explained_variance(self):
        """Get explained variance per component."""
        if not self.is_fitted:
            raise ValueError("Must fit first")
        
        return self.pca.explained_variance_ratio_
    
    def get_cumulative_variance(self):
        """Get cumulative explained variance."""
        return np.cumsum(self.pca.explained_variance_ratio_)
    
    def find_optimal_components(self, threshold=0.85):
        """
        Find number of components for variance threshold.
        
        Parameters
        ----------
        threshold : float
            Variance threshold (0-1)
        
        Returns
        -------
        n_components : int
            Optimal number of components
        """
        cumvar = self.get_cumulative_variance()
        return np.argmax(cumvar >= threshold) + 1
    
    def get_component_loadings(self, component_idx=0):
        """
        Get feature loadings for a component.
        
        Parameters
        ----------
        component_idx : int
            Component index
        
        Returns
        -------
        loadings : DataFrame
            Feature loadings sorted by absolute value
        """
        if not self.is_fitted:
            raise ValueError("Must fit first")
        
        loadings = self.pca.components_[component_idx]
        
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(loadings))]
        else:
            feature_names = self.feature_names
        
        loading_df = pd.DataFrame({
            'feature': feature_names,
            'loading': loadings,
            'abs_loading': np.abs(loadings)
        }).sort_values('abs_loading', ascending=False)
        
        return loading_df
    
    def plot_explained_variance(self, figsize=(12, 5)):
        """Plot explained variance and cumulative variance."""
        if not self.is_fitted:
            raise ValueError("Must fit first")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Individual variance
        n_components = len(self.pca.explained_variance_ratio_)
        ax1.bar(range(1, n_components + 1), self.pca.explained_variance_ratio_)
        ax1.set_xlabel('Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Scree Plot')
        ax1.grid(alpha=0.3)
        
        # Cumulative variance
        cumvar = self.get_cumulative_variance()
        ax2.plot(range(1, n_components + 1), cumvar, 'o-', linewidth=2)
        ax2.axhline(y=0.85, color='r', linestyle='--', label='85% threshold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Variance Explained')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_component_loadings(self, component_idx=0, top_n=20, figsize=(10, 8)):
        """Plot top feature loadings for a component."""
        loadings = self.get_component_loadings(component_idx).head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['red' if x < 0 else 'blue' for x in loadings['loading']]
        ax.barh(range(len(loadings)), loadings['loading'], color=colors)
        ax.set_yticks(range(len(loadings)))
        ax.set_yticklabels(loadings['feature'])
        ax.set_xlabel('Loading')
        ax.set_title(f'Component {component_idx + 1} - Top {top_n} Features\n'
                    f'Variance Explained: {self.pca.explained_variance_ratio_[component_idx]:.1%}')
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def plot_biplot(self, X_transformed, component_x=0, component_y=1, figsize=(10, 8)):
        """
        Create biplot showing samples and feature vectors.
        
        Parameters
        ----------
        X_transformed : array
            Transformed data from fit_transform
        component_x : int
            Component for x-axis
        component_y : int
            Component for y-axis
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot samples
        ax.scatter(X_transformed[:, component_x], X_transformed[:, component_y], 
                  alpha=0.5, s=30)
        
        # Plot feature vectors
        if self.feature_names is not None:
            loadings = self.pca.components_[[component_x, component_y]].T
            
            # Scale arrows for visibility
            scale = np.max(np.abs(X_transformed[:, [component_x, component_y]])) / np.max(np.abs(loadings))
            
            for i, feature in enumerate(self.feature_names[:20]):  # Limit to 20 for clarity
                ax.arrow(0, 0, loadings[i, 0] * scale, loadings[i, 1] * scale,
                        color='red', alpha=0.5, head_width=0.1)
                ax.text(loadings[i, 0] * scale * 1.1, loadings[i, 1] * scale * 1.1,
                       feature, color='red', fontsize=8)
        
        ax.set_xlabel(f'PC{component_x + 1} ({self.pca.explained_variance_ratio_[component_x]:.1%})')
        ax.set_ylabel(f'PC{component_y + 1} ({self.pca.explained_variance_ratio_[component_y]:.1%})')
        ax.set_title('PCA Biplot')
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        
        plt.tight_layout()
        return fig
    
    def calculate_reconstruction_error(self, X):
        """Calculate reconstruction error for original data."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        X_transformed = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        
        mse = np.mean((X_scaled - X_reconstructed) ** 2)
        
        return mse
