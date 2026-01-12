"""
Manifold Learning Module

Nonlinear dimensionality reduction techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
from sklearn.preprocessing import StandardScaler


class ManifoldReducer:
    """
    Unified interface for manifold learning methods.

    Parameters
    ----------
    method : str
        'tsne', 'isomap', 'lle', or 'mds'
    n_components : int
        Number of dimensions (typically 2 or 3)
    **kwargs : dict
        Method-specific parameters
    """

    def __init__(self, method='tsne', n_components=2, **kwargs):
        self.method = method
        self.n_components = n_components
        self.kwargs = kwargs
        self.reducer = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.embedding_ = None

    def fit_transform(self, X):
        """
        Fit and transform data.

        Parameters
        ----------
        X : array-like or DataFrame
            Input data

        Returns
        -------
        X_embedded : array
            Embedded data
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Standardize
        X_scaled = self.scaler.fit_transform(X)

        # Create reducer
        if self.method == 'tsne':
            self.reducer = TSNE(n_components=self.n_components, **self.kwargs)
        elif self.method == 'isomap':
            self.reducer = Isomap(n_components=self.n_components, **self.kwargs)
        elif self.method == 'lle':
            self.reducer = LocallyLinearEmbedding(n_components=self.n_components, **self.kwargs)
        elif self.method == 'mds':
            self.reducer = MDS(n_components=self.n_components, **self.kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Fit and transform
        self.embedding_ = self.reducer.fit_transform(X_scaled)
        self.is_fitted = True

        return self.embedding_

    def plot_2d(self, labels=None, figsize=(10, 8), title=None):
        """
        Plot 2D embedding.

        Parameters
        ----------
        labels : array-like, optional
            Labels for coloring points
        figsize : tuple
            Figure size
        title : str, optional
            Custom title
        """
        if not self.is_fitted:
            raise ValueError("Must fit_transform first")

        if self.n_components != 2:
            raise ValueError("Only available for 2D embeddings")

        fig, ax = plt.subplots(figsize=figsize)

        if labels is not None:
            scatter = ax.scatter(self.embedding_[:, 0], self.embedding_[:, 1],
                               c=labels, cmap='tab10', alpha=0.6, s=30)
            plt.colorbar(scatter, ax=ax, label='Label')
        else:
            ax.scatter(self.embedding_[:, 0], self.embedding_[:, 1],
                      alpha=0.6, s=30)

        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')

        if title is None:
            title = f'{self.method.upper()} - 2D Embedding'
        ax.set_title(title)

        ax.grid(alpha=0.3)
        plt.tight_layout()

        return fig

    def plot_3d(self, labels=None, figsize=(10, 8), title=None):
        """
        Plot 3D embedding.

        Parameters
        ----------
        labels : array-like, optional
            Labels for coloring points
        figsize : tuple
            Figure size
        title : str, optional
            Custom title
        """
        if not self.is_fitted:
            raise ValueError("Must fit_transform first")

        if self.n_components != 3:
            raise ValueError("Only available for 3D embeddings")

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        if labels is not None:
            scatter = ax.scatter(self.embedding_[:, 0],
                               self.embedding_[:, 1],
                               self.embedding_[:, 2],
                               c=labels, cmap='tab10', alpha=0.6, s=30)
            plt.colorbar(scatter, ax=ax, label='Label')
        else:
            ax.scatter(self.embedding_[:, 0],
                      self.embedding_[:, 1],
                      self.embedding_[:, 2],
                      alpha=0.6, s=30)

        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')

        if title is None:
            title = f'{self.method.upper()} - 3D Embedding'
        ax.set_title(title)

        plt.tight_layout()

        return fig


class TSNEOptimizer:
    """
    Helper class for optimizing t-SNE parameters.

    t-SNE is sensitive to hyperparameters. This class helps find good settings.
    """

    @staticmethod
    def find_optimal_perplexity(X, perplexities=None, n_components=2, random_state=42):
        """
        Try different perplexity values and evaluate.

        Parameters
        ----------
        X : array-like
            Input data
        perplexities : list, optional
            Perplexity values to try
        n_components : int
            Number of dimensions
        random_state : int
            Random seed

        Returns
        -------
        results : dict
            Perplexity -> embedding
        """
        if perplexities is None:
            n_samples = X.shape[0]
            perplexities = [5, 10, 30, 50, min(100, n_samples // 3)]

        results = {}
        for perp in perplexities:
            if perp >= X.shape[0]:
                continue

            tsne = TSNE(n_components=n_components, perplexity=perp,
                       random_state=random_state, n_iter=1000)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            embedding = tsne.fit_transform(X_scaled)
            results[perp] = embedding

        return results

    @staticmethod
    def plot_perplexity_comparison(results, labels=None, figsize=(15, 10)):
        """
        Plot embeddings for different perplexities.

        Parameters
        ----------
        results : dict
            From find_optimal_perplexity
        labels : array-like, optional
            Labels for coloring
        figsize : tuple
            Figure size
        """
        n_plots = len(results)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_plots > 1 else [axes]

        for idx, (perp, embedding) in enumerate(results.items()):
            ax = axes[idx]

            if labels is not None:
                scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                                   c=labels, cmap='tab10', alpha=0.6, s=20)
            else:
                ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=20)

            ax.set_title(f'Perplexity = {perp}')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.grid(alpha=0.3)

        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        return fig


def compare_manifold_methods(X, labels=None, methods=None, n_components=2, figsize=(15, 10)):
    """
    Compare multiple manifold learning methods.

    Parameters
    ----------
    X : array-like
        Input data
    labels : array-like, optional
        Labels for coloring
    methods : list, optional
        Methods to compare
    n_components : int
        Number of dimensions
    figsize : tuple
        Figure size

    Returns
    -------
    embeddings : dict
        Method -> embedding
    """
    if methods is None:
        methods = ['tsne', 'isomap', 'lle', 'mds']

    embeddings = {}

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, method in enumerate(methods):
        if idx >= len(axes):
            break

        try:
            reducer = ManifoldReducer(method=method, n_components=n_components,
                                     random_state=42)
            embedding = reducer.fit_transform(X)
            embeddings[method] = embedding

            ax = axes[idx]
            if labels is not None:
                scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                                   c=labels, cmap='tab10', alpha=0.6, s=20)
            else:
                ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=20)

            ax.set_title(f'{method.upper()}')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.grid(alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, f'{method} failed:\n{str(e)}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{method.upper()} - Error')

    plt.tight_layout()

    return embeddings, fig
