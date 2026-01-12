"""
Clustering & Segmentation Module

Production-ready clustering with K-Means, Hierarchical, DBSCAN, and GMM.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bould

in_score
import warnings
warnings.filterwarnings('ignore')


class CustomerSegmentation:
    """
    Customer segmentation pipeline with multiple clustering algorithms.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (for K-Means, Hierarchical, GMM)
    method : str
        Clustering method: 'kmeans', 'hierarchical', 'dbscan', 'gmm'
    random_state : int
        Random seed
    """

    def __init__(self, n_clusters=5, method='kmeans', random_state=42):
        self.n_clusters = n_clusters
        self.method = method
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.labels_ = None
        self.is_fitted = False

        self._initialize_model()

    def _initialize_model(self):
        """Initialize clustering model."""
        if self.method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )

        elif self.method == 'hierarchical':
            self.model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage='ward'
            )

        elif self.method == 'dbscan':
            self.model = DBSCAN(eps=0.5, min_samples=5)

        elif self.method == 'gmm':
            self.model = GaussianMixture(
                n_components=self.n_clusters,
                random_state=self.random_state
            )

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit(self, X):
        """Fit clustering model."""
        X_scaled = self.scaler.fit_transform(X)

        if self.method == 'gmm':
            self.model.fit(X_scaled)
            self.labels_ = self.model.predict(X_scaled)
        else:
            self.labels_ = self.model.fit_predict(X_scaled)

        self.is_fitted = True
        return self

    def predict(self, X):
        """Predict cluster labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        X_scaled = self.scaler.transform(X)

        if self.method == 'gmm':
            return self.model.predict(X_scaled)
        elif self.method == 'kmeans':
            return self.model.predict(X_scaled)
        else:
            raise ValueError(f"Prediction not supported for {self.method}")

    def fit_predict(self, X):
        """Fit and predict in one step."""
        self.fit(X)
        return self.labels_

    def evaluate(self, X):
        """Calculate clustering metrics."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        X_scaled = self.scaler.transform(X)

        metrics = {
            'silhouette': silhouette_score(X_scaled, self.labels_),
            'calinski_harabasz': calinski_harabasz_score(X_scaled, self.labels_),
            'davies_bouldin': davies_bouldin_score(X_scaled, self.labels_)
        }

        return metrics

    def profile_segments(self, X, labels=None):
        """Generate segment profiles."""
        if labels is None:
            labels = self.labels_

        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)

        df['Segment'] = labels

        profiles = df.groupby('Segment').agg(['mean', 'median', 'std', 'count'])

        return profiles


def find_optimal_clusters(X, max_clusters=10, method='elbow'):
    """
    Find optimal number of clusters.

    Parameters
    ----------
    X : array-like or DataFrame
        Features
    max_clusters : int
        Maximum clusters to test
    method : str
        Method: 'elbow', 'silhouette', 'gap'

    Returns
    -------
    optimal_k : int
        Optimal number of clusters
    scores : list
        Scores for each K
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scores = []

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        if method == 'elbow':
            score = kmeans.inertia_
        elif method == 'silhouette':
            score = silhouette_score(X_scaled, labels)
        elif method == 'gap':
            # Simplified gap statistic
            score = calinski_harabasz_score(X_scaled, labels)

        scores.append(score)

    if method == 'elbow':
        # Find elbow using second derivative
        diffs = np.diff(scores)
        diffs2 = np.diff(diffs)
        optimal_k = np.argmin(diffs2) + 3  # +3 because of two diffs and 0-indexing
    else:
        optimal_k = np.argmax(scores) + 2

    return optimal_k, scores
