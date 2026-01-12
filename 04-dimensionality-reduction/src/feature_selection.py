"""
Feature Selection Module

Comprehensive feature selection methods for dimensionality reduction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, f_regression,
    mutual_info_classif, mutual_info_regression, RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV


class FeatureSelector:
    """
    Unified interface for multiple feature selection methods.

    Parameters
    ----------
    method : str
        Selection method: 'variance', 'univariate', 'mutual_info', 'rfe', 'lasso', 'tree'
    task : str
        'classification' or 'regression'
    n_features : int
        Number of features to select
    """

    def __init__(self, method='mutual_info', task='classification', n_features=10):
        self.method = method
        self.task = task
        self.n_features = n_features
        self.selector = None
        self.feature_names = None
        self.scores_ = None
        self.is_fitted = False

    def fit(self, X, y=None):
        """
        Fit feature selector.

        Parameters
        ----------
        X : array-like or DataFrame
            Features
        y : array-like, optional
            Target (required for supervised methods)
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        if self.method == 'variance':
            self._fit_variance_threshold(X)
        elif self.method == 'univariate':
            self._fit_univariate(X, y)
        elif self.method == 'mutual_info':
            self._fit_mutual_info(X, y)
        elif self.method == 'rfe':
            self._fit_rfe(X, y)
        elif self.method == 'lasso':
            self._fit_lasso(X, y)
        elif self.method == 'tree':
            self._fit_tree(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.is_fitted = True
        return self

    def _fit_variance_threshold(self, X):
        """Variance threshold - remove low variance features."""
        # Determine threshold automatically
        variances = np.var(X, axis=0)
        threshold = np.percentile(variances, (1 - self.n_features / X.shape[1]) * 100)

        self.selector = VarianceThreshold(threshold=threshold)
        self.selector.fit(X)
        self.scores_ = variances

    def _fit_univariate(self, X, y):
        """Univariate statistical tests (ANOVA F-test)."""
        if y is None:
            raise ValueError("y required for univariate selection")

        score_func = f_classif if self.task == 'classification' else f_regression
        self.selector = SelectKBest(score_func=score_func, k=self.n_features)
        self.selector.fit(X, y)
        self.scores_ = self.selector.scores_

    def _fit_mutual_info(self, X, y):
        """Mutual information between features and target."""
        if y is None:
            raise ValueError("y required for mutual info selection")

        mi_func = mutual_info_classif if self.task == 'classification' else mutual_info_regression
        self.selector = SelectKBest(score_func=mi_func, k=self.n_features)
        self.selector.fit(X, y)
        self.scores_ = self.selector.scores_

    def _fit_rfe(self, X, y):
        """Recursive Feature Elimination."""
        if y is None:
            raise ValueError("y required for RFE")

        if self.task == 'classification':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)

        self.selector = RFE(estimator=estimator, n_features_to_select=self.n_features)
        self.selector.fit(X, y)
        # RFE ranking: 1 = selected, >1 = not selected
        self.scores_ = 1.0 / self.selector.ranking_

    def _fit_lasso(self, X, y):
        """L1 regularization (Lasso) for feature selection."""
        if y is None:
            raise ValueError("y required for Lasso")

        lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
        lasso.fit(X, y)

        self.selector = SelectFromModel(lasso, max_features=self.n_features, prefit=True)
        self.scores_ = np.abs(lasso.coef_)

    def _fit_tree(self, X, y):
        """Tree-based feature importance."""
        if y is None:
            raise ValueError("y required for tree-based selection")

        if self.task == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        model.fit(X, y)

        self.selector = SelectFromModel(model, max_features=self.n_features, prefit=True)
        self.scores_ = model.feature_importances_

    def transform(self, X):
        """Transform data to selected features."""
        if not self.is_fitted:
            raise ValueError("Must fit first")

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        X_selected = self.selector.transform(X_array)

        # Return DataFrame if input was DataFrame
        if isinstance(X, pd.DataFrame):
            selected_features = self.get_selected_features()
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index)

        return X_selected

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    def get_selected_features(self):
        """Get names of selected features."""
        if not self.is_fitted:
            raise ValueError("Must fit first")

        mask = self.selector.get_support()
        return [f for f, selected in zip(self.feature_names, mask) if selected]

    def get_feature_scores(self):
        """
        Get feature importance scores.

        Returns
        -------
        scores_df : DataFrame
            Features with scores, sorted by importance
        """
        if not self.is_fitted:
            raise ValueError("Must fit first")

        mask = self.selector.get_support()

        scores_df = pd.DataFrame({
            'feature': self.feature_names,
            'score': self.scores_,
            'selected': mask
        }).sort_values('score', ascending=False)

        return scores_df

    def plot_feature_scores(self, top_n=20, figsize=(10, 8)):
        """Plot top feature scores."""
        scores_df = self.get_feature_scores().head(top_n)

        fig, ax = plt.subplots(figsize=figsize)

        colors = ['green' if s else 'lightgray' for s in scores_df['selected']]
        ax.barh(range(len(scores_df)), scores_df['score'], color=colors)
        ax.set_yticks(range(len(scores_df)))
        ax.set_yticklabels(scores_df['feature'])
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Feature Importance - {self.method.upper()}\n'
                    f'Top {top_n} Features (green = selected)')
        ax.grid(alpha=0.3, axis='x')

        plt.tight_layout()
        return fig


def compare_feature_selectors(X, y, task='classification', n_features=10):
    """
    Compare multiple feature selection methods.

    Parameters
    ----------
    X : array-like or DataFrame
        Features
    y : array-like
        Target
    task : str
        'classification' or 'regression'
    n_features : int
        Number of features to select

    Returns
    -------
    comparison_df : DataFrame
        Selected features by each method
    """
    methods = ['univariate', 'mutual_info', 'rfe', 'lasso', 'tree']

    results = {}
    for method in methods:
        try:
            selector = FeatureSelector(method=method, task=task, n_features=n_features)
            selector.fit(X, y)
            results[method] = selector.get_selected_features()
        except Exception as e:
            print(f"Error with {method}: {e}")
            results[method] = []

    # Create comparison DataFrame
    if isinstance(X, pd.DataFrame):
        all_features = X.columns.tolist()
    else:
        all_features = [f"feature_{i}" for i in range(X.shape[1])]

    comparison_df = pd.DataFrame(index=all_features)
    for method, selected in results.items():
        comparison_df[method] = [f in selected for f in all_features]

    # Add consensus column
    comparison_df['n_methods'] = comparison_df.sum(axis=1)
    comparison_df = comparison_df.sort_values('n_methods', ascending=False)

    return comparison_df


def plot_feature_selection_comparison(comparison_df, figsize=(12, 10)):
    """Plot heatmap comparing feature selection methods."""
    fig, ax = plt.subplots(figsize=figsize)

    # Select features that were chosen by at least one method
    selected_features = comparison_df[comparison_df['n_methods'] > 0].copy()

    # Plot heatmap (excluding n_methods column)
    sns.heatmap(selected_features.iloc[:, :-1].astype(int),
                cmap='RdYlGn', cbar_kws={'label': 'Selected'},
                linewidths=0.5, ax=ax)

    ax.set_xlabel('Selection Method')
    ax.set_ylabel('Feature')
    ax.set_title('Feature Selection Comparison\n(Green = Selected by Method)')

    plt.tight_layout()
    return fig
