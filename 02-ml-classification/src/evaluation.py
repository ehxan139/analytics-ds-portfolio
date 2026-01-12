"""
Model Evaluation

Comprehensive evaluation metrics, visualizations, and business interpretation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from dataclasses import dataclass


@dataclass
class ClassificationMetrics:
    """Container for classification evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: np.ndarray

    def summary(self):
        """Print formatted summary."""
        print("\n" + "="*60)
        print("CLASSIFICATION METRICS")
        print("="*60)
        print(f"Accuracy:  {self.accuracy:.4f}")
        print(f"Precision: {self.precision:.4f}")
        print(f"Recall:    {self.recall:.4f}")
        print(f"F1 Score:  {self.f1:.4f}")
        print(f"ROC-AUC:   {self.roc_auc:.4f}")
        print(f"PR-AUC:    {self.pr_auc:.4f}")
        print("\nConfusion Matrix:")
        print(self.confusion_matrix)
        print("="*60)


def evaluate_classifier(y_true, y_pred, y_pred_proba=None):
    """
    Comprehensive classifier evaluation.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities

    Returns
    -------
    metrics : ClassificationMetrics
        Evaluation metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Probability-based metrics
    if y_pred_proba is not None:
        if y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]

        roc_auc = roc_auc_score(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
    else:
        roc_auc = None
        pr_auc = None

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=cm
    )


def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False, figsize=(8, 6)):
    """
    Plot confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : list, optional
        Class labels
    normalize : bool
        Normalize by row (true labels)
    figsize : tuple
        Figure size
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))

    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_pred_proba, figsize=(8, 6)):
    """
    Plot ROC curve.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    figsize : tuple
        Figure size
    """
    if y_pred_proba.ndim == 2:
        y_pred_proba = y_pred_proba[:, 1]

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_precision_recall_curve(y_true, y_pred_proba, figsize=(8, 6)):
    """
    Plot Precision-Recall curve.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    figsize : tuple
        Figure size
    """
    if y_pred_proba.ndim == 2:
        y_pred_proba = y_pred_proba[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {pr_auc:.3f})')

    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    ax.plot([0, 1], [baseline, baseline], 'k--', linewidth=1, label=f'Baseline ({baseline:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_threshold_metrics(y_true, y_pred_proba, figsize=(10, 6)):
    """
    Plot precision, recall, F1 vs threshold.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    figsize : tuple
        Figure size
    """
    if y_pred_proba.ndim == 2:
        y_pred_proba = y_pred_proba[:, 1]

    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(thresholds, precisions, label='Precision', linewidth=2)
    ax.plot(thresholds, recalls, label='Recall', linewidth=2)
    ax.plot(thresholds, f1_scores, label='F1 Score', linewidth=2)

    # Optimal F1 threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    ax.axvline(optimal_threshold, color='red', linestyle='--',
               label=f'Optimal Threshold ({optimal_threshold:.2f})')

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Classification Metrics vs Threshold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """
    Find optimal classification threshold.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    metric : str
        Metric to optimize: 'f1', 'precision', 'recall'

    Returns
    -------
    optimal_threshold : float
        Optimal threshold
    optimal_score : float
        Score at optimal threshold
    """
    if y_pred_proba.ndim == 2:
        y_pred_proba = y_pred_proba[:, 1]

    thresholds = np.linspace(0, 1, 1000)
    scores = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        scores.append(score)

    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_score = scores[optimal_idx]

    return optimal_threshold, optimal_score


def calculate_business_metrics(y_true, y_pred, cost_fp=1.0, cost_fn=1.0, revenue_tp=0.0):
    """
    Calculate business-focused metrics.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    cost_fp : float
        Cost of false positive
    cost_fn : float
        Cost of false negative
    revenue_tp : float
        Revenue from true positive

    Returns
    -------
    business_metrics : dict
        Business impact metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    total_cost = fp * cost_fp + fn * cost_fn
    total_revenue = tp * revenue_tp
    net_profit = total_revenue - total_cost

    return {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn),
        'total_cost': total_cost,
        'total_revenue': total_revenue,
        'net_profit': net_profit,
        'roi': (net_profit / total_cost * 100) if total_cost > 0 else float('inf')
    }


def plot_calibration_curve(y_true, y_pred_proba, n_bins=10, figsize=(8, 6)):
    """
    Plot calibration curve.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    n_bins : int
        Number of bins
    figsize : tuple
        Figure size
    """
    if y_pred_proba.ndim == 2:
        y_pred_proba = y_pred_proba[:, 1]

    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Calculate empirical probabilities
    empirical_probs = []
    for i in range(n_bins):
        mask = (y_pred_proba >= bins[i]) & (y_pred_proba < bins[i+1])
        if mask.sum() > 0:
            empirical_probs.append(y_true[mask].mean())
        else:
            empirical_probs.append(np.nan)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(bin_centers, empirical_probs, 'o-', linewidth=2, label='Model')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectly Calibrated')

    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Empirical Probability')
    ax.set_title('Calibration Curve')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def create_evaluation_report(y_true, y_pred, y_pred_proba=None, output_dir='evaluation_results/'):
    """
    Create comprehensive evaluation report with all visualizations.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities
    output_dir : str
        Output directory for plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Metrics
    metrics = evaluate_classifier(y_true, y_pred, y_pred_proba)
    metrics.summary()

    # Confusion matrix
    fig = plot_confusion_matrix(y_true, y_pred)
    fig.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    if y_pred_proba is not None:
        # ROC curve
        fig = plot_roc_curve(y_true, y_pred_proba)
        fig.savefig(f'{output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

        # PR curve
        fig = plot_precision_recall_curve(y_true, y_pred_proba)
        fig.savefig(f'{output_dir}/pr_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Threshold analysis
        fig = plot_threshold_metrics(y_true, y_pred_proba)
        fig.savefig(f'{output_dir}/threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Calibration
        fig = plot_calibration_curve(y_true, y_pred_proba)
        fig.savefig(f'{output_dir}/calibration_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\nEvaluation report saved to {output_dir}")
