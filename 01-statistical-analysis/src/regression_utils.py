"""
Regression Analysis Utilities

Linear and multiple regression with diagnostics and business interpretation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RegressionResults:
    """Container for regression analysis results."""
    coefficients: pd.Series
    intercept: float
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_pvalue: float
    residuals: np.ndarray
    predictions: np.ndarray
    rmse: float
    mae: float
    feature_pvalues: pd.Series
    confidence_intervals: pd.DataFrame

    def summary(self):
        """Print formatted regression summary."""
        print("\n" + "="*70)
        print("REGRESSION ANALYSIS SUMMARY")
        print("="*70)
        print(f"\nR²: {self.r_squared:.4f}")
        print(f"Adjusted R²: {self.adj_r_squared:.4f}")
        print(f"F-statistic: {self.f_statistic:.4f} (p-value: {self.f_pvalue:.4e})")
        print(f"RMSE: {self.rmse:.4f}")
        print(f"MAE: {self.mae:.4f}")

        print("\n" + "-"*70)
        print("COEFFICIENTS")
        print("-"*70)
        print(f"{'Feature':<20} {'Coefficient':>12} {'p-value':>12} {'95% CI':>25}")
        print("-"*70)

        for feat in self.coefficients.index:
            coef = self.coefficients[feat]
            pval = self.feature_pvalues[feat]
            ci_lower = self.confidence_intervals.loc[feat, 'lower']
            ci_upper = self.confidence_intervals.loc[feat, 'upper']
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"{feat:<20} {coef:>12.4f} {pval:>12.4e} [{ci_lower:>8.4f}, {ci_upper:>8.4f}] {sig}")

        print("-"*70)
        print("Significance codes: *** p<0.001, ** p<0.01, * p<0.05")
        print("="*70)


def linear_regression_analysis(X, y, feature_names=None, alpha=0.05):
    """
    Perform comprehensive linear regression analysis.

    Parameters
    ----------
    X : array-like or DataFrame
        Feature matrix
    y : array-like
        Target variable
    feature_names : list, optional
        Feature names for reporting
    alpha : float
        Significance level for confidence intervals

    Returns
    -------
    results : RegressionResults
        Comprehensive regression results
    """
    # Convert to numpy arrays
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    if feature_names is None:
        feature_names = [f"X{i+1}" for i in range(X.shape[1])]

    # Fit model
    model = LinearRegression()
    model.fit(X, y)

    # Predictions
    y_pred = model.predict(X)
    residuals = y - y_pred

    # Model statistics
    n = len(y)
    k = X.shape[1]

    r2 = r2_score(y, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

    # RMSE and MAE
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    # F-statistic
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum(residuals**2)
    ss_regression = ss_total - ss_residual

    ms_regression = ss_regression / k
    ms_residual = ss_residual / (n - k - 1)
    f_stat = ms_regression / ms_residual
    f_pvalue = 1 - stats.f.cdf(f_stat, k, n - k - 1)

    # Standard errors and t-statistics
    mse = ss_residual / (n - k - 1)
    var_coef = mse * np.linalg.inv(X.T @ X).diagonal()
    se_coef = np.sqrt(var_coef)

    t_stats = model.coef_ / se_coef
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))

    # Confidence intervals
    t_crit = stats.t.ppf(1 - alpha/2, n - k - 1)
    ci_lower = model.coef_ - t_crit * se_coef
    ci_upper = model.coef_ + t_crit * se_coef

    # Package results
    coefficients = pd.Series(model.coef_, index=feature_names)
    feature_pvalues = pd.Series(p_values, index=feature_names)
    confidence_intervals = pd.DataFrame({
        'lower': ci_lower,
        'upper': ci_upper
    }, index=feature_names)

    return RegressionResults(
        coefficients=coefficients,
        intercept=model.intercept_,
        r_squared=r2,
        adj_r_squared=adj_r2,
        f_statistic=f_stat,
        f_pvalue=f_pvalue,
        residuals=residuals,
        predictions=y_pred,
        rmse=rmse,
        mae=mae,
        feature_pvalues=feature_pvalues,
        confidence_intervals=confidence_intervals
    )


def check_regression_assumptions(X, residuals, y_pred):
    """
    Check linear regression assumptions.

    Parameters
    ----------
    X : array-like
        Feature matrix
    residuals : array-like
        Model residuals
    y_pred : array-like
        Predicted values

    Returns
    -------
    diagnostics : dict
        Diagnostic test results
    """
    diagnostics = {}

    # 1. Normality of residuals (Shapiro-Wilk)
    stat, p = stats.shapiro(residuals)
    diagnostics['normality'] = {
        'test': 'Shapiro-Wilk',
        'statistic': stat,
        'p_value': p,
        'is_normal': p > 0.05
    }

    # 2. Homoscedasticity (Breusch-Pagan test approximation)
    # Test if residuals squared relate to predictions
    from scipy.stats import pearsonr
    corr, p = pearsonr(y_pred, residuals**2)
    diagnostics['homoscedasticity'] = {
        'correlation': corr,
        'p_value': p,
        'is_homoscedastic': p > 0.05
    }

    # 3. Autocorrelation (Durbin-Watson statistic)
    diff_residuals = np.diff(residuals)
    dw_stat = np.sum(diff_residuals**2) / np.sum(residuals**2)
    diagnostics['autocorrelation'] = {
        'durbin_watson': dw_stat,
        'interpretation': 'No autocorrelation' if 1.5 < dw_stat < 2.5 else 'Possible autocorrelation'
    }

    # 4. Multicollinearity (VIF)
    vif_values = []
    for i in range(X.shape[1]):
        X_i = X[:, i]
        X_not_i = np.delete(X, i, axis=1)
        r2_i = r2_score(X_i, LinearRegression().fit(X_not_i, X_i).predict(X_not_i))
        vif = 1 / (1 - r2_i) if r2_i < 0.999 else float('inf')
        vif_values.append(vif)

    diagnostics['multicollinearity'] = {
        'vif_values': vif_values,
        'max_vif': max(vif_values),
        'has_multicollinearity': max(vif_values) > 10
    }

    return diagnostics


def calculate_prediction_intervals(X, y, X_new, alpha=0.05):
    """
    Calculate prediction intervals for new observations.

    Parameters
    ----------
    X : array-like
        Training features
    y : array-like
        Training target
    X_new : array-like
        New observations to predict
    alpha : float
        Significance level

    Returns
    -------
    predictions : array
        Point predictions
    lower_bounds : array
        Lower prediction interval bounds
    upper_bounds : array
        Upper prediction interval bounds
    """
    # Fit model
    model = LinearRegression()
    model.fit(X, y)

    # Predictions
    y_pred_new = model.predict(X_new)

    # Calculate prediction standard error
    y_pred_train = model.predict(X)
    residuals = y - y_pred_train
    n = len(y)
    k = X.shape[1]

    mse = np.sum(residuals**2) / (n - k - 1)

    # Leverage for new points
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    X_new_with_intercept = np.column_stack([np.ones(len(X_new)), X_new])

    hat_matrix_diag = np.sum(X_new_with_intercept @ np.linalg.inv(X_with_intercept.T @ X_with_intercept) * X_new_with_intercept, axis=1)

    # Prediction standard error
    se_pred = np.sqrt(mse * (1 + hat_matrix_diag))

    # Prediction intervals
    t_crit = stats.t.ppf(1 - alpha/2, n - k - 1)
    lower_bounds = y_pred_new - t_crit * se_pred
    upper_bounds = y_pred_new + t_crit * se_pred

    return y_pred_new, lower_bounds, upper_bounds


def feature_importance_regression(X, y, feature_names=None):
    """
    Calculate feature importance for regression.

    Parameters
    ----------
    X : array-like or DataFrame
        Feature matrix
    y : array-like
        Target variable
    feature_names : list, optional
        Feature names

    Returns
    -------
    importance : pd.DataFrame
        Feature importance metrics
    """
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    if feature_names is None:
        feature_names = [f"X{i+1}" for i in range(X.shape[1])]

    # Standardize features for fair comparison
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    y_std = (y - y.mean()) / y.std()

    # Fit model
    model = LinearRegression()
    model.fit(X_std, y_std)

    # Standardized coefficients (beta weights)
    beta_weights = model.coef_

    # Absolute importance
    abs_importance = np.abs(beta_weights)

    # Relative importance (normalized to sum to 1)
    rel_importance = abs_importance / abs_importance.sum()

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'beta_weight': beta_weights,
        'abs_importance': abs_importance,
        'rel_importance_pct': rel_importance * 100
    }).sort_values('abs_importance', ascending=False)

    return importance_df


def stepwise_selection(X, y, feature_names=None, threshold_in=0.05, threshold_out=0.1, verbose=True):
    """
    Perform stepwise feature selection for regression.

    Parameters
    ----------
    X : array-like or DataFrame
        Feature matrix
    y : array-like
        Target variable
    feature_names : list, optional
        Feature names
    threshold_in : float
        P-value threshold for adding features
    threshold_out : float
        P-value threshold for removing features
    verbose : bool
        Print selection progress

    Returns
    -------
    selected_features : list
        Selected feature indices
    """
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    if feature_names is None:
        feature_names = [f"X{i+1}" for i in range(X.shape[1])]

    n_features = X.shape[1]
    selected = []

    while True:
        changed = False

        # Forward step: add best feature
        excluded = [i for i in range(n_features) if i not in selected]
        best_pval = threshold_in
        best_feature = None

        for feature in excluded:
            current_features = selected + [feature]
            X_subset = X[:, current_features]

            results = linear_regression_analysis(X_subset, y)
            pval = results.feature_pvalues.iloc[-1]

            if pval < best_pval:
                best_pval = pval
                best_feature = feature

        if best_feature is not None:
            selected.append(best_feature)
            if verbose:
                print(f"Added feature: {feature_names[best_feature]} (p={best_pval:.4f})")
            changed = True

        # Backward step: remove worst feature
        if len(selected) > 0:
            X_subset = X[:, selected]
            results = linear_regression_analysis(X_subset, y)

            worst_pval = 0
            worst_idx = None

            for i, pval in enumerate(results.feature_pvalues):
                if pval > worst_pval:
                    worst_pval = pval
                    worst_idx = i

            if worst_pval > threshold_out:
                removed_feature = selected.pop(worst_idx)
                if verbose:
                    print(f"Removed feature: {feature_names[removed_feature]} (p={worst_pval:.4f})")
                changed = True

        if not changed:
            break

    return [feature_names[i] for i in selected]
