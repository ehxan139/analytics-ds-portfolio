"""
Hypothesis Testing Module

Statistical tests for business decision-making with effect sizes and confidence intervals.
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class TestResult:
    """Container for hypothesis test results."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    conclusion: str
    
    def summary(self):
        """Print formatted summary."""
        print(f"\n{self.test_name}")
        print("=" * 60)
        print(f"Test Statistic: {self.statistic:.4f}")
        print(f"P-value: {self.p_value:.4f}")
        if self.effect_size is not None:
            print(f"Effect Size: {self.effect_size:.4f}")
        if self.confidence_interval is not None:
            print(f"95% CI: ({self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f})")
        print(f"\nConclusion: {self.conclusion}")
        print("=" * 60)


def check_normality(data, alpha=0.05):
    """
    Test normality using Shapiro-Wilk test.
    
    Parameters
    ----------
    data : array-like
        Data to test
    alpha : float
        Significance level
    
    Returns
    -------
    is_normal : bool
        True if data appears normal
    p_value : float
        Test p-value
    """
    stat, p_value = stats.shapiro(data)
    is_normal = p_value > alpha
    
    return is_normal, p_value


def check_equal_variance(group1, group2, alpha=0.05):
    """
    Test equality of variances using Levene's test.
    
    Parameters
    ----------
    group1, group2 : array-like
        Groups to compare
    alpha : float
        Significance level
    
    Returns
    -------
    equal_var : bool
        True if variances are equal
    p_value : float
        Test p-value
    """
    stat, p_value = stats.levene(group1, group2)
    equal_var = p_value > alpha
    
    return equal_var, p_value


def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size.
    
    Parameters
    ----------
    group1, group2 : array-like
        Groups to compare
    
    Returns
    -------
    d : float
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return d


def two_sample_ttest(group1, group2, alpha=0.05, equal_var=None):
    """
    Perform two-sample t-test with effect size.
    
    Parameters
    ----------
    group1, group2 : array-like
        Groups to compare
    alpha : float
        Significance level
    equal_var : bool, optional
        Assume equal variance? If None, test automatically
    
    Returns
    -------
    result : TestResult
        Test results with effect size and CI
    """
    # Check assumptions
    norm1, _ = check_normality(group1)
    norm2, _ = check_normality(group2)
    
    if equal_var is None:
        equal_var, _ = check_equal_variance(group1, group2)
    
    # Perform t-test
    statistic, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
    
    # Effect size
    d = cohens_d(group1, group2)
    
    # Confidence interval for mean difference
    mean_diff = np.mean(group1) - np.mean(group2)
    se_diff = np.sqrt(np.var(group1, ddof=1)/len(group1) + np.var(group2, ddof=1)/len(group2))
    df = len(group1) + len(group2) - 2
    t_crit = stats.t.ppf(1 - alpha/2, df)
    ci = (mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff)
    
    # Conclusion
    if p_value < alpha:
        conclusion = f"Significant difference detected (p={p_value:.4f} < {alpha})"
    else:
        conclusion = f"No significant difference (p={p_value:.4f} >= {alpha})"
    
    if not (norm1 and norm2):
        conclusion += "\nWarning: Normality assumption violated. Consider Mann-Whitney U test."
    
    return TestResult(
        test_name="Two-Sample T-Test",
        statistic=statistic,
        p_value=p_value,
        effect_size=d,
        confidence_interval=ci,
        conclusion=conclusion
    )


def paired_ttest(before, after, alpha=0.05):
    """
    Perform paired t-test for before/after comparisons.
    
    Parameters
    ----------
    before, after : array-like
        Paired observations
    alpha : float
        Significance level
    
    Returns
    -------
    result : TestResult
        Test results
    """
    differences = np.array(after) - np.array(before)
    
    # Check normality of differences
    is_normal, _ = check_normality(differences)
    
    # Perform test
    statistic, p_value = stats.ttest_rel(before, after)
    
    # Effect size
    d = np.mean(differences) / np.std(differences, ddof=1)
    
    # CI for mean difference
    se = stats.sem(differences)
    df = len(differences) - 1
    t_crit = stats.t.ppf(1 - alpha/2, df)
    ci = (np.mean(differences) - t_crit * se, np.mean(differences) + t_crit * se)
    
    # Conclusion
    if p_value < alpha:
        conclusion = f"Significant change detected (p={p_value:.4f} < {alpha})"
    else:
        conclusion = f"No significant change (p={p_value:.4f} >= {alpha})"
    
    if not is_normal:
        conclusion += "\nWarning: Normality assumption violated. Consider Wilcoxon signed-rank test."
    
    return TestResult(
        test_name="Paired T-Test",
        statistic=statistic,
        p_value=p_value,
        effect_size=d,
        confidence_interval=ci,
        conclusion=conclusion
    )


def one_way_anova(*groups, alpha=0.05):
    """
    Perform one-way ANOVA for multiple group comparison.
    
    Parameters
    ----------
    *groups : array-like
        Groups to compare (2 or more)
    alpha : float
        Significance level
    
    Returns
    -------
    result : TestResult
        ANOVA results with eta-squared
    """
    # Perform ANOVA
    statistic, p_value = stats.f_oneway(*groups)
    
    # Calculate eta-squared (effect size)
    # eta² = SS_between / SS_total
    grand_mean = np.mean(np.concatenate(groups))
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = sum(np.sum((g - grand_mean)**2) for g in groups)
    eta_squared = ss_between / ss_total
    
    # Conclusion
    if p_value < alpha:
        conclusion = f"Significant difference between groups (p={p_value:.4f} < {alpha})\n"
        conclusion += "Recommendation: Perform post-hoc tests (e.g., Tukey HSD) to identify which groups differ."
    else:
        conclusion = f"No significant difference between groups (p={p_value:.4f} >= {alpha})"
    
    return TestResult(
        test_name="One-Way ANOVA",
        statistic=statistic,
        p_value=p_value,
        effect_size=eta_squared,
        confidence_interval=None,
        conclusion=conclusion
    )


def chi_square_test(contingency_table, alpha=0.05):
    """
    Perform chi-square test of independence.
    
    Parameters
    ----------
    contingency_table : array-like
        2D contingency table
    alpha : float
        Significance level
    
    Returns
    -------
    result : TestResult
        Chi-square test results with Cramér's V
    """
    # Perform test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Calculate Cramér's V (effect size)
    n = np.sum(contingency_table)
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    
    # Conclusion
    if p_value < alpha:
        conclusion = f"Significant association detected (p={p_value:.4f} < {alpha})\n"
        conclusion += f"Cramér's V = {cramers_v:.3f} "
        if cramers_v < 0.1:
            conclusion += "(weak association)"
        elif cramers_v < 0.3:
            conclusion += "(moderate association)"
        else:
            conclusion += "(strong association)"
    else:
        conclusion = f"No significant association (p={p_value:.4f} >= {alpha})"
    
    return TestResult(
        test_name="Chi-Square Test of Independence",
        statistic=chi2,
        p_value=p_value,
        effect_size=cramers_v,
        confidence_interval=None,
        conclusion=conclusion
    )


def mann_whitney_u(group1, group2, alpha=0.05):
    """
    Non-parametric alternative to two-sample t-test.
    
    Parameters
    ----------
    group1, group2 : array-like
        Groups to compare
    alpha : float
        Significance level
    
    Returns
    -------
    result : TestResult
        Mann-Whitney U test results
    """
    statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    # Rank-biserial correlation (effect size)
    n1, n2 = len(group1), len(group2)
    rank_biserial = 1 - (2*statistic) / (n1 * n2)
    
    # Conclusion
    if p_value < alpha:
        conclusion = f"Significant difference in distributions (p={p_value:.4f} < {alpha})"
    else:
        conclusion = f"No significant difference in distributions (p={p_value:.4f} >= {alpha})"
    
    return TestResult(
        test_name="Mann-Whitney U Test",
        statistic=statistic,
        p_value=p_value,
        effect_size=rank_biserial,
        confidence_interval=None,
        conclusion=conclusion
    )
