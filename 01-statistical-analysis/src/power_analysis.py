"""
Power Analysis and Sample Size Calculations

Determine required sample sizes and assess statistical power.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class PowerAnalysisResult:
    """Container for power analysis results."""
    test_type: str
    effect_size: float
    alpha: float
    power: float
    sample_size: int
    recommendation: str

    def summary(self):
        """Print formatted summary."""
        print(f"\n{self.test_type} - Power Analysis")
        print("=" * 60)
        print(f"Effect Size: {self.effect_size:.3f}")
        print(f"Significance Level (α): {self.alpha:.3f}")
        print(f"Statistical Power: {self.power:.3f}")
        print(f"Required Sample Size: {self.sample_size}")
        print(f"\nRecommendation: {self.recommendation}")
        print("=" * 60)


def power_ttest(effect_size, n, alpha=0.05, alternative='two-sided'):
    """
    Calculate power for t-test.

    Parameters
    ----------
    effect_size : float
        Cohen's d effect size
    n : int
        Sample size per group
    alpha : float
        Significance level
    alternative : str
        'two-sided', 'greater', or 'less'

    Returns
    -------
    power : float
        Statistical power (1 - β)
    """
    # Degrees of freedom
    df = 2 * n - 2

    # Non-centrality parameter
    ncp = effect_size * np.sqrt(n / 2)

    # Critical value
    if alternative == 'two-sided':
        t_crit = stats.t.ppf(1 - alpha/2, df)
    elif alternative == 'greater':
        t_crit = stats.t.ppf(1 - alpha, df)
    else:  # 'less'
        t_crit = stats.t.ppf(alpha, df)

    # Power calculation
    if alternative == 'two-sided':
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    else:
        power = 1 - stats.nct.cdf(t_crit, df, ncp)

    return power


def sample_size_ttest(effect_size, power=0.80, alpha=0.05, alternative='two-sided'):
    """
    Calculate required sample size for t-test.

    Parameters
    ----------
    effect_size : float
        Cohen's d effect size
    power : float
        Desired statistical power
    alpha : float
        Significance level
    alternative : str
        'two-sided', 'greater', or 'less'

    Returns
    -------
    result : PowerAnalysisResult
        Power analysis results
    """
    # Binary search for sample size
    n_low, n_high = 2, 10000

    while n_high - n_low > 1:
        n_mid = (n_low + n_high) // 2
        current_power = power_ttest(effect_size, n_mid, alpha, alternative)

        if current_power < power:
            n_low = n_mid
        else:
            n_high = n_mid

    n_required = n_high
    actual_power = power_ttest(effect_size, n_required, alpha, alternative)

    # Recommendation
    if effect_size < 0.2:
        rec = f"Small effect size ({effect_size:.2f}). Large sample needed. Consider if effect is practically meaningful."
    elif effect_size < 0.5:
        rec = f"Medium effect size ({effect_size:.2f}). Moderate sample required."
    else:
        rec = f"Large effect size ({effect_size:.2f}). Smaller sample sufficient."

    rec += f"\n\nTotal participants needed: {2 * n_required} ({n_required} per group)"

    return PowerAnalysisResult(
        test_type="Two-Sample T-Test",
        effect_size=effect_size,
        alpha=alpha,
        power=actual_power,
        sample_size=2 * n_required,
        recommendation=rec
    )


def minimum_detectable_effect(n, power=0.80, alpha=0.05, alternative='two-sided'):
    """
    Calculate minimum detectable effect size for given sample.

    Parameters
    ----------
    n : int
        Sample size per group
    power : float
        Desired statistical power
    alpha : float
        Significance level
    alternative : str
        'two-sided', 'greater', or 'less'

    Returns
    -------
    mde : float
        Minimum detectable effect (Cohen's d)
    """
    # Binary search for effect size
    d_low, d_high = 0.01, 5.0

    while d_high - d_low > 0.001:
        d_mid = (d_low + d_high) / 2
        current_power = power_ttest(d_mid, n, alpha, alternative)

        if current_power < power:
            d_low = d_mid
        else:
            d_high = d_mid

    return d_high


def power_proportion_test(p1, p2, n, alpha=0.05):
    """
    Calculate power for two-proportion z-test.

    Parameters
    ----------
    p1 : float
        Proportion in group 1
    p2 : float
        Proportion in group 2
    n : int
        Sample size per group
    alpha : float
        Significance level

    Returns
    -------
    power : float
        Statistical power
    """
    # Pooled proportion
    p_pooled = (p1 + p2) / 2

    # Standard error under null
    se_null = np.sqrt(2 * p_pooled * (1 - p_pooled) / n)

    # Standard error under alternative
    se_alt = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)

    # Critical value
    z_crit = stats.norm.ppf(1 - alpha/2)

    # Non-centrality parameter
    ncp = abs(p1 - p2) / se_alt

    # Power
    power = 1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)

    return power


def sample_size_proportion_test(p1, p2, power=0.80, alpha=0.05):
    """
    Calculate required sample size for two-proportion test.

    Parameters
    ----------
    p1 : float
        Expected proportion in group 1
    p2 : float
        Expected proportion in group 2
    power : float
        Desired statistical power
    alpha : float
        Significance level

    Returns
    -------
    result : PowerAnalysisResult
        Power analysis results
    """
    # Binary search
    n_low, n_high = 10, 100000

    while n_high - n_low > 1:
        n_mid = (n_low + n_high) // 2
        current_power = power_proportion_test(p1, p2, n_mid, alpha)

        if current_power < power:
            n_low = n_mid
        else:
            n_high = n_mid

    n_required = n_high
    actual_power = power_proportion_test(p1, p2, n_required, alpha)

    effect_size = abs(p1 - p2)

    rec = f"Detecting difference: {p1:.1%} vs {p2:.1%} (Δ = {effect_size:.1%})"
    rec += f"\n\nTotal participants needed: {2 * n_required} ({n_required} per group)"

    return PowerAnalysisResult(
        test_type="Two-Proportion Z-Test",
        effect_size=effect_size,
        alpha=alpha,
        power=actual_power,
        sample_size=2 * n_required,
        recommendation=rec
    )


def power_anova(effect_size_f, k, n, alpha=0.05):
    """
    Calculate power for one-way ANOVA.

    Parameters
    ----------
    effect_size_f : float
        Effect size f (Cohen's f)
    k : int
        Number of groups
    n : int
        Sample size per group
    alpha : float
        Significance level

    Returns
    -------
    power : float
        Statistical power
    """
    # Degrees of freedom
    df1 = k - 1
    df2 = k * (n - 1)

    # Non-centrality parameter
    ncp = effect_size_f**2 * k * n

    # Critical F-value
    f_crit = stats.f.ppf(1 - alpha, df1, df2)

    # Power
    power = 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)

    return power


def sample_size_anova(effect_size_f, k, power=0.80, alpha=0.05):
    """
    Calculate required sample size for ANOVA.

    Parameters
    ----------
    effect_size_f : float
        Effect size f (Cohen's f)
    k : int
        Number of groups
    power : float
        Desired statistical power
    alpha : float
        Significance level

    Returns
    -------
    result : PowerAnalysisResult
        Power analysis results
    """
    # Binary search
    n_low, n_high = 2, 10000

    while n_high - n_low > 1:
        n_mid = (n_low + n_high) // 2
        current_power = power_anova(effect_size_f, k, n_mid, alpha)

        if current_power < power:
            n_low = n_mid
        else:
            n_high = n_mid

    n_required = n_high
    actual_power = power_anova(effect_size_f, k, n_required, alpha)

    # Recommendations
    if effect_size_f < 0.1:
        rec = f"Small effect (f={effect_size_f:.2f}). Large sample needed."
    elif effect_size_f < 0.25:
        rec = f"Medium effect (f={effect_size_f:.2f}). Moderate sample required."
    else:
        rec = f"Large effect (f={effect_size_f:.2f}). Smaller sample sufficient."

    rec += f"\n\nTotal participants needed: {k * n_required} ({n_required} per group across {k} groups)"

    return PowerAnalysisResult(
        test_type=f"One-Way ANOVA ({k} groups)",
        effect_size=effect_size_f,
        alpha=alpha,
        power=actual_power,
        sample_size=k * n_required,
        recommendation=rec
    )


def cohens_d_to_f(d):
    """Convert Cohen's d to Cohen's f."""
    return d / 2


def cohens_f_to_d(f):
    """Convert Cohen's f to Cohen's d."""
    return 2 * f


def effect_size_from_means(means, std_pooled):
    """
    Calculate Cohen's f from group means.

    Parameters
    ----------
    means : array-like
        Group means
    std_pooled : float
        Pooled standard deviation

    Returns
    -------
    f : float
        Cohen's f effect size
    """
    grand_mean = np.mean(means)
    variance_means = np.var(means)
    f = np.sqrt(variance_means) / std_pooled

    return f
