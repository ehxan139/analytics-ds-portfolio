"""
A/B Testing Framework

Classical and Bayesian A/B testing with comprehensive statistical analysis.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional


@dataclass
class ABTestResult:
    """Container for A/B test results."""
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    relative_lift: float
    p_value: float
    confidence_interval: tuple
    is_significant: bool
    sample_size_control: int
    sample_size_treatment: int
    recommendation: str
    
    def summary(self):
        """Print formatted summary."""
        print("\n" + "="*70)
        print("A/B TEST RESULTS")
        print("="*70)
        print(f"\nControl Rate:    {self.control_rate:.2%}")
        print(f"Treatment Rate:  {self.treatment_rate:.2%}")
        print(f"\nAbsolute Lift:   {self.absolute_lift:+.2%}")
        print(f"Relative Lift:   {self.relative_lift:+.1%}")
        print(f"\nP-value:         {self.p_value:.4f}")
        print(f"95% CI:          [{self.confidence_interval[0]:.2%}, {self.confidence_interval[1]:.2%}]")
        print(f"Significant:     {'YES' if self.is_significant else 'NO'}")
        print(f"\nSample Sizes:    Control={self.sample_size_control:,}, Treatment={self.sample_size_treatment:,}")
        print(f"\n{self.recommendation}")
        print("="*70)


class ABTest:
    """
    A/B test for conversion rates using two-proportion z-test.
    
    Parameters
    ----------
    control_conversions : int
        Number of conversions in control
    control_samples : int
        Total samples in control
    treatment_conversions : int
        Number of conversions in treatment
    treatment_samples : int
        Total samples in treatment
    alpha : float
        Significance level
    """
    
    def __init__(self, control_conversions, control_samples, 
                 treatment_conversions, treatment_samples, alpha=0.05):
        self.control_conv = control_conversions
        self.control_n = control_samples
        self.treatment_conv = treatment_conversions
        self.treatment_n = treatment_samples
        self.alpha = alpha
        
        self.control_rate = control_conversions / control_samples
        self.treatment_rate = treatment_conversions / treatment_samples
    
    def analyze(self):
        """
        Perform two-proportion z-test.
        
        Returns
        -------
        result : ABTestResult
            Test results
        """
        # Pooled proportion
        p_pooled = (self.control_conv + self.treatment_conv) / (self.control_n + self.treatment_n)
        
        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/self.control_n + 1/self.treatment_n))
        
        # Z-statistic
        z_stat = (self.treatment_rate - self.control_rate) / se
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Confidence interval for difference
        se_diff = np.sqrt(
            self.control_rate * (1 - self.control_rate) / self.control_n +
            self.treatment_rate * (1 - self.treatment_rate) / self.treatment_n
        )
        z_crit = stats.norm.ppf(1 - self.alpha/2)
        diff = self.treatment_rate - self.control_rate
        ci_lower = diff - z_crit * se_diff
        ci_upper = diff + z_crit * se_diff
        
        # Lifts
        absolute_lift = self.treatment_rate - self.control_rate
        relative_lift = (absolute_lift / self.control_rate) if self.control_rate > 0 else 0
        
        # Significance
        is_significant = p_value < self.alpha
        
        # Recommendation
        if is_significant:
            if relative_lift > 0:
                rec = f"✅ SHIP IT: Treatment shows {abs(relative_lift):.1%} improvement (p={p_value:.4f})"
            else:
                rec = f"❌ DON'T SHIP: Treatment shows {abs(relative_lift):.1%} decline (p={p_value:.4f})"
        else:
            rec = f"⚠️  INCONCLUSIVE: No significant difference detected (p={p_value:.4f}). Consider running longer or larger sample."
        
        return ABTestResult(
            control_rate=self.control_rate,
            treatment_rate=self.treatment_rate,
            absolute_lift=absolute_lift,
            relative_lift=relative_lift,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            sample_size_control=self.control_n,
            sample_size_treatment=self.treatment_n,
            recommendation=rec
        )
    
    def calculate_sample_ratio_mismatch(self):
        """
        Check for sample ratio mismatch (SRM).
        
        Returns
        -------
        srm_p_value : float
            P-value for SRM test (should be > 0.001)
        has_srm : bool
            True if SRM detected
        """
        expected_control = (self.control_n + self.treatment_n) / 2
        expected_treatment = (self.control_n + self.treatment_n) / 2
        
        # Chi-square test
        observed = [self.control_n, self.treatment_n]
        expected = [expected_control, expected_treatment]
        
        chi2, p_value = stats.chisquare(observed, expected)
        
        has_srm = p_value < 0.001  # Conservative threshold
        
        return p_value, has_srm


class BayesianABTest:
    """
    Bayesian A/B test using Beta-Binomial model for conversion rates.
    
    Parameters
    ----------
    prior_alpha : float
        Prior alpha for Beta distribution
    prior_beta : float
        Prior beta for Beta distribution
    """
    
    def __init__(self, prior_alpha=1, prior_beta=1):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
    
    def analyze(self, control_conversions, control_samples,
                treatment_conversions, treatment_samples):
        """
        Perform Bayesian analysis.
        
        Returns
        -------
        result : dict
            Bayesian analysis results
        """
        # Posterior distributions
        control_alpha = self.prior_alpha + control_conversions
        control_beta = self.prior_beta + (control_samples - control_conversions)
        
        treatment_alpha = self.prior_alpha + treatment_conversions
        treatment_beta = self.prior_beta + (treatment_samples - treatment_conversions)
        
        # Monte Carlo sampling
        n_samples = 100000
        control_samples_mc = np.random.beta(control_alpha, control_beta, n_samples)
        treatment_samples_mc = np.random.beta(treatment_alpha, treatment_beta, n_samples)
        
        # Probability treatment > control
        prob_treatment_better = (treatment_samples_mc > control_samples_mc).mean()
        
        # Expected lift
        lift_samples = (treatment_samples_mc - control_samples_mc) / control_samples_mc
        expected_lift = lift_samples.mean()
        lift_ci_lower = np.percentile(lift_samples, 2.5)
        lift_ci_upper = np.percentile(lift_samples, 97.5)
        
        return {
            'prob_treatment_better': prob_treatment_better,
            'expected_lift': expected_lift,
            'lift_credible_interval': (lift_ci_lower, lift_ci_upper),
            'control_posterior': (control_alpha, control_beta),
            'treatment_posterior': (treatment_alpha, treatment_beta)
        }
    
    def plot_posteriors(self, control_alpha, control_beta, treatment_alpha, treatment_beta):
        """Plot posterior distributions."""
        import matplotlib.pyplot as plt
        
        x = np.linspace(0, 1, 1000)
        control_post = stats.beta.pdf(x, control_alpha, control_beta)
        treatment_post = stats.beta.pdf(x, treatment_alpha, treatment_beta)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, control_post, label='Control', linewidth=2)
        ax.plot(x, treatment_post, label='Treatment', linewidth=2)
        ax.set_xlabel('Conversion Rate')
        ax.set_ylabel('Probability Density')
        ax.set_title('Posterior Distributions')
        ax.legend()
        ax.grid(alpha=0.3)
        
        return fig


def sequential_test(data_stream, alpha=0.05, spending_function='obrien_fleming'):
    """
    Sequential testing with alpha spending function.
    
    Parameters
    ----------
    data_stream : generator
        Stream of (control_conv, control_n, treatment_conv, treatment_n) tuples
    alpha : float
        Overall significance level
    spending_function : str
        Alpha spending: 'obrien_fleming' or 'pocock'
    
    Returns
    -------
    decision : str
        'continue', 'stop_success', or 'stop_futility'
    """
    # Placeholder for sequential testing logic
    # In practice, would implement group sequential design
    pass


def calculate_business_value(control_rate, treatment_rate, traffic_per_day,
                            revenue_per_conversion, implementation_cost=0):
    """
    Calculate business value of A/B test result.
    
    Parameters
    ----------
    control_rate : float
        Control conversion rate
    treatment_rate : float
        Treatment conversion rate
    traffic_per_day : int
        Daily traffic
    revenue_per_conversion : float
        Revenue per conversion
    implementation_cost : float
        One-time cost to implement
    
    Returns
    -------
    business_metrics : dict
        Business impact metrics
    """
    daily_conversions_control = traffic_per_day * control_rate
    daily_conversions_treatment = traffic_per_day * treatment_rate
    
    daily_revenue_control = daily_conversions_control * revenue_per_conversion
    daily_revenue_treatment = daily_conversions_treatment * revenue_per_conversion
    
    daily_lift = daily_revenue_treatment - daily_revenue_control
    annual_lift = daily_lift * 365
    
    roi = ((annual_lift - implementation_cost) / implementation_cost * 100) if implementation_cost > 0 else float('inf')
    
    return {
        'daily_lift': daily_lift,
        'monthly_lift': daily_lift * 30,
        'annual_lift': annual_lift,
        'implementation_cost': implementation_cost,
        'net_annual_value': annual_lift - implementation_cost,
        'roi_percent': roi,
        'breakeven_days': implementation_cost / daily_lift if daily_lift > 0 else float('inf')
    }
