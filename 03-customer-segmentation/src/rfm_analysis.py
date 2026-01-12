"""
RFM (Recency, Frequency, Monetary) Analysis

Customer segmentation based on purchase behavior.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class RFMAnalyzer:
    """
    RFM analysis for customer segmentation.

    RFM scores customers on:
    - Recency: Days since last purchase
    - Frequency: Number of purchases
    - Monetary: Total spending
    """

    def __init__(self, r_bins=5, f_bins=5, m_bins=5):
        self.r_bins = r_bins
        self.f_bins = f_bins
        self.m_bins = m_bins
        self.segment_map = self._create_segment_map()

    def calculate_rfm(self, df, customer_id='CustomerID', date='Date', amount='Amount', reference_date=None):
        """
        Calculate RFM scores from transaction data.

        Parameters
        ----------
        df : DataFrame
            Transaction data
        customer_id : str
            Customer ID column
        date : str
            Date column
        amount : str
            Transaction amount column
        reference_date : datetime, optional
            Reference date for recency (default: max date + 1 day)

        Returns
        -------
        rfm_df : DataFrame
            RFM scores per customer
        """
        df = df.copy()
        df[date] = pd.to_datetime(df[date])

        if reference_date is None:
            reference_date = df[date].max() + timedelta(days=1)

        # Calculate RFM metrics
        rfm = df.groupby(customer_id).agg({
            date: lambda x: (reference_date - x.max()).days,  # Recency
            customer_id: 'count',  # Frequency
            amount: 'sum'  # Monetary
        }).reset_index()

        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

        # Calculate RFM scores (1-5, where 5 is best)
        rfm['R_Score'] = pd.qcut(rfm['Recency'], q=self.r_bins, labels=False, duplicates='drop')
        rfm['R_Score'] = self.r_bins - rfm['R_Score']  # Invert: lower recency = higher score

        rfm['F_Score'] = pd.qcut(rfm['Frequency'], q=self.f_bins, labels=False, duplicates='drop')
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=self.m_bins, labels=False, duplicates='drop')

        # Combined RFM score
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

        # Assign segments
        rfm['Segment'] = rfm['RFM_Score'].map(self.segment_map)
        rfm['Segment'] = rfm['Segment'].fillna('Other')

        return rfm

    def _create_segment_map(self):
        """Create RFM score to segment mapping."""
        segment_map = {
            # Champions: Best customers
            '444': 'Champions', '443': 'Champions', '434': 'Champions', '433': 'Champions',
            '344': 'Champions', '343': 'Champions', '334': 'Champions', '333': 'Champions',

            # Loyal Customers: Regular buyers
            '442': 'Loyal', '432': 'Loyal', '342': 'Loyal', '332': 'Loyal',
            '424': 'Loyal', '423': 'Loyal', '324': 'Loyal', '323': 'Loyal',

            # Potential Loyalists: Recent high-value
            '441': 'Potential Loyalist', '431': 'Potential Loyalist',
            '341': 'Potential Loyalist', '331': 'Potential Loyalist',
            '422': 'Potential Loyalist', '421': 'Potential Loyalist',
            '322': 'Potential Loyalist', '321': 'Potential Loyalist',

            # At Risk: High value but declining
            '244': 'At Risk', '243': 'At Risk', '234': 'At Risk', '233': 'At Risk',
            '144': 'At Risk', '143': 'At Risk', '134': 'At Risk', '133': 'At Risk',

            # Hibernating: Low recent activity
            '242': 'Hibernating', '232': 'Hibernating', '142': 'Hibernating', '132': 'Hibernating',
            '224': 'Hibernating', '223': 'Hibernating', '124': 'Hibernating', '123': 'Hibernating',

            # Lost: No recent activity
            '241': 'Lost', '231': 'Lost', '141': 'Lost', '131': 'Lost',
            '222': 'Lost', '221': 'Lost', '122': 'Lost', '121': 'Lost',
            '111': 'Lost', '112': 'Lost', '113': 'Lost', '114': 'Lost',
            '211': 'Lost', '212': 'Lost', '213': 'Lost', '214': 'Lost',
        }

        return segment_map

    def get_segment_strategies(self, rfm_df):
        """
        Get marketing strategies for each segment.

        Parameters
        ----------
        rfm_df : DataFrame
            RFM data with segments

        Returns
        -------
        strategies : dict
            Marketing strategies per segment
        """
        segment_counts = rfm_df['Segment'].value_counts()

        strategies = {
            'Champions': {
                'description': 'Best customers - high value, recent, frequent',
                'percentage': (segment_counts.get('Champions', 0) / len(rfm_df) * 100),
                'strategy': 'Reward them. VIP treatment, early access, exclusive offers.',
                'channels': 'Email, SMS, personalized outreach',
                'expected_roi': '8:1'
            },
            'Loyal': {
                'description': 'Regular customers with moderate value',
                'percentage': (segment_counts.get('Loyal', 0) / len(rfm_df) * 100),
                'strategy': 'Upsell and cross-sell. Loyalty programs, product bundles.',
                'channels': 'Email campaigns, in-app notifications',
                'expected_roi': '5:1'
            },
            'Potential Loyalist': {
                'description': 'Recent customers with growth potential',
                'percentage': (segment_counts.get('Potential Loyalist', 0) / len(rfm_df) * 100),
                'strategy': 'Engage early. Onboarding programs, educational content.',
                'channels': 'Email nurture sequences, retargeting ads',
                'expected_roi': '4:1'
            },
            'At Risk': {
                'description': 'High-value customers showing decline',
                'percentage': (segment_counts.get('At Risk', 0) / len(rfm_df) * 100),
                'strategy': 'Win them back. Surveys, special offers, re-engagement.',
                'channels': 'Personalized emails, phone calls, win-back discounts',
                'expected_roi': '3:1'
            },
            'Hibernating': {
                'description': 'Low recent activity, moderate historical value',
                'percentage': (segment_counts.get('Hibernating', 0) / len(rfm_df) * 100),
                'strategy': 'Re-activate. Discount offers, new product announcements.',
                'channels': 'Email blasts, social media ads',
                'expected_roi': '2:1'
            },
            'Lost': {
                'description': 'No recent activity, likely churned',
                'percentage': (segment_counts.get('Lost', 0) / len(rfm_df) * 100),
                'strategy': 'Last attempt or let go. Aggressive win-back or suppress.',
                'channels': 'Final email campaign, surveys to understand why',
                'expected_roi': '1:1'
            }
        }

        return strategies

    def calculate_segment_value(self, rfm_df):
        """Calculate total value per segment."""
        segment_value = rfm_df.groupby('Segment').agg({
            'CustomerID': 'count',
            'Monetary': ['sum', 'mean'],
            'Frequency': 'mean',
            'Recency': 'mean'
        }).round(2)

        segment_value.columns = ['Customer_Count', 'Total_Revenue', 'Avg_Revenue', 'Avg_Frequency', 'Avg_Recency']

        # Calculate segment percentage
        segment_value['Percentage'] = (segment_value['Customer_Count'] / len(rfm_df) * 100).round(1)

        return segment_value.sort_values('Total_Revenue', ascending=False)
