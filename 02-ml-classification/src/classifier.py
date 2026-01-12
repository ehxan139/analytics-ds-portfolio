"""
Classification Pipeline

Production-ready classification with preprocessing, training, evaluation, and interpretation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')


class ClassificationPipeline:
    """
    End-to-end classification pipeline with preprocessing and model training.

    Parameters
    ----------
    model_type : str
        Model type: 'logistic', 'random_forest', 'xgboost', 'svm', 'ensemble'
    handle_imbalance : bool or str
        Handle class imbalance: False, 'smote', 'undersample', or 'class_weight'
    random_state : int
        Random seed
    """

    def __init__(self, model_type='xgboost', handle_imbalance='smote', random_state=42):
        self.model_type = model_type
        self.handle_imbalance = handle_imbalance
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_fitted = False

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the classification model."""
        if self.model_type == 'logistic':
            if self.handle_imbalance == 'class_weight':
                self.model = LogisticRegression(
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=self.random_state
                )
            else:
                self.model = LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state
                )

        elif self.model_type == 'random_forest':
            if self.handle_imbalance == 'class_weight':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    class_weight='balanced',
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1
                )

        elif self.model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )

        elif self.model_type == 'svm':
            if self.handle_imbalance == 'class_weight':
                self.model = SVC(
                    kernel='rbf',
                    class_weight='balanced',
                    probability=True,
                    random_state=self.random_state
                )
            else:
                self.model = SVC(
                    kernel='rbf',
                    probability=True,
                    random_state=self.random_state
                )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _preprocess_features(self, X):
        """Preprocess features."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.copy()

            # Handle categorical variables
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

            X = X.values

        return X

    def fit(self, X, y, validation_split=0.2):
        """
        Fit the classification pipeline.

        Parameters
        ----------
        X : array-like or DataFrame
            Features
        y : array-like
            Target labels
        validation_split : float
            Fraction for validation set

        Returns
        -------
        self : ClassificationPipeline
            Fitted pipeline
        """
        # Preprocess
        X_processed = self._preprocess_features(X)
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y_encoded,
            test_size=validation_split,
            stratify=y_encoded,
            random_state=self.random_state
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Handle imbalance
        if self.handle_imbalance == 'smote':
            smote = SMOTE(random_state=self.random_state)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

        elif self.handle_imbalance == 'undersample':
            undersampler = RandomUnderSampler(random_state=self.random_state)
            X_train_scaled, y_train = undersampler.fit_resample(X_train_scaled, y_train)

        # Train model
        if self.model_type == 'xgboost' and self.handle_imbalance:
            # Calculate scale_pos_weight for XGBoost
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            self.model.set_params(scale_pos_weight=scale_pos_weight)

        self.model.fit(X_train_scaled, y_train)

        # Validation score
        val_score = self.model.score(X_val_scaled, y_val)
        print(f"Validation Accuracy: {val_score:.4f}")

        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Make predictions.

        Parameters
        ----------
        X : array-like or DataFrame
            Features

        Returns
        -------
        predictions : array
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_processed = self._preprocess_features(X)
        X_scaled = self.scaler.transform(X_processed)

        y_pred = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like or DataFrame
            Features

        Returns
        -------
        probabilities : array
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_processed = self._preprocess_features(X)
        X_scaled = self.scaler.transform(X_processed)

        return self.model.predict_proba(X_scaled)

    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.

        Parameters
        ----------
        X : array-like or DataFrame
            Features
        y : array-like
            Target labels
        cv : int
            Number of folds

        Returns
        -------
        scores : dict
            Cross-validation scores
        """
        X_processed = self._preprocess_features(X)
        y_encoded = self.label_encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X_processed)

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        scores = cross_val_score(
            self.model, X_scaled, y_encoded,
            cv=skf, scoring='accuracy'
        )

        return {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }

    def get_feature_importance(self, top_n=20):
        """
        Get feature importance rankings.

        Parameters
        ----------
        top_n : int
            Number of top features to return

        Returns
        -------
        importance_df : DataFrame
            Feature importance rankings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
        else:
            feature_names = self.feature_names

        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Linear models
            importances = np.abs(self.model.coef_[0])
        else:
            raise ValueError("Model does not support feature importance")

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)

        return importance_df

    def save(self, filepath):
        """Save the fitted pipeline."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, filepath)

    @classmethod
    def load(cls, filepath):
        """Load a saved pipeline."""
        data = joblib.load(filepath)

        pipeline = cls(model_type=data['model_type'])
        pipeline.model = data['model']
        pipeline.scaler = data['scaler']
        pipeline.label_encoder = data['label_encoder']
        pipeline.feature_names = data['feature_names']
        pipeline.is_fitted = True

        return pipeline


def compare_models(X, y, model_types=['logistic', 'random_forest', 'xgboost'], cv=5):
    """
    Compare multiple models using cross-validation.

    Parameters
    ----------
    X : array-like or DataFrame
        Features
    y : array-like
        Target labels
    model_types : list
        List of model types to compare
    cv : int
        Number of cross-validation folds

    Returns
    -------
    comparison_df : DataFrame
        Model comparison results
    """
    results = []

    for model_type in model_types:
        print(f"Training {model_type}...")

        pipeline = ClassificationPipeline(model_type=model_type)
        cv_scores = pipeline.cross_validate(X, y, cv=cv)

        results.append({
            'model': model_type,
            'mean_accuracy': cv_scores['mean'],
            'std_accuracy': cv_scores['std']
        })

    comparison_df = pd.DataFrame(results).sort_values('mean_accuracy', ascending=False)

    return comparison_df
