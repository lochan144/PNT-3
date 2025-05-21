"""
Module for GNSS LOS/NLOS classification.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class GNSSClassifier:
    """Class for GNSS LOS/NLOS classification."""

    def __init__(self, model_type: str = 'rf', model_params: Optional[Dict] = None):
        """
        Initialize the classifier.

        Args:
            model_type (str): Type of model to use ('rf', 'svm', 'nn', or 'gb')
            model_params (Optional[Dict]): Model hyperparameters
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = self._create_model()
        self.scaler = StandardScaler()
        self.is_trained = False

    def _create_model(self) -> Any:
        """
        Create a new model instance based on model_type.

        Returns:
            Any: Initialized model
        """
        if self.model_type == 'rf':
            default_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
            params = {**default_params, **self.model_params}
            return RandomForestClassifier(**params)

        elif self.model_type == 'svm':
            default_params = {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'probability': True,
                'random_state': 42
            }
            params = {**default_params, **self.model_params}
            return SVC(**params)

        elif self.model_type == 'nn':
            default_params = {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'max_iter': 1000,
                'random_state': 42
            }
            params = {**default_params, **self.model_params}
            return MLPClassifier(**params)

        elif self.model_type == 'gb':
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            }
            params = {**default_params, **self.model_params}
            return GradientBoostingClassifier(**params)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, X: pd.DataFrame, y: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict:
        """
        Train the classifier.

        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Training labels
            X_val (Optional[pd.DataFrame]): Validation features
            y_val (Optional[pd.Series]): Validation labels

        Returns:
            Dict: Training metrics

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If training fails
        """
        # Input validation
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Training features (X) must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise ValueError("Training labels (y) must be a pandas Series")
        
        # Check for empty data
        if len(X) == 0:
            raise ValueError("Training features (X) is empty")
        if len(y) == 0:
            raise ValueError("Training labels (y) is empty")
        
        # Check for mismatched dimensions
        if len(X) != len(y):
            raise ValueError(f"Mismatched dimensions: X has {len(X)} samples but y has {len(y)} samples")

        # Data validation and cleaning
        if X.isnull().any().any():
            logger.warning("Training features contain NaN values. Filling with median values.")
            X = X.fillna(X.median())

        if y.isnull().any():
            logger.warning("Training labels contain NaN values. Removing corresponding samples.")
            valid_mask = ~y.isnull()
            X = X[valid_mask]
            y = y[valid_mask]

        # Convert labels to string and standardize
        y = y.astype(str).str.upper()
        
        # Check class distribution
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            raise ValueError(f"Training data must contain at least two different classes. Found only: {', '.join(class_counts.index)}")
        
        logger.info(f"Class distribution in training data: {class_counts.to_dict()}")
        
        # Check for extreme class imbalance
        min_class_ratio = class_counts.min() / len(y)
        if min_class_ratio < 0.1:  # Less than 10% of samples in minority class
            logger.warning(f"Severe class imbalance detected. Minority class has only {min_class_ratio:.1%} of samples")

        # Validate validation data if provided
        if X_val is not None and y_val is not None:
            if not isinstance(X_val, pd.DataFrame):
                raise ValueError("Validation features (X_val) must be a pandas DataFrame")
            if not isinstance(y_val, pd.Series):
                raise ValueError("Validation labels (y_val) must be a pandas Series")
            
            if X_val.isnull().any().any():
                logger.warning("Validation features contain NaN values. Filling with median values.")
                X_val = X_val.fillna(X_val.median())

            if y_val.isnull().any():
                logger.warning("Validation labels contain NaN values. Removing corresponding samples.")
                valid_mask = ~y_val.isnull()
                X_val = X_val[valid_mask]
                y_val = y_val[valid_mask]

            # Convert validation labels to string
            y_val = y_val.astype(str).str.upper()

        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train the model
            self.model.fit(X_scaled, y)
            self.is_trained = True

            # Calculate training metrics
            train_metrics = self._calculate_metrics(X_scaled, y)

            # Calculate validation metrics if validation data is provided
            val_metrics = None
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                val_metrics = self._calculate_metrics(X_val_scaled, y_val)

            return {
                'train': train_metrics,
                'validation': val_metrics,
                'data_info': {
                    'training_samples': len(y),
                    'validation_samples': len(y_val) if y_val is not None else 0,
                    'feature_count': X.shape[1],
                    'class_distribution': class_counts.to_dict()
                }
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise RuntimeError(f"Model training failed: {str(e)}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X (pd.DataFrame): Features to predict on

        Returns:
            np.ndarray: Predicted labels
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X (pd.DataFrame): Features to predict on

        Returns:
            np.ndarray: Prediction probabilities
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluate the model on test data.

        Args:
            X (pd.DataFrame): Test features
            y (pd.Series): Test labels

        Returns:
            Dict: Evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        X_scaled = self.scaler.transform(X)
        return self._calculate_metrics(X_scaled, y)

    def _calculate_metrics(self, X: np.ndarray, y: pd.Series) -> Dict:
        """
        Calculate performance metrics.

        Args:
            X (np.ndarray): Features
            y (pd.Series): True labels

        Returns:
            Dict: Performance metrics
        """
        y_pred = self.model.predict(X)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='binary'),
            'recall': recall_score(y, y_pred, average='binary'),
            'f1_score': f1_score(y, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }

    def save_model(self, model_dir: str) -> None:
        """
        Save the trained model and scaler.

        Args:
            model_dir (str): Directory to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(self.model, model_dir / 'model.joblib')
        
        # Save scaler
        joblib.dump(self.scaler, model_dir / 'scaler.joblib')

        # Save model type and parameters
        model_info = {
            'model_type': self.model_type,
            'model_params': self.model_params
        }
        joblib.dump(model_info, model_dir / 'model_info.joblib')

    def load_model(self, model_dir: str) -> None:
        """
        Load a trained model and scaler.

        Args:
            model_dir (str): Directory containing the saved model
        """
        model_dir = Path(model_dir)

        # Load model info
        model_info = joblib.load(model_dir / 'model_info.joblib')
        self.model_type = model_info['model_type']
        self.model_params = model_info['model_params']

        # Load model and scaler
        self.model = joblib.load(model_dir / 'model.joblib')
        self.scaler = joblib.load(model_dir / 'scaler.joblib')
        self.is_trained = True

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores if the model supports it.

        Returns:
            Dict[str, float]: Feature importance scores
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained to get feature importance")

        if hasattr(self.model, 'feature_importances_'):
            return {'importance': self.model.feature_importances_}
        elif hasattr(self.model, 'coef_'):
            return {'importance': abs(self.model.coef_[0])}
        else:
            raise ValueError(f"Model type {self.model_type} does not provide feature importance scores") 