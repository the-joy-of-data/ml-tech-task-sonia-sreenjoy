import numpy as np
import lightgbm as lgb
from typing import Dict, Optional, Tuple
import logging
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalTextClassifier:
    """
    LightGBM Classifier with optimizations for imbalanced German medical text.

    Key features:
    - Handles class imbalance via scale_pos_weight
    - Early stopping to prevent overfitting on small dataset
    """
    def __init__(
        self,
        class_weights: Optional[Dict[int, float]] = None,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        random_state: int = 42
    ):
        """
        Args:
            class_weights: Dict mapping class labels to weights
            n_estimators: boosting rounds
            learning_rate: Step size (grad descent)
            max_depth: max depth of tree
            num_leaves: max leaves per tree
            random_state: Random seed
        """
        self.class_weights = class_weights or {0: 1.0, 1: 1.0}
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.random_state = random_state

        self.model = None
        self.feature_importances_ = None

    def _compute_scale_pos_weight(self) -> float:
        """Compute LightGBM's scale_pos_weight parameter."""
        return self.class_weights[1] / self.class_weights[0]

    def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            verbose: int = 50
        ):
        """
        Training Handler : LightGBM classifier with early stopping.

        Args:
            X_train: train features
            y_train: train labels
            X_val: valid. features (for early stopping)
            y_val: valid. labels
            verbose: print train progress logs
        """
        logger.info("Training LightGBM classifier...")
        logger.info(f"Training samples: {len(X_train)}, Features: {X_train.shape[1]}")

        # compute scale_pos_weight
        scale_pos_weight = self._compute_scale_pos_weight()
        logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

        # model parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'scale_pos_weight': scale_pos_weight,
            'min_child_samples': 20,   # prevent overfitting
            'reg_alpha': 0.1,          # L1 reg
            'reg_lambda': 0.1,         # L2 reg
            'random_state': self.random_state,
            'verbose': -1
        }

        # create LightGBM dataset from train data
        train_data = lgb.Dataset(X_train, label=y_train)

        # Early stopping given that we have validation set
        valid_sets = [train_data]
        valid_names = ['train']
        callbacks = []

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
            callbacks.append(lgb.early_stopping(stopping_rounds=50))
            logger.info(f"Validation samples: {len(X_val)}")

        if verbose > 0:
            callbacks.append(lgb.log_evaluation(period=verbose))

        # train model
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        # store feature importances
        self.feature_importances_ = self.model.feature_importance(importance_type='gain')

        logger.info(f"Training complete. Best iteration: {self.model.best_iteration}")
        logger.info(f"Best score: {self.model.best_score}")

    def train_with_cross_validation(
        X: np.ndarray,
        y: np.ndarray,
        class_weights: Dict[int, float],
        n_folds: int = 5
    ) -> Tuple[float, float]:
        """
        Train model with stratified K-fold cross-validation.
        Returns mean and std of validation F1 scores.

        Args:
            X: Feature matrix
            y: Labels
            class_weights: Dictionary mapping class labels to weights
            n_folds: Number of CV folds

        Returns:
            (mean_f1, std_f1)
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        f1_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            model = MedicalTextClassifier(class_weights=class_weights)
            model.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold, verbose=0)

            y_pred = model.predict(X_val_fold)
            f1 = f1_score(y_val_fold, y_pred, pos_label=1)
            f1_scores.append(f1)

            logger.info(f"Fold {fold+1}/{n_folds} - F1: {f1:.4f}")

        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)

        logger.info(f"Cross-validation F1: {mean_f1:.4f} ± {std_f1:.4f}")
        return mean_f1, std_f1

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Array of shape (n_samples, 2) with probabilities for each class (binary labels)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        proba_positive = self.model.predict(X)
        proba_negative = 1 - proba_positive #lgbm returns pos class probabilities only

        return np.column_stack([proba_negative, proba_positive])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix
            threshold: Decision threshold (default: 0.5)

        Returns:
            Array of predicted labels (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)

    def get_feature_importance(
        self,
        feature_names: Optional[list] = None,
        top_k: int = 20
    ) -> Dict[str, float]:
        """
        Get top K most important features.

        Args:
            feature_names: List of feature names
            top_k: Number of top features to return

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.feature_importances_ is None:
            raise ValueError("Model not trained!")

        # sort features by importance and grab top_k indices
        indices = np.argsort(self.feature_importances_)[::-1][:top_k]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importances_))]

        importance_dict = {
            feature_names[i]: float(self.feature_importances_[i])
            for i in indices
        }

        return importance_dict

    def save(self, model_dir: str):
        """Save trained model to disk."""
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        #TODO : Create ENV variable for saved model names
        model_file = model_path / "lightgbm_model.txt"
        self.model.save_model(str(model_file))

        metadata = {
            'class_weights': self.class_weights,
            'feature_importances': self.feature_importances_.tolist(),
            'n_features': len(self.feature_importances_)
        }

        #TODO : Create ENV variable for model metadata file names
        metadata_file = model_path / "model_metadata.joblib"
        joblib.dump(metadata, metadata_file)

        logger.info(f"Model saved to {model_dir}")

    def load(self, model_dir: str):
        """Load trained model from disk."""
        model_path = Path(model_dir)

        #TODO : Create ENV variable for saved model names
        model_file = model_path / "lightgbm_model.txt"
        self.model = lgb.Booster(model_file=str(model_file))

        #TODO : Create ENV variable for model metadata file names
        metadata_file = model_path / "model_metadata.joblib"
        metadata = joblib.load(metadata_file)
        self.class_weights = metadata['class_weights']
        self.feature_importances_ = np.array(metadata['feature_importances'])

        logger.info(f"Model loaded from {model_dir}")