import pandas as pd
import numpy as np

from typing import Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, DatasetDict

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalDataLoader:
    """
    Loads and preprocesses German medical text dataset.

    CSV Files : Quick Inference
    HF Dataset : Training, Inference and Eval

    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_split: str = "train",
        text_column: str = "text",
        label_column: str = "is_medical",
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Args:
            data_path: Path to local CSV file
            dataset_name: HuggingFace data card
            dataset_split: Split to load from HF dataset (default: "train")
            text_column: Name of text column in dataset
            label_column: Name of label column in dataset
            test_size: Proportion of data for testing (default: 0.2)
            random_state: Random seed
        """
        if data_path is None and dataset_name is None:
            raise ValueError("Missing Data : Provide data_path (CSV) or dataset_name (HuggingFace)")

        if data_path is not None and dataset_name is not None:
            raise ValueError("Conflicting Data : Provide only one of data_path or dataset_name, not both")

        self.data_path = data_path
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split

        self.text_column = text_column
        self.label_column = label_column

        self.test_size = test_size

        self.random_state = random_state
        self.class_distribution = None

    def load_data(self) -> pd.DataFrame:
        """Load data handler."""
        if self.data_path is not None:
            return self._load_from_csv()
        else:
            return self._load_from_huggingface()

    def _load_from_csv(self) -> pd.DataFrame:
        """Load dataset from local CSV file."""
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(df)} samples from {self.data_path}")

            # Data should have important cols (text und is_medical)
            required_cols = [self.text_column, self.label_column]

            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Dataset must contain columns: {required_cols}")

            # Clean up col names
            df = df.rename(columns={
                self.text_column: 'text',
                self.label_column: 'is_medical'
            })

            # Convert labels from text to bool
            df['is_medical'] = df['is_medical'].astype(bool).astype(int)

            # Dataset is HEAVILY imbalanced (30-70 split), so we need log class distr
            self.class_distribution = df['is_medical'].value_counts().to_dict()

            logger.info(f"Class distribution: Medical={self.class_distribution.get(1, 0)}, "
                       f"Non-Medical={self.class_distribution.get(0, 0)}")

            return df

        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")
            raise

    def _load_from_huggingface(self) -> pd.DataFrame:
        """Load dataset from HuggingFace Hub."""
        try:
            logger.info(f"Loading HuggingFace dataset: {self.dataset_name}")
            logger.info(f"Split: {self.dataset_split}")

            # Dataset directly from HF data card
            dataset = load_dataset(self.dataset_name, split=self.dataset_split)

            logger.info(f"Loaded {len(dataset)} samples from HuggingFace")
            logger.info(f"Dataset features: {dataset.features}")

            df = dataset.to_pandas()

            # Validate important cols (text und is_medical)
            if self.text_column not in df.columns:
                raise ValueError(
                    f"Text column '{self.text_column}' not found. "
                    f"Available columns: {list(df.columns)}"
                )

            if self.label_column not in df.columns:
                raise ValueError(
                    f"Label column '{self.label_column}' not found. "
                    f"Available columns: {list(df.columns)}"
                )

            # Clean up col names
            df = df.rename(columns={
                self.text_column: 'text',
                self.label_column: 'is_medical'
            })

            # Convert labels to binary int (1 / 0 for True / False)
            if df['is_medical'].dtype == bool:
                df['is_medical'] = df['is_medical'].astype(int)
            else:
                # TODO : Write a label mapping method for this later
                logger.error(f"Labels are not in bool format, clean this up or write method for it.")

            # Dataset is HEAVILY imbalanced (30-70 split), so we need log class distr
            # TODO : write a small method for this, since we do it for both data loaders
            self.class_distribution = df['is_medical'].value_counts().to_dict()
            logger.info(f"Class distribution: Medical={self.class_distribution.get(1, 0)}, "
                       f"Non-Medical={self.class_distribution.get(0, 0)}")

            return df

        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset: {e}")
            raise

    def preprocess_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize German text."""
        df = df.copy()

        # Remove whitespaces
        df['text'] = df['text'].str.strip()

        # Remove multiple spaces
        df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)

        # Remove empty samples
        df = df[df['text'].str.len() > 0]

        logger.info(f"Preprocessed {len(df)} samples")
        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Stratified train/test split to maintain class balance.

        Returns:
            X_train, X_test, y_train, y_test
        """
        X = df['text']
        y = df['is_medical']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state
        )

        logger.info(f"Train samples: {len(X_train)} (Medical: {y_train.sum()}, Non-Medical: {len(y_train)-y_train.sum()})")
        logger.info(f"Test samples: {len(X_test)} (Medical: {y_test.sum()}, Non-Medical: {len(y_test)-y_test.sum()})")

        return X_train, X_test, y_train, y_test

    def get_class_weights(self, y: pd.Series) -> dict:
        """
        Compute class weights for imbalanced dataset.

        Returns:
            Dictionary mapping class labels to weights
        """
        n_samples = len(y)
        n_classes = len(np.unique(y))
        class_counts = np.bincount(y)

        weights = n_samples / (n_classes * class_counts)
        weight_dict = {i: weights[i] for i in range(n_classes)}

        logger.info(f"Class weights: {weight_dict}")
        return weight_dict

    def prepare_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
        """
        Pipeline Orchestrator : load → preprocess → split → compute weights

        Returns:
            X_train, X_test, y_train, y_test, class_weights
        """
        # TODO : Expand docstring here for more explanation on why custom split_data method
        df = self.load_data() #load
        df = self.preprocess_text(df)  #preprocess

        X_train, X_test, y_train, y_test = self.split_data(df)  #split using custom method
        class_weights = self.get_class_weights(y_train) #get class weights

        return X_train, X_test, y_train, y_test, class_weights
