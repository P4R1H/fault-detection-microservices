"""
Data Loading Utilities for TrainTicket Microservice Benchmark

This module handles loading and initial processing of the TrainTicket dataset
with 41 microservices, 88+ KPIs, and 14 fault types.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from loguru import logger


class TrainTicketDataset(Dataset):
    """
    PyTorch Dataset for TrainTicket microservice metrics.

    The dataset should contain time-series windows of metrics with labels
    indicating normal or anomalous behavior.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        window_size: int = 256,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize dataset.

        Args:
            data: Time-series data of shape (num_samples, num_features) or (num_samples, window_size, num_features)
            labels: Binary labels (0=normal, 1=anomaly) of shape (num_samples,)
            window_size: Size of time-series window
            feature_names: Names of features
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.window_size = window_size
        self.feature_names = feature_names or [f"feature_{i}" for i in range(data.shape[-1])]

        # Validate shapes
        if len(self.data.shape) == 2:
            # Data is (num_samples, num_features), needs to be windowed
            logger.warning(f"Data has shape {self.data.shape}, windowing not applied yet")
        elif len(self.data.shape) == 3:
            # Data is already windowed (num_samples, window_size, num_features)
            assert self.data.shape[1] == window_size, \
                f"Window size mismatch: expected {window_size}, got {self.data.shape[1]}"

        assert len(self.labels) == len(self.data), \
            f"Data and labels length mismatch: {len(self.data)} vs {len(self.labels)}"

        logger.info(f"Dataset created: {len(self)} samples, {self.data.shape[-1]} features")
        logger.info(f"Anomaly ratio: {self.labels.sum().item() / len(self.labels):.2%}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Tuple of (data, label)
        """
        return self.data[idx], self.labels[idx]

    def get_positive_ratio(self) -> float:
        """Get ratio of positive (anomaly) samples."""
        return self.labels.sum().item() / len(self.labels)


class TrainTicketLoader:
    """
    Loader for TrainTicket dataset with preprocessing and splitting.
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        window_size: int = 256,
        stride: int = 128,
        random_seed: int = 42,
    ):
        """
        Initialize data loader.

        Args:
            data_dir: Directory containing TrainTicket data
            window_size: Size of sliding window
            stride: Stride for sliding window
            random_seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.random_seed = random_seed

        logger.info(f"TrainTicketLoader initialized with data_dir: {self.data_dir}")

    def load_csv(
        self,
        file_path: str,
        timestamp_col: str = "timestamp",
        label_col: str = "label",
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load data from CSV file.

        Args:
            file_path: Path to CSV file
            timestamp_col: Name of timestamp column
            label_col: Name of label column
            feature_cols: List of feature column names (if None, use all except timestamp and label)

        Returns:
            Tuple of (features_df, labels_series)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {file_path}\n"
                f"Please ensure TrainTicket data is downloaded to {self.data_dir}"
            )

        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)

        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")

        # Handle timestamp if present
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.sort_values(timestamp_col)
            logger.info(f"Sorted by timestamp: {df[timestamp_col].min()} to {df[timestamp_col].max()}")

        # Extract labels
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in data")

        labels = df[label_col]

        # Extract features
        if feature_cols is None:
            # Use all columns except timestamp and label
            exclude_cols = [timestamp_col, label_col]
            feature_cols = [col for col in df.columns if col not in exclude_cols]

        features = df[feature_cols]

        logger.info(f"Features: {len(feature_cols)} columns")
        logger.info(f"Labels: {labels.value_counts().to_dict()}")

        return features, labels

    def create_windows(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows from time-series data.

        Args:
            features: Feature array of shape (num_timesteps, num_features)
            labels: Label array of shape (num_timesteps,)

        Returns:
            Tuple of (windowed_features, window_labels)
            - windowed_features: shape (num_windows, window_size, num_features)
            - window_labels: shape (num_windows,) - label from last timestep in window
        """
        num_timesteps, num_features = features.shape
        num_windows = (num_timesteps - self.window_size) // self.stride + 1

        windowed_features = []
        window_labels = []

        for i in range(num_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size

            if end_idx > num_timesteps:
                break

            window = features[start_idx:end_idx]
            # Use label from the last timestep in the window
            label = labels[end_idx - 1]

            windowed_features.append(window)
            window_labels.append(label)

        windowed_features = np.array(windowed_features)
        window_labels = np.array(window_labels)

        logger.info(f"Created {len(windowed_features)} windows")
        logger.info(f"Window shape: {windowed_features.shape}")

        return windowed_features, window_labels

    def split_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/val/test sets.

        Args:
            features: Feature array
            labels: Label array
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            stratify: Whether to stratify split by labels

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        # First split: train vs. (val + test)
        stratify_labels = labels if stratify else None

        X_train, X_temp, y_train, y_temp = train_test_split(
            features,
            labels,
            test_size=(val_ratio + test_ratio),
            random_state=self.random_seed,
            stratify=stratify_labels,
        )

        # Second split: val vs. test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        stratify_temp = y_temp if stratify else None

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=(1 - val_ratio_adjusted),
            random_state=self.random_seed,
            stratify=stratify_temp,
        )

        logger.info(f"Data split:")
        logger.info(f"  Train: {len(X_train)} samples ({len(X_train)/len(features):.1%})")
        logger.info(f"  Val:   {len(X_val)} samples ({len(X_val)/len(features):.1%})")
        logger.info(f"  Test:  {len(X_test)} samples ({len(X_test)/len(features):.1%})")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_few_shot_subset(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        num_samples: int,
        stratify: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a few-shot subset from training data.

        Args:
            X_train: Training features
            y_train: Training labels
            num_samples: Number of samples for few-shot learning
            stratify: Maintain class balance

        Returns:
            Tuple of (X_subset, y_subset)
        """
        if num_samples >= len(X_train):
            logger.warning(f"Requested {num_samples} samples but only {len(X_train)} available")
            return X_train, y_train

        indices = np.arange(len(X_train))
        stratify_labels = y_train if stratify else None

        subset_indices, _ = train_test_split(
            indices,
            train_size=num_samples,
            random_state=self.random_seed,
            stratify=stratify_labels,
        )

        X_subset = X_train[subset_indices]
        y_subset = y_train[subset_indices]

        logger.info(f"Created few-shot subset: {num_samples} samples")
        logger.info(f"  Anomaly ratio: {y_subset.sum() / len(y_subset):.2%}")

        return X_subset, y_subset


def create_dataloaders(
    train_dataset: TrainTicketDataset,
    val_dataset: TrainTicketDataset,
    test_dataset: TrainTicketDataset,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test sets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(f"DataLoaders created:")
    logger.info(f"  Train: {len(train_loader)} batches")
    logger.info(f"  Val:   {len(val_loader)} batches")
    logger.info(f"  Test:  {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage (requires actual data file)
    logger.info("TrainTicket Data Loader - Example Usage")

    # Initialize loader
    loader = TrainTicketLoader(
        data_dir="data/raw",
        window_size=256,
        stride=128,
    )

    # Check if data exists
    example_file = Path("data/raw/trainticket.csv")
    if not example_file.exists():
        logger.warning(f"Example data file not found: {example_file}")
        logger.info("Please download TrainTicket dataset and place it in data/raw/")
        logger.info("Expected format: CSV with timestamp, features, and label columns")
    else:
        logger.info(f"Found data file: {example_file}")

        # Load data
        features, labels = loader.load_csv(str(example_file))

        logger.info(f"Data loaded successfully!")
        logger.info(f"  Features shape: {features.shape}")
        logger.info(f"  Labels shape: {labels.shape}")
