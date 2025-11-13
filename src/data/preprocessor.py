"""
Time-Series Preprocessing for MOMENT Foundation Model

This module handles:
- Normalization (Z-score, MinMax, Robust)
- Patching (PatchTST-style for MOMENT)
- Missing value imputation
- Feature scaling
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from loguru import logger
import torch


class TimeSeriesPreprocessor:
    """
    Comprehensive preprocessor for time-series data.
    """

    def __init__(
        self,
        window_size: int = 256,
        normalization: str = "zscore",
        per_feature_norm: bool = True,
        missing_value_strategy: str = "forward_fill",
        patch_len: Optional[int] = 16,
        patch_stride: Optional[int] = 8,
    ):
        """
        Initialize preprocessor.

        Args:
            window_size: Size of time-series windows
            normalization: Normalization strategy ("zscore", "minmax", "robust", "none")
            per_feature_norm: Normalize each feature independently
            missing_value_strategy: Strategy for handling missing values
            patch_len: Length of patches for MOMENT (None to disable patching)
            patch_stride: Stride for patching
        """
        self.window_size = window_size
        self.normalization = normalization
        self.per_feature_norm = per_feature_norm
        self.missing_value_strategy = missing_value_strategy
        self.patch_len = patch_len
        self.patch_stride = patch_stride

        # Scalers (fitted on training data)
        self.scalers: Optional[List[Any]] = None
        self.global_scaler: Optional[Any] = None
        self.is_fitted = False

        logger.info(f"TimeSeriesPreprocessor initialized:")
        logger.info(f"  Window size: {window_size}")
        logger.info(f"  Normalization: {normalization}")
        logger.info(f"  Per-feature: {per_feature_norm}")
        logger.info(f"  Patching: {patch_len if patch_len else 'Disabled'}")

    def _create_scaler(self) -> Any:
        """Create a scaler based on normalization strategy."""
        if self.normalization == "zscore":
            return StandardScaler()
        elif self.normalization == "minmax":
            return MinMaxScaler()
        elif self.normalization == "robust":
            return RobustScaler()
        elif self.normalization == "none":
            return None
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")

    def handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """
        Handle missing values in time-series data.

        Args:
            data: Input data with potential NaN values

        Returns:
            Data with missing values handled
        """
        if not np.isnan(data).any():
            return data

        num_missing = np.isnan(data).sum()
        logger.warning(f"Found {num_missing} missing values ({num_missing/data.size:.2%})")

        if self.missing_value_strategy == "forward_fill":
            # Forward fill along time axis (axis=0 for 2D, axis=1 for 3D)
            if len(data.shape) == 2:
                # Shape: (timesteps, features)
                df = pd.DataFrame(data)
                filled = df.fillna(method='ffill').fillna(method='bfill').values
            elif len(data.shape) == 3:
                # Shape: (samples, timesteps, features)
                filled = np.zeros_like(data)
                for i in range(data.shape[0]):
                    df = pd.DataFrame(data[i])
                    filled[i] = df.fillna(method='ffill').fillna(method='bfill').values
            else:
                raise ValueError(f"Unsupported data shape: {data.shape}")

        elif self.missing_value_strategy == "interpolate":
            if len(data.shape) == 2:
                df = pd.DataFrame(data)
                filled = df.interpolate(method='linear', axis=0).fillna(0).values
            elif len(data.shape) == 3:
                filled = np.zeros_like(data)
                for i in range(data.shape[0]):
                    df = pd.DataFrame(data[i])
                    filled[i] = df.interpolate(method='linear', axis=0).fillna(0).values
            else:
                raise ValueError(f"Unsupported data shape: {data.shape}")

        elif self.missing_value_strategy == "zero":
            filled = np.nan_to_num(data, nan=0.0)

        else:
            raise ValueError(f"Unknown missing value strategy: {self.missing_value_strategy}")

        logger.info(f"Missing values handled using '{self.missing_value_strategy}'")
        return filled

    def fit(self, train_data: np.ndarray) -> 'TimeSeriesPreprocessor':
        """
        Fit normalization scalers on training data.

        Args:
            train_data: Training data of shape (num_samples, num_features) or
                       (num_samples, window_size, num_features)

        Returns:
            Self
        """
        if self.normalization == "none":
            self.is_fitted = True
            logger.info("No normalization applied (normalization='none')")
            return self

        # Handle missing values first
        train_data = self.handle_missing_values(train_data)

        # Reshape for fitting
        if len(train_data.shape) == 3:
            # (num_samples, window_size, num_features) -> (num_samples * window_size, num_features)
            num_samples, window_size, num_features = train_data.shape
            train_data_flat = train_data.reshape(-1, num_features)
        elif len(train_data.shape) == 2:
            # (timesteps, num_features)
            train_data_flat = train_data
            num_features = train_data.shape[1]
        else:
            raise ValueError(f"Unsupported data shape: {train_data.shape}")

        if self.per_feature_norm:
            # Fit separate scaler for each feature
            self.scalers = []
            for i in range(num_features):
                scaler = self._create_scaler()
                scaler.fit(train_data_flat[:, i:i+1])
                self.scalers.append(scaler)
            logger.info(f"Fitted {num_features} per-feature scalers")
        else:
            # Fit single global scaler
            self.global_scaler = self._create_scaler()
            self.global_scaler.fit(train_data_flat)
            logger.info("Fitted global scaler")

        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scalers.

        Args:
            data: Data to transform

        Returns:
            Normalized data
        """
        if self.normalization == "none":
            return data

        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        # Handle missing values
        data = self.handle_missing_values(data)

        # Store original shape
        original_shape = data.shape

        # Reshape for transformation
        if len(data.shape) == 3:
            num_samples, window_size, num_features = data.shape
            data_flat = data.reshape(-1, num_features)
        elif len(data.shape) == 2:
            data_flat = data
            num_features = data.shape[1]
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")

        # Transform
        if self.per_feature_norm:
            transformed = np.zeros_like(data_flat)
            for i in range(num_features):
                transformed[:, i:i+1] = self.scalers[i].transform(data_flat[:, i:i+1])
        else:
            transformed = self.global_scaler.transform(data_flat)

        # Reshape back to original shape
        if len(original_shape) == 3:
            transformed = transformed.reshape(original_shape)

        return transformed

    def fit_transform(self, train_data: np.ndarray) -> np.ndarray:
        """Fit scalers and transform training data."""
        self.fit(train_data)
        return self.transform(train_data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale.

        Args:
            data: Normalized data

        Returns:
            Data in original scale
        """
        if self.normalization == "none":
            return data

        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        # Store original shape
        original_shape = data.shape

        # Reshape for transformation
        if len(data.shape) == 3:
            num_samples, window_size, num_features = data.shape
            data_flat = data.reshape(-1, num_features)
        elif len(data.shape) == 2:
            data_flat = data
            num_features = data.shape[1]
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")

        # Inverse transform
        if self.per_feature_norm:
            inverse = np.zeros_like(data_flat)
            for i in range(num_features):
                inverse[:, i:i+1] = self.scalers[i].inverse_transform(data_flat[:, i:i+1])
        else:
            inverse = self.global_scaler.inverse_transform(data_flat)

        # Reshape back to original shape
        if len(original_shape) == 3:
            inverse = inverse.reshape(original_shape)

        return inverse

    def create_patches(self, data: np.ndarray) -> np.ndarray:
        """
        Create patches from time-series windows for MOMENT.

        PatchTST-style patching: Divide each window into overlapping patches.

        Args:
            data: Windowed data of shape (num_samples, window_size, num_features)

        Returns:
            Patched data of shape (num_samples, num_patches, patch_len, num_features)
        """
        if self.patch_len is None:
            logger.info("Patching disabled")
            return data

        if len(data.shape) != 3:
            raise ValueError(f"Expected 3D data (samples, window, features), got {data.shape}")

        num_samples, window_size, num_features = data.shape

        if window_size != self.window_size:
            logger.warning(f"Window size mismatch: expected {self.window_size}, got {window_size}")

        # Calculate number of patches
        num_patches = (window_size - self.patch_len) // self.patch_stride + 1

        # Create patches
        patches = []
        for i in range(num_patches):
            start_idx = i * self.patch_stride
            end_idx = start_idx + self.patch_len

            if end_idx > window_size:
                break

            patch = data[:, start_idx:end_idx, :]  # (num_samples, patch_len, num_features)
            patches.append(patch)

        # Stack patches: (num_samples, num_patches, patch_len, num_features)
        patched_data = np.stack(patches, axis=1)

        logger.info(f"Created patches: {patched_data.shape}")
        logger.info(f"  Num patches per window: {num_patches}")
        logger.info(f"  Patch length: {self.patch_len}")
        logger.info(f"  Patch stride: {self.patch_stride}")

        return patched_data

    def process_pipeline(
        self,
        data: np.ndarray,
        is_train: bool = False,
    ) -> np.ndarray:
        """
        Full preprocessing pipeline.

        Args:
            data: Input data
            is_train: Whether this is training data (fit scalers)

        Returns:
            Preprocessed data
        """
        logger.info(f"Processing {'training' if is_train else 'inference'} data")
        logger.info(f"Input shape: {data.shape}")

        # Step 1: Handle missing values
        data = self.handle_missing_values(data)

        # Step 2: Normalize
        if is_train:
            data = self.fit_transform(data)
        else:
            data = self.transform(data)

        # Step 3: Create patches (if enabled)
        if self.patch_len is not None and len(data.shape) == 3:
            data = self.create_patches(data)

        logger.info(f"Output shape: {data.shape}")
        return data


if __name__ == "__main__":
    # Test preprocessor
    logger.info("Testing TimeSeriesPreprocessor")

    # Create dummy data
    num_samples = 100
    window_size = 256
    num_features = 88

    np.random.seed(42)
    train_data = np.random.randn(num_samples, window_size, num_features)
    test_data = np.random.randn(20, window_size, num_features)

    # Add some missing values
    train_data[0, 10:15, 5] = np.nan

    # Initialize preprocessor
    preprocessor = TimeSeriesPreprocessor(
        window_size=256,
        normalization="zscore",
        per_feature_norm=True,
        patch_len=16,
        patch_stride=8,
    )

    # Process data
    train_processed = preprocessor.process_pipeline(train_data, is_train=True)
    test_processed = preprocessor.process_pipeline(test_data, is_train=False)

    logger.info(f"Train processed shape: {train_processed.shape}")
    logger.info(f"Test processed shape: {test_processed.shape}")

    logger.info("Preprocessor test completed!")
