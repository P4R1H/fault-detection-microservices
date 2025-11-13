"""
Logging Utilities for Experiment Tracking

This module provides comprehensive logging for training, evaluation, and debugging.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from loguru import logger
import json


class ExperimentLogger:
    """
    Comprehensive logger for ML experiments with file and console output.
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        level: str = "INFO",
        console_output: bool = True,
    ):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for log files
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
            console_output: Whether to print to console
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"

        # Remove default logger
        logger.remove()

        # Add console handler
        if console_output:
            logger.add(
                sys.stdout,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
                level=level,
                colorize=True,
            )

        # Add file handler
        logger.add(
            self.log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation="100 MB",  # Rotate after 100MB
            retention="30 days",  # Keep logs for 30 days
        )

        # Metrics log file (JSON format for easy parsing)
        self.metrics_file = self.log_dir / f"{experiment_name}_{timestamp}_metrics.jsonl"

        logger.info(f"Experiment Logger Initialized: {experiment_name}")
        logger.info(f"Log file: {self.log_file}")
        logger.info(f"Metrics file: {self.metrics_file}")

    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        logger.info("="*60)
        logger.info("EXPERIMENT CONFIGURATION")
        logger.info("="*60)

        # Save full config to JSON
        config_file = self.log_dir / f"{self.experiment_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Configuration saved to: {config_file}")

        # Log key settings
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, dict):
                    logger.info(f"{key}:")
                    for sub_key, sub_value in value.items():
                        logger.info(f"  {sub_key}: {sub_value}")
                else:
                    logger.info(f"{key}: {value}")

        logger.info("="*60)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        phase: str = "train",
        prefix: str = "",
    ):
        """
        Log metrics to both console and metrics file.

        Args:
            metrics: Dictionary of metric names and values
            step: Training step/epoch
            phase: Phase of training (train, val, test)
            prefix: Optional prefix for metric names
        """
        # Prepare metrics dict
        metrics_dict = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "step": step,
        }

        # Add metrics with optional prefix
        for key, value in metrics.items():
            metric_name = f"{prefix}{key}" if prefix else key
            metrics_dict[metric_name] = value

        # Log to console
        metric_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                 for k, v in metrics.items()])
        logger.info(f"[{phase.upper()}] Step {step} | {metric_str}")

        # Append to metrics file (JSONL format)
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics_dict) + '\n')

    def log_model_summary(
        self,
        model_name: str,
        num_params: int,
        num_trainable: int,
        memory_mb: float,
    ):
        """Log model summary information."""
        logger.info("="*60)
        logger.info(f"MODEL SUMMARY: {model_name}")
        logger.info("="*60)
        logger.info(f"Total Parameters:     {num_params:,}")
        logger.info(f"Trainable Parameters: {num_trainable:,}")
        logger.info(f"Trainable %:          {100.0 * num_trainable / num_params:.2f}%")
        logger.info(f"Memory Footprint:     {memory_mb:.2f} MB")
        logger.info("="*60)

    def log_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
    ):
        """Log end-of-epoch summary."""
        logger.info("="*60)
        logger.info(f"EPOCH {epoch} SUMMARY")
        logger.info("="*60)

        logger.info("Training Metrics:")
        for key, value in train_metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        if val_metrics:
            logger.info("Validation Metrics:")
            for key, value in val_metrics.items():
                logger.info(f"  {key}: {value:.4f}")

        logger.info("="*60)

    def log_final_results(
        self,
        test_metrics: Dict[str, float],
        best_epoch: Optional[int] = None,
        best_val_metric: Optional[float] = None,
    ):
        """Log final test results."""
        logger.info("")
        logger.info("="*60)
        logger.info("FINAL TEST RESULTS")
        logger.info("="*60)

        if best_epoch is not None:
            logger.info(f"Best Epoch: {best_epoch}")
        if best_val_metric is not None:
            logger.info(f"Best Validation Metric: {best_val_metric:.4f}")

        logger.info("\nTest Metrics:")
        for key, value in test_metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        logger.info("="*60)
        logger.info("")

    def log_comparison(
        self,
        model_results: Dict[str, Dict[str, float]],
        metric_name: str = "f1",
    ):
        """
        Log comparison between multiple models.

        Args:
            model_results: Dict mapping model names to their metrics
            metric_name: Primary metric for comparison
        """
        logger.info("")
        logger.info("="*60)
        logger.info(f"MODEL COMPARISON (Primary Metric: {metric_name})")
        logger.info("="*60)

        # Sort models by primary metric
        sorted_models = sorted(
            model_results.items(),
            key=lambda x: x[1].get(metric_name, 0),
            reverse=True
        )

        for rank, (model_name, metrics) in enumerate(sorted_models, 1):
            logger.info(f"\n#{rank} {model_name}")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")

        logger.info("="*60)
        logger.info("")


def setup_logging(
    experiment_name: str,
    log_dir: str = "logs",
    level: str = "INFO",
) -> ExperimentLogger:
    """
    Setup logging for an experiment.

    Args:
        experiment_name: Name of the experiment
        log_dir: Directory for log files
        level: Logging level

    Returns:
        ExperimentLogger instance
    """
    return ExperimentLogger(
        experiment_name=experiment_name,
        log_dir=log_dir,
        level=level,
    )


if __name__ == "__main__":
    # Test logging
    exp_logger = setup_logging("test_experiment")

    # Test config logging
    config = {
        "model": "MOMENT-1-large",
        "batch_size": 8,
        "learning_rate": 1e-4,
        "hardware": {
            "device": "cuda",
            "vram": "8GB",
        }
    }
    exp_logger.log_config(config)

    # Test metrics logging
    exp_logger.log_metrics(
        metrics={"loss": 0.5, "f1": 0.85, "precision": 0.88},
        step=100,
        phase="train"
    )

    # Test model summary
    exp_logger.log_model_summary(
        model_name="MOMENT",
        num_params=385_000_000,
        num_trainable=1_000_000,
        memory_mb=1500.0
    )

    # Test final results
    exp_logger.log_final_results(
        test_metrics={"f1": 0.92, "precision": 0.94, "recall": 0.90},
        best_epoch=8,
        best_val_metric=0.89
    )

    logger.info("Logging test completed!")
