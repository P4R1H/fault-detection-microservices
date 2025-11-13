"""
Configuration Management for MOMENT Anomaly Detection Pipeline

This module handles all configuration settings for the project, including
model parameters, training settings, and hardware constraints.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import yaml
import json
from pathlib import Path


@dataclass
class HardwareConfig:
    """Hardware-specific configuration for RTX 4070 (8GB VRAM)."""

    device: str = "cuda"
    gpu_id: int = 0
    vram_gb: int = 8
    use_mixed_precision: bool = True  # FP16 for memory savings
    gradient_checkpointing: bool = True  # MANDATORY for 8GB VRAM
    max_batch_size: int = 16
    recommended_batch_size: int = 8
    num_workers: int = 4  # DataLoader workers
    pin_memory: bool = True

    def __post_init__(self):
        """Validate hardware configuration."""
        if self.vram_gb <= 8 and self.max_batch_size > 16:
            print(f"Warning: Batch size {self.max_batch_size} may cause OOM on {self.vram_gb}GB VRAM")
            print(f"Recommended: {self.recommended_batch_size}")


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""

    # Paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    dataset_name: str = "trainticket"

    # Time series parameters
    window_size: int = 256  # 5-10 min context
    stride: int = 128  # 50% overlap
    num_features: int = 88  # From Phase 1 feature engineering

    # PatchTST-style patching for MOMENT
    patch_len: int = 16
    patch_stride: int = 8

    # Normalization
    normalization: str = "zscore"  # "zscore", "minmax", "robust"
    per_feature_norm: bool = True

    # Missing values
    missing_value_strategy: str = "forward_fill"  # "forward_fill", "interpolate", "zero"

    # Splits
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Fault types (from TrainTicket)
    num_fault_types: int = 14
    fault_types: List[str] = field(default_factory=lambda: [
        "cpu_load", "memory_leak", "network_delay", "disk_io",
        "database_slow", "service_crash", "pod_failure", "node_failure",
        "replica_failure", "config_error", "dependency_failure",
        "cascading_failure", "load_spike", "normal"
    ])

    def __post_init__(self):
        """Validate data configuration."""
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "Train/val/test ratios must sum to 1.0"


@dataclass
class MOMENTConfig:
    """MOMENT Foundation Model configuration."""

    # Model selection
    model_name: str = "AutonLab/MOMENT-1-large"
    model_size: str = "large"  # "small", "base", "large"
    num_parameters: int = 385_000_000  # 385M params

    # Task configuration
    task: str = "classification"  # "classification", "forecasting", "anomaly_detection"
    num_classes: int = 2  # Binary: normal vs. anomaly

    # Architecture
    d_model: int = 1024  # Hidden dimension (large model)
    n_heads: int = 16
    n_layers: int = 24
    dropout: float = 0.1

    # Quantization (for memory savings)
    load_in_8bit: bool = False  # Enable if OOM occurs
    load_in_4bit: bool = False  # More aggressive if needed

    # Cache
    use_cache: bool = True
    cache_dir: str = "data/model_cache"


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration for few-shot fine-tuning."""

    # LoRA parameters
    r: int = 8  # Rank (reduced from 16 for memory)
    lora_alpha: int = 16  # Scaling factor
    lora_dropout: float = 0.1

    # Target modules (attention layers)
    target_modules: List[str] = field(default_factory=lambda: [
        "query", "value", "key", "dense"  # Transformer attention layers
    ])

    # Fine-tuning strategy
    bias: str = "none"  # "none", "all", "lora_only"
    task_type: str = "SEQ_CLS"  # Sequence classification

    # Inference mode
    inference_mode: bool = False


@dataclass
class TrainingConfig:
    """Training and fine-tuning configuration."""

    # Zero-shot (no training)
    zero_shot: bool = True

    # Few-shot configurations
    few_shot_samples: List[int] = field(default_factory=lambda: [100, 500, 1000])
    full_training_samples: int = 10000

    # Training hyperparameters
    num_epochs: int = 10
    learning_rate: float = 1e-4  # LoRA learning rate
    weight_decay: float = 0.01
    warmup_steps: int = 100

    # Optimizer
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Learning rate schedule
    lr_scheduler: str = "cosine"  # "linear", "cosine", "constant"

    # Early stopping
    early_stopping: bool = True
    patience: int = 3
    min_delta: float = 0.001

    # Gradient management
    max_grad_norm: float = 1.0  # Gradient clipping

    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500


@dataclass
class EvaluationConfig:
    """Evaluation metrics and visualization configuration."""

    # Primary metrics
    primary_metric: str = "f1"  # "f1", "precision", "recall"
    metrics: List[str] = field(default_factory=lambda: [
        "f1", "precision", "recall", "accuracy",
        "roc_auc", "pr_auc", "fpr_at_95_recall"
    ])

    # Threshold
    classification_threshold: float = 0.5
    optimize_threshold: bool = True  # Optimize on validation set

    # Point-adjusted evaluation (for time series)
    point_adjusted: bool = True
    adjustment_window: int = 10  # timesteps

    # Confusion matrix
    save_confusion_matrix: bool = True

    # Visualization
    plot_learning_curves: bool = True
    plot_attention_heatmaps: bool = True
    plot_tsne_embeddings: bool = True

    # Results
    results_dir: str = "data/results"
    save_predictions: bool = True


@dataclass
class BaselineConfig:
    """Configuration for baseline models (Phase 1)."""

    # Isolation Forest
    isolation_forest_enabled: bool = True
    if_n_estimators: int = 100
    if_contamination: float = 0.05  # 5% anomaly ratio

    # Random Forest
    random_forest_enabled: bool = True
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    rf_class_weight: str = "balanced"

    # LSTM Autoencoder
    lstm_ae_enabled: bool = True
    lstm_hidden_dim: int = 64
    lstm_latent_dim: int = 16
    lstm_num_layers: int = 2
    lstm_sequence_len: int = 50

    # CatBoost (Phase 2 planned baseline)
    catboost_enabled: bool = False
    cb_iterations: int = 1000
    cb_learning_rate: float = 0.03
    cb_depth: int = 6


@dataclass
class ExperimentConfig:
    """Complete experiment configuration combining all configs."""

    # Experiment metadata
    experiment_name: str = "moment_fewshot_anomaly_detection"
    project_name: str = "microservice_fault_detection"
    random_seed: int = 42

    # Sub-configurations
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    data: DataConfig = field(default_factory=DataConfig)
    moment: MOMENTConfig = field(default_factory=MOMENTConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)

    # Paths
    checkpoint_dir: str = "data/checkpoints"
    log_dir: str = "logs"

    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        print(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'experiment_name': self.experiment_name,
            'project_name': self.project_name,
            'random_seed': self.random_seed,
            'hardware': self.hardware.__dict__,
            'data': self.data.__dict__,
            'moment': self.moment.__dict__,
            'lora': self.lora.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'baselines': self.baselines.__dict__,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary."""
        return cls(
            experiment_name=config_dict.get('experiment_name', 'default'),
            project_name=config_dict.get('project_name', 'default'),
            random_seed=config_dict.get('random_seed', 42),
            hardware=HardwareConfig(**config_dict.get('hardware', {})),
            data=DataConfig(**config_dict.get('data', {})),
            moment=MOMENTConfig(**config_dict.get('moment', {})),
            lora=LoRAConfig(**config_dict.get('lora', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            baselines=BaselineConfig(**config_dict.get('baselines', {})),
            checkpoint_dir=config_dict.get('checkpoint_dir', 'data/checkpoints'),
            log_dir=config_dict.get('log_dir', 'logs'),
        )


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration optimized for RTX 4070 (8GB VRAM)."""
    return ExperimentConfig()


if __name__ == "__main__":
    # Example: Create and save default configuration
    config = get_default_config()
    config.save("configs/default_config.yaml")
    print("Default configuration created!")

    # Print summary
    print(f"\nExperiment: {config.experiment_name}")
    print(f"Hardware: {config.hardware.device} ({config.hardware.vram_gb}GB VRAM)")
    print(f"Batch size: {config.hardware.recommended_batch_size}")
    print(f"Mixed precision: {config.hardware.use_mixed_precision}")
    print(f"Model: {config.moment.model_name}")
    print(f"LoRA rank: {config.lora.r}")
    print(f"Few-shot samples: {config.training.few_shot_samples}")
