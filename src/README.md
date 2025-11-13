# Source Code Structure

This directory contains all source code for the MOMENT Foundation Model anomaly detection pipeline.

## Directory Organization

```
src/
├── data/               # Data loading and preprocessing
│   ├── loader.py       # Dataset loading utilities
│   ├── preprocessor.py # Time-series preprocessing (windowing, normalization)
│   └── augmentation.py # Data augmentation (if needed)
│
├── models/             # Model architectures
│   ├── moment_wrapper.py    # MOMENT model wrapper with memory optimization
│   ├── anomaly_detector.py  # Anomaly detection head
│   └── model_factory.py     # Model creation factory
│
├── training/           # Training and fine-tuning
│   ├── zero_shot.py    # Zero-shot evaluation
│   ├── few_shot.py     # Few-shot LoRA fine-tuning
│   ├── trainer.py      # Generic trainer class
│   └── lora_config.py  # LoRA configuration
│
├── evaluation/         # Evaluation and metrics
│   ├── metrics.py      # F1, Precision, Recall, AUC
│   ├── visualizer.py   # Plotting and visualization
│   └── comparator.py   # Baseline comparison
│
├── utils/              # Utilities
│   ├── config.py       # Configuration management
│   ├── memory.py       # Memory profiling and optimization
│   ├── logger.py       # Logging setup
│   └── checkpoint.py   # Model checkpointing
│
└── baselines/          # Baseline implementations
    ├── isolation_forest.py  # Isolation Forest baseline
    ├── lstm_ae.py           # LSTM Autoencoder baseline
    └── random_forest.py     # Random Forest baseline
```

## Module Descriptions

### data/
Handles all data-related operations including:
- Loading TrainTicket benchmark data
- Time-series windowing (256 timesteps)
- Normalization and standardization
- Patching for MOMENT (PatchTST-style)
- Train/val/test splitting

### models/
Contains model architectures:
- MOMENT wrapper with memory optimization (gradient checkpointing, mixed precision)
- Custom anomaly detection heads
- Model factory for easy instantiation

### training/
Training logic:
- Zero-shot inference (no training)
- Few-shot LoRA fine-tuning
- Generic trainer with memory management
- Hyperparameter configurations

### evaluation/
Evaluation and analysis:
- Comprehensive metrics (F1, Precision, Recall, PR-AUC, ROC-AUC)
- Visualization (learning curves, confusion matrices, attention heatmaps)
- Comparison with baselines

### utils/
Helper utilities:
- Configuration management (YAML/JSON)
- Memory profiling and GPU monitoring
- Logging and experiment tracking
- Checkpoint saving/loading

### baselines/
Baseline model implementations for comparison:
- Isolation Forest (from Phase 1)
- LSTM Autoencoder (from Phase 1)
- Random Forest (from Phase 1)

## Design Principles

1. **Modularity**: Each component is self-contained and reusable
2. **Type Hints**: All functions have type annotations
3. **Docstrings**: Google-style docstrings for all classes and functions
4. **Memory Efficiency**: Optimized for RTX 4070 (8GB VRAM)
5. **Reproducibility**: Seed setting and deterministic operations
6. **Logging**: Comprehensive logging for debugging

## Usage Examples

```python
# Load and preprocess data
from src.data.loader import load_trainticket_data
from src.data.preprocessor import TimeSeriesPreprocessor

data = load_trainticket_data('data/raw/trainticket.csv')
preprocessor = TimeSeriesPreprocessor(window_size=256, patch_len=16)
train, val, test = preprocessor.process(data)

# Zero-shot evaluation
from src.training.zero_shot import ZeroShotEvaluator

evaluator = ZeroShotEvaluator(model_name='AutonLab/MOMENT-1-large')
results = evaluator.evaluate(test)

# Few-shot fine-tuning
from src.training.few_shot import FewShotTrainer

trainer = FewShotTrainer(
    model_name='AutonLab/MOMENT-1-large',
    lora_rank=8,
    batch_size=8,
    use_fp16=True
)
trainer.train(train_subset=100)  # 100 samples
```

## Development Workflow

1. Start with small data subset for testing
2. Profile memory usage with `src/utils/memory.py`
3. Test on CPU first, then GPU
4. Monitor GPU with: `watch -n 0.5 nvidia-smi`
5. Log all experiments to `logs/`
6. Save checkpoints to `data/checkpoints/`
