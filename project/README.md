# AIOps Microservice Anomaly Detection & Root Cause Analysis

**Status**: Initialization Phase
**Last Updated**: 2025-11-14

## Project Overview

This project implements a comprehensive multimodal AIOps system for microservice anomaly detection and root cause analysis. The system integrates three data modalities (metrics, logs, traces) using modern deep learning techniques including foundation models, graph neural networks, and causal inference.

### Key Features
- **Multimodal Fusion**: Combines metrics, logs, and traces for robust RCA
- **Foundation Models**: Leverages Chronos-Bolt-Tiny or TCN for time-series forecasting
- **Causal Discovery**: Uses PCMCI algorithm for causal graph inference
- **Graph Learning**: 2-layer GNN on service dependency graphs
- **Comprehensive Evaluation**: 10+ ablations and 5+ baseline comparisons
- **Production-Ready**: Modular, reproducible, well-documented

## Project Structure

```
/project
├── /data                    # RCAEval TrainTicket dataset
├── /src                     # Source code
│   ├── /metrics_module      # Time-series anomaly detection
│   ├── /logs_module         # Log parsing and analysis
│   ├── /traces_module       # Service graph + GNN
│   ├── /causal_module       # PCMCI causal inference
│   ├── /fusion_module       # Multimodal fusion
│   ├── /baselines           # Baseline implementations
│   ├── /models              # Shared model components
│   └── /utils               # Common utilities
├── /experiments             # Experimental results
├── /docs                    # Academic documentation
│   ├── final_report.md
│   ├── architecture.md
│   ├── literature_review.md
│   ├── methodology.md
│   ├── results.md
│   ├── ablations.md
│   └── experiments.md
└── /context                 # Project memory (internal)
    ├── memory.md
    ├── task_list.md
    ├── decisions.md
    └── notes.md
```

## Quick Start

**Note**: This project is currently in the initialization phase. Setup instructions will be added as implementation progresses.

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Required libraries: PyTorch, tigramite, Drain3, PyG/DGL

### Installation
```bash
# Installation instructions will be added
```

### Usage
```bash
# Usage instructions will be added
```

## Dataset

This project uses the **RCAEval TrainTicket dataset**, a standard benchmark for microservice root cause analysis containing:
- System metrics (CPU, memory, latency, etc.)
- Application logs
- Distributed traces
- Ground-truth fault labels

## Methodology

### Architecture Components

1. **Metrics Module**: Time-series forecasting using Chronos-Bolt-Tiny (zero-shot) or TCN
2. **Logs Module**: Drain3 parser + template embeddings + anomaly scoring
3. **Traces Module**: Service dependency graph construction + 2-layer GNN
4. **Causal Module**: PCMCI-based causal graph discovery
5. **Fusion Module**: Cross-modal attention for intermediate fusion

### Ablation Studies

The project includes comprehensive ablation studies:
- Single modality: Metrics-only, Logs-only, Traces-only
- Pairwise: Metrics+Logs, Metrics+Traces, Logs+Traces
- Full system: All modalities
- Architecture: With/without GNN, with/without PCMCI
- Foundation models: With/without pretrained models

### Baselines

Comparison against 5+ state-of-the-art baselines including:
- BARO
- Statistical methods
- Deep learning approaches
- (Additional baselines TBD)

## Results

Results will be added as experiments are completed.

## Documentation

Full academic documentation is available in the `/docs` folder:
- **final_report.md**: Complete project report
- **architecture.md**: System architecture and design
- **methodology.md**: Detailed methodology and experimental setup
- **results.md**: Experimental results and analysis
- **ablations.md**: Ablation study results
- **literature_review.md**: Related work and references

## Contributing

This is a university research project. For questions or collaboration inquiries, please contact the project maintainer.

## License

TBD

## Acknowledgments

- RCAEval benchmark dataset
- Chronos foundation model by Amazon
- PCMCI algorithm by Jakob Runge et al.
- Drain3 log parser

---

**Project Status**: Awaiting initial materials and dataset access
