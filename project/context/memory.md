# Project Memory

**Last Updated**: 2025-11-14

## What I Know

### Project Scope
- **Domain**: AIOps for Microservice Anomaly Detection + Root Cause Analysis
- **Dataset**: RCAEval TrainTicket dataset
- **Modalities**: Metrics, Logs, Traces (multimodal fusion)
- **Key Technologies**:
  - Forecasting: Chronos-Bolt-Tiny OR TCN
  - Causal Discovery: PCMCI (via tigramite)
  - Graph Learning: 2-layer GNN (GCN/GAT) on service dependency graphs
  - Log Parsing: Drain3
  - Intermediate multimodal fusion with cross-modal attention

### Research Objectives
1. Implement anomaly detection across three modalities
2. Build causal inference pipeline using PCMCI
3. Construct service dependency graphs from traces
4. Develop intermediate fusion architecture
5. Conduct comprehensive ablation studies (10+ configurations)
6. Compare against 5+ baselines (including BARO)
7. Produce publication-grade documentation

### Technical Architecture (Preliminary)
- **Metrics Module**: Time-series forecasting for anomaly detection
- **Logs Module**: Template extraction → embeddings → anomaly scoring
- **Traces Module**: Dependency graph construction → GNN-based RCA
- **Causal Module**: PCMCI for causal graph discovery
- **Fusion Module**: Separate encoders + cross-modal attention → unified scoring

### Deliverables Required
- Clean, modular codebase
- 10+ ablation experiments
- 5+ baseline comparisons
- Tables, plots, comparative analysis
- Academic-grade documentation in /docs
- Reproducible experimental setup

## What I Don't Know Yet (Waiting for User Input)

### Critical Information Needed
1. **Mid-semester report** - Understanding current progress and requirements
2. **Project proposals** - Detailed scope and expectations
3. **Research notes** - Domain-specific insights
4. **Literature review** - Papers to cite and build upon (2020-2025)
5. **Datasets/Links** - Access to RCAEval TrainTicket dataset
6. **Compute Resources** - Available GPU/memory constraints
7. **Dependencies** - Any pre-installed libraries or version constraints
8. **Deadlines** - Project completion timeline

### Ambiguities to Resolve
- Preferred forecasting model (Chronos-Bolt-Tiny vs TCN)
- GNN architecture preference (GCN vs GAT)
- Evaluation metrics priority (Precision@k, F1, NDCG?)
- Baseline implementations (from scratch vs existing repos)
- Hyperparameter search budget
- Documentation style guide (if any specific format required)

## Key Facts Remembered
- Project is university-grade major project
- Must be reproducible and modular
- Academic format required for all docs
- Citations needed (user will add bib entries later)
- All code must go in correct folders
- Context files must be updated after each major step
- NEVER hallucinate missing information - always ask user

## Current State
**Status**: Initialized - Awaiting materials from user
**Next Step**: Receive and analyze user-provided materials
