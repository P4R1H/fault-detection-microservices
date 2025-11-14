# Architectural Decisions & Rationale

**Last Updated**: 2025-11-14

## Decision Log

### D001: Project Structure
**Date**: 2025-11-14
**Decision**: Modular architecture with separate modules for each modality
**Rationale**:
- Enables independent development and testing of each component
- Facilitates ablation studies (can easily disable/enable modules)
- Improves code maintainability
- Supports parallel development if needed

**Structure**:
```
/src
    /metrics_module    # Time-series forecasting & anomaly detection
    /logs_module       # Log parsing, embeddings, anomaly scoring
    /traces_module     # Service graph + GNN for RCA
    /causal_module     # PCMCI causal discovery
    /fusion_module     # Multimodal fusion architecture
    /baselines         # Baseline implementations
    /models            # Shared model components
    /utils             # Common utilities
```

### D002: Context Management
**Date**: 2025-11-14
**Decision**: Maintain four separate context files (memory.md, task_list.md, decisions.md, notes.md)
**Rationale**:
- **memory.md**: High-level knowledge state - what we know and don't know
- **task_list.md**: Execution tracking - what's done, what's next
- **decisions.md**: Design rationale - why we chose specific approaches
- **notes.md**: Quick reminders, failures, lessons learned
- Separation allows quick reference to different types of information
- Enables better organization than single monolithic file

### D003: Documentation Strategy
**Date**: 2025-11-14
**Decision**: Separate polished docs (/docs) from working context (/context)
**Rationale**:
- /context = working memory (informal, evolving)
- /docs = publication-ready (formal, polished)
- Clear boundary prevents mixing draft notes with final deliverables
- Easier to share final docs without exposing internal working notes

---

## Pending Decisions (Awaiting User Input)

### PD001: Forecasting Model Selection
**Question**: Chronos-Bolt-Tiny vs TCN?
**Options**:
- **Chronos-Bolt-Tiny**: Zero-shot foundation model, no training needed, modern approach
- **TCN**: Temporal Convolutional Network, requires training, proven baseline
**Factors to Consider**:
- Compute budget
- Training time availability
- Desire for foundation model comparison
- Dataset size compatibility
**Status**: Waiting for user preference or dataset characteristics

### PD002: GNN Architecture
**Question**: GCN vs GAT?
**Options**:
- **GCN**: Simpler, faster, good baseline
- **GAT**: Attention mechanism, more expressive, potentially better for RCA
**Status**: Waiting for user preference

### PD003: Baseline Selection
**Question**: Which 5+ baselines to compare?
**Known**: BARO is required
**Potential Others**:
- Statistical methods (ARIMA, Isolation Forest)
- Deep learning baselines (LSTM, Transformer)
- Existing RCA methods (MicroRCA, etc.)
**Status**: Waiting for literature review to identify key baselines

### PD004: Evaluation Metrics
**Question**: Primary metrics for RCA performance?
**Options**:
- Precision@k (k=1,3,5)
- F1-score
- NDCG
- Mean Average Precision
**Status**: Waiting for user requirements and literature review

---

## Implementation Constraints

### Known Constraints
1. Must use RCAEval TrainTicket dataset
2. Must use PCMCI via tigramite
3. Must use Drain3 for log parsing
4. Must implement 10+ ablation configurations
5. Must compare against 5+ baselines
6. Must maintain reproducibility

### Unknown Constraints (To Be Determined)
- Compute resources (GPU memory, CPU cores)
- Time budget for experiments
- Library version constraints
- Code style requirements
- Specific evaluation protocols from literature

---

## Notes
- This document will grow as implementation progresses
- Each major decision should be logged with rationale
- Failed approaches should be documented to avoid repetition
