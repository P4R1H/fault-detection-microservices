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
- **Chronos-Bolt-Tiny**: Zero-shot foundation model, 20M params, 100MB VRAM, 250x faster inference, no training needed
- **TCN**: Temporal Convolutional Network, <10M params, requires training, 3-5x faster than LSTM, more control

**Recommendation Based on Research**:
**IMPLEMENT BOTH** for comprehensive comparison:
1. **Start with Chronos-Bolt-Tiny** (Week 1-2):
   - Immediate results (zero-shot)
   - Establishes baseline quickly
   - Demonstrates modern foundation model usage
   - Minimal VRAM (100MB)
   - Great for ablation: "with vs without pretrained foundation model"

2. **Add TCN** (Week 2-3):
   - Full control over architecture
   - Domain-specific fine-tuning possible
   - Comparison shows understanding of trade-offs
   - Lightweight enough to train alongside Chronos

**Justification**: Having both provides excellent ablation study ("with/without foundation model") and shows technical breadth. Research notes confirm both fit 8GB VRAM constraint. This is exactly what distinguishes A+ projects.

**Status**: RECOMMENDED APPROACH - Both models complement each other

### PD002: GNN Architecture
**Question**: GCN vs GAT?
**Options**:
- **GCN**: Simpler, faster, similar performance on small graphs (<30 nodes), 1-3MB memory, 15-line implementation
- **GAT**: Attention mechanism, more expressive, interpretable, 3-8MB memory

**Recommendation Based on Research**:
**Start with 2-layer GCN, prepare GAT upgrade if time permits**

**Rationale from Literature**:
- 2022 survey confirms GCN remains most popular for service dependency graphs
- For 10-30 node microservices (TrainTicket ~41 services), both perform similarly
- GCN advantage: Development time saved, simpler debugging
- GAT advantage: Interpretable attention weights (good for presentations)

**Configuration**:
- **GCN**: 2-3 layers, hidden_dim=64, dropout=0.3-0.5, lr=0.01, weight_decay=5e-4
- **GAT** (if implemented): 2-3 layers, 4-8 heads (first layer), 1 head (output), hidden_dim=16-32 per head

**Upgrade Triggers**:
- If GCN performance plateaus
- If need interpretability for report visualizations
- If heterogeneous service types require different attention weights

**Status**: START WITH GCN - Upgrade to GAT if needed and time allows

### PD003: Baseline Selection
**Question**: Which 5+ baselines to compare?

**DECIDED - 7 Baselines** (Based on RCAEval and Literature):
1. **BARO** (FSE'24) - Bayesian online change point detection - REQUIRED
2. **Random Forest** (Phase 1) - Existing supervised baseline - ALREADY HAVE
3. **Isolation Forest** (Phase 1) - Unsupervised density - ALREADY HAVE
4. **Statistical (3-sigma/ARIMA)** - Classical approach - SIMPLE TO ADD
5. **MicroRCA** (2020) - Graph-based PageRank - MEDIUM COMPLEXITY
6. **Granger-Lasso** - Fast causal baseline - 5-MIN IMPLEMENTATION
7. **RCAEval built-in methods** - Use existing implementations - FREE

**Rationale**:
- 3 baselines already exist from Phase 1 (RF, IF, LSTM-AE)
- Statistical methods quick to implement
- RCAEval provides 15 baseline implementations - leverage these
- Granger-Lasso provides causal baseline before PCMCI
- Total 7+ baselines demonstrates thoroughness

**Status**: DECIDED - Implementation priority: Simple statistical → Granger → BARO → MicroRCA

### PD004: Evaluation Metrics
**Question**: Primary metrics for RCA performance?

**DECIDED - RCAEval Standard Metrics**:
1. **AC@k** (Accuracy at k) - Is ground truth in top-k predictions? (k=1,3,5)
2. **Avg@k** - Weighted by rank: 1/rank if found in top-k, else 0
3. **MRR** (Mean Reciprocal Rank) - Average of 1/rank across cases
4. **For Anomaly Detection**: NAB scoring (Standard, Reward Low FP, Reward Low FN)
5. **Standard ML**: F1-score, Precision, Recall, AUC-ROC

**Rationale**:
- AC@k is standard for RCA evaluation in literature
- MRR provides single-number summary
- NAB scoring rewards early detection (critical for AIOps)
- Using RCAEval standards ensures comparability
- Multiple metrics prevent cherry-picking

**Statistical Significance**:
- Report mean ± std from 3-5 runs with different seeds
- Paired t-tests (p<0.05) for method comparisons
- Wilcoxon signed-rank for non-normal distributions

**Status**: DECIDED - Use RCAEval standard metrics (AC@k, Avg@k, MRR) + NAB

---

## Major Architectural Decisions (NEW - Based on Research)

### D004: Dataset Selection - RCAEval RE2-TrainTicket
**Date**: 2025-11-14
**Decision**: Use RCAEval RE2-TrainTicket as primary dataset (270 multimodal failure cases)
**Rationale**:
- **Only public dataset** with multimodal microservice fault data + causal ground truth
- Recent benchmark (WWW'25, ASE 2024) - cutting-edge positioning
- Complete ground truth: root cause service AND indicator
- Three modalities synchronized: metrics (77-376), logs (8.6-26.9M lines), traces (39.6-76.7M)
- 15 baseline RCA methods already implemented (free comparisons)
- Ideal size for academic project (270 cases vs SockShop DIY ~weeks of work)
- 4.21GB compressed - manageable download
- Simple installation: `pip install RCAEval[default]`

**Alternatives Rejected**:
- SockShop standalone: Requires 1-2 weeks deployment + fault injection + labeling
- Alibaba traces: 2TB, lacks fault scenarios, weeks of preprocessing
- Google/Azure datasets: VM-level, not microservices

**Impact**: Foundation for entire project - all experiments use this dataset

### D005: Replace LSTM-AE with Modern Alternatives
**Date**: 2025-11-14
**Decision**: Replace LSTM-AE with BOTH Chronos-Bolt-Tiny AND TCN
**Rationale**:
- **Research consensus**: LSTM obsolete in 2025 (Phase 1 identified 25.4s bottleneck)
- Transformers: 250x faster inference
- TCN: 3-5x faster training, 5-10% better F1
- Chronos: Zero-shot (no training), 100MB VRAM, modern foundation model
- **Both models fit 8GB VRAM** and provide excellent ablation study

**Evidence from Literature**:
- Yahoo S5: LSTM F1=0.82 vs Transformer F1=0.91
- SMD: LSTM F1=0.74 vs Anomaly Transformer F1=0.89
- TCN receptive field: 381 timesteps with 7 layers vs LSTM struggling >200
- Chronos trained on 100+ billion time series observations

**Implementation Strategy**:
1. Week 1-2: Chronos-Bolt-Tiny for immediate zero-shot results
2. Week 2-3: TCN for domain-specific training
3. Week 4: Compare both in ablation study

**Impact**: Solves Phase 1 computational bottleneck, demonstrates modern techniques

### D006: PCMCI for Causal Discovery
**Date**: 2025-11-14
**Decision**: Use PCMCI/PCMCIplus from tigramite library as primary causal method
**Rationale**:
- **Gold standard** for time series causal discovery
- Two-stage procedure explicitly handles temporal structure (PC algorithm doesn't)
- Detection power >80% in high-dimensional cases
- RUN framework (AAAI 2024): AC@1=0.63 on Sock Shop using neural Granger + PCMCI concepts
- ASE 2024 evaluation: No single method dominates, but PCMCI consistently handles temporal dependencies

**Configuration**:
- tau_max=3-5 (fault propagation within 3-5 sampling intervals)
- pc_alpha=0.1-0.2 (liberal for parent discovery)
- alpha_level=0.01-0.05 (conservative for final graph)
- Start with ParCorr test (linear), upgrade to GPDC if needed (nonlinear)

**Baseline Comparison**: Granger-Lasso (5-min implementation) as fast causal baseline

**Alternatives Considered**:
- NOTEARS: Lacks temporal modeling
- PC algorithm: Order-dependent, ignores time structure
- LiNGAM: Too restrictive (linear non-Gaussian)
- DoWhy: Requires pre-specified graphs (not discovery)

**Impact**: Core component for RCA - translates anomaly detection to root cause localization

### D007: Intermediate Multimodal Fusion Architecture
**Date**: 2025-11-14
**Decision**: Implement intermediate fusion with separate encoders + cross-modal attention
**Rationale**:
- **Early fusion**: Fails on heterogeneous sampling rates (1-min metrics, irregular logs, ms traces)
- **Late fusion**: Misses cross-modal correlations (network delay causing CPU spike needs joint reasoning)
- **Intermediate fusion**: Best of both worlds (2024 research consensus)

**Architecture**:
```
Metrics → TCN/Chronos Encoder →
Logs → Drain + BERT/Embedding →  Cross-Modal Attention → Unified Representation → RCA
Traces → GNN Encoder →
```

**Time Alignment Strategy**:
- Base 1-minute windows
- Aggregate metrics: mean/max/p99
- Count log events, embed templates (Drain)
- Sample representative spans from traces
- Cross-reference via trace_id

**Evidence**: 2024 papers (FAMOS, MULAN) demonstrate intermediate fusion superiority

**Impact**: Enables leveraging all three modalities effectively

### D008: 2-Layer GCN as Primary Graph Architecture
**Date**: 2025-11-14
**Decision**: Start with 2-layer GCN, prepare GAT as optional upgrade
**Rationale**:
- 2022 survey: GCN most popular for service dependency graphs
- TrainTicket: ~41 services (<30 nodes) → GCN and GAT perform similarly
- GCN: 15-line implementation, 1-3MB memory, faster debugging
- GAT: Better for interpretability (attention visualization), 3-8MB memory

**Configuration**:
- 2-3 layers (more causes over-smoothing on small graphs)
- hidden_dim=64
- dropout=0.3-0.5
- learning_rate=0.01
- weight_decay=5e-4

**Graph Construction**:
- Parse distributed traces (OpenTelemetry format)
- Extract parent-child span relationships
- Nodes = services, Edges = calls
- Node features: response time (mean, p50, p90, p99), CPU, memory, request rate, error rate
- Edge features: call frequency, latency, error rate

**Upgrade Trigger**: If need interpretable attention for presentation visuals

**Impact**: Captures service dependency topology for fault propagation modeling

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
