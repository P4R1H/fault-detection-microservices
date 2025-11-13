# Active State - MOMENT Foundation Model for Microservice Anomaly Detection

**Last Updated**: 2025-11-13
**Current Phase**: Phase 2 - Foundation Model Implementation
**Branch**: claude/moment-anomaly-detection-phase2-01Rv2krwYTGmKdLD6yVQJVoi

---

## Current Status

**INITIALIZATION PHASE - COMPLETED ✅**
- ✅ Background analysis completed
- ✅ Todo tracking initialized
- ✅ Project infrastructure created (active_state.md, requirements.txt, src/)
- ✅ Data pipeline implemented
- ⏳ Model implementation pending (next phase)

**What's Been Built:**
1. **Configuration System** (`src/utils/config.py`)
   - Hardware config optimized for 8GB VRAM
   - Complete experiment configuration with YAML support
   - LoRA, training, evaluation configs

2. **Memory Management** (`src/utils/memory.py`)
   - GPU/CPU memory monitoring
   - Memory optimization utilities
   - Context managers for memory tracking
   - OOM prevention checks

3. **Logging System** (`src/utils/logger.py`)
   - Experiment tracking with timestamps
   - Metrics logging (JSONL format)
   - Model summary logging
   - Comparison utilities

4. **Data Pipeline** (`src/data/`)
   - `loader.py`: TrainTicket dataset loader with windowing and splitting
   - `preprocessor.py`: Normalization, patching, missing value handling
   - Ready for 256-timestep windows with PatchTST-style patching

---

## Next Immediate Step

**CHECKPOINT: Dataset Acquisition Required**

Before proceeding with model implementation, I need the TrainTicket dataset. The data pipeline is ready to process:
- CSV format with timestamp, features (88 columns), and labels
- Expected format: Time-series data with 14 fault types
- Minimum 10K samples (as per Phase 1)

**Options:**
1. You provide the dataset location/download link
2. You download and push to `data/raw/`
3. I can create a synthetic dataset for testing (not ideal for final results)

**Once dataset is available, next steps:**
- Implement MOMENT model wrapper
- Implement zero-shot inference
- Implement few-shot LoRA fine-tuning

---

## Key Decisions Log

### Decision 1: Memory Budget Strategy (2025-11-13)
**Context**: RTX 4070 has **8GB VRAM** (NOT 12GB as in task.md)
**Decision**: Aggressive memory optimization required
- Use MOMENT-1-large with 8-bit quantization if needed
- Batch size: Start at 8, max 16 (not 32)
- Gradient checkpointing: MANDATORY
- Mixed precision (FP16): MANDATORY
- LoRA rank: Start at 8 (not 16) to save memory

**Rationale**: 385M parameter model + gradients will consume ~6-7GB alone. Must leave headroom.

---

### Decision 2: Project Structure Philosophy (2025-11-13)
**Decision**: Modular Python scripts over notebooks
- `src/data/` - Data loading and preprocessing
- `src/models/` - MOMENT wrapper, baselines
- `src/training/` - Zero-shot, few-shot trainers
- `src/evaluation/` - Metrics, visualization
- `src/utils/` - Memory management, config

**Rationale**: Production-grade code, version control friendly, reusable components.

---

### Decision 3: Dataset Acquisition Strategy (2025-11-13)
**Decision**: ASK user before downloading large datasets
**Rationale**: TrainTicket data may be >1GB. User will download and push to repo.

---

## Phase 1 Recap (From midsem_report.txt)

**What Worked:**
- 88-dimensional feature engineering (65-70% predictive power from rolling stats)
- Comprehensive evaluation framework
- Multi-model ensemble approach

**Critical Issues Identified:**
1. **Overfitting**: Random Forest F1=1.0, AUC=1.0 on validation (sample/feature ratio = 113:1)
2. **Latency**: LSTM-AE training time = 25.4s (too slow for real-time)
3. **Generalization Risk**: High variance with limited data (10K samples)

**Phase 2 Pivot Justification:**
- Foundation models solve data scarcity (pre-trained on 352M timestamps)
- Zero-shot capability achieves F1~0.70-0.75 with ZERO training data
- Few-shot (100-1K samples) can reach F1~0.88-0.95
- Transfer learning reduces required samples by 60-80%

---

## Implementation Constraints

### Hardware
- **GPU**: NVIDIA RTX 4070
- **VRAM**: 8GB (CRITICAL CONSTRAINT)
- **Strategy**: Quantization, gradient checkpointing, FP16, small batches

### Software
- Python 3.8+
- PyTorch 2.0+ with CUDA
- HuggingFace ecosystem
- Time-series specific libraries

### Data
- TrainTicket benchmark (41 microservices, 88+ KPIs)
- 14 fault types
- Expected format: Time-series windows (256 timesteps × 88 features)
- Split: 70/15/15 train/val/test

---

## Expected Deliverables

### Code Artifacts
1. ✅ `context/active_state.md` (this file)
2. ⏳ `requirements.txt`
3. ⏳ `src/` directory with modular structure
4. ⏳ Data preprocessing pipeline
5. ⏳ Zero-shot evaluation script
6. ⏳ Few-shot LoRA fine-tuning script
7. ⏳ Baseline reproductions (Isolation Forest, LSTM-AE)
8. ⏳ Evaluation and visualization scripts

### Documentation
1. ⏳ `docs/methodology_v2.md` - Formal methodology
2. ⏳ Architecture diagrams (if needed)
3. ⏳ Training logs and results

### Experimental Results
1. Zero-shot performance (0 samples)
2. Few-shot learning curves (100, 500, 1K samples)
3. Comparison with Phase 1 baselines
4. Memory usage profiling
5. Inference latency measurements

---

## Blockers and Risks

### Current Blockers
- None (initialization phase)

### Anticipated Risks
1. **OOM Errors**: 385M params on 8GB VRAM
   - Mitigation: Quantization, checkpointing, micro-batching
2. **Dataset Availability**: TrainTicket data location unknown
   - Mitigation: User will provide/download
3. **MOMENT API Changes**: momentfm library may differ from docs
   - Mitigation: Check GitHub issues, use official examples

---

## Session Recovery Protocol

**If session restarts, read this section first:**

1. Check git status: `git status`
2. Review recent commits: `git log -3`
3. Check last todo status in this file
4. Check `src/` for completed modules
5. Look for training checkpoints in `data/checkpoints/`
6. Resume from last pending task in todo list

---

## Notes for Future Self

- Always update this file before asking user for help
- Commit stable milestones before pushing
- Keep memory profiling active during training
- Log all hyperparameters for reproducibility
- Test on small data subset first before full runs
