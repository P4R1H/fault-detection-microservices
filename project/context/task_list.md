# Task List

**Last Updated**: 2025-11-14

## Status: PHASE 0 - Initialization

### Current Tasks
- [x] Create folder structure
- [x] Initialize context files
- [ ] Receive mid-semester report from user
- [ ] Receive project proposals from user
- [ ] Receive research notes from user
- [ ] Receive literature review from user
- [ ] Receive dataset access/links from user

---

## PHASE 1: Analysis & Planning (Pending)
- [ ] Read and analyze all provided materials
- [ ] Extract core requirements and constraints
- [ ] Design detailed system architecture
- [ ] Create data pipeline specification
- [ ] Define evaluation metrics
- [ ] List all dependencies
- [ ] Create detailed implementation timeline

## PHASE 2: Data Pipeline (Pending)
- [ ] Download/access RCAEval TrainTicket dataset
- [ ] Implement data loaders for metrics
- [ ] Implement data loaders for logs
- [ ] Implement data loaders for traces
- [ ] Verify data integrity
- [ ] Create train/val/test splits
- [ ] Document data statistics

## PHASE 3: Metrics Module (Pending)
- [ ] Implement Chronos-Bolt-Tiny integration OR TCN model
- [ ] Implement forecasting pipeline
- [ ] Implement anomaly detection logic
- [ ] Implement baseline: BARO
- [ ] Implement additional baselines (statistical methods)
- [ ] Create evaluation harness
- [ ] Run metrics-only experiments

## PHASE 4: Logs Module (Pending)
- [ ] Implement Drain3 parser integration
- [ ] Implement template extraction pipeline
- [ ] Implement template embeddings
- [ ] Implement anomaly scoring
- [ ] Create evaluation pipeline
- [ ] Run logs-only experiments

## PHASE 5: Traces Module (Pending)
- [ ] Parse trace data
- [ ] Build service dependency graph
- [ ] Extract graph features
- [ ] Implement 2-layer GNN (GCN or GAT)
- [ ] Implement RCA scoring mechanism
- [ ] Create evaluation pipeline
- [ ] Run traces-only experiments

## PHASE 6: Causal Inference (Pending)
- [ ] Integrate tigramite library
- [ ] Implement PCMCI pipeline
- [ ] Discover causal graphs
- [ ] Compare with RCAEval ground-truth
- [ ] Evaluate causal discovery quality
- [ ] Document findings

## PHASE 7: Multimodal Fusion (Pending)
- [ ] Design fusion architecture
- [ ] Implement separate encoders
- [ ] Implement cross-modal attention
- [ ] Implement unified scoring model
- [ ] Train fusion model
- [ ] Evaluate full system

## PHASE 8: Ablations & Baselines (Pending)
- [ ] Metrics-only
- [ ] Logs-only
- [ ] Traces-only
- [ ] Metrics + Logs
- [ ] Metrics + Traces
- [ ] Logs + Traces
- [ ] All modalities
- [ ] Without GNN
- [ ] Without PCMCI
- [ ] With/without pretrained foundation model
- [ ] Additional meaningful ablations

## PHASE 9: Baselines Comparison (Pending)
- [ ] Implement/integrate BARO
- [ ] Implement/integrate baseline 2
- [ ] Implement/integrate baseline 3
- [ ] Implement/integrate baseline 4
- [ ] Implement/integrate baseline 5+
- [ ] Run all baseline experiments
- [ ] Create comparison tables

## PHASE 10: Analysis & Visualization (Pending)
- [ ] Generate all results tables
- [ ] Create performance plots
- [ ] Create ablation comparison charts
- [ ] Statistical significance testing
- [ ] Error analysis
- [ ] Case studies

## PHASE 11: Documentation (Pending)
- [ ] Write final_report.md
- [ ] Write architecture.md with diagrams
- [ ] Write literature_review.md
- [ ] Write methodology.md
- [ ] Write results.md
- [ ] Write ablations.md
- [ ] Write experiments.md
- [ ] Write limitations.md
- [ ] Polish README.md
- [ ] Code documentation review

## PHASE 12: Final Review (Pending)
- [ ] Reproducibility check
- [ ] Code quality review
- [ ] Documentation completeness check
- [ ] Experimental results validation
- [ ] Final polishing

---

## Blockers
- **Awaiting user materials**: Cannot proceed with detailed planning until mid-semester report, proposals, research notes, literature review, and dataset access are provided.

## Notes
- Task list will be refined after receiving and analyzing user materials
- Estimates will be added once compute resources are known
- Dependencies between phases may require reordering
