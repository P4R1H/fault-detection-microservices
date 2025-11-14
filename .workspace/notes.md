# Scratch Notes & Brainstorming

Quick notes, ideas, and temporary information. Clean up regularly.

---

## 2025-11-14: Initial Setup

### User Requirements Summary
- Clean codebase organization
- Working memory for resumption
- Keep docs/ formal only
- Data folder local (30GB)
- Make tests parallel/fast
- Target: A+ grade via report quality

### Key Numbers to Remember
- **270 failure cases** (90 per system)
- **60-70% grade from report** (not code!)
- **5+ baselines** needed for comparison
- **8GB VRAM** constraint
- **Sub-100ms latency** requirement
- **~500MB total model memory** budget

### Quick Command References
```bash
# Dataset verification
python scripts/verify_dataset.py --parallel

# Run tests
pytest tests/ -v --tb=short

# Run EDA
python scripts/eda_analysis.py --all --parallel

# Future: Run ablations
python experiments/run_ablations.py --parallel --n_jobs=16
```

### Library Versions to Track
- PyTorch Geometric: v2.3+
- tigramite: >=5.1.0.3
- drain3: >=0.9.11
- chronos-forecasting: >=1.0.0
- transformers: 4.30.0-4.33.0

### Implementation Shortcuts
- Use HuggingFace Chronos directly (no training!)
- PCMCI examples in tigramite docs
- PyG GCN is literally 15 lines
- Drain3 has streaming API

### Visualization Ideas
- Service graphs with Graphviz DOT
- Interactive plots with PyVis (for presentations)
- Seaborn for publication quality
- Plotly for interactive dashboards

### Things to Double-Check
- [ ] Scenario-based splitting prevents data leakage?
- [ ] All modalities time-aligned properly?
- [ ] Ground truth format matches our predictions?
- [ ] Statistical tests appropriate for sample size?

### Questions Forming
1. How to handle missing modalities in some cases?
2. What if GCN doesn't work on small graphs?
3. Should we use pretrained log embeddings or train own?
4. How to visualize attention weights effectively?

### Useful Links (from docs)
- RCAEval: doi.org/10.5281/zenodo.14590730
- Chronos: huggingface.co/amazon/chronos-bolt-tiny
- Tigramite: github.com/jakobrunge/tigramite
- PyG Docs: pytorch-geometric.readthedocs.io

---

## Random Insights

- **Ablations matter more than novelty** for grades
- **Error bars from 5 seeds** = statistical rigor
- **Honest limitations** = academic maturity
- **Professional figures** = visual communication skills
- User has local dataset = skip download time!

---

## TODO Quick Capture (move to todo.md later)
- Verify dataset exists locally
- Ask user about "new technique"
- Create parallel test script
- Set up config YAML templates

---

**Clean this file weekly!**
