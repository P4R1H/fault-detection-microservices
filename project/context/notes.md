# Quick Notes & Reminders

**Last Updated**: 2025-11-14

## Today's Progress
- ✅ Created complete project folder structure
- ✅ Initialized all context files
- ✅ Ready to receive user materials

## Immediate TODOs
- [ ] Wait for user to provide mid-semester report
- [ ] Wait for user to provide project proposals
- [ ] Wait for user to provide research notes
- [ ] Wait for user to provide literature review
- [ ] Wait for user to provide dataset access

## Quick Reminders
- **NEVER hallucinate** - always ask when info is missing
- **Update context files** after each major step
- **Keep /docs polished** - academic grade only
- **Code goes in /src** - organized by module
- **Experiments in /experiments** - with timestamps
- User will add bibliography entries later (just cite in text for now)

## Technical Notes

### RCAEval TrainTicket Dataset
- Standard benchmark for microservice RCA
- Contains metrics, logs, traces
- Has ground-truth fault labels
- Need to verify access and format

### Key Libraries to Use
- `tigramite` - PCMCI causal discovery
- `Drain3` - Log parsing
- PyTorch/TensorFlow - Deep learning (TBD based on Chronos requirements)
- PyG or DGL - GNN implementation
- Standard scientific stack: numpy, pandas, scikit-learn

### Foundation Models
- Chronos-Bolt-Tiny: Amazon's zero-shot time-series model
- Need to check: Hugging Face availability, input format, inference API

## Questions Log
- None yet - waiting for initial materials

## Failures & Lessons
- None yet

## Ideas & Explorations
- Consider attention visualization for interpretability
- May want to analyze which modality contributes most to RCA
- Could create interactive dashboard for results (if time permits)
- Service dependency graph visualization could be valuable

## References to Check Later
- RCAEval paper for dataset details
- BARO baseline paper
- PCMCI original papers
- Chronos model card and documentation
- Recent AIOps surveys (2023-2025)

---

*This file is for quick, informal notes. Structured information goes in memory.md, task_list.md, or decisions.md.*
