# Legacy internal-surrogate Phase 1B

This folder labels the completed `phase1/outputs/phase1b_deepen/` run as legacy diagnostic data.

Phase 1B used the older single-timestep internal-surrogate ranking. It is preserved and should not be deleted or reverted, but it should not be treated as final attack success without semantic/final-edit validation.

Current inspection:

- Phase 1B exists and completed.
- Completion marker: `phase1/outputs/phase1b_deepen/DONE.json`
- Completed combinations: 4
- Completed starts: 48
- Checkpoint rows: 816
- All 48 best rows were old-budget-admissible.
- Aggregate files exist:
  - `phase1/outputs/phase1b_deepen/phase1b_all_candidates.csv`
  - `phase1/outputs/phase1b_deepen/phase1b_all_checkpoints.csv`
  - `phase1/outputs/phase1b_deepen/phase1b_top_candidates.csv`
  - `phase1/outputs/phase1b_deepen/phase1b_summary.json`
  - `phase1/outputs/phase1b_deepen/phase1b_top_sheet.jpg`
- A legacy `phase1b_decision_report.md` was not present in the pushed artifacts.

Best old-score candidates were led by:

1. `add_a_small_beard` / `unet_prediction` / `strong`, old attack score about `0.291`
2. `add_a_small_beard` / `unet_prediction` / `strong`, old attack score about `0.290`
3. `make_the_person_smile_slightly` / `vae_conditioning` / `strong`, old attack score about `0.280`
4. `make_the_person_smile_slightly` / `vae_conditioning` / `strong`, old attack score about `0.235`
5. `add_a_small_beard` / `vae_conditioning` / `strong`, old attack score about `0.217`

Visual note: the top sheet shows measurable output differences, but many clean edits and perturbed edits still visibly preserve the requested edit. This is the same legacy failure mode: useful internal/objective movement, but not convincing final-edit failure.

New Phase 1C improves on this by:

- ranking checkpoints by final-edit-aware semantic scoring,
- labeling metric-only candidates separately,
- adding multi-timestep internal objectives,
- keeping full final edits only at checkpoints,
- preserving Phase 1A/1B outputs for semantic rescoring.

Reference sheet:

- `phase1/outputs/phase1b_deepen/phase1b_top_sheet.jpg`
