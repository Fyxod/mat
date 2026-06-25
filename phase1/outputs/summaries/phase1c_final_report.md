# Phase 1C final-edit-aligned report

Status: code and reporting workflow are prepared; A6000 Phase 1C results have not been run yet.

## Legacy Phase 1A/1B interpretation

Phase 1B exists and is preserved as `phase1/outputs/phase1b_deepen/`. It is labeled as legacy internal-surrogate diagnostic data in `phase1/outputs/legacy_internal_surrogate_phase1b/README.md`.

Visual inspection of `phase1/outputs/phase1b_deepen/phase1b_top_sheet.jpg` shows the old failure mode: output metrics changed, but clean edits and perturbed edits often still visibly preserve the requested edit. Treat Phase 1B as diagnostic unless semantic rescoring says otherwise.

## Pending A6000 steps

Run:

1. `python -m phase1.scripts.rescore_legacy_phase1ab --root /home/interns/Desktop/mat`
2. `python -m phase1.scripts.run_phase1c_screening --root /home/interns/Desktop/mat`
3. `python -m phase1.scripts.run_phase1d_deepen --root /home/interns/Desktop/mat`
4. `python -m phase1.scripts.summarize_phase1c --root /home/interns/Desktop/mat`

After those commands, this report should be regenerated with actual semantic rescore, Phase 1C, Phase 1D, and final-validation conclusions.
