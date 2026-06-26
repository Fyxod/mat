# Phase 2C targeted headphone failure probe report

Phase 2C did not find a visible headphone edit failure. The best candidates only weaken or shift the headphones. Do not spend more A6000 time on the same InstructPix2Pix/headphones setup.

## Handoff from Phase 2B

Phase 2B improved the weak `add headphones` signal, but the perturbed edits still visibly retained headphones. It is preserved as diagnostic data and should not be treated as a strong attack success.

## Counts

- Phase 2A decision counts: {'metric_only_candidate': 415, 'reject_input_damage': 15, 'weak_candidate': 14}
- Phase 2B decision counts: {'metric_only_candidate': 37, 'reject_input_damage': 10, 'weak_candidate': 50}
- Phase 2C decision counts: {'metric_only_candidate': 529, 'reject_input_damage': 52, 'weak_candidate': 314}
- Phase 2C part counts: {'amplification': 315, 'region_ablations': 60, 'semantic_heavy_cem': 520}
- Phase 2C expected evaluations: {'amplification': 315, 'region_ablations': 60, 'semantic_heavy_cem': 520, 'total': 895}

## Files to inspect

- `phase2/outputs/phase2c_probe/phase2c_visible_failure_sheet.jpg`
- `phase2/outputs/phase2c_probe/phase2c_semantic_top_sheet.jpg`
- `phase2/outputs/phase2c_probe/phase2c_decision_report.md`

## Decision rule

If the perturbed edit still clearly has headphones, call it weak or metric-only even if semantic_drop improved.
