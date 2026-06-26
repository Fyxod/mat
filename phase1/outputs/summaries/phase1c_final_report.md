# Phase 1C final-edit-aligned report

## Interpretation rule

A candidate is only convincing if the clean original edit succeeds, the perturbed input remains close enough, and the perturbed edit visibly weakens or fails. If final images look the same, this report treats the row as metric-only even when CSV scores improved.

## Legacy Phase 1A/1B semantic rescore

- phase1a: {'metric_only_candidate': 106, 'weak_candidate': 2}
- phase1b: {'metric_only_candidate': 48}

## Phase 1C screening

- Decision counts: {'metric_only_candidate': 68, 'reject_input_damage': 1, 'weak_candidate': 27}

## Phase 1D deepening

- Decision counts: {}
- Final validation candidates: 0

## Next interpretation

The white-box internal objectives produced measurable output differences but not a convincing visible edit failure yet, unless manual inspection of the sheets says otherwise.
