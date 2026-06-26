# Phase 2 final-edit geometric CEM report

## Phase 1 handoff

Phase 1A/1B/1C are preserved as diagnostic internal-objective results. The current conclusion is not to run more Phase 1C/1D with the same objective family.

## Phase 2A probe

- Decision counts: {'metric_only_candidate': 415, 'reject_input_damage': 15, 'weak_candidate': 14}

## Phase 2B

- Decision counts: {'metric_only_candidate': 37, 'reject_input_damage': 10, 'weak_candidate': 50}
- Visual verdict: weak headphone weakening/shift only, not a convincing visible edit failure.

## Phase 2C

Phase 2C did not find a visible headphone edit failure. The best candidates only weaken or shift the headphones. Do not spend more A6000 time on the same InstructPix2Pix/headphones setup.
- Decision counts: {'metric_only_candidate': 529, 'reject_input_damage': 52, 'weak_candidate': 314}

## Interpretation rule

If the final clean and perturbed edited images look the same, treat the row as metric-only even if the CSV score is high.
