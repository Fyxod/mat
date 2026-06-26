# Phase 3 final report

## Phase 2C handoff

Phase 2C is treated as negative for the exact InstructPix2Pix / face_001 / add-headphones / region-local TPS-DCT-mesh setup. Do not spend more A6000 time deepening that line unless explicitly requested.

## Phase 3 clean discovery

- Detected face folders: 8
- New image folders: 7
- Selected clean-success cases for breadth probe: 24
- Rejected clean cases: 19
- Prompt-restricted faces: ['face_007', 'face_008']

## Phase 3 breadth probe

- Candidate count: 816
- Decision counts: {'metric_only_candidate': 773, 'reject_input_damage': 43}
- metric_only_candidate: face_002 / add headphones / igs=1.5 semantic_drop=0.0092, score=0.3147
- metric_only_candidate: face_003 / add headphones / igs=1.5 semantic_drop=0.0088, score=0.3037
- metric_only_candidate: face_003 / add headphones / igs=1.5 semantic_drop=0.0087, score=0.2938
- reject_input_damage: face_003 / add headphones / igs=1.5 semantic_drop=0.0084, score=-0.0242
- reject_input_damage: face_002 / add headphones / igs=1.0 semantic_drop=0.0068, score=-0.0624
- metric_only_candidate: face_003 / add headphones / igs=1.5 semantic_drop=0.0065, score=0.2057
- reject_input_damage: face_003 / add headphones / igs=1.5 semantic_drop=0.0064, score=0.0549
- metric_only_candidate: face_002 / add headphones / igs=1.0 semantic_drop=0.0060, score=0.3795

## Decision rule

If Phase 3 finds visible/strong candidates, run a narrow Phase 3C only on those image/prompt/settings cases. If it finds only weak or metric-only rows, inspect the sheet and avoid automatic deepening.
