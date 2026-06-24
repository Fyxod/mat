# Prompt filtering report

- Candidate settings evaluated: 128
- Automatically selected settings: 6
- Final manually audited prompt/settings: 6

The automatic score tied many plausible and implausible outputs because identity was not available during the grid. The contact sheet was therefore used as the required final gate.

## Accepted after visual audit

- add black sunglasses — IGS 1.5, GS 7.5
- add round glasses — IGS 1.5, GS 7.5
- add a small earring — IGS 2.0, GS 7.5
- add headphones — IGS 1.5, GS 7.5
- add a small beard — IGS 1.5, GS 7.5
- make the person smile slightly — IGS 2.0, GS 7.5

## Explicitly rejected despite the automatic score

- add a black baseball cap: substantial identity/skin-tone drift.
- add a red beanie: global red color collapse.
- add a blue beanie: substantial identity drift.
- add thin eyeglasses: gender/identity drift.
- add light stubble: fabricated extra face / global semantic artifact.

The Phase 1A and Phase 1B runners use only the manually audited selection in `phase1/configs/phase1_selected_prompts.json`.
