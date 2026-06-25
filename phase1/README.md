# Phase 1: true InstructPix2Pix geometric white-box attack

## What is optimized

The geometry module exposes three differentiable warp families:

- `face_local_tps`: a thin-plate-spline control lattice inside a soft face ellipse.
- `dct_lowfreq`: low-frequency displacement coefficients.
- `combined_tps_dct`: the default sum of those fields.

The field is capped in pixel units and penalized for visual change, displacement magnitude, total variation, and low/negative Jacobian determinants. No pixel-space perturbation exists in this code path.

All VAE, text encoder, UNet, and scheduler model weights are frozen. Adam receives only geometry parameters.

## White-box objective

For a fixed prompt, empty prompt, noisy latent, and scheduler timestep, the optimizer differentiates through:

```text
text/image -> VAE latent -> [fixed noisy latent, image latent] -> InstructPix2Pix UNet -> objective
```

The code follows upstream InstructPix2Pix's 8-channel UNet contract. The primary objectives are:

- `edit_direction`: maximize the mismatch between clean and perturbed `(UNet(prompt) - UNet(empty))` edit directions.
- `unet_prediction`: maximize the prompt-conditioned UNet prediction mismatch.
- `vae_conditioning`: maximize VAE image-conditioning latent mismatch.
- `hybrid_edit_unet_vae`: available for follow-up work.

Prompt discovery is mandatory because a failed clean edit cannot be counted as an attack success.

## Stage commands

All commands accept `--root` and are intended to run as modules:

```bash
python -m phase1.scripts.smoke_instruct_whitebox --root /workspace/mat
python -m phase1.scripts.run_prompt_discovery --root /workspace/mat
python -m phase1.scripts.run_clean_baselines --root /workspace/mat
python -m phase1.scripts.run_phase1a_screening --root /workspace/mat
python -m phase1.scripts.run_phase1b_deepen --root /workspace/mat
python -m phase1.scripts.validate_top_candidates --root /workspace/mat
python -m phase1.scripts.summarize_phase1 --root /workspace/mat
```

For normal A6000 work, use `a6000_run` instead; it creates timestamped logs and outer run markers.

## Phase 1C: final-edit-aligned follow-up

Phase 1A/1B are preserved as legacy internal-surrogate diagnostics. Their old `attack_score` measures output pixel disruption and is not sufficient evidence of attack success when clean and perturbed edits look semantically the same.

Phase 1C keeps the same geometry-only white-box setup, but adds:

- multi-timestep internal objectives:
  - `multi_timestep_edit_direction`
  - `multi_timestep_unet_prediction`
  - `multi_timestep_hybrid`
- checkpoint-level final edit generation and semantic scoring,
- optional CLIP prompt-margin scoring for clean-edit success versus perturbed-edit weakening,
- process-level parallel workers for A6000 runs,
- legacy Phase 1A/1B semantic rescoring.

Run the new flow on the A6000:

```bash
python -m phase1.scripts.rescore_legacy_phase1ab --root /home/interns/Desktop/mat
python -m phase1.scripts.check_clip_semantic --root /home/interns/Desktop/mat --require
python -m phase1.scripts.run_phase1c_screening --root /home/interns/Desktop/mat
python -m phase1.scripts.run_phase1d_deepen --root /home/interns/Desktop/mat
python -m phase1.scripts.summarize_phase1c --root /home/interns/Desktop/mat
```

Phase 1C now refuses to start if CLIP semantic scoring cannot load. Phase 1D automatically skips if Phase 1C finds only metric-only candidates.

## Resumption and artifacts

Every prompt case, baseline, attack start, validation candidate, and A6000 mode writes `DONE.json` or `FAILED.json`. A rerun skips complete work unless `--force` is explicitly passed.

- `outputs/prompt_discovery`: all clean candidates, filtering sheet, selected settings.
- `outputs/baselines`: clean edits for only selected settings.
- `outputs/phase1a_screening`: 3 objectives × 2 budgets × 3 starts; 150 iterations each.
- `outputs/phase1b_deepen`: top four distinct combinations; 12 starts × 400 iterations unless the logged timing fallback activates.
- `outputs/legacy_internal_surrogate_phase1b`: label/report for preserved Phase 1B diagnostics.
- `outputs/semantic_rescore`: semantic rescoring of old Phase 1A/1B candidates.
- `outputs/phase1c_screening`: focused final-edit-aligned multi-timestep screening.
- `outputs/phase1d_deepen`: deepening only for semantic Phase 1C candidates.
- `outputs/final_validation_phase1c`: optional identity validation for strong Phase 1D candidates.
- `outputs/final_validation`: top candidate panels and optional DeepFace identity comparisons.
- `outputs/summaries`: graph-ready CSV/JSON/Markdown handoff.
- `outputs/debug_bundles`: environment, logs, tree, configuration and error archive.

Each history row carries loss/objective, all penalties, gradient norm, displacement statistics, and Jacobian statistics. Checkpoint rows carry input/output image metrics and the attack score:

```text
output_disruption - input_damage_penalty
```

Input damage penalizes SSIM below the budget target and displacement above the configured cap.
