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

## Resumption and artifacts

Every prompt case, baseline, attack start, validation candidate, and A6000 mode writes `DONE.json` or `FAILED.json`. A rerun skips complete work unless `--force` is explicitly passed.

- `outputs/prompt_discovery`: all clean candidates, filtering sheet, selected settings.
- `outputs/baselines`: clean edits for only selected settings.
- `outputs/phase1a_screening`: 3 objectives × 2 budgets × 3 starts; 150 iterations each.
- `outputs/phase1b_deepen`: top four distinct combinations; 12 starts × 400 iterations unless the logged timing fallback activates.
- `outputs/final_validation`: top candidate panels and optional DeepFace identity comparisons.
- `outputs/summaries`: graph-ready CSV/JSON/Markdown handoff.
- `outputs/debug_bundles`: environment, logs, tree, configuration and error archive.

Each history row carries loss/objective, all penalties, gradient norm, displacement statistics, and Jacobian statistics. Checkpoint rows carry input/output image metrics and the attack score:

```text
output_disruption - input_damage_penalty
```

Input damage penalizes SSIM below the budget target and displacement above the configured cap.
