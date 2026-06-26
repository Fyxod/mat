# Phase 3: breadth search for vulnerable InstructPix2Pix geometric cases

Phase 3 stops deepening the saturated `face_001 / add headphones` setup from
Phase 2C.  It instead searches across multiple face images, localized edit
prompts, and InstructPix2Pix image-guidance settings to find cases where a
geometry-only input warp causes a visible final-edit failure.

The perturbation constraint is unchanged:

- no pixel noise
- no adversarial patches
- no finetuning, LoRA, or model-weight training
- only coordinate/geometry warps

Workflow:

1. `run_phase3_prompt_discovery` generates clean InstructPix2Pix edits for all
   prompt-compatible image/prompt/settings combinations.
2. Clean cases are rejected unless the clean edit appears semantically positive
   by CLIP margin and avoids obvious global collapse.
3. `run_phase3_breadth_probe` attacks only selected clean-success cases with a
   small final-edit CEM probe.
4. `summarize_phase3` reports whether any image/prompt/settings combination is
   worth a later narrow Phase 3C deepening.

If the perturbed final edit still clearly succeeds, the candidate must be
treated as weak or metric-only even if its CSV score is high.
