# Phase 4: landmark / semantic-structure geometric attacks

Phase 4 is a materially different InstructPix2Pix-only geometry attack family.
Previous phases used smooth region-local TPS/DCT/mesh fields.  Phase 4 instead
builds coordinate warps from semantic face-part actions:

- eyes and nose bridge for glasses/sunglasses
- mouth, chin, and lower face for smile/beard
- jaw, head sides, and head top for headphones

The constraint stays strict:

- no pixel noise
- no adversarial patches
- no finetuning, LoRA, or model-weight training
- only geometric coordinate transformations

The workflow is:

1. Detect face landmarks on `data/face_*/instruct_512.png`.
2. Convert landmarks into semantic regions and action anchors.
3. Run a final-edit CEM existence probe with landmark semantic TPS and
   landmark piecewise-affine warps.
4. Only run tightening if Phase 4A finds a visible or existence-level success.

If the perturbed final image still clearly contains the requested edit, the row
is weak or metric-only regardless of numeric score.
