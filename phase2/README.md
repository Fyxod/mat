# Phase 2: InstructPix2Pix final-edit geometric search

Phase 2 is the follow-up to the Phase 1A/1B/1C internal-objective diagnostics.
Those phases remain preserved, but they did not produce convincing visible edit
failures. Phase 2 therefore scores the actual final InstructPix2Pix edited
image directly.

The perturbation is still geometry-only:

- no pixel noise
- no adversarial patch
- no finetuning
- no LoRA
- no model-weight training

The loop is:

```text
sample geometric candidate
apply a region-local geometric warp to the original input
run the real InstructPix2Pix edit on the perturbed input
score final clean edit vs final perturbed edit
update CEM from elite candidates
```

The first probe prioritizes `add headphones`, because Phase 1C found the only
weak semantic signal there. It also runs a small check on sunglasses, round
glasses, and beard prompts.

## A6000 commands

```bash
cd /home/interns/Desktop/mat
git pull origin main

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m phase2.scripts.run_phase2a_probe --root /home/interns/Desktop/mat

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m phase2.scripts.run_phase2b_cem --root /home/interns/Desktop/mat

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m phase2.scripts.summarize_phase2 --root /home/interns/Desktop/mat
```

Phase 2B automatically skips if Phase 2A finds only metric-only candidates.

