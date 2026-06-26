# MAT: geometric attacks on InstructPix2Pix

This repository is a focused research workflow: can visually small, face-local geometric warps disrupt a cleanly working InstructPix2Pix edit?

The attack changes only geometric transformation parameters. It does not add pixel noise, adversarial patches, LoRA, finetuning, or train model weights.

The canonical input is [data/face_001/instruct_512.png](data/face_001/instruct_512.png).

- [Phase 1](phase1/README.md) preserves the true white-box internal-objective diagnostics.
- [Phase 2](phase2/README.md) adds final-edit black-box CEM search over geometry-only warps.

## Windows development

This Windows machine is for code changes, import checks, and very small diagnostics only. Do not run Phase 1 sweeps or Phase 2 final-edit CEM on a laptop GPU.

From the project root:

```powershell
python -m compileall phase1 phase2 scripts
python scripts/check_env.py
```

The real smoke test and all heavy runs belong on the A6000.

## A6000 workflow

On the Ubuntu workstation:

```bash
cd /home/interns/Desktop
git clone https://github.com/fyxod/mat.git
cd /home/interns/Desktop/mat

bash scripts/install_linux_a6000.sh
```

The install script prints the exact micromamba command prefix. With the default location, run:

```bash
$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 python scripts/check_env.py

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 python -m phase1.scripts.a6000_run --root /home/interns/Desktop/mat --mode smoke
$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 python -m phase1.scripts.a6000_run --root /home/interns/Desktop/mat --mode prompt_discovery
$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 python -m phase1.scripts.a6000_run --root /home/interns/Desktop/mat --mode baselines
$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 python -m phase1.scripts.a6000_run --root /home/interns/Desktop/mat --mode scale_probe
$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 python -m phase1.scripts.a6000_run --root /home/interns/Desktop/mat --mode phase1a --force
$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 python -m phase1.scripts.a6000_run --root /home/interns/Desktop/mat --mode phase1b
$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 python -m phase1.scripts.a6000_run --root /home/interns/Desktop/mat --mode final_validation
$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 python -m phase1.scripts.a6000_run --root /home/interns/Desktop/mat --mode summarize
```

For Phase 2 final-edit search:

```bash
$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 python -m phase2.scripts.run_phase2a_probe --root /home/interns/Desktop/mat
$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 python -m phase2.scripts.run_phase2b_cem --root /home/interns/Desktop/mat
$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 python -m phase2.scripts.summarize_phase2 --root /home/interns/Desktop/mat
```

Each mode records a timestamped log and a success/failure marker. It skips completed work until called with `--force`.

If a mode fails:

```bash
$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 python -m phase1.scripts.a6000_collect_debug_bundle --root /home/interns/Desktop/mat
```

Then push the reports and artifacts:

```bash
git add phase1/outputs phase1/configs/phase1_selected_prompts.json
git commit -m "Add A6000 Phase 1 results"
git push origin main
```

Do not push model caches or environments. The output policy preserves reports, graph-ready CSVs, selected artifacts, sheets, final validation, and debug bundles.
