"""Collect a portable A6000 diagnostic bundle after any failure."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path

from phase1.src.utils import outputs_root, project_root, timestamp_slug, write_json, write_text


def _command(command: list[str], root: Path) -> str:
    try:
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
        return result.stdout + result.stderr
    except Exception as error:
        return f"Command unavailable: {command!r}\n{error}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a self-contained Phase 1 debug bundle.")
    parser.add_argument("--root", required=True, help="MAT project root")
    args = parser.parse_args()
    root = project_root(args.root)
    bundle = outputs_root(root) / "debug_bundles" / f"debug_{timestamp_slug()}"
    bundle.mkdir(parents=True, exist_ok=False)

    write_text(bundle / "environment.txt", _command(["bash", "-lc", "env | sort"], root))
    write_text(bundle / "pip_freeze.txt", _command(["python", "-m", "pip", "freeze"], root))
    write_text(bundle / "torch_cuda.txt", _command(["python", "-c", "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"], root))
    write_text(bundle / "gpu_info.txt", _command(["nvidia-smi"], root))
    write_text(bundle / "git_status.txt", _command(["git", "status", "-sb"], root))

    latest_logs = bundle / "latest_logs"
    latest_logs.mkdir()
    logs = sorted((outputs_root(root) / "logs").glob("*.log"), key=lambda item: item.stat().st_mtime, reverse=True)[:8]
    for log in logs:
        shutil.copy2(log, latest_logs / log.name)

    tree = []
    for item in sorted(outputs_root(root).rglob("*")):
        if len(tree) >= 4000:
            tree.append("... output tree truncated after 4000 entries")
            break
        tree.append(item.relative_to(root).as_posix())
    write_text(bundle / "output_tree.txt", "\n".join(tree) + "\n")

    configs = {}
    for path in sorted((root / "phase1" / "configs").glob("*.json")):
        configs[path.name] = path.read_text(encoding="utf-8")
    write_json(bundle / "config_snapshot.json", configs)

    failures = sorted(outputs_root(root).rglob("FAILED.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    summary = "No FAILED.json markers found.\n"
    if failures:
        latest = failures[0]
        summary = f"Latest failure: {latest.relative_to(root)}\n\n{latest.read_text(encoding='utf-8')}"
    write_text(bundle / "latest_error_summary.txt", summary)

    archive = bundle.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(bundle, arcname=bundle.name)
    print(bundle)
    print(archive)


if __name__ == "__main__":
    main()
