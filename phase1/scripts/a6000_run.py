"""Timestamped, resumable A6000 runner for every Phase 1 stage."""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Callable

from phase1.src.runners import (
    run_clean_baselines,
    run_final_validation,
    run_phase1a,
    run_phase1b,
    run_prompt_discovery,
    run_scale_probe,
    run_smoke,
    run_summary,
)
from phase1.src.phase1c_runner import (
    check_clip_semantic_preflight,
    rescore_legacy_phase1ab,
    run_phase1c_screening,
    run_phase1d_deepen,
    summarize_phase1c,
)
from phase1.src.utils import mark_done, mark_failed, outputs_root, project_root, timestamp_slug, write_json


class Tee:
    def __init__(self, file):
        self.file = file
        self.console = sys.__stdout__

    def write(self, value: str) -> int:
        self.console.write(value)
        self.file.write(value)
        return len(value)

    def flush(self) -> None:
        self.console.flush()
        if not self.file.closed:
            self.file.flush()

    def close(self) -> None:
        """Compatibility with logging handlers that retain this tee at exit.

        The surrounding context manager owns and closes the underlying log
        stream.  At interpreter shutdown absl/TensorFlow may subsequently call
        ``close`` on the stale tee; treating that as a no-op avoids a harmless
        but distracting atexit traceback.
        """
        return None


def _run_one(root: Path, mode: str, force: bool, identity: bool) -> object:
    steps: dict[str, Callable[[], object]] = {
        "smoke": lambda: run_smoke(root, force=force),
        "prompt_discovery": lambda: run_prompt_discovery(root, force=force, identity_enabled=identity),
        "baselines": lambda: run_clean_baselines(root, force=force, identity_enabled=identity),
        "phase1a": lambda: run_phase1a(root, force=force),
        "scale_probe": lambda: run_scale_probe(root, force=force),
        "phase1b": lambda: run_phase1b(root, force=force),
        "check_clip_semantic": lambda: check_clip_semantic_preflight(root, require_available=True),
        "rescore_legacy_phase1ab": lambda: rescore_legacy_phase1ab(root, force=force),
        "phase1c": lambda: run_phase1c_screening(root, force=force),
        "phase1d": lambda: run_phase1d_deepen(root, force=force),
        "final_validation": lambda: run_final_validation(root, force=force),
        "summarize": lambda: run_summary(root),
        "summarize_phase1c": lambda: summarize_phase1c(root),
    }
    if mode not in steps:
        raise ValueError(f"Unsupported mode: {mode}")
    return steps[mode]()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a timestamped, resumable A6000 Phase 1 mode.")
    parser.add_argument("--root", required=True, help="MAT project root")
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "smoke",
            "prompt_discovery",
            "baselines",
            "scale_probe",
            "phase1a",
            "phase1b",
            "rescore_legacy_phase1ab",
            "check_clip_semantic",
            "phase1c",
            "phase1d",
            "final_validation",
            "summarize",
            "summarize_phase1c",
            "all",
        ),
    )
    parser.add_argument("--force", action="store_true", help="Recompute otherwise-completed work")
    parser.add_argument("--identity", action="store_true", help="Try optional SFace during discovery and baselines")
    args = parser.parse_args()
    root = project_root(args.root)
    log_dir = outputs_root(root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{timestamp_slug()}_{args.mode}.log"
    marker = outputs_root(root) / "run_markers" / args.mode

    with log_path.open("w", encoding="utf-8") as stream, contextlib.redirect_stdout(Tee(stream)), contextlib.redirect_stderr(Tee(stream)):
        print(f"MAT A6000 mode={args.mode} root={root} force={args.force}")
        try:
            modes = (
                ("smoke", "prompt_discovery", "baselines", "phase1a", "phase1b", "final_validation", "summarize")
                if args.mode == "all"
                else (args.mode,)
            )
            results = {}
            for mode in modes:
                print(f"\n=== Starting {mode} ===")
                results[mode] = _run_one(root, mode, args.force, args.identity)
                print(f"=== Completed {mode} ===")
            write_json(marker / "result.json", {"mode": args.mode, "results": results, "log_path": str(log_path)})
            mark_done(marker, {"mode": args.mode, "log_path": str(log_path)})
            print("\nDONE. Next step:")
            next_step = {
                "smoke": "Run --mode prompt_discovery.",
                "prompt_discovery": "Inspect prompt_discovery_sheet.jpg, then run --mode baselines.",
                "baselines": "Run --mode phase1a.",
                "scale_probe": "Push the scale-probe outputs for review before rerunning Phase 1A.",
                "phase1a": "Run --mode phase1b.",
                "phase1b": "Run --mode final_validation.",
                "check_clip_semantic": "If this passed, run --mode rescore_legacy_phase1ab or --mode phase1c.",
                "rescore_legacy_phase1ab": "Inspect semantic_rescore_report.md, then run --mode phase1c.",
                "phase1c": "Inspect phase1c_semantic_top_sheet.jpg, then run --mode phase1d only if semantic candidates exist.",
                "phase1d": "Inspect phase1d_decision_report.md, then run --mode summarize_phase1c.",
                "final_validation": "Run --mode summarize.",
                "summarize": "Commit the selected outputs and push them to GitHub.",
                "summarize_phase1c": "Commit the selected Phase 1C outputs and push them to GitHub.",
                "all": "Commit the selected outputs and push them to GitHub.",
            }
            print(next_step[args.mode])
        except Exception as error:
            mark_failed(marker, error)
            print(f"\nFAILED: {error}", file=sys.stderr)
            print("Run: python -m phase1.scripts.a6000_collect_debug_bundle --root " + str(root), file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
