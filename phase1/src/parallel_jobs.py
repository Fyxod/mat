"""Optional process-level job execution for A6000 Phase 1C/1D runs."""
from __future__ import annotations

import math
import multiprocessing as mp
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .utils import append_run_note, write_json


def split_batches(items: Sequence[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    worker_count = max(1, min(int(workers), len(items) or 1))
    batch_size = int(math.ceil(len(items) / worker_count)) if items else 1
    return [list(items[index:index + batch_size]) for index in range(0, len(items), batch_size)]


def run_serial_or_parallel(
    *,
    root: Path,
    jobs: Sequence[dict[str, Any]],
    worker_entry: Callable[[str, list[dict[str, Any]], bool], list[dict[str, Any]]],
    force: bool,
    parallel_config: dict[str, Any],
    notes_title: str,
) -> list[dict[str, Any]]:
    """Run job dictionaries serially or in spawned worker processes."""
    if not jobs:
        return []
    if not bool(parallel_config.get("parallel_experimental", False)):
        return worker_entry(str(root), list(jobs), force)

    requested = int(parallel_config.get("parallel_workers", 2))
    maximum = int(parallel_config.get("max_parallel_workers", requested))
    workers = max(1, min(requested, maximum, len(jobs)))
    start_method = str(parallel_config.get("worker_start_method", "spawn"))
    batches = split_batches(jobs, workers)
    try:
        context = mp.get_context(start_method)
        with context.Pool(processes=len(batches)) as pool:
            results = pool.starmap(worker_entry, [(str(root), batch, force) for batch in batches])
        flattened: list[dict[str, Any]] = []
        for batch_rows in results:
            flattened.extend(batch_rows)
        append_run_note(
            root,
            notes_title,
            f"Process-level parallel execution completed with {len(batches)} worker processes and {len(jobs)} jobs.",
        )
        return flattened
    except Exception as error:
        failure_root = root / "phase1" / "outputs" / "summaries"
        write_json(
            failure_root / "phase1_parallel_failure.json",
            {
                "notes_title": notes_title,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "requested_workers": requested,
                "used_workers": workers,
            },
        )
        append_run_note(
            root,
            notes_title,
            f"Parallel execution failed ({error}); falling back to serial because retry_serial_on_parallel_failure=true.",
        )
        if bool(parallel_config.get("retry_serial_on_parallel_failure", True)):
            return worker_entry(str(root), list(jobs), force)
        raise


__all__ = ["run_serial_or_parallel", "split_batches"]
