"""Shared controls for deterministic process-level parallelism."""

from __future__ import annotations

import os


def resolve_worker_count(requested: int, task_count: int) -> int:
    """Resolve a CLI worker request without creating idle processes.

    ``requested=0`` keeps one logical CPU available for the operating system.
    The result is always between one and the number of independent tasks.
    """

    if requested < 0:
        raise ValueError("workers must be zero or a positive integer.")
    if task_count <= 0:
        raise ValueError("task_count must be positive.")

    if requested == 0:
        logical_cpus = os.cpu_count() or 1
        requested = max(1, logical_cpus - 1)
    return max(1, min(requested, task_count))


__all__ = ["resolve_worker_count"]
