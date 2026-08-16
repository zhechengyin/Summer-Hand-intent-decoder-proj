"""Session-balanced sampling used to select the frozen Indy model."""

from __future__ import annotations

from collections import Counter

import numpy as np


def balanced_allocations(
    items: list[str], total: int, rng: np.random.Generator
) -> dict[str, int]:
    """Allocate an exact total as evenly as possible across named items."""
    if not items:
        raise ValueError("Cannot balance an empty item list")
    if total < 0:
        raise ValueError("Allocation total cannot be negative")
    base, remainder = divmod(total, len(items))
    allocation = {item: base for item in items}
    if remainder:
        for index in rng.permutation(len(items))[:remainder]:
            allocation[items[int(index)]] += 1
    return allocation


def draw_session_balanced_indices(
    sessions: list[str],
    session_lengths: dict[str, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    """Draw one fixed-size epoch with equal expected exposure per session."""
    if set(sessions) != set(session_lengths):
        raise ValueError("sessions and session_lengths must contain the same names")
    if any(session_lengths[session] <= 0 for session in sessions):
        raise ValueError("Every session must contain at least one training window")

    offsets: dict[str, int] = {}
    cursor = 0
    for session in sessions:
        offsets[session] = cursor
        cursor += session_lengths[session]
    epoch_size = cursor
    session_draws = balanced_allocations(sessions, epoch_size, rng)

    blocks = []
    for session in sessions:
        local = rng.integers(0, session_lengths[session], size=session_draws[session])
        blocks.append(local.astype(np.int64) + offsets[session])
    indices = np.concatenate(blocks)
    rng.shuffle(indices)

    month_draws: Counter[str] = Counter()
    for session, count in session_draws.items():
        date = session.split("_")[1]
        month_draws[f"{date[:4]}-{date[4:6]}"] += count

    if len(indices) != epoch_size or sum(session_draws.values()) != epoch_size:
        raise AssertionError("Session-balanced sampler changed the epoch size")
    if np.any(indices < 0) or np.any(indices >= epoch_size):
        raise AssertionError("Sampler produced an out-of-range window index")
    return indices, session_draws, dict(sorted(month_draws.items()))
