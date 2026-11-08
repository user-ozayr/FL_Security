"""Deterministic, round-scoped client selection."""
import numpy as np

def select_clients(num_clients: int, fraction: float, strategy: str = "random", seed: int | None = None) -> list[int]:
    if isinstance(num_clients, bool) or not isinstance(num_clients, int) or num_clients <= 0:
        raise ValueError("num_clients must be a positive integer")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    count = min(num_clients, max(1, int(num_clients * fraction)))
    clients = list(range(num_clients))
    if strategy == "fixed":
        return clients[:count]
    if strategy != "random":
        raise ValueError(f"unsupported selection strategy: {strategy}")
    return sorted(int(cid) for cid in np.random.default_rng(seed).choice(clients, count, replace=False))
