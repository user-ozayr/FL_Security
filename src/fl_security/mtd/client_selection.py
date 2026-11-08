"""Round-scoped client selection with reproducible and secret-keyed randomness."""
from __future__ import annotations

import hashlib
import secrets

from fl_security.mtd.randomness import HMACCounterRNG


def select_clients(
    num_clients: int,
    fraction: float,
    strategy: str = "random",
    seed: int | None = None,
    rng: HMACCounterRNG | None = None,
) -> list[int]:
    if isinstance(num_clients, bool) or not isinstance(num_clients, int) or num_clients <= 0:
        raise ValueError("num_clients must be a positive integer")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    count = min(num_clients, max(1, int(num_clients * fraction)))
    clients = list(range(num_clients))
    if strategy == "fixed":
        return clients[:count]
    if strategy != "random":
        raise ValueError(f"unsupported selection strategy: {strategy}")

    if rng is None:
        # A missing seed uses OS randomness. A supplied seed is intentionally
        # deterministic for repeatable simulations, not a deployment secret.
        key = (
            secrets.token_bytes(32)
            if seed is None
            else hashlib.sha256(
                b"fl-security-mtd-simulation-seed-v1:" + str(seed).encode("utf-8")
            ).digest()
        )
        rng = HMACCounterRNG(key)
    return sorted(int(cid) for cid in rng.sample(clients, count))
