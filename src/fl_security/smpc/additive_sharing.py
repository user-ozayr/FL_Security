"""Finite-field additive sharing for simulated secure aggregation."""
from __future__ import annotations

import secrets
from functools import lru_cache
from typing import Iterable, List, Sequence

import numpy as np

DEFAULT_FIELD_MODULUS = 2**61 - 1


@lru_cache(maxsize=128)
def _is_prime(value: int) -> bool:
    """Deterministic Miller-Rabin test for integers below 2**64."""
    if value < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small_primes:
        if value % prime == 0:
            return value == prime
    odd_part, power_of_two = value - 1, 0
    while odd_part % 2 == 0:
        odd_part //= 2
        power_of_two += 1
    # This base set is deterministic for n < 2**64.
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        base %= value
        if base == 0:
            continue
        witness = pow(base, odd_part, value)
        if witness in (1, value - 1):
            continue
        for _ in range(power_of_two - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def validate_modulus(modulus: int) -> int:
    if isinstance(modulus, bool) or not isinstance(modulus, int):
        raise TypeError("modulus must be an integer")
    if modulus <= 2 or modulus > 2**62 - 1:
        raise ValueError("modulus must be a prime in [3, 2**62-1]")
    if not _is_prime(modulus):
        raise ValueError("modulus must be prime to define the configured finite field")
    return modulus


def _modular_sum(left: np.ndarray, right: np.ndarray, modulus: int) -> np.ndarray:
    # Each input is a residue and 2*(q-1) fits in signed int64.
    return np.remainder(left + right, modulus).astype(np.int64, copy=False)


def _secure_random_field_elements(count: int, modulus: int) -> np.ndarray:
    """Generate unbiased field elements using OS-backed cryptographic bytes."""
    if count == 0:
        return np.empty((0,), dtype=np.int64)
    span = 1 << 64
    limit = span - (span % modulus)
    output = np.empty(count, dtype=np.int64)
    written = 0
    # Bound temporary allocations for large tensors (e.g. CIFAR-10's dense layer).
    max_batch = 1_000_000
    while written < count:
        remaining = count - written
        batch_size = min(max_batch, max(64, remaining + max(16, remaining // 16)))
        candidates = np.frombuffer(secrets.token_bytes(batch_size * 8), dtype=np.uint64)
        accepted = candidates if limit == span else candidates[candidates < np.uint64(limit)]
        take = accepted[:remaining]
        if take.size:
            output[written:written + take.size] = np.remainder(
                take, np.uint64(modulus)
            ).astype(np.int64)
            written += int(take.size)
    return output


def generate_additive_shares(
    secret: np.ndarray,
    num_parties: int,
    modulus: int = DEFAULT_FIELD_MODULUS,
) -> List[np.ndarray]:
    """Split an integer array into uniformly random additive shares in Z_q."""
    q = validate_modulus(modulus)
    if isinstance(num_parties, bool) or not isinstance(num_parties, int) or num_parties < 2:
        raise ValueError("num_parties must be an integer >= 2")
    values = np.asarray(secret)
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("secret must contain integers")
    if values.dtype == np.uint64 and values.size and np.any(
        values > np.uint64(np.iinfo(np.int64).max)
    ):
        raise OverflowError("secret is outside int64 range")
    secret_mod = np.remainder(values.astype(np.int64), q)
    shares, running = [], np.zeros(secret_mod.shape, dtype=np.int64)
    for _ in range(num_parties - 1):
        share = _secure_random_field_elements(secret_mod.size, q).reshape(secret_mod.shape)
        shares.append(share)
        running = _modular_sum(running, share, q)
    shares.append(np.remainder(secret_mod - running, q).astype(np.int64))
    return shares


def reconstruct_additive_shares(
    shares: Iterable[np.ndarray],
    modulus: int = DEFAULT_FIELD_MODULUS,
) -> np.ndarray:
    """Reconstruct a value from all additive shares in Z_q."""
    q = validate_modulus(modulus)
    items = [np.asarray(share) for share in shares]
    if len(items) < 2:
        raise ValueError("at least two shares are required")
    shape = items[0].shape
    if any(
        item.shape != shape or not np.issubdtype(item.dtype, np.integer)
        for item in items
    ):
        raise ValueError("shares must be same-shaped integer arrays")
    total = np.zeros(shape, dtype=np.int64)
    for item in items:
        total = _modular_sum(total, np.remainder(item, q), q)
    return total


def aggregate_shared_updates(
    encoded_updates: Sequence[np.ndarray],
    committees: Sequence[Sequence[int]],
    modulus: int = DEFAULT_FIELD_MODULUS,
) -> np.ndarray:
    """Route shares to MTD committees and reconstruct only the total sum.

    Logical aggregators sum the shares addressed to them. No client update is
    reconstructed individually. This in-process routine validates the
    arithmetic and routing only; it does not hide updates from a coordinator
    that already receives the original client arrays.
    """
    q = validate_modulus(modulus)
    updates = [np.asarray(update) for update in encoded_updates]
    assignments = [list(group) for group in committees]
    if not updates or len(updates) != len(assignments):
        raise ValueError("non-empty updates and one committee per update are required")
    shape = updates[0].shape
    if any(
        update.shape != shape or not np.issubdtype(update.dtype, np.integer)
        for update in updates
    ):
        raise ValueError("updates must be same-shaped integer arrays")
    partials = {}
    for update, committee in zip(updates, assignments):
        if len(committee) < 2 or len(set(committee)) != len(committee):
            raise ValueError("each committee must contain at least two distinct parties")
        if any(
            isinstance(party_id, bool) or not isinstance(party_id, int) or party_id < 0
            for party_id in committee
        ):
            raise ValueError("party IDs must be non-negative integers")
        for party_id, share in zip(
            committee, generate_additive_shares(update, len(committee), q)
        ):
            if party_id not in partials:
                partials[party_id] = np.zeros(shape, dtype=np.int64)
            partials[party_id] = _modular_sum(partials[party_id], share, q)
    if len(partials) < 2:
        raise ValueError("aggregation requires at least two logical parties")
    return reconstruct_additive_shares(partials.values(), q)
