"""Deterministic HMAC-based pseudorandom stream for round-scoped MTD.

When the key is derived from FL_MTD_SECRET, outputs are unpredictable to
parties that do not know that secret. Without FL_MTD_SECRET, the caller uses a
public experiment seed for reproducible simulation only.
"""
from __future__ import annotations

import hashlib
import hmac


class HMACCounterRNG:
    """A small HMAC-SHA256 counter-mode PRF with unbiased randbelow sampling."""

    def __init__(self, key: bytes) -> None:
        if not isinstance(key, bytes) or len(key) < 16:
            raise ValueError("MTD random key must contain at least 16 bytes")
        self._key = hmac.new(
            key, b"fl-security-mtd-counter-rng-v1", hashlib.sha256
        ).digest()
        self._counter = 0
        self._buffer = bytearray()

    def _read(self, count: int) -> bytes:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("byte count must be a non-negative integer")
        while len(self._buffer) < count:
            if self._counter >= 2**128:
                raise OverflowError("MTD random stream counter exhausted")
            block = hmac.new(
                self._key,
                b"block\x00" + self._counter.to_bytes(16, "big"),
                hashlib.sha256,
            ).digest()
            self._counter += 1
            self._buffer.extend(block)
        result = bytes(self._buffer[:count])
        del self._buffer[:count]
        return result

    def randbelow(self, upper: int) -> int:
        """Return an unbiased integer in range(upper)."""
        if isinstance(upper, bool) or not isinstance(upper, int) or upper <= 0:
            raise ValueError("upper bound must be a positive integer")
        width = max(1, (upper.bit_length() + 7) // 8)
        span = 1 << (8 * width)
        limit = span - (span % upper)
        while True:
            candidate = int.from_bytes(self._read(width), "big")
            if candidate < limit:
                return candidate % upper

    def shuffle(self, values: list) -> None:
        """In-place Fisher-Yates shuffle."""
        for index in range(len(values) - 1, 0, -1):
            other = self.randbelow(index + 1)
            values[index], values[other] = values[other], values[index]

    def sample(self, population, count: int) -> list:
        """Select count unique elements without replacement."""
        values = list(population)
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or count > len(values)
        ):
            raise ValueError("sample size must be between zero and population size")
        for index in range(count):
            other = index + self.randbelow(len(values) - index)
            values[index], values[other] = values[other], values[index]
        return values[:count]
