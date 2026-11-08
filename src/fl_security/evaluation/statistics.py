"""Small-sample confidence intervals for repeated federated experiments."""
from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Iterable

# Two-sided 95% Student-t critical values for df=1..30.
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def summarize_values(values: Iterable[float]) -> dict[str, float | int]:
    """Return n, mean, sample std and a two-sided 95% Student-t interval."""
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    count = len(clean)
    if count == 0:
        return {
            "n": 0, "mean": float("nan"), "std": float("nan"),
            "ci95_lower": float("nan"), "ci95_upper": float("nan"),
        }
    avg = mean(clean)
    if count == 1:
        return {
            "n": 1, "mean": avg, "std": float("nan"),
            "ci95_lower": float("nan"), "ci95_upper": float("nan"),
        }
    deviation = stdev(clean)
    critical = _T975.get(count - 1, 1.96)
    margin = critical * deviation / math.sqrt(count)
    return {
        "n": count,
        "mean": avg,
        "std": deviation,
        "ci95_lower": avg - margin,
        "ci95_upper": avg + margin,
    }
