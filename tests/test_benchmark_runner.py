import math
import sys
from pathlib import Path

import pandas as pd

# run_benchmark.py is a root-level CLI script, not an installed package module.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from run_benchmark import _summarize_by_round


def test_benchmark_summary_aggregates_seed_metrics_by_round():
    frame = pd.DataFrame([
        {"condition": "baseline", "round": 1, "seed": 1, "global_accuracy": 0.50},
        {"condition": "baseline", "round": 1, "seed": 2, "global_accuracy": 0.60},
        {"condition": "baseline", "round": 2, "seed": 1, "global_accuracy": 0.65},
        {"condition": "baseline", "round": 2, "seed": 2, "global_accuracy": 0.75},
    ])
    summary = _summarize_by_round(frame, ["condition", "round"])
    first_round = summary[
        (summary["condition"] == "baseline")
        & (summary["round"] == 1)
        & (summary["metric"] == "global_accuracy")
    ].iloc[0]
    assert first_round["n"] == 2
    assert math.isclose(first_round["mean"], 0.55, rel_tol=1e-12)
    assert first_round["ci95_lower"] < first_round["mean"] < first_round["ci95_upper"]
