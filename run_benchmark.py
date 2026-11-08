"""Run paired multi-seed FL experiments and report confidence intervals.

Example:
    python run_benchmark.py \
      --configs configs/baseline_sign_flip.yaml configs/mtd_sign_flip.yaml \
                 configs/smpc_sign_flip.yaml configs/hybrid.yaml \
      --baseline baseline_sign_flip --seeds 5
"""
from __future__ import annotations

import argparse
import copy
import os
import re
from pathlib import Path

import pandas as pd

from fl_security.config import load_config
from fl_security.evaluation.statistics import summarize_values
from fl_security.federated.flower_simulation import run_flower_simulation
from fl_security.smpc.local_demo import run_with_local_aggregators


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")
    if not result:
        raise ValueError(f"Cannot create a file-safe name from {value!r}")
    return result


def _summarize_by_round(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    excluded = {"condition", "config", "seed", "round", "attack_type"}
    metric_columns = [
        column for column in frame.select_dtypes(include="number").columns
        if column not in excluded
    ]
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        metadata = dict(zip(group_columns, key_values))
        for metric in metric_columns:
            rows.append({
                **metadata,
                "metric": metric,
                **summarize_values(group[metric].tolist()),
            })
    return pd.DataFrame(rows)


def run_benchmark(
    config_paths: list[str],
    seeds: list[int],
    baseline_condition: str | None,
    output_dir: Path,
    use_local_aggregators: bool,
    skip_existing: bool,
) -> dict[str, Path]:
    if len(config_paths) < 2:
        raise ValueError("Provide at least two configs to compare.")
    loaded = []
    for config_path in config_paths:
        config = load_config(config_path)
        condition = Path(config_path).stem
        loaded.append((condition, config_path, config))

    conditions = [entry[0] for entry in loaded]
    if len(set(conditions)) != len(conditions):
        raise ValueError("Config filenames must have unique stems.")
    if baseline_condition is None:
        baseline_condition = conditions[0]
    if baseline_condition not in conditions:
        raise ValueError(
            f"Baseline {baseline_condition!r} is not one of the config stems: {conditions}"
        )

    raw_frames = []
    metadata_rows = []
    for condition, config_path, base_config in loaded:
        condition_dir = output_dir / "runs" / _slug(condition)
        condition_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            run_config = copy.deepcopy(base_config)
            run_name = f"{_slug(condition)}_seed{seed}"
            run_config["experiment"]["seed"] = int(seed)
            run_config["experiment"]["name"] = run_name
            evaluation = run_config.setdefault("evaluation", {})
            evaluation["output_dir"] = str(condition_dir)
            evaluation["save_csv"] = True
            evaluation["save_plots"] = False

            csv_path = condition_dir / f"{run_name}_results.csv"
            if not (skip_existing and csv_path.exists()):
                if use_local_aggregators and run_config.get("smpc", {}).get("enabled", False):
                    run_with_local_aggregators(run_config)
                else:
                    run_flower_simulation(run_config)

            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Experiment {condition!r}, seed {seed} completed without {csv_path}"
                )
            frame = pd.read_csv(csv_path)
            frame["condition"] = condition
            frame["config"] = str(config_path)
            frame["seed"] = int(seed)
            raw_frames.append(frame)

            attack_config = run_config.get("attacks", {})
            mtd_config = run_config.get("mtd", {})
            smpc_config = run_config.get("smpc", {})
            metadata_rows.append({
                "condition": condition,
                "config": str(config_path),
                "seed": int(seed),
                "experiment_name": run_name,
                "mtd_enabled": bool(mtd_config.get("enabled", False)),
                "mtd_secret_seeded": bool(os.environ.get("FL_MTD_SECRET")),
                "smpc_enabled": bool(smpc_config.get("enabled", False)),
                "attack_enabled": bool(attack_config.get("enabled", False)),
                "attack_type": (
                    attack_config.get("attack_type", "none")
                    if attack_config.get("enabled", False) else "none"
                ),
                "compromised_fraction": float(attack_config.get("compromised_fraction", 0.0)),
                "num_clients": int(run_config["federated_learning"]["num_clients"]),
                "client_fraction": float(run_config["federated_learning"].get("fraction_train", 1.0)),
                "local_epochs": int(run_config["federated_learning"].get("local_epochs", 1)),
                "num_aggregation_parties": int(smpc_config.get("num_aggregation_parties", 0)),
                "committee_size": int(mtd_config.get("committee_size", 0)),
                "fixed_point_scale": int(smpc_config.get("fixed_point_scale", 0)),
                "field_modulus": int(smpc_config.get("field_modulus", 0)),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.concat(raw_frames, ignore_index=True)
    raw_path = output_dir / "all_rounds.csv"
    raw.to_csv(raw_path, index=False)

    summary = _summarize_by_round(raw, ["condition", "round"])
    summary_path = output_dir / "summary_by_round.csv"
    summary.to_csv(summary_path, index=False)

    metadata_path = output_dir / "condition_metadata.csv"
    pd.DataFrame(metadata_rows).drop_duplicates(
        subset=["condition", "seed"]
    ).to_csv(metadata_path, index=False)

    paired_path = output_dir / "paired_accuracy_differences.csv"
    paired_rows = []
    baseline = raw[raw["condition"] == baseline_condition][
        ["seed", "round", "global_accuracy", "global_loss"]
    ].rename(columns={
        "global_accuracy": "baseline_accuracy",
        "global_loss": "baseline_loss",
    })
    for condition in conditions:
        if condition == baseline_condition:
            continue
        comparison = raw[raw["condition"] == condition][
            ["seed", "round", "global_accuracy", "global_loss"]
        ].rename(columns={
            "global_accuracy": "comparison_accuracy",
            "global_loss": "comparison_loss",
        })
        paired = comparison.merge(baseline, on=["seed", "round"], how="inner")
        paired["accuracy_difference"] = (
            paired["comparison_accuracy"] - paired["baseline_accuracy"]
        )
        paired["loss_difference"] = paired["comparison_loss"] - paired["baseline_loss"]
        for round_num, group in paired.groupby("round"):
            for metric in ("accuracy_difference", "loss_difference"):
                paired_rows.append({
                    "baseline": baseline_condition,
                    "comparison": condition,
                    "round": int(round_num),
                    "metric": metric,
                    **summarize_values(group[metric].tolist()),
                })
    pd.DataFrame(paired_rows).to_csv(paired_path, index=False)

    return {
        "raw": raw_path,
        "summary": summary_path,
        "metadata": metadata_path,
        "paired": paired_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run paired multi-seed Flower FL security comparisons."
    )
    parser.add_argument("--configs", nargs="+", required=True, help="YAML configs to compare")
    parser.add_argument("--baseline", default=None, help="Baseline config filename stem")
    parser.add_argument("--seeds", type=int, default=5, help="Number of paired seeds")
    parser.add_argument("--start-seed", type=int, default=None, help="First seed; defaults to baseline seed")
    parser.add_argument("--output-dir", default="./results/benchmark")
    parser.add_argument(
        "--no-local-aggregators",
        action="store_true",
        help="Use externally configured aggregation services instead of local processes",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Reuse run CSVs already on disk")
    args = parser.parse_args()

    if args.seeds < 1:
        parser.error("--seeds must be >= 1")
    baseline_path = next(
        (path for path in args.configs if Path(path).stem == args.baseline),
        None,
    ) if args.baseline else args.configs[0]
    if baseline_path is None:
        parser.error("--baseline must match one of the --configs filename stems")
    baseline_config = load_config(baseline_path)
    first_seed = (
        int(args.start_seed)
        if args.start_seed is not None
        else int(baseline_config["experiment"].get("seed", 42))
    )
    seeds = list(range(first_seed, first_seed + args.seeds))

    paths = run_benchmark(
        config_paths=args.configs,
        seeds=seeds,
        baseline_condition=args.baseline or Path(baseline_path).stem,
        output_dir=Path(args.output_dir),
        use_local_aggregators=not args.no_local_aggregators,
        skip_existing=args.skip_existing,
    )
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
