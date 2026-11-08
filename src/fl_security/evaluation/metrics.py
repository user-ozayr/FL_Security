"""Experiment logging with constrained output filenames and safe CSV export."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from fl_security.evaluation.csv_safety import sanitize_csv_dataframe


class ExperimentLogger:
    def __init__(self, output_dir="./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records = []

    def log_round(self, metrics: dict):
        self.records.append(metrics)

    def _safe_output_path(self, filename: str, required_suffix: str) -> Path:
        if (
            not isinstance(filename, str)
            or not filename
            or filename in {".", ".."}
            or "\x00" in filename
            or "/" in filename
            or "\\" in filename
            or ":" in filename
            or Path(filename).is_absolute()
            or Path(filename).name != filename
            or Path(filename).suffix.lower() != required_suffix
        ):
            raise ValueError(
                f"output filename must be a single {required_suffix} filename"
            )

        root = self.output_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
        target = root / filename
        if target.is_symlink() or target.resolve(strict=False).parent != root:
            raise ValueError("output filename must remain inside output_dir")
        return target

    def _open_output(self, filename: str, suffix: str, binary: bool = False):
        target = self._safe_output_path(filename, suffix)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target, flags, 0o600)
        if binary:
            return os.fdopen(descriptor, "wb")
        return os.fdopen(descriptor, "w", encoding="utf-8", newline="")

    def save_to_csv(self, filename="experiment_results.csv"):
        frame = sanitize_csv_dataframe(pd.DataFrame(self.records))
        with self._open_output(filename, ".csv") as output:
            frame.to_csv(output, index=False)

    def plot_metrics(self, filename="accuracy_vs_round.png"):
        frame = pd.DataFrame(self.records)
        if "round" not in frame.columns or "global_accuracy" not in frame.columns:
            return
        figure = plt.figure(figsize=(8, 5))
        try:
            plt.plot(frame["round"], frame["global_accuracy"], marker="o", linestyle="-")
            plt.title("Global Test Accuracy vs. Communication Round")
            plt.xlabel("Round")
            plt.ylabel("Accuracy")
            plt.grid(True)
            with self._open_output(filename, ".png", binary=True) as output:
                figure.savefig(output, format="png")
        finally:
            plt.close(figure)
