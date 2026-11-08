import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class ExperimentLogger:
    def __init__(self, output_dir="./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records = []

    def log_round(self, metrics: dict):
        self.records.append(metrics)

    def save_to_csv(self, filename="experiment_results.csv"):
        df = pd.DataFrame(self.records)
        df.to_csv(self.output_dir / filename, index=False)

    def plot_metrics(self, filename="accuracy_vs_round.png"):
        df = pd.DataFrame(self.records)
        if "round" in df.columns and "global_accuracy" in df.columns:
            plt.figure(figsize=(8, 5))
            plt.plot(df["round"], df["global_accuracy"], marker='o', linestyle='-')
            plt.title("Global Test Accuracy vs. Communication Round")
            plt.xlabel("Round")
            plt.ylabel("Accuracy")
            plt.grid(True)
            plt.savefig(self.output_dir / filename)
            plt.close()
