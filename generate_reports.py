import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def main():
    os.makedirs("results", exist_ok=True)
    csv_files = glob.glob("*_results.csv")
    
    summary_data = []
    
    plt.figure(figsize=(10, 6))
    for file in csv_files:
        df = pd.read_csv(file)
        exp_name = file.replace("_results.csv", "")
        # Move CSV to results directory
        os.rename(file, os.path.join("results", file))
        
        if "global_accuracy" in df.columns and "round" in df.columns:
            plt.plot(df["round"], df["global_accuracy"] * 100, marker='o', label=exp_name)
            
            final_acc = df["global_accuracy"].iloc[-1] * 100
            final_loss = df["global_loss"].iloc[-1] if "global_loss" in df.columns else 0.0
            summary_data.append({
                "Experiment": exp_name,
                "Final Accuracy (%)": f"{final_acc:.2f}%",
                "Final Loss": f"{final_loss:.4f}"
            })

    plt.title("Federated Learning Benchmark - Accuracy vs Rounds")
    plt.xlabel("Communication Round")
    plt.ylabel("Global Test Accuracy (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/all_experiments_accuracy.png")
    plt.close()

    # Generate final text report
    with open("results/final_report.txt", "w") as f:
        f.write("=== FEDERATED LEARNING SECURITY EXPERIMENTAL BENCHMARK REPORT ===\n\n")
        f.write("1. Architecture & Framework:\n")
        f.write("   - Custom orchestration simulation loop over CIFAR-10 dataset.\n")
        f.write("   - Consistent CNN architecture with IID partitioning across 5 clients.\n")
        f.write("   - SMPC implementation explicitly designated as a single-process prototype.\n\n")
        
        f.write("2. Benchmark Execution Summary:\n")
        for item in summary_data:
            f.write(f"   - Experiment: {item['Experiment']} | Final Accuracy: {item['Final Accuracy (%)']} | Final Loss: {item['Final Loss']}\n")
        
        f.write("\n3. Reproducibility Commands:\n")
        f.write("   - Run full suite: ./run_all.sh\n")
        f.write("   - Run individual config: python run_experiment.py --config configs/<name>.yaml\n")
        f.write("   - Generate plots & report: python generate_reports.py\n\n")
        f.write("4. Known Limitations:\n")
        f.write("   - SMPC is a single-process additive sharing prototype (does not provide networked cryptographic multi-party privacy).\n")
        f.write("   - Limited to 3 rounds and 5 clients for lightweight validation runs.\n")

    print("Reports and plots successfully generated in 'results/' directory!")

if __name__ == "__main__":
    main()
