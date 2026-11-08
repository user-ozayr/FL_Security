import os
import glob
import pandas as pd
import numpy as np

print("=== RESULTS DIRECTORY INSPECTION ===")
files = os.listdir("results")
print("Files in results/:", files)

csv_files = glob.glob("results/*.csv")
print(f"\nFound {len(csv_files)} CSV files.")

artifact_report = []
for file in sorted(csv_files):
    df = pd.read_csv(file)
    filename = os.path.basename(file)
    num_records = len(df)
    rounds = df["round"].max() if "round" in df.columns else "N/A"
    has_nan = df.isna().any().any()
    has_inf = np.isinf(df.select_dtypes(include=np.number)).any().any() if not df.empty else False
    
    print(f"\nFile: {filename}")
    print(f"  - Records: {num_records}")
    print(f"  - Max Round: {rounds}")
    print(f"  - Contains NaN: {has_nan}")
    print(f"  - Contains Inf: {has_inf}")
    print(f"  - Columns: {list(df.columns)}")
    
    artifact_report.append({
        "filename": filename,
        "records": num_records,
        "rounds": rounds,
        "has_nan": has_nan,
        "has_inf": has_inf
    })

print("\nArtifact inspection complete.")
