#!/usr/bin/env bash
set -euo pipefail

echo "Starting FL security benchmark suite..."
for cfg in configs/*.yaml; do
    echo "========================================="
    echo "Running experiment: $cfg"
    echo "========================================="
    # SMPC configurations start and stop local party processes automatically.
    python run_experiment.py --config "$cfg" --local-aggregators
done
echo "Benchmark suite completed."
