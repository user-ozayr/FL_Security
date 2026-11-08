#!/bin/bash
set -e
echo "Starting full benchmark suite execution..."
for cfg in configs/*.yaml; do
    echo "========================================="
    echo "Running experiment with config: $cfg"
    echo "========================================="
    python run_experiment.py --config "$cfg"
done
echo "All experiments completed successfully!"
