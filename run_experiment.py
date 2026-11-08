"""Launch a federated-learning experiment."""
import argparse
from fl_security.config import load_config
from fl_security.federated.flower_simulation import run_flower_simulation
from fl_security.smpc.local_demo import run_with_local_aggregators


def main():
    parser = argparse.ArgumentParser(description="Run FL Security with Flower")
    parser.add_argument("--config", required=True, help="Path to YAML configuration")
    parser.add_argument(
        "--local-aggregators",
        action="store_true",
        help="Start one local aggregation-party process per configured party (SMPC only)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    framework = config.get("federated_learning", {}).get("framework", "flower")
    if framework != "flower":
        raise ValueError(f"Unsupported framework: {framework!r}; Flower is required")
    print(f"Starting Flower experiment: {config['experiment']['name']}")
    if args.local_aggregators:
        run_with_local_aggregators(config)
    else:
        run_flower_simulation(config)


if __name__ == "__main__":
    main()
