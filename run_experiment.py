"""Launch the Flower-based FL security experiment."""
import argparse
from fl_security.config import load_config
from fl_security.federated.flower_simulation import run_flower_simulation

def main():
    parser = argparse.ArgumentParser(description="Run FL Security with Flower")
    parser.add_argument("--config", required=True, help="Path to YAML configuration")
    args = parser.parse_args()
    config = load_config(args.config)
    framework = config.get("federated_learning", {}).get("framework", "flower")
    if framework != "flower":
        raise ValueError(f"Unsupported framework: {framework!r}; Flower is required")
    print(f"Starting Flower experiment: {config['experiment']['name']}")
    run_flower_simulation(config)

if __name__ == "__main__":
    main()
