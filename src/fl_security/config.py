import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config

def validate_config(config: dict):
    assert "experiment" in config, "Missing 'experiment' section in config"
    assert "federated_learning" in config, "Missing 'federated_learning' section"
    fl = config["federated_learning"]
    assert fl["num_clients"] > 0, "num_clients must be positive"
    if "mtd" in config and config["mtd"].get("enabled", False):
        assert "committee_size" in config["mtd"], "MTD committee size required"
    if "smpc" in config and config["smpc"].get("enabled", False):
        assert "num_aggregation_parties" in config["smpc"], "SMPC aggregation parties required"
