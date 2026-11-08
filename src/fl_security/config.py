from pathlib import Path
import yaml

def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config

def validate_config(config: dict):
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    for section in ("experiment", "federated_learning"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"missing or invalid {section!r} section")
    experiment, fl = config["experiment"], config["federated_learning"]
    if not isinstance(experiment.get("rounds"), int) or experiment["rounds"] <= 0:
        raise ValueError("experiment.rounds must be a positive integer")
    clients = fl.get("num_clients")
    if isinstance(clients, bool) or not isinstance(clients, int) or clients < 2:
        raise ValueError("federated_learning.num_clients must be an integer >= 2")
    fraction = fl.get("fraction_train", 1.0)
    if not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
        raise ValueError("federated_learning.fraction_train must be in (0, 1]")
    if fl.get("framework", "flower") != "flower":
        raise ValueError("federated_learning.framework must be 'flower'")
    mtd, smpc = config.get("mtd", {}), config.get("smpc", {})
    if mtd.get("enabled", False) and int(mtd.get("committee_size", 3)) < 2:
        raise ValueError("mtd.committee_size must be >= 2")
    if smpc.get("enabled", False):
        if int(smpc.get("num_aggregation_parties", 0)) < 2:
            raise ValueError("smpc.num_aggregation_parties must be >= 2")
        if int(smpc.get("fixed_point_scale", 0)) <= 0:
            raise ValueError("smpc.fixed_point_scale must be positive")
        if "threshold" in smpc or smpc.get("dropout_recovery", False):
            raise ValueError("This additive-sharing prototype has no Shamir threshold/dropout recovery; remove unsupported SMPC options.")
    fraction_bad = not 0 <= float(config.get("attacks", {}).get("compromised_fraction", 0.0)) <= 1
    if fraction_bad:
        raise ValueError("attacks.compromised_fraction must be in [0, 1]")
