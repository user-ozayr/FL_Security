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
    if isinstance(experiment.get("rounds"), bool) or not isinstance(experiment.get("rounds"), int) or experiment["rounds"] <= 0:
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
        parties = int(smpc.get("num_aggregation_parties", 0))
        committee = int(mtd.get("committee_size", 3))
        minimum = int(smpc.get("min_clients_per_round", 2))
        if parties < 2:
            raise ValueError("smpc.num_aggregation_parties must be >= 2")
        if minimum < 2:
            raise ValueError("smpc.min_clients_per_round must be >= 2")
        if int(smpc.get("fixed_point_scale", 0)) <= 0:
            raise ValueError("smpc.fixed_point_scale must be positive")
        if "threshold" in smpc or smpc.get("dropout_recovery", False):
            raise ValueError("This additive-sharing prototype has no Shamir threshold/dropout recovery; remove unsupported options.")
        selected_count = max(1, int(clients * fraction))
        effective_committee = min(committee, parties)
        if selected_count < minimum:
            raise ValueError("client fraction is too low for smpc.min_clients_per_round")
        if selected_count * effective_committee < 2 * parties:
            raise ValueError(
                "committee configuration must route each aggregation party shares from "
                "at least two selected clients per round"
            )

    compromised_fraction = float(config.get("attacks", {}).get("compromised_fraction", 0.0))
    if not 0 <= compromised_fraction <= 1:
        raise ValueError("attacks.compromised_fraction must be in [0, 1]")
