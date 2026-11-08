from pathlib import Path

import pytest

from fl_security.config import load_config, validate_config


@pytest.mark.parametrize("path", sorted(Path("configs").glob("*.yaml")))
def test_all_committed_experiment_configs_validate(path):
    config = load_config(str(path))
    assert config["federated_learning"]["framework"] == "flower"


def test_secure_config_rejects_a_client_fraction_below_minimum_cohort():
    config = {
        "experiment": {"name": "too-small", "seed": 1, "rounds": 1},
        "federated_learning": {
            "framework": "flower",
            "num_clients": 5,
            "fraction_train": 0.4,
        },
        "mtd": {"enabled": True, "committee_size": 3},
        "smpc": {
            "enabled": True,
            "num_aggregation_parties": 4,
            "min_clients_per_round": 3,
            "fixed_point_scale": 1_000_000,
        },
    }
    with pytest.raises(ValueError, match="too low"):
        validate_config(config)
