from collections import Counter

import pytest

from fl_security.mtd.controller import MTDController


def _secure_mtd_config():
    return {
        "experiment": {"seed": 42},
        "federated_learning": {"num_clients": 5},
        "mtd": {
            "enabled": True,
            "selection_strategy": "random",
            "committee_size": 3,
            "change_every_round": True,
        },
        "smpc": {
            "enabled": True,
            "num_aggregation_parties": 4,
            "min_clients_per_round": 3,
        },
    }


def test_round_committees_are_balanced_and_non_singleton():
    controller = MTDController(_secure_mtd_config())
    round_config = controller.generate_round_config(1, num_clients=5, fraction=0.6)

    assert len(round_config["selected_clients"]) == 3
    assert all(len(group) == 3 and len(set(group)) == 3
               for group in round_config["committees"].values())
    party_loads = Counter(
        party_id
        for group in round_config["committees"].values()
        for party_id in group
    )
    assert set(party_loads) == {0, 1, 2, 3}
    assert min(party_loads.values()) >= 2
    assert max(party_loads.values()) - min(party_loads.values()) <= 1


def test_mtd_changes_client_and_committee_assignments_by_round():
    controller = MTDController(_secure_mtd_config())
    round_one = controller.generate_round_config(1, num_clients=5, fraction=0.6)
    round_two = controller.generate_round_config(2, num_clients=5, fraction=0.6)

    assert round_one["selected_clients"] != round_two["selected_clients"] or (
        round_one["committees"] != round_two["committees"]
    )


def test_smpc_refuses_round_with_too_few_selected_clients():
    controller = MTDController(_secure_mtd_config())
    with pytest.raises(ValueError, match="at least 3 selected clients"):
        controller.generate_round_config(1, num_clients=5, fraction=0.4)
