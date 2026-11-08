import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fl_security.attacks.byzantine import ByzantineAttacker
from fl_security.federated.aggregation import weighted_fedavg
from fl_security.federated.training import train_local
from fl_security.mtd.controller import MTDController
from fl_security.smpc.fixed_point import FixedPointEncoder


def test_sign_flip_client_updates_reach_secure_fedavg():
    config = {
        "experiment": {"seed": 7},
        "federated_learning": {"num_clients": 2},
        "mtd": {"enabled": True, "committee_size": 2},
        "smpc": {"enabled": True, "num_aggregation_parties": 2, "min_clients_per_round": 2},
        "attacks": {
            "enabled": True,
            "attack_type": "sign_flip",
            "compromised_fraction": 1.0,
            "activation_rounds": [1],
            "strength": 1.0,
        },
    }
    attacker = ByzantineAttacker(config, num_clients=2)
    round_config = MTDController(config).generate_round_config(1, 2, 1.0)
    global_state = {"weight": torch.tensor([0.0, 0.0])}
    honest_states = [
        {"weight": torch.tensor([1.0, -2.0])},
        {"weight": torch.tensor([3.0, 4.0])},
    ]
    poisoned = [
        attacker.perturb_update(cid, 1, global_state, local)
        for cid, local in enumerate(honest_states)
    ]
    secure_average = weighted_fedavg(
        poisoned, [1, 1], global_state,
        encoder=FixedPointEncoder(1_000_000),
        committees={index: round_config["committees"][cid]
                    for index, cid in enumerate(round_config["selected_clients"])},
    )
    torch.testing.assert_close(
        secure_average["weight"], torch.tensor([-2.0, -1.0]), atol=1e-6, rtol=0
    )


def test_label_flip_is_applied_inside_local_training():
    config = {
        "experiment": {"seed": 11},
        "attacks": {
            "enabled": True,
            "attack_type": "label_flip",
            "compromised_fraction": 1.0,
            "activation_rounds": [1],
        },
    }
    attacker = ByzantineAttacker(config, num_clients=1)
    model = nn.Linear(2, 2)
    dataset = TensorDataset(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]),
        torch.tensor([0, 1, 0, 1]),
    )
    calls = []

    def flip_batch(labels):
        calls.append(labels.clone())
        return torch.as_tensor(
            attacker.perturb_labels(0, 1, labels, num_classes=2),
            dtype=torch.long,
            device=labels.device,
        )

    train_local(
        model, DataLoader(dataset, batch_size=2, shuffle=False),
        epochs=1, lr=0.05, label_transform=flip_batch,
    )
    assert len(calls) == 2
    assert torch.equal(calls[0], torch.tensor([0, 1]))
    assert torch.equal(flip_batch(torch.tensor([0, 1])), torch.tensor([1, 0]))
