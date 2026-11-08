from types import SimpleNamespace
import os

import pandas as pd
import torch
from torch.utils.data import TensorDataset

import flwr.simulation
import fl_security.federated.flower_simulation as flower_runner
from flwr.common import parameters_to_ndarrays

from fl_security.smpc.local_demo import run_with_local_aggregators


def test_flower_strategy_uses_client_to_party_shares_not_plaintext_updates(
    monkeypatch, tmp_path
):
    """Exercise the actual Flower NumPyClient/Strategy path with only Ray mocked."""
    torch.set_num_threads(1)
    images = torch.randn(6, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    trainset = TensorDataset(images, labels)
    testset = TensorDataset(images[:4], labels[:4])
    client_partitions = [
        TensorDataset(images[offset:offset + 2], labels[offset:offset + 2])
        for offset in (0, 2, 4)
    ]
    monkeypatch.setattr(
        flower_runner, "load_cifar10_data", lambda: (trainset, testset)
    )
    monkeypatch.setattr(
        flower_runner, "partition_data", lambda *args, **kwargs: client_partitions
    )

    def deterministic_flower_scheduler(client_fn, num_clients, config, strategy, **kwargs):
        # Coordinator-only secrets must not be inherited by Ray/client workers.
        assert "FL_SMPC_AGGREGATORS_JSON" not in os.environ
        assert "FL_MTD_SECRET" not in os.environ
        assert "FL_SMPC_CONTROL_TOKEN" not in os.environ
        assert "FL_SMPC_CLIENT_KEYS_JSON" not in os.environ
        assert "FL_SMPC_CLIENT_TOKENS_JSON" in os.environ
        proxies = {
            str(cid): SimpleNamespace(cid=str(cid)) for cid in range(num_clients)
        }
        client_manager = SimpleNamespace(all=lambda: proxies)
        parameters = strategy.initialize_parameters(client_manager)

        for server_round in range(1, config.num_rounds + 1):
            instructions = strategy.configure_fit(
                server_round, parameters, client_manager
            )
            results = []
            for proxy, fit_ins in instructions:
                client = client_fn(proxy.cid)
                fit_res = client.fit(fit_ins)
                # In SMPC mode, a Flower FitRes must contain no client tensors.
                assert parameters_to_ndarrays(fit_res.parameters) == []
                assert fit_res.metrics["shares_uploaded"] == 1
                results.append((proxy, fit_res))

            parameters, fit_metrics = strategy.aggregate_fit(
                server_round, results, []
            )
            assert parameters is not None
            assert fit_metrics["smpc_enabled"] == 1
            assert fit_metrics["selected_clients"] == 3
            assert fit_metrics["aggregators_used"] == 3

            loss, evaluation_metrics = strategy.evaluate(server_round, parameters)
            assert loss >= 0
            assert 0.0 <= evaluation_metrics["accuracy"] <= 1.0

        return None

    monkeypatch.setattr(
        flwr.simulation, "start_simulation", deterministic_flower_scheduler
    )
    config = {
        "experiment": {"name": "flower_secure_smoke", "seed": 123, "rounds": 1},
        "federated_learning": {
            "framework": "flower",
            "dataset": "cifar10",
            "num_clients": 3,
            "fraction_train": 1.0,
            "local_epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "partitioning": "iid",
        },
        "mtd": {
            "enabled": True,
            "selection_strategy": "random",
            "committee_size": 2,
            "change_every_round": True,
        },
        "smpc": {
            "enabled": True,
            "scheme": "additive_sharing",
            "fixed_point_scale": 100_000,
            "field_modulus": 2305843009213693951,
            "num_aggregation_parties": 3,
            "min_clients_per_round": 2,
        },
        "attacks": {
            "enabled": False,
            "attack_type": "sign_flip",
            "compromised_fraction": 0.0,
            "strength": 1.0,
        },
        "evaluation": {
            "output_dir": str(tmp_path),
            "save_csv": True,
            "save_plots": False,
        },
    }

    mtd_secret = "mtd-secret-for-test-" + ("x" * 32)
    monkeypatch.setenv("FL_MTD_SECRET", mtd_secret)
    run_with_local_aggregators(config)
    assert os.environ["FL_MTD_SECRET"] == mtd_secret

    results = pd.read_csv(tmp_path / "flower_secure_smoke_results.csv")
    assert len(results) == 1
    assert results.loc[0, "smpc_enabled"] == 1
    assert results.loc[0, "selected_clients"] == 3
    assert results.loc[0, "completed_clients"] == 3
    assert results.loc[0, "share_payload_bytes"] > 0
    assert results.loc[0, "aggregator_cleanup_failures"] == 0
