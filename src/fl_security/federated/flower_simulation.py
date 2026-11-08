"""Flower-backed simulation adapter for the FL/MTD/additive-sharing pipeline."""
from __future__ import annotations
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from fl_security.attacks.byzantine import ByzantineAttacker
from fl_security.evaluation.metrics import ExperimentLogger
from fl_security.federated.aggregation import weighted_fedavg
from fl_security.federated.dataset import load_cifar10_data, partition_data
from fl_security.federated.model import CIFAR10CNN
from fl_security.federated.training import evaluate_model, train_local
from fl_security.mtd.controller import MTDController
from fl_security.smpc.additive_sharing import DEFAULT_FIELD_MODULUS

def run_flower_simulation(config: dict) -> None:
    """Run Flower client training and a coordinator-side sharing arithmetic simulation."""
    try:
        from flwr.client import NumPyClient
        from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays
        from flwr.server import ServerConfig
        from flwr.server.strategy import Strategy
        from flwr.simulation import start_simulation
    except ImportError as exc:
        raise RuntimeError(
            "Flower simulation dependencies are missing. Install with pip install -e .[simulation]."
        ) from exc

    experiment, fl_config = config["experiment"], config["federated_learning"]
    num_clients, rounds = int(fl_config["num_clients"]), int(experiment["rounds"])
    seed = int(experiment.get("seed", 42))
    torch.manual_seed(seed)
    trainset, testset = load_cifar10_data()
    client_datasets = partition_data(trainset, num_clients, fl_config.get("partitioning", "iid"), seed)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attacker, mtd = ByzantineAttacker(config, num_clients), MTDController(config)
    smpc = config.get("smpc", {})
    use_sharing_simulation = bool(smpc.get("enabled", False))
    modulus = int(smpc.get("field_modulus", DEFAULT_FIELD_MODULUS))
    encoder = None
    if use_sharing_simulation:
        from fl_security.smpc.fixed_point import FixedPointEncoder
        encoder = FixedPointEncoder(int(smpc.get("fixed_point_scale", 1_000_000)))

    global_model = CIFAR10CNN()
    initial_state = global_model.state_dict()
    state_keys = list(initial_state)
    initial_arrays = [tensor.detach().cpu().numpy() for tensor in initial_state.values()]
    initial_parameters = ndarrays_to_parameters(initial_arrays)
    logger = ExperimentLogger(output_dir=config.get("evaluation", {}).get("output_dir", "./results"))

    class ResearchNumPyClient(NumPyClient):
        def __init__(self, cid: int):
            self.cid, self.dataset = cid, client_datasets[cid]

        def get_parameters(self, config):
            return [array.copy() for array in initial_arrays]

        def fit(self, parameters, fit_config):
            round_num = int(fit_config["server_round"])
            state = {key: torch.from_numpy(array.copy()) for key, array in zip(state_keys, parameters)}
            model = CIFAR10CNN()
            model.load_state_dict(state)
            torch.manual_seed(seed + round_num * 100_000 + self.cid)
            loader = DataLoader(self.dataset, batch_size=int(fl_config.get("batch_size", 32)), shuffle=True)
            label_transform = None
            if attacker.is_malicious(self.cid, round_num) and attacker.attack_type == "label_flip":
                def label_transform(labels):
                    values = attacker.perturb_labels(self.cid, round_num, labels, num_classes=10)
                    return torch.as_tensor(values, dtype=torch.long, device=labels.device)
            local_state = train_local(
                model, loader, epochs=int(fl_config.get("local_epochs", 1)),
                lr=float(fl_config.get("learning_rate", 0.01)),
                device=device, label_transform=label_transform,
            )
            local_state = attacker.perturb_update(self.cid, round_num, state, local_state)
            # This legacy Flower FitRes returns clear model arrays to the server.
            # The later share arithmetic does NOT protect this communication hop.
            return (
                [local_state[key].detach().cpu().numpy() for key in state_keys],
                len(self.dataset),
                {"client_id": int(self.cid)},
            )

        def evaluate(self, parameters, config):
            return 0.0, 0, {"client_id": int(self.cid)}

    class ResearchStrategy(Strategy):
        """Flower strategy with MTD sampling and committee-routed sharing arithmetic."""
        def __init__(self):
            self.last_round_config, self.last_selected, self.last_completed = {}, [], []
            self.last_malicious, self.last_aggregation_seconds, self.last_share_bytes = [], 0.0, 0

        def initialize_parameters(self, client_manager):
            return initial_parameters

        def configure_fit(self, server_round, parameters, client_manager):
            available = client_manager.all()
            self.last_round_config = mtd.generate_round_config(
                server_round, num_clients, float(fl_config.get("fraction_train", 1.0))
            )
            self.last_selected = list(self.last_round_config["selected_clients"])
            return [
                (available[str(cid)], FitIns(parameters, {"server_round": int(server_round), "client_id": int(cid)}))
                for cid in self.last_selected if str(cid) in available
            ]

        def aggregate_fit(self, server_round, results, failures):
            if not results:
                self.last_completed = []
                return None, {}
            round_config = self.last_round_config
            client_ids = [int(proxy.cid) for proxy, _ in results]
            self.last_completed = client_ids
            states, counts, committees = [], [], {}
            for cid, (_, fit_res) in zip(client_ids, results):
                arrays = parameters_to_ndarrays(fit_res.parameters)
                states.append({key: torch.from_numpy(array.copy()) for key, array in zip(state_keys, arrays)})
                counts.append(int(fit_res.num_examples))
                committees[len(states) - 1] = round_config["committees"][cid]
            self.last_malicious = [cid for cid in client_ids if attacker.is_malicious(cid, server_round)]
            self.last_share_bytes = 0
            if use_sharing_simulation:
                # Encoded shares are int64, even though Flower FitRes arrays are float32.
                self.last_share_bytes = sum(
                    sum(t.numel() for t in state.values() if torch.is_floating_point(t))
                    * 8 * len(round_config["committees"][cid])
                    for cid, state in zip(client_ids, states)
                )
            started = time.perf_counter()
            aggregated = weighted_fedavg(
                states, counts, global_model.state_dict(), encoder=encoder,
                committees=committees if use_sharing_simulation else None, modulus=modulus,
            )
            self.last_aggregation_seconds = time.perf_counter() - started
            global_model.load_state_dict(aggregated)
            return (
                ndarrays_to_parameters([tensor.detach().cpu().numpy() for tensor in global_model.state_dict().values()]),
                {
                    "selected_clients": int(len(client_ids)),
                    "malicious_clients": int(len(self.last_malicious)),
                    "additive_sharing_simulation_enabled": int(use_sharing_simulation),
                    "client_updates_delivered_in_clear_to_flower_server": 1,
                    "share_payload_bytes_estimate": int(self.last_share_bytes),
                    "aggregation_seconds": float(self.last_aggregation_seconds),
                },
            )

        def configure_evaluate(self, server_round, parameters, client_manager):
            return []

        def aggregate_evaluate(self, server_round, results, failures):
            return None, {}

        def evaluate(self, server_round, parameters):
            arrays = parameters_to_ndarrays(parameters)
            model = CIFAR10CNN()
            model.load_state_dict({key: torch.from_numpy(array.copy()) for key, array in zip(state_keys, arrays)})
            loss, accuracy = evaluate_model(model, testloader, device=device)
            logger.log_round({
                "round": int(server_round),
                "global_loss": float(loss),
                "global_accuracy": float(accuracy),
                "selected_clients": int(len(self.last_selected)),
                "completed_clients": int(len(self.last_completed)),
                "malicious_selected": int(len(self.last_malicious)),
                "additive_sharing_simulation_enabled": int(use_sharing_simulation),
                "client_updates_delivered_in_clear_to_flower_server": 1,
                "committee_count": int(len(self.last_round_config.get("committees", {}))),
                "share_payload_bytes_estimate": int(self.last_share_bytes),
                "aggregation_seconds": float(self.last_aggregation_seconds),
                "client_failures_or_omissions": int(max(0, len(self.last_selected) - len(self.last_completed))),
            })
            return float(loss), {"accuracy": float(accuracy)}

        def summary(self):
            return None

    def client_fn(cid: str):
        return ResearchNumPyClient(int(cid)).to_client()

    start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=rounds),
        strategy=ResearchStrategy(),
        client_resources={
            "num_cpus": float(config.get("flower", {}).get("client_cpus", 1)),
            "num_gpus": float(config.get("flower", {}).get("client_gpus", 0)),
        },
        ray_init_args={"include_dashboard": False, "ignore_reinit_error": True},
    )
    evaluation, name = config.get("evaluation", {}), experiment.get("name", "fl_security_experiment")
    if evaluation.get("save_csv", True):
        logger.save_to_csv(f"{name}_results.csv")
    if evaluation.get("save_plots", True):
        logger.plot_metrics(f"{name}_accuracy.png")
    print(f"Flower experiment '{name}' completed.")
