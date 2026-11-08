"""Flower simulation with client-side additive sharing to remote aggregation parties."""
from __future__ import annotations
import json
import os
import re
import time
import uuid
from numbers import Integral
import numpy as np
import torch
from torch.utils.data import DataLoader
from fl_security.attacks.byzantine import ByzantineAttacker
from fl_security.evaluation.metrics import ExperimentLogger
from fl_security.federated.aggregation import reconstruct_weighted_average, weighted_fedavg
from fl_security.federated.robust_aggregation import RobustUpdateDetector
from fl_security.federated.dataset import load_cifar10_data, partition_data
from fl_security.federated.model import CIFAR10CNN
from fl_security.federated.training import evaluate_model, train_local
from fl_security.mtd.controller import MTDController
from fl_security.smpc.additive_sharing import DEFAULT_FIELD_MODULUS
from fl_security.smpc.client_sharing import acknowledge_party, finalize_party, retrieve_party_share, upload_client_update


def _load_party_configuration() -> dict[str, dict]:
    raw = os.environ.get("FL_SMPC_AGGREGATORS_JSON", "")
    if not raw:
        return {}
    config = json.loads(raw)
    if not isinstance(config, dict):
        raise ValueError("FL_SMPC_AGGREGATORS_JSON must be a JSON object keyed by party ID")
    normalized = {}
    for party_id, party in config.items():
        if not isinstance(party, dict):
            raise ValueError(f"aggregator party {party_id} must be a JSON object")
        url, token = str(party.get("url", "")).rstrip("/"), str(party.get("control_token", ""))
        if not url or not token:
            raise ValueError(f"aggregator party {party_id} requires url and control_token")
        normalized[str(int(party_id))] = {"url": url, "control_token": token}
    return normalized


def _load_client_tokens(client_id: int) -> dict[str, str]:
    """Load credentials from the client runtime, never from Flower round messages."""
    raw = os.environ.get("FL_SMPC_CLIENT_TOKENS_JSON", "")
    if not raw:
        raise RuntimeError("FL_SMPC_CLIENT_TOKENS_JSON is required in each client runtime")
    config = json.loads(raw)
    if not isinstance(config, dict):
        raise ValueError("FL_SMPC_CLIENT_TOKENS_JSON must be a JSON object")
    nested = config.get(str(client_id))
    if isinstance(nested, dict):
        return {str(pid): str(token) for pid, token in nested.items()}
    if all(isinstance(value, str) for value in config.values()):
        return {str(pid): str(token) for pid, token in config.items()}
    raise ValueError(f"no client-local credentials found for client {client_id}")



def _resolve_authenticated_client_ids(results, proxy_id_by_client, selected_client_ids):
    """Bind every FitRes identity to its registered Flower proxy.

    The client_id metric is client supplied and is never trusted on its own.
    Flower's registered proxy ID is the simulation/runtime identity anchor.
    """
    if not isinstance(proxy_id_by_client, dict) or not proxy_id_by_client:
        raise RuntimeError("Flower proxy identity map has not been initialized")

    proxy_to_logical = {}
    for logical_id, proxy_id in proxy_id_by_client.items():
        logical = int(logical_id)
        proxy = str(proxy_id)
        if proxy in proxy_to_logical:
            raise RuntimeError("Flower proxy identity map contains duplicate proxies")
        proxy_to_logical[proxy] = logical

    selected = {int(client_id) for client_id in selected_client_ids}
    resolved = []
    for client_proxy, fit_res in results:
        proxy_id = getattr(client_proxy, "cid", None)
        logical_id = proxy_to_logical.get(str(proxy_id))
        if proxy_id is None or logical_id is None:
            raise RuntimeError("Flower returned a result from an unregistered proxy")
        if logical_id not in selected:
            raise RuntimeError("Flower returned a result from a client not selected this round")

        metrics = getattr(fit_res, "metrics", {})
        reported_id = metrics.get("client_id") if isinstance(metrics, dict) else None
        if (
            isinstance(reported_id, bool)
            or not isinstance(reported_id, Integral)
            or int(reported_id) != logical_id
        ):
            raise RuntimeError(
                "Client-reported identity does not match its registered Flower proxy"
            )
        resolved.append(logical_id)

    if len(set(resolved)) != len(resolved):
        raise RuntimeError("Flower returned duplicate client results for a round")
    return resolved

def run_flower_simulation(config: dict) -> None:
    """Run Flower client training; SMPC clients upload shares directly to party services."""
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
    client_datasets = partition_data(
        trainset, num_clients, fl_config.get("partitioning", "iid"), seed
    )
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attacker, mtd = ByzantineAttacker(config, num_clients), MTDController(config)

    smpc_config = config.get("smpc", {})
    use_smpc = bool(smpc_config.get("enabled", False))
    modulus = int(smpc_config.get("field_modulus", DEFAULT_FIELD_MODULUS))
    min_clients = int(smpc_config.get("min_clients_per_round", 2))
    encoder, party_config = None, {}
    robust_config = config.get("defences", {}).get("robust_aggregation", {})
    robust_enabled = bool(robust_config.get("enabled", False))
    if use_smpc and robust_enabled:
        raise ValueError(
            "Individual-update robust screening is not available with the current "
            "sum-only SMPC backend. Do not enable it until robust scoring and "
            "selection are implemented inside MPC."
        )
    if robust_enabled and num_clients < int(robust_config.get("min_survivors", 2)):
        raise ValueError("robust aggregation requires at least min_survivors clients")

    if use_smpc:
        from fl_security.smpc.fixed_point import FixedPointEncoder
        encoder = FixedPointEncoder(int(smpc_config.get("fixed_point_scale", 1_000_000)))
        party_config = _load_party_configuration()
        required = set(range(int(smpc_config["num_aggregation_parties"])))
        if set(map(int, party_config)) != required:
            raise RuntimeError(
                "SMPC is enabled but FL_SMPC_AGGREGATORS_JSON does not configure every party. "
                "Use --local-aggregators or configure external aggregation services."
            )

    global_model = CIFAR10CNN()
    initial_state = global_model.state_dict()
    state_keys = list(initial_state)
    initial_arrays = [tensor.detach().cpu().numpy() for tensor in initial_state.values()]
    initial_parameters = ndarrays_to_parameters(initial_arrays)
    logger = ExperimentLogger(output_dir=config.get("evaluation", {}).get("output_dir", "./results"))
    slug = re.sub(r"[^a-zA-Z0-9_-]", "-", str(experiment.get("name", "fl")))
    experiment_prefix = f"{slug}-{seed}"

    class ResearchNumPyClient(NumPyClient):
        """Client whose experiment identity is supplied by the server."""

        def __init__(self, flower_cid: str):
            self.flower_cid = str(flower_cid)

        def get_parameters(self, client_config):
            return [array.copy() for array in initial_arrays]

        def fit(self, parameters, fit_config):
            client_id = int(fit_config["client_id"])
            dataset = client_datasets[client_id]
            round_num = int(fit_config["server_round"])

            state = {
                key: torch.from_numpy(array.copy())
                for key, array in zip(state_keys, parameters)
            }
            model = CIFAR10CNN()
            model.load_state_dict(state)
            torch.manual_seed(seed + round_num * 100_000 + client_id)

            loader = DataLoader(
                dataset,
                batch_size=int(fl_config.get("batch_size", 32)),
                shuffle=True,
            )

            label_transform = None
            if attacker.is_malicious(client_id, round_num) and attacker.attack_type == "label_flip":
                def label_transform(labels):
                    values = attacker.perturb_labels(
                        client_id, round_num, labels, num_classes=10
                    )
                    return torch.as_tensor(
                        values, dtype=torch.long, device=labels.device
                    )

            local_state = train_local(
                model,
                loader,
                epochs=int(fl_config.get("local_epochs", 1)),
                lr=float(fl_config.get("learning_rate", 0.01)),
                device=device,
                label_transform=label_transform,
            )

            bad_training = [
                key for key, value in local_state.items()
                if torch.is_floating_point(value)
                and not torch.isfinite(value).all()
            ]

            if bad_training:
                raise RuntimeError(
                    f"Client {client_id}: non-finite values produced "
                    f"during local training in tensors: {bad_training}")
            local_state = attacker.perturb_update(
                client_id, round_num, state, local_state
            )

            bad_attack = [
                key for key, value in local_state.items()
                if torch.is_floating_point(value)
                and not torch.isfinite(value).all()
            ]

            if bad_attack:
                raise RuntimeError(
                    f"Client {client_id}: non-finite values produced "
                    f"after applying {attacker.attack_type}: {bad_attack}"
                )

            if bool(fit_config.get("smpc_enabled", False)):
                committee = json.loads(fit_config["committee_ids_json"])
                endpoints = json.loads(fit_config["aggregator_endpoints_json"])
                tokens = _load_client_tokens(client_id)

                payload_bytes = upload_client_update(
                    local_state,
                    num_examples=len(dataset),
                    client_id=client_id,
                    round_id=str(fit_config["round_id"]),
                    committee_ids=committee,
                    aggregator_endpoints=endpoints,
                    client_tokens=tokens,
                    encoder=encoder,
                    modulus=int(fit_config["field_modulus"]),
                )

                return [], len(dataset), {
                    "client_id": client_id,
                    "shares_uploaded": 1,
                    "share_payload_bytes": int(payload_bytes),
                }

            return (
                [
                    local_state[key].detach().cpu().numpy()
                    for key in state_keys
                ],
                len(dataset),
                {"client_id": client_id},
            )

        def evaluate(self, parameters, client_config):
            return 0.0, 0, {"flower_client_id": self.flower_cid}

    class ResearchStrategy(Strategy):
        """Flower strategy using MTD and committee-routed secret sharing."""
        def __init__(self):
            self.last_round_config = {}
            self.proxy_id_by_client = None
            self.last_selected, self.last_completed, self.last_malicious = [], [], []
            self.last_aggregation_seconds, self.last_share_bytes = 0.0, 0
            self.last_cleanup_failures, self.last_round_id = 0, ""
            self.last_robust_report = {}
            self.robust_detector = (
                RobustUpdateDetector(
                    max_malicious=int(robust_config.get("max_malicious", 1)),
                    residual_z_threshold=float(robust_config.get("residual_z_threshold", 3.5)),
                    min_cosine_similarity=float(robust_config.get("min_cosine_similarity", 0.0)),
                    max_norm_ratio=float(robust_config.get("max_norm_ratio", 5.0)),
                    min_survivors=int(robust_config.get("min_survivors", 2)),
                    history_decay=float(robust_config.get("history_decay", 0.8)),
                    min_history_rounds=int(robust_config.get("min_history_rounds", 2)),
                    history_cosine_threshold=float(
                        robust_config.get("history_cosine_threshold", -0.25)
                    ),
                )
                if robust_enabled else None
            )

        def _round_parties(self):
            if not use_smpc:
                return []
            committees = self.last_round_config.get("committees", {})
            return sorted({
                int(pid) for cid in self.last_selected for pid in committees.get(cid, [])
            })

        def _clear_round_parties(self):
            self.last_cleanup_failures = 0
            for party_id in self._round_parties():
                try:
                    acknowledge_party(
                        party_config[str(party_id)]["url"],
                        self.last_round_id,
                        party_config[str(party_id)]["control_token"],
                    )
                except Exception as exc:
                    self.last_cleanup_failures += 1
                    print(f"Warning: failed to clear aggregator {party_id} round state: {exc}")

        def initialize_parameters(self, client_manager):
            return initial_parameters

        def configure_fit(self, server_round, parameters, client_manager):
            available = client_manager.all()

            # Flower generates opaque proxy IDs. Assign stable logical IDs
            # 0..num_clients-1 to the sorted proxy IDs once per simulation.
            if self.proxy_id_by_client is None:
                proxy_ids = sorted(available.keys())
                if len(proxy_ids) != num_clients:
                    raise RuntimeError(
                        f"Expected {num_clients} registered Flower clients, "
                        f"but found {len(proxy_ids)}: {proxy_ids}"
                    )
                self.proxy_id_by_client = {
                    logical_id: proxy_id
                    for logical_id, proxy_id in enumerate(proxy_ids)
                }
                print(
                    "[DEBUG] Flower proxy mapping (logical ID -> Flower ID): "
                    f"{self.proxy_id_by_client}",
                    flush=True,
                )

            self.last_round_config = mtd.generate_round_config(
                server_round,
                num_clients,
                float(fl_config.get("fraction_train", 1.0)),
            )
            self.last_selected = list(
                self.last_round_config["selected_clients"]
            )
            self.last_round_id = (
                f"{experiment_prefix}-r{server_round}-{uuid.uuid4().hex}"
            )

            print(
                f"[DEBUG] MTD selected logical clients: {self.last_selected}",
                flush=True,
            )

            instructions = []
            for cid in self.last_selected:
                proxy_id = self.proxy_id_by_client.get(int(cid))
                proxy = available.get(proxy_id) if proxy_id is not None else None
                if proxy is None:
                    print(
                        f"[WARNING] Logical client {cid} is not currently available",
                        flush=True,
                    )
                    continue

                client_config = {
                    "server_round": int(server_round),
                    "client_id": int(cid),
                    "smpc_enabled": bool(use_smpc),
                }

                if use_smpc:
                    committee = self.last_round_config["committees"][cid]
                    client_config.update({
                        "round_id": self.last_round_id,
                        "committee_ids_json": json.dumps(
                            committee, separators=(",", ":")
                        ),
                        "aggregator_endpoints_json": json.dumps({
                            str(pid): party_config[str(pid)]["url"]
                            for pid in committee
                        }, separators=(",", ":")),
                        "field_modulus": int(modulus),
                    })

                instructions.append((proxy, FitIns(parameters, client_config)))

            return instructions

        def aggregate_fit(self, server_round, results, failures):
            if not results:
                self.last_completed = []
                self._clear_round_parties()
                return None, {}
            committees = self.last_round_config["committees"]
            client_ids = _resolve_authenticated_client_ids(
                results, self.proxy_id_by_client, self.last_selected
            )
            reported_counts = [int(fit_res.num_examples) for _, fit_res in results]
            counts = [len(client_datasets[cid]) for cid in client_ids]
            count_mismatches = [
                (cid, reported, expected)
                for cid, reported, expected in zip(client_ids, reported_counts, counts)
                if reported != expected
            ]
            self.last_completed = client_ids
            self.last_malicious = [
                cid for cid in client_ids if attacker.is_malicious(cid, server_round)
            ]
            self.last_share_bytes = sum(
                int(fit_res.metrics.get("share_payload_bytes", 0)) for _, fit_res in results
            )
            started = time.perf_counter()

            if use_smpc:
                try:
                    if count_mismatches:
                        raise RuntimeError(
                            "Client-reported example counts do not match server-side partitions: "
                            f"{count_mismatches}"
                        )
                    if len(client_ids) < min_clients:
                        raise RuntimeError(
                            f"Only {len(client_ids)} clients completed; SMPC requires {min_clients}. "
                            "Aborting this round to avoid a low-cardinality aggregate."
                        )
                    for _, fit_res in results:
                        if fit_res.metrics.get("shares_uploaded") != 1:
                            raise RuntimeError("Flower client did not confirm share uploads")
                        if parameters_to_ndarrays(fit_res.parameters):
                            raise RuntimeError("SMPC mode received plaintext parameters; refusing fallback")

                    parties_used = list(range(int(smpc_config["num_aggregation_parties"])))
                    participants_by_party = {
                        party_id: [cid for cid in client_ids if party_id in committees[cid]]
                        for party_id in parties_used
                    }
                    weak_parties = [
                        party_id for party_id, members in participants_by_party.items()
                        if len(members) < 2
                    ]
                    if weak_parties:
                        raise RuntimeError(
                            "SMPC round aborted: parties received fewer than two client shares: "
                            f"{weak_parties}. Retry with more successful clients."
                        )

                    # Stage 1: parties sum their client shares and re-share their
                    # partial sums directly to all parties. No first-stage partial
                    # sum is ever returned to the coordinator.
                    for party_id in parties_used:
                        finalize_party(
                            party_config[str(party_id)]["url"],
                            self.last_round_id,
                            participants_by_party[party_id],
                            party_config[str(party_id)]["control_token"],
                        )

                    # Stage 2: the coordinator sees only independent random shares
                    # of the global sum, not the committee-partial equations.
                    party_partials = [
                        retrieve_party_share(
                            party_config[str(party_id)]["url"],
                            self.last_round_id,
                            parties_used,
                            party_config[str(party_id)]["control_token"],
                        )
                        for party_id in parties_used
                    ]
                    aggregated = reconstruct_weighted_average(
                        party_partials, global_model.state_dict(), sum(counts), encoder, modulus
                    )
                    global_model.load_state_dict(aggregated)
                finally:
                    # Also removes partial uploads from failed clients and cached party sums.
                    self._clear_round_parties()
            else:
                if count_mismatches:
                    raise RuntimeError(
                        "Client-reported example counts do not match server-side partitions: "
                        f"{count_mismatches}"
                    )
                client_states = []
                for _, fit_res in results:
                    arrays = parameters_to_ndarrays(fit_res.parameters)
                    client_states.append({
                        key: torch.from_numpy(array.copy())
                        for key, array in zip(state_keys, arrays)
                    })
                if robust_enabled:
                    aggregated, self.last_robust_report = self.robust_detector.aggregate(
                        client_states,
                        counts,
                        global_model.state_dict(),
                        client_ids=client_ids,
                    )
                else:
                    self.last_robust_report = {}
                    aggregated = weighted_fedavg(
                        client_states, counts, global_model.state_dict()
                    )
                global_model.load_state_dict(aggregated)

            self.last_aggregation_seconds = time.perf_counter() - started
            return (
                ndarrays_to_parameters([
                    tensor.detach().cpu().numpy() for tensor in global_model.state_dict().values()
                ]),
                {
                    "selected_clients": int(len(client_ids)),
                    "malicious_clients": int(len(self.last_malicious)),
                    "smpc_enabled": int(use_smpc),
                    "share_payload_bytes": int(self.last_share_bytes),
                    "aggregation_seconds": float(self.last_aggregation_seconds),
                    "aggregators_used": (
                        int(len({
                            int(pid) for cid in client_ids
                            for pid in committees[cid]
                        }))
                        if use_smpc else 0
                    ),
                    "aggregator_cleanup_failures": int(self.last_cleanup_failures),
                },
            )

        def configure_evaluate(self, server_round, parameters, client_manager):
            return []

        def aggregate_evaluate(self, server_round, results, failures):
            return None, {}

        def evaluate(self, server_round, parameters):
            arrays = parameters_to_ndarrays(parameters)
            model = CIFAR10CNN()
            model.load_state_dict({
                key: torch.from_numpy(array.copy()) for key, array in zip(state_keys, arrays)
            })
            loss, accuracy = evaluate_model(model, testloader, device=device)
            logger.log_round({
                "round": int(server_round),
                "global_loss": float(loss),
                "global_accuracy": float(accuracy),
                "selected_clients": int(len(self.last_selected)),
                "completed_clients": int(len(self.last_completed)),
                "malicious_selected": int(len(self.last_malicious)),
                "smpc_enabled": int(use_smpc),
                "committees_used": (
                    int(len(self.last_round_config.get("committees", {})))
                    if use_smpc else 0
                ),
                "mtd_secret_seeded": int(self.last_round_config.get("secret_seeded", False)),
                "aggregators_used": (
                    int(len({
                        int(pid) for cid in self.last_completed
                        for pid in self.last_round_config.get("committees", {}).get(cid, [])
                    }))
                    if use_smpc else 0
                ),
                "share_payload_bytes": int(self.last_share_bytes),
                "aggregation_seconds": float(self.last_aggregation_seconds),
                "aggregator_cleanup_failures": int(self.last_cleanup_failures),
                "client_failures_or_omissions": int(
                    max(0, len(self.last_selected) - len(self.last_completed))
                ),
                "robust_aggregation_enabled": int(robust_enabled),
                "robust_aggregation_rule": self.last_robust_report.get(
                    "aggregation_rule", "fedavg"
                ),
                "detected_client_count": int(
                    self.last_robust_report.get("detected_client_count", 0)
                ),
                "detected_client_ids": json.dumps(
                    self.last_robust_report.get("detected_client_ids", [])
                ),
                "accepted_client_ids": json.dumps(
                    self.last_robust_report.get("accepted_client_ids", self.last_completed)
                ),
                "robust_fallback_used": int(
                    self.last_robust_report.get("fallback_used", False)
                ),
                "flagged_client_count": int(
                    self.last_robust_report.get("flagged_client_count", 0)
                ),
                "history_updated_client_ids": json.dumps(
                    self.last_robust_report.get("history_updated_client_ids", [])
                ),
                "detection_true_positives": int(
                    len(set(self.last_robust_report.get("detected_client_ids", []))
                        & set(self.last_malicious))
                ),
                "detection_false_positives": int(
                    len(set(self.last_robust_report.get("detected_client_ids", []))
                        - set(self.last_malicious))
                ),
                "detection_false_negatives": int(
                    len(set(self.last_malicious)
                        - set(self.last_robust_report.get("detected_client_ids", [])))
                ),
                "detection_precision": (
                    len(set(self.last_robust_report.get("detected_client_ids", []))
                        & set(self.last_malicious))
                    / max(1, len(self.last_robust_report.get("detected_client_ids", [])))
                ) if robust_enabled else None,
                "detection_recall": (
                    len(set(self.last_robust_report.get("detected_client_ids", []))
                        & set(self.last_malicious))
                    / max(1, len(self.last_malicious))
                ) if robust_enabled else None,
                "robust_client_scores_json": json.dumps(
                    self.last_robust_report.get("client_scores", []),
                    separators=(",", ":"),
                ),
            })
            return float(loss), {"accuracy": float(accuracy)}

        def summary(self):
            return None

    def client_fn(cid: str):
        return ResearchNumPyClient(cid).to_client()

    # These are coordinator-only secrets. The strategy has already loaded
    # the values it needs, so remove them before Ray starts worker processes.
    # Client auth credentials are handled separately (and locally remain
    # shared by the synthetic simulation runtime; see the README warning).
    server_only_names = (
        "FL_SMPC_AGGREGATORS_JSON",
        "FL_MTD_SECRET",
        "FL_SMPC_CONTROL_TOKEN",
        "FL_SMPC_CLIENT_KEYS_JSON",
    )
    server_only_env = {
        name: os.environ.pop(name)
        for name in server_only_names
        if name in os.environ
    }
    try:
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
    finally:
        os.environ.update(server_only_env)
    evaluation, name = config.get("evaluation", {}), experiment.get("name", "fl_security_experiment")
    if evaluation.get("save_csv", True):
        logger.save_to_csv(f"{slug}_results.csv")
    if evaluation.get("save_plots", True):
        logger.plot_metrics(f"{slug}_accuracy.png")
    print(f"Flower experiment '{name}' completed.")
