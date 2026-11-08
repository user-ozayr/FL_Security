"""Start independent local aggregation-party processes for an end-to-end demo.

These processes are isolated from one another, but share one host/kernel with
the Flower simulation. For cross-organization assurance, deploy each party
on a separately administered host and protect HTTP with TLS/mTLS.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import secrets
import socket
import time
import urllib.error
import urllib.request

from fl_security.federated.flower_simulation import run_flower_simulation
from fl_security.smpc.additive_sharing import DEFAULT_FIELD_MODULUS
from fl_security.smpc.aggregator_server import serve_aggregator


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_healthy(url: str, process: mp.Process, timeout: float = 15.0) -> None:
    deadline, last_error = time.monotonic() + timeout, None
    while time.monotonic() < deadline:
        if not process.is_alive():
            raise RuntimeError("aggregation-party process exited during startup")
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=1.0) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            time.sleep(0.1)
    raise TimeoutError(f"aggregation party did not become healthy: {last_error}")


def run_with_local_aggregators(config: dict) -> None:
    """Run Flower with one local process per configured SMPC aggregation party."""
    smpc = config.get("smpc", {})
    if not smpc.get("enabled", False):
        raise ValueError("local aggregator mode requires smpc.enabled=true")
    party_count = int(smpc["num_aggregation_parties"])
    client_count = int(config["federated_learning"]["num_clients"])
    modulus = int(smpc.get("field_modulus", DEFAULT_FIELD_MODULUS))
    context = mp.get_context("spawn")
    processes = []
    server_config, client_config = {}, {str(cid): {} for cid in range(client_count)}

    protected_names = (
        "FL_SMPC_AGGREGATORS_JSON",
        "FL_SMPC_CLIENT_TOKENS_JSON",
        "FL_SMPC_CONTROL_TOKEN",
        "FL_SMPC_CLIENT_KEYS_JSON",
        "FL_SMPC_PEER_ENDPOINTS_JSON",
        "FL_SMPC_PEER_TOKENS_JSON",
        "FL_MTD_SECRET",
    )
    previous_env = {
        name: os.environ.pop(name)
        for name in protected_names
        if name in os.environ
    }

    try:
        ports = {party_id: _free_port() for party_id in range(party_count)}
        urls = {
            str(party_id): f"http://127.0.0.1:{ports[party_id]}"
            for party_id in range(party_count)
        }
        # One symmetric high-entropy secret per pair, used for both directions.
        pair_tokens = {
            (left, right): secrets.token_urlsafe(32)
            for left in range(party_count)
            for right in range(left + 1, party_count)
        }
        peer_tokens = {
            party_id: {
                str(peer): pair_tokens[(min(party_id, peer), max(party_id, peer))]
                for peer in range(party_count)
                if peer != party_id
            }
            for party_id in range(party_count)
        }

        for party_id in range(party_count):
            client_keys = {
                str(cid): secrets.token_urlsafe(32)
                for cid in range(client_count)
            }
            control_token = secrets.token_urlsafe(32)
            process = context.Process(
                target=serve_aggregator,
                args=(
                    "127.0.0.1",
                    ports[party_id],
                    client_keys,
                    control_token,
                    modulus,
                    party_id,
                    party_count,
                    {str(peer): url for peer, url in urls.items() if int(peer) != party_id},
                    peer_tokens[party_id],
                ),
                name=f"fl-smpc-aggregator-{party_id}",
            )
            process.start()
            processes.append(process)
            server_config[str(party_id)] = {
                "url": urls[str(party_id)],
                "control_token": control_token,
            }
            for client_id, token in client_keys.items():
                client_config[client_id][str(party_id)] = token

        for party_id, party in server_config.items():
            _wait_healthy(party["url"], processes[int(party_id)])

        os.environ["FL_SMPC_AGGREGATORS_JSON"] = json.dumps(
            server_config, separators=(",", ":")
        )
        os.environ["FL_SMPC_CLIENT_TOKENS_JSON"] = json.dumps(
            client_config, separators=(",", ":")
        )
        if "FL_MTD_SECRET" in previous_env:
            os.environ["FL_MTD_SECRET"] = previous_env["FL_MTD_SECRET"]

        run_flower_simulation(config)
    finally:
        for name in protected_names:
            os.environ.pop(name, None)
        os.environ.update(previous_env)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
