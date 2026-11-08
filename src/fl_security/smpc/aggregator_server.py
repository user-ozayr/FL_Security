"""HTTP aggregation-party service with two-stage additive sharing.

Stage 1: each party sums client shares addressed to it.
Stage 2: each party re-shares that partial sum across all parties. The
coordinator receives only the second-stage shares of the global sum, not the
first-stage partial sums. This prevents the coordinator from solving the
committee-partial equations to isolate client updates.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Mapping

import numpy as np

from fl_security.smpc.additive_sharing import (
    DEFAULT_FIELD_MODULUS,
    generate_additive_shares,
    validate_modulus,
)


def _signature(secret: str, message: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def _encode_arrays(arrays: Mapping[str, np.ndarray]) -> tuple[bytes, str]:
    keys = list(arrays)
    if not keys:
        raise ValueError("cannot encode an empty tensor mapping")
    stream = io.BytesIO()
    np.savez(
        stream,
        **{
            f"arr_{index}": np.asarray(arrays[key], dtype=np.int64)
            for index, key in enumerate(keys)
        },
    )
    return stream.getvalue(), json.dumps(keys, separators=(",", ":"))


def _decode_arrays(body: bytes, keys_json: str) -> Dict[str, np.ndarray]:
    keys = json.loads(keys_json)
    if not isinstance(keys, list) or not keys or any(not isinstance(key, str) for key in keys):
        raise ValueError("invalid tensor-key header")
    with np.load(io.BytesIO(body), allow_pickle=False) as archive:
        if set(archive.files) != {f"arr_{index}" for index in range(len(keys))}:
            raise ValueError("archive tensor count does not match tensor-key header")
        arrays = {}
        for index, key in enumerate(keys):
            array = np.asarray(archive[f"arr_{index}"])
            if array.dtype != np.int64:
                raise ValueError("share tensors must use int64 residues")
            arrays[key] = array.copy()
    return arrays


def _post_binary(url: str, body: bytes, headers: dict[str, str], timeout: float = 60.0) -> None:
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/octet-stream", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status not in (200, 201):
                raise RuntimeError(f"peer returned HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"peer share transfer failed: {exc.code}: {detail}") from exc


class AggregatorState:
    """In-memory party state with client-share and party-reshare stages."""

    def __init__(
        self,
        client_keys: Mapping[str, str],
        control_token: str,
        modulus: int = DEFAULT_FIELD_MODULUS,
        party_id: int = 0,
        party_count: int = 1,
        peer_endpoints: Mapping[str, str] | None = None,
        peer_tokens: Mapping[str, str] | None = None,
        max_body_bytes: int = 512 * 1024 * 1024,
    ):
        if not client_keys or any(not value for value in client_keys.values()):
            raise ValueError("client_keys must contain per-client secrets")
        if not control_token:
            raise ValueError("control_token is required")
        if isinstance(party_id, bool) or not isinstance(party_id, int) or party_id < 0:
            raise ValueError("party_id must be a non-negative integer")
        if isinstance(party_count, bool) or not isinstance(party_count, int) or party_count < 2:
            raise ValueError("party_count must be an integer >= 2")
        if party_id >= party_count:
            raise ValueError("party_id must be less than party_count")

        self.client_keys = {str(cid): str(token) for cid, token in client_keys.items()}
        self.control_token = control_token
        self.modulus = validate_modulus(modulus)
        self.party_id = party_id
        self.party_count = party_count
        self.peer_endpoints = {str(pid): str(url).rstrip("/") for pid, url in (peer_endpoints or {}).items()}
        self.peer_tokens = {str(pid): str(token) for pid, token in (peer_tokens or {}).items()}
        expected_peers = {str(pid) for pid in range(party_count) if pid != party_id}
        if set(self.peer_endpoints) != expected_peers:
            raise ValueError("peer_endpoints must contain every other aggregation party")
        if set(self.peer_tokens) != expected_peers or any(not token for token in self.peer_tokens.values()):
            raise ValueError("peer_tokens must contain a secret for every other aggregation party")

        self.max_body_bytes = int(max_body_bytes)
        self.rounds: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        self.digests: dict[str, dict[str, str]] = {}
        # Outgoing shares are cached for idempotent retries after partial network failures.
        self.outgoing: dict[str, dict[int, tuple[bytes, str, str]]] = {}
        # One re-shared Stage-1 partial from each source party.
        self.reshared: dict[str, dict[int, dict[str, np.ndarray]]] = {}
        self.reshare_digests: dict[str, dict[int, str]] = {}
        self.finalized: dict[str, tuple[tuple[str, ...], tuple[int, ...]]] = {}
        self.output_cache: dict[str, tuple[bytes, str]] = {}
        self.lock = threading.RLock()

    def _validate_round_id(self, round_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", round_id):
            raise ValueError("invalid round ID")

    def submit_share(
        self, round_id: str, client_id: str, body: bytes, keys_json: str, signature: str
    ) -> bool:
        self._validate_round_id(round_id)
        if client_id not in self.client_keys:
            raise PermissionError("unknown client")
        message = (
            round_id.encode() + b"\n" + client_id.encode() + b"\n"
            + keys_json.encode() + b"\n" + body
        )
        if not hmac.compare_digest(signature, _signature(self.client_keys[client_id], message)):
            raise PermissionError("invalid client signature")
        arrays = _decode_arrays(body, keys_json)
        for array in arrays.values():
            if np.any(array < 0) or np.any(array >= self.modulus):
                raise ValueError("share element outside field")
        digest = hashlib.sha256(body + keys_json.encode()).hexdigest()
        with self.lock:
            if round_id in self.finalized:
                raise RuntimeError("round already finalized")
            shares = self.rounds.setdefault(round_id, {})
            digests = self.digests.setdefault(round_id, {})
            if client_id in shares:
                if digests[client_id] == digest:
                    return False
                raise RuntimeError("client submitted a different share twice")
            shares[client_id], digests[client_id] = arrays, digest
        return True

    def accept_reshare(
        self,
        round_id: str,
        source_party: int,
        body: bytes,
        keys_json: str,
        signature: str,
    ) -> bool:
        self._validate_round_id(round_id)
        if source_party == self.party_id or source_party < 0 or source_party >= self.party_count:
            raise ValueError("invalid source party ID")
        token = self.peer_tokens.get(str(source_party))
        if not token:
            raise PermissionError("unknown source party")
        message = (
            b"reshare\n" + round_id.encode() + b"\n" + str(source_party).encode()
            + b"\n" + str(self.party_id).encode() + b"\n" + keys_json.encode() + b"\n" + body
        )
        if not hmac.compare_digest(signature, _signature(token, message)):
            raise PermissionError("invalid party-to-party signature")
        arrays = _decode_arrays(body, keys_json)
        for array in arrays.values():
            if np.any(array < 0) or np.any(array >= self.modulus):
                raise ValueError("re-share element outside field")
        digest = hashlib.sha256(body + keys_json.encode()).hexdigest()
        with self.lock:
            current = self.reshared.setdefault(round_id, {})
            digests = self.reshare_digests.setdefault(round_id, {})
            if source_party in current:
                if digests[source_party] == digest:
                    return False
                raise RuntimeError("source party submitted a conflicting re-share")
            current[source_party] = arrays
            digests[source_party] = digest
        return True

    def finalize_to_peers(
        self,
        round_id: str,
        expected_client_ids: list[str],
        signature: str,
    ) -> None:
        """Sum this party's client shares then re-share the sum to every party."""
        self._validate_round_id(round_id)
        canonical = json.dumps(sorted(set(str(cid) for cid in expected_client_ids)), separators=(",", ":"))
        message = b"finalize\n" + round_id.encode() + b"\n" + canonical.encode()
        if not hmac.compare_digest(signature, _signature(self.control_token, message)):
            raise PermissionError("invalid coordinator signature")
        clients = tuple(json.loads(canonical))
        if len(clients) < 1:
            raise ValueError("party must have at least one contributing client")

        with self.lock:
            if round_id in self.finalized:
                previous_clients, previous_sources = self.finalized[round_id]
                if previous_clients != clients:
                    raise RuntimeError("round already finalized for a different client set")
                if set(self.reshared.get(round_id, {})) == set(previous_sources):
                    return
            received = self.rounds.get(round_id, {})
            missing = [cid for cid in clients if cid not in received]
            if missing:
                raise RuntimeError(f"missing client shares for IDs: {missing}")
            tensor_keys = list(received[clients[0]])
            key_set = set(tensor_keys)
            shapes = {key: received[clients[0]][key].shape for key in tensor_keys}
            partial = {key: np.zeros(shapes[key], dtype=np.int64) for key in tensor_keys}
            for cid in clients:
                current = received[cid]
                if set(current) != key_set:
                    raise ValueError("client share tensor keys do not match")
                for key in tensor_keys:
                    if current[key].shape != shapes[key]:
                        raise ValueError(f"tensor shape mismatch for {key}")
                    partial[key] = np.remainder(
                        partial[key] + current[key], self.modulus
                    ).astype(np.int64)
            # Create a stable cache once; retries reuse identical outgoing shares.
            outgoing = self.outgoing.setdefault(round_id, {})
            if not outgoing:
                shares = {
                    peer_id: {}
                    for peer_id in range(self.party_count)
                }
                for key, value in partial.items():
                    pieces = generate_additive_shares(value, self.party_count, self.modulus)
                    for peer_id, piece in enumerate(pieces):
                        shares[peer_id][key] = piece
                for peer_id in range(self.party_count):
                    body, keys_json = _encode_arrays(shares[peer_id])
                    if peer_id == self.party_id:
                        token = "internal"
                    else:
                        token = self.peer_tokens[str(peer_id)]
                    outgoing[peer_id] = (body, keys_json, token)
            self.finalized[round_id] = (clients, tuple(range(self.party_count)))
            outgoing = dict(outgoing)

        # Deliver outside the lock. Target endpoints are idempotent by source ID.
        for peer_id, (body, keys_json, token) in outgoing.items():
            if peer_id == self.party_id:
                self._accept_own_reshare(round_id, body, keys_json)
                continue
            message = (
                b"reshare\n" + round_id.encode() + b"\n" + str(self.party_id).encode()
                + b"\n" + str(peer_id).encode() + b"\n" + keys_json.encode() + b"\n" + body
            )
            _post_binary(
                self.peer_endpoints[str(peer_id)] + f"/v1/rounds/{round_id}/reshare",
                body,
                {
                    "X-Source-Party": str(self.party_id),
                    "X-Tensor-Keys": keys_json,
                    "X-Signature": _signature(token, message),
                },
            )
        # Individual client shares are no longer needed once all re-share
        # deliveries have succeeded; retain only the outgoing retry cache.
        with self.lock:
            self.rounds.pop(round_id, None)
            self.digests.pop(round_id, None)

    def _accept_own_reshare(self, round_id: str, body: bytes, keys_json: str) -> None:
        arrays = _decode_arrays(body, keys_json)
        with self.lock:
            current = self.reshared.setdefault(round_id, {})
            digests = self.reshare_digests.setdefault(round_id, {})
            digest = hashlib.sha256(body + keys_json.encode()).hexdigest()
            if self.party_id in current and digests[self.party_id] != digest:
                raise RuntimeError("conflicting internal re-share")
            current[self.party_id] = arrays
            digests[self.party_id] = digest

    def get_final_share(
        self,
        round_id: str,
        expected_source_parties: list[int],
        signature: str,
    ) -> tuple[bytes, str]:
        """Return this party's fresh random share of the global aggregate."""
        canonical = json.dumps(sorted(set(int(pid) for pid in expected_source_parties)), separators=(",", ":"))
        message = b"result\n" + round_id.encode() + b"\n" + canonical.encode()
        if not hmac.compare_digest(signature, _signature(self.control_token, message)):
            raise PermissionError("invalid coordinator signature")
        sources = tuple(json.loads(canonical))
        if sources != tuple(range(self.party_count)):
            raise ValueError("result reconstruction must include every source aggregation party")
        with self.lock:
            if round_id in self.output_cache:
                return self.output_cache[round_id]
            received = self.reshared.get(round_id, {})
            missing = [source for source in sources if source not in received]
            if missing:
                raise RuntimeError(f"missing party re-shares from: {missing}")
            tensor_keys = list(received[sources[0]])
            key_set = set(tensor_keys)
            shapes = {key: received[sources[0]][key].shape for key in tensor_keys}
            total = {key: np.zeros(shapes[key], dtype=np.int64) for key in tensor_keys}
            for source in sources:
                current = received[source]
                if set(current) != key_set:
                    raise ValueError("party re-share tensor keys do not match")
                for key in tensor_keys:
                    if current[key].shape != shapes[key]:
                        raise ValueError(f"re-share tensor shape mismatch for {key}")
                    total[key] = np.remainder(
                        total[key] + current[key], self.modulus
                    ).astype(np.int64)
            body, keys_json = _encode_arrays(total)
            self.output_cache[round_id] = (body, keys_json)
            return body, keys_json

    def acknowledge(self, round_id: str, signature: str) -> None:
        message = b"ack\n" + round_id.encode()
        if not hmac.compare_digest(signature, _signature(self.control_token, message)):
            raise PermissionError("invalid coordinator signature")
        with self.lock:
            self.rounds.pop(round_id, None)
            self.digests.pop(round_id, None)
            self.outgoing.pop(round_id, None)
            self.reshared.pop(round_id, None)
            self.reshare_digests.pop(round_id, None)
            self.finalized.pop(round_id, None)
            self.output_cache.pop(round_id, None)


def make_handler(state: AggregatorState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FLSMPCParty/2.0"

        def log_message(self, fmt, *args):
            return

        def _send(self, status: int, body: bytes = b"", content_type: str = "application/json", headers=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, b'{"status":"ok"}')
            else:
                self._send(404, b'{"error":"not found"}')

        def do_POST(self):
            parts = self.path.strip("/").split("/")
            if len(parts) != 4 or parts[:2] != ["v1", "rounds"]:
                self._send(404, b'{"error":"not found"}')
                return
            round_id, action = parts[2], parts[3]
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > state.max_body_bytes:
                    self._send(413, b'{"error":"invalid body size"}')
                    return
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("truncated request body")
                signature = self.headers.get("X-Signature", "")

                if action == "shares":
                    client_id = self.headers.get("X-Client-ID", "")
                    keys_json = self.headers.get("X-Tensor-Keys", "")
                    created = state.submit_share(round_id, client_id, body, keys_json, signature)
                    response = b'{"status":"accepted"}' if created else b'{"status":"already-accepted"}'
                    self._send(201 if created else 200, response)
                    return

                if action == "reshare":
                    source = int(self.headers.get("X-Source-Party", "-1"))
                    keys_json = self.headers.get("X-Tensor-Keys", "")
                    created = state.accept_reshare(round_id, source, body, keys_json, signature)
                    response = b'{"status":"accepted"}' if created else b'{"status":"already-accepted"}'
                    self._send(201 if created else 200, response)
                    return

                if action == "finalize":
                    request = json.loads(body)
                    clients = request.get("expected_client_ids")
                    if not isinstance(clients, list) or any(not isinstance(cid, str) for cid in clients):
                        raise ValueError("expected_client_ids must be a list of strings")
                    state.finalize_to_peers(round_id, clients, signature)
                    self._send(200, b'{"status":"reshared"}')
                    return

                if action == "result":
                    request = json.loads(body)
                    sources = request.get("expected_source_parties")
                    if not isinstance(sources, list) or any(
                        isinstance(pid, bool) or not isinstance(pid, int) for pid in sources
                    ):
                        raise ValueError("expected_source_parties must be a list of integers")
                    result, keys_json = state.get_final_share(round_id, sources, signature)
                    self._send(200, result, "application/octet-stream", {"X-Tensor-Keys": keys_json})
                    return

                if action == "ack":
                    state.acknowledge(round_id, signature)
                    self._send(200, b'{"status":"cleared"}')
                    return

                self._send(404, b'{"error":"not found"}')
            except PermissionError as exc:
                self._send(403, json.dumps({"error": str(exc)}).encode())
            except RuntimeError as exc:
                self._send(409, json.dumps({"error": str(exc)}).encode())
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                self._send(400, json.dumps({"error": str(exc)}).encode())
            except Exception:
                self._send(500, b'{"error":"internal server error"}')

    return Handler


def create_server(host: str, port: int, state: AggregatorState) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(state))
    server.daemon_threads = True
    return server


def serve_aggregator(
    host: str,
    port: int,
    client_keys: dict,
    control_token: str,
    modulus: int,
    party_id: int,
    party_count: int,
    peer_endpoints: dict,
    peer_tokens: dict,
):
    server = create_server(
        host,
        port,
        AggregatorState(
            client_keys, control_token, modulus,
            party_id=party_id, party_count=party_count,
            peer_endpoints=peer_endpoints, peer_tokens=peer_tokens,
        ),
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Run one additive-SMPC aggregation party")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--party-id", type=int, required=True)
    parser.add_argument("--party-count", type=int, required=True)
    parser.add_argument("--field-modulus", type=int, default=DEFAULT_FIELD_MODULUS)
    args = parser.parse_args()
    client_keys = json.loads(os.environ.get("FL_SMPC_CLIENT_KEYS_JSON", "{}"))
    control_token = os.environ.get("FL_SMPC_CONTROL_TOKEN", "")
    peer_endpoints = json.loads(os.environ.get("FL_SMPC_PEER_ENDPOINTS_JSON", "{}"))
    peer_tokens = json.loads(os.environ.get("FL_SMPC_PEER_TOKENS_JSON", "{}"))
    server = create_server(
        args.host,
        args.port,
        AggregatorState(
            client_keys, control_token, args.field_modulus,
            party_id=args.party_id,
            party_count=args.party_count,
            peer_endpoints=peer_endpoints,
            peer_tokens=peer_tokens,
        ),
    )
    print(f"Aggregator party {args.party_id} listening on {args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
