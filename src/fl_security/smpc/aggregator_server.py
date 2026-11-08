"""HTTP aggregation-party service for client-side additive secret sharing."""
from __future__ import annotations
import argparse
import hashlib
import hmac
import io
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Mapping
import numpy as np
from fl_security.smpc.additive_sharing import DEFAULT_FIELD_MODULUS, validate_modulus

def _signature(secret: str, message: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()

def _encode_arrays(arrays: Mapping[str, np.ndarray]) -> tuple[bytes, str]:
    keys = list(arrays)
    stream = io.BytesIO()
    np.savez(stream, **{f"arr_{i}": np.asarray(arrays[key], dtype=np.int64) for i, key in enumerate(keys)})
    return stream.getvalue(), json.dumps(keys, separators=(",", ":"))

def _decode_arrays(body: bytes, keys_json: str) -> Dict[str, np.ndarray]:
    keys = json.loads(keys_json)
    if not isinstance(keys, list) or not keys or any(not isinstance(key, str) for key in keys):
        raise ValueError("invalid tensor-key header")
    with np.load(io.BytesIO(body), allow_pickle=False) as archive:
        if set(archive.files) != {f"arr_{i}" for i in range(len(keys))}:
            raise ValueError("archive tensor count does not match tensor-key header")
        arrays = {}
        for i, key in enumerate(keys):
            array = np.asarray(archive[f"arr_{i}"])
            if array.dtype != np.int64:
                raise ValueError("share tensors must use int64 residues")
            arrays[key] = array.copy()
    return arrays

class AggregatorState:
    """In-memory party state. Use persistent storage for long-running deployments."""
    def __init__(self, client_keys: Mapping[str, str], control_token: str,
                 modulus: int = DEFAULT_FIELD_MODULUS,
                 max_body_bytes: int = 512 * 1024 * 1024):
        if not client_keys or any(not value for value in client_keys.values()):
            raise ValueError("client_keys must contain per-client secrets")
        if not control_token:
            raise ValueError("control_token is required")
        self.client_keys = {str(cid): str(token) for cid, token in client_keys.items()}
        self.control_token = control_token
        self.modulus = validate_modulus(modulus)
        self.max_body_bytes = int(max_body_bytes)
        self.rounds: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        self.digests: dict[str, dict[str, str]] = {}
        self.finalized: dict[str, tuple[tuple[str, ...], bytes, str]] = {}
        self.lock = threading.RLock()

    def submit_share(self, round_id: str, client_id: str, body: bytes,
                     keys_json: str, signature: str) -> bool:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", round_id):
            raise ValueError("invalid round ID")
        if client_id not in self.client_keys:
            raise PermissionError("unknown client")
        message = round_id.encode() + b"\n" + client_id.encode() + b"\n" + keys_json.encode() + b"\n" + body
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

    def finalize(self, round_id: str, expected_client_ids: list[str],
                 signature: str) -> tuple[bytes, str]:
        canonical = json.dumps(sorted(set(str(cid) for cid in expected_client_ids)), separators=(",", ":"))
        message = b"finalize\n" + round_id.encode() + b"\n" + canonical.encode()
        if not hmac.compare_digest(signature, _signature(self.control_token, message)):
            raise PermissionError("invalid coordinator signature")
        clients = tuple(json.loads(canonical))
        if len(clients) < 2:
            raise ValueError("each party aggregate must include at least two clients")
        with self.lock:
            if round_id in self.finalized:
                previous, body, keys_json = self.finalized[round_id]
                if previous != clients:
                    raise RuntimeError("round was already finalized for a different participant set")
                return body, keys_json
            received = self.rounds.get(round_id, {})
            missing = [cid for cid in clients if cid not in received]
            if missing:
                raise RuntimeError(f"missing client shares for IDs: {missing}")
            tensor_keys = list(received[clients[0]])
            key_set = set(tensor_keys)
            shapes = {key: received[clients[0]][key].shape for key in tensor_keys}
            totals = {key: np.zeros(shapes[key], dtype=np.int64) for key in tensor_keys}
            for cid in clients:
                current = received[cid]
                if set(current) != key_set:
                    raise ValueError("client share tensor keys do not match")
                for key in tensor_keys:
                    if current[key].shape != shapes[key]:
                        raise ValueError(f"tensor shape mismatch for {key}")
                    totals[key] = np.remainder(totals[key] + current[key], self.modulus).astype(np.int64)
            body, keys_json = _encode_arrays(totals)
            self.finalized[round_id] = (clients, body, keys_json)
            self.rounds.pop(round_id, None)
            self.digests.pop(round_id, None)
            return body, keys_json

    def acknowledge(self, round_id: str, signature: str) -> None:
        message = b"ack\n" + round_id.encode()
        if not hmac.compare_digest(signature, _signature(self.control_token, message)):
            raise PermissionError("invalid coordinator signature")
        with self.lock:
            self.rounds.pop(round_id, None)
            self.digests.pop(round_id, None)
            self.finalized.pop(round_id, None)

def make_handler(state: AggregatorState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FLSMPCParty/1.0"
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
                if action == "finalize":
                    request = json.loads(body)
                    clients = request.get("expected_client_ids")
                    if not isinstance(clients, list) or any(not isinstance(cid, str) for cid in clients):
                        raise ValueError("expected_client_ids must be a list of strings")
                    result, keys_json = state.finalize(round_id, clients, signature)
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

def serve_aggregator(host: str, port: int, client_keys: dict, control_token: str, modulus: int):
    server = create_server(host, port, AggregatorState(client_keys, control_token, modulus))
    try:
        server.serve_forever()
    finally:
        server.server_close()

def main():
    parser = argparse.ArgumentParser(description="Run one additive-SMPC aggregation party")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--party-id", type=int, required=True)
    parser.add_argument("--field-modulus", type=int, default=DEFAULT_FIELD_MODULUS)
    args = parser.parse_args()
    client_keys = json.loads(os.environ.get("FL_SMPC_CLIENT_KEYS_JSON", "{}"))
    control_token = os.environ.get("FL_SMPC_CONTROL_TOKEN", "")
    server = create_server(args.host, args.port, AggregatorState(client_keys, control_token, args.field_modulus))
    print(f"Aggregator party {args.party_id} listening on {args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
