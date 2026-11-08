"""Client-side sharing and coordinator control for two-stage additive aggregation."""
from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
from typing import Mapping, Sequence

import numpy as np
import torch

from fl_security.smpc.additive_sharing import (
    DEFAULT_FIELD_MODULUS,
    generate_additive_shares,
    validate_modulus,
)
from fl_security.smpc.aggregator_server import _encode_arrays, _signature
from fl_security.smpc.fixed_point import FixedPointEncoder


def upload_client_update(
    state: Mapping[str, torch.Tensor],
    num_examples: int,
    client_id: int,
    round_id: str,
    committee_ids: Sequence[int],
    aggregator_endpoints: Mapping[str, str],
    client_tokens: Mapping[str, str],
    encoder: FixedPointEncoder,
    modulus: int = DEFAULT_FIELD_MODULUS,
    timeout: float = 60.0,
    retries: int = 3,
) -> int:
    """Create shares client-side and send each share only to its assigned party."""
    q = validate_modulus(modulus)
    if num_examples <= 0:
        raise ValueError("num_examples must be positive")
    parties = [int(pid) for pid in committee_ids]
    if len(parties) < 2 or len(set(parties)) != len(parties):
        raise ValueError("committee_ids must contain at least two distinct parties")
    shares_by_party = {party: {} for party in parties}
    for key, value in state.items():
        if not torch.is_floating_point(value):
            continue
        weighted = value.detach().cpu().to(torch.float64) * int(num_examples)
        encoded = encoder.encode(weighted).numpy()
        if np.any(np.abs(encoded.astype(object)) >= q // 2):
            raise OverflowError(f"weighted parameter {key!r} exceeds centred field range")
        residues = np.remainder(encoded, q).astype(np.int64)
        for party, share in zip(parties, generate_additive_shares(residues, len(parties), q)):
            shares_by_party[party][key] = share

    total_bytes = 0
    for party in parties:
        endpoint, token = aggregator_endpoints.get(str(party)), client_tokens.get(str(party))
        if not endpoint or not token:
            raise ValueError(f"missing endpoint or client token for aggregation party {party}")
        body, keys_json = _encode_arrays(shares_by_party[party])
        message = (
            str(round_id).encode() + b"\n" + str(client_id).encode()
            + b"\n" + keys_json.encode() + b"\n" + body
        )
        signature = _signature(token, message)
        request = urllib.request.Request(
            endpoint.rstrip("/") + f"/v1/rounds/{round_id}/shares",
            data=body,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Client-ID": str(client_id),
                "X-Tensor-Keys": keys_json,
                "X-Signature": signature,
            },
            method="POST",
        )
        last_error = None
        for attempt in range(max(1, retries)):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    if response.status not in (200, 201):
                        raise RuntimeError(f"party {party} returned HTTP {response.status}")
                last_error = None
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read(4096).decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"share upload to party {party} failed: {exc.code}: {detail}"
                ) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt + 1 < max(1, retries):
                    time.sleep(min(0.25 * (2 ** attempt), 2.0))
        if last_error is not None:
            raise RuntimeError(
                f"share upload to party {party} failed after retries: {last_error}"
            ) from last_error
        total_bytes += len(body)
    return total_bytes


def _post_json(
    endpoint: str,
    route: str,
    payload: dict,
    signature: str,
    timeout: float,
) -> tuple[bytes, Mapping[str, str]]:
    body = json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + route,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Signature": signature,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(), response.headers
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"aggregator request failed: {exc.code}: {detail}") from exc


def finalize_party(
    endpoint: str,
    round_id: str,
    expected_client_ids: Sequence[int],
    control_token: str,
    timeout: float = 120.0,
) -> None:
    """Ask a party to sum its client shares and re-share that partial to all peers.

    No tensor data is returned to the coordinator at this stage. Each source
    party sends random additive re-shares directly to the other parties.
    """
    client_ids = sorted(set(str(cid) for cid in expected_client_ids))
    if not client_ids:
        raise ValueError("a party must receive at least one client share")
    canonical = json.dumps(client_ids, separators=(",", ":"))
    message = b"finalize\n" + str(round_id).encode() + b"\n" + canonical.encode()
    signature = _signature(control_token, message)
    response, _ = _post_json(
        endpoint,
        f"/v1/rounds/{round_id}/finalize",
        {"expected_client_ids": client_ids},
        signature,
        timeout,
    )
    status = json.loads(response)
    if status.get("status") != "reshared":
        raise RuntimeError("aggregation party did not confirm Stage-2 re-sharing")


def retrieve_party_share(
    endpoint: str,
    round_id: str,
    expected_source_party_ids: Sequence[int],
    control_token: str,
    timeout: float = 120.0,
) -> dict[str, np.ndarray]:
    """Retrieve one random share of the global aggregate from a party."""
    sources = sorted(set(int(pid) for pid in expected_source_party_ids))
    canonical = json.dumps(sources, separators=(",", ":"))
    message = b"result\n" + str(round_id).encode() + b"\n" + canonical.encode()
    signature = _signature(control_token, message)
    payload, headers = _post_json(
        endpoint,
        f"/v1/rounds/{round_id}/result",
        {"expected_source_parties": sources},
        signature,
        timeout,
    )
    keys_json = headers.get("X-Tensor-Keys", "")
    keys = json.loads(keys_json)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != {f"arr_{index}" for index in range(len(keys))}:
            raise ValueError("aggregator result schema is invalid")
        return {key: archive[f"arr_{index}"].copy() for index, key in enumerate(keys)}


def acknowledge_party(
    endpoint: str,
    round_id: str,
    control_token: str,
    timeout: float = 10.0,
) -> None:
    """Tell a party the aggregate was consumed so it can clear its round state."""
    message = b"ack\n" + str(round_id).encode()
    signature = _signature(control_token, message)
    response, _ = _post_json(
        endpoint,
        f"/v1/rounds/{round_id}/ack",
        {"ack": True},
        signature,
        timeout,
    )
    if json.loads(response).get("status") != "cleared":
        raise RuntimeError("aggregator did not confirm round-state cleanup")
