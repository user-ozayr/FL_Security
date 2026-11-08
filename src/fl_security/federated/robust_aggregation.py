"""Robust client-update screening and aggregation for poisoning experiments.

The detector operates on individual plaintext updates. It must not be enabled
with the current sum-only SMPC backend: doing so would either reveal updates to
the coordinator or falsely imply that anomaly detection is happening on shares.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch

from fl_security.federated.aggregation import weighted_fedavg


def _validate_states(
    client_states: Sequence[Mapping[str, torch.Tensor]],
    num_examples: Sequence[int],
    global_state: Mapping[str, torch.Tensor],
) -> None:
    if not client_states or len(client_states) != len(num_examples):
        raise ValueError("client_states and num_examples must be non-empty and aligned")
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in num_examples):
        raise ValueError("each client example count must be a positive integer")
    expected = set(global_state)
    if any(set(state) != expected for state in client_states):
        raise ValueError("all client states must match global_state keys")
    for index, state in enumerate(client_states):
        for key, base in global_state.items():
            value = state[key]
            if value.shape != base.shape:
                raise ValueError(f"client {index} tensor shape mismatch for {key!r}")
            if torch.is_floating_point(base):
                delta = value.detach().to(device="cpu", dtype=torch.float64) - base.detach().to(
                    device="cpu", dtype=torch.float64
                )
                if not torch.isfinite(delta).all():
                    raise ValueError(f"client {index} update contains NaN or infinity in {key!r}")


def _flatten_delta(
    state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    parts = [
        (state[key].detach().to(device="cpu", dtype=torch.float64)
         - base.detach().to(device="cpu", dtype=torch.float64)).reshape(-1)
        for key, base in global_state.items()
        if torch.is_floating_point(base)
    ]
    if not parts:
        raise ValueError("global model has no floating-point parameters")
    return torch.cat(parts)


def _coordinate_median_fallback(
    client_states: Sequence[Mapping[str, torch.Tensor]],
    global_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Coordinate-wise median of model deltas; integer buffers stay unchanged."""
    result: dict[str, torch.Tensor] = {}
    for key, base in global_state.items():
        if not torch.is_floating_point(base):
            result[key] = base.detach().clone()
            continue
        deltas = torch.stack([
            state[key].detach().to(device="cpu", dtype=torch.float64)
            - base.detach().to(device="cpu", dtype=torch.float64)
            for state in client_states
        ])
        median_delta = torch.median(deltas, dim=0).values
        result[key] = (base.detach().cpu().to(torch.float64) + median_delta).to(
            device=base.device, dtype=base.dtype
        )
    return result


def robust_filter_and_aggregate(
    client_states: Sequence[Mapping[str, torch.Tensor]],
    num_examples: Sequence[int],
    global_state: Mapping[str, torch.Tensor],
    *,
    client_ids: Sequence[int] | None = None,
    max_malicious: int = 1,
    residual_z_threshold: float = 3.5,
    min_cosine_similarity: float = 0.0,
    max_norm_ratio: float = 5.0,
    min_survivors: int = 2,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Screen update outliers and FedAvg the remaining clients.

    Detection uses distance from the coordinate-wise median update, a robust
    median/MAD score, cosine similarity to the median direction, and a norm
    ratio. The detector is heuristic: it can produce false positives for
    heterogeneous honest clients and can miss coordinated/adaptive poisoning.

    If screening leaves fewer than min_survivors, the function falls back to a
    coordinate-wise median over all submitted updates rather than returning an
    empty or single-client aggregate.
    """
    _validate_states(client_states, num_examples, global_state)
    count = len(client_states)
    if isinstance(max_malicious, bool) or not isinstance(max_malicious, int):
        raise TypeError("max_malicious must be an integer")
    if max_malicious < 0 or max_malicious >= count:
        raise ValueError("max_malicious must be in [0, number of clients)")
    if min_survivors < 2 or min_survivors > count:
        raise ValueError("min_survivors must be between 2 and the number of clients")
    if not math.isfinite(residual_z_threshold) or residual_z_threshold <= 0:
        raise ValueError("residual_z_threshold must be finite and positive")
    if not math.isfinite(min_cosine_similarity) or not -1.0 <= min_cosine_similarity <= 1.0:
        raise ValueError("min_cosine_similarity must be in [-1, 1]")
    if not math.isfinite(max_norm_ratio) or max_norm_ratio < 1.0:
        raise ValueError("max_norm_ratio must be finite and >= 1")

    ids = list(range(count)) if client_ids is None else [int(cid) for cid in client_ids]
    if len(ids) != count or len(set(ids)) != count:
        raise ValueError("client_ids must be unique and aligned with client_states")

    vectors = torch.stack([_flatten_delta(state, global_state) for state in client_states])
    centre = torch.median(vectors, dim=0).values
    residuals = torch.linalg.vector_norm(vectors - centre, dim=1)
    norms = torch.linalg.vector_norm(vectors, dim=1)
    centre_norm = torch.linalg.vector_norm(centre)

    residual_median = torch.median(residuals)
    residual_mad = torch.median(torch.abs(residuals - residual_median))
    # A zero MAD is common when several clients submit almost identical
    # updates. A small scale floor avoids division by zero without hiding
    # clearly non-zero deviations.
    scale = max(float(residual_mad.item()) * 1.4826, 1e-12)
    robust_z = (residuals - residual_median) / scale
    cosine = torch.nn.functional.cosine_similarity(
        vectors, centre.unsqueeze(0).expand_as(vectors), dim=1, eps=1e-12
    )
    norm_median = max(float(torch.median(norms).item()), 1e-12)
    norm_ratios = norms / norm_median

    reports = []
    flagged_indices: set[int] = set()
    for index in range(count):
        residual_outlier = float(robust_z[index].item()) > residual_z_threshold
        direction_outlier = (
            float(centre_norm.item()) > 1e-12
            and float(cosine[index].item()) < min_cosine_similarity
        )
        magnitude_outlier = float(norm_ratios[index].item()) > max_norm_ratio
        flagged = residual_outlier or direction_outlier or magnitude_outlier
        if flagged:
            flagged_indices.add(index)
        reports.append({
            "client_id": ids[index],
            "update_norm": float(norms[index].item()),
            "distance_from_median": float(residuals[index].item()),
            "robust_z": float(robust_z[index].item()),
            "cosine_to_median": float(cosine[index].item()),
            "norm_ratio": float(norm_ratios[index].item()),
            "flagged": bool(flagged),
            "reasons": [
                name for condition, name in (
                    (residual_outlier, "median_residual_outlier"),
                    (direction_outlier, "opposed_update_direction"),
                    (magnitude_outlier, "excessive_update_norm"),
                ) if condition
            ],
        })

    # The configured malicious-client count bounds only how many suspicious
    # candidates are removed; it is not used to force a fixed number of flags.
    ranked = sorted(
        flagged_indices,
        key=lambda index: (
            float(robust_z[index].item()),
            1.0 - float(cosine[index].item()),
            float(norm_ratios[index].item()),
        ),
        reverse=True,
    )
    rejected = set(ranked[:max_malicious])
    survivors = [index for index in range(count) if index not in rejected]

    fallback = len(survivors) < min_survivors
    if fallback:
        aggregated = _coordinate_median_fallback(client_states, global_state)
        survivors = list(range(count))
        rule = "coordinate_median_fallback"
    else:
        aggregated = weighted_fedavg(
            [client_states[index] for index in survivors],
            [num_examples[index] for index in survivors],
            global_state,
        )
        rule = "screened_sample_weighted_fedavg"

    rejected_ids = [ids[index] for index in sorted(rejected)]
    for item in reports:
        item["rejected"] = item["client_id"] in rejected_ids
    return aggregated, {
        "aggregation_rule": rule,
        "detected_client_ids": rejected_ids,
        "detected_client_count": len(rejected_ids),
        "accepted_client_ids": [ids[index] for index in survivors],
        "accepted_client_count": len(survivors),
        "fallback_used": fallback,
        "client_scores": reports,
    }
