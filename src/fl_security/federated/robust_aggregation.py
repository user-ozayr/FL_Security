"""Robust client-update screening and aggregation for poisoning experiments.

This detector evaluates individual plaintext updates. It must not be enabled
with the current sum-only SMPC backend: robust distance, median, and comparison
operations are not currently evaluated inside MPC.
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


class RobustUpdateDetector:
    """Stateful robust detector with cohort and historical contribution checks.

    Cross-sectional checks:
      * median residual with a MAD-normalized robust z-score;
      * cosine alignment with the current cohort median direction;
      * update norm relative to the cohort median norm.

    Temporal contribution consistency:
      * cosine similarity with an EMA of this client's previously accepted
        normalized update directions.
      * A temporal inconsistency is considered only after min_history_rounds
        accepted observations. Suspicious updates do not update their history.

    This is a heuristic screening layer, not Byzantine-secure computation.
    """

    def __init__(
        self,
        *,
        max_malicious: int = 1,
        residual_z_threshold: float = 3.5,
        min_cosine_similarity: float = 0.0,
        max_norm_ratio: float = 5.0,
        min_survivors: int = 2,
        history_decay: float = 0.8,
        min_history_rounds: int = 2,
        history_cosine_threshold: float = -0.25,
    ) -> None:
        if isinstance(max_malicious, bool) or not isinstance(max_malicious, int) or max_malicious < 0:
            raise ValueError("max_malicious must be a non-negative integer")
        if not math.isfinite(residual_z_threshold) or residual_z_threshold <= 0:
            raise ValueError("residual_z_threshold must be finite and positive")
        if not math.isfinite(min_cosine_similarity) or not -1.0 <= min_cosine_similarity <= 1.0:
            raise ValueError("min_cosine_similarity must be in [-1, 1]")
        if not math.isfinite(max_norm_ratio) or max_norm_ratio < 1.0:
            raise ValueError("max_norm_ratio must be finite and >= 1")
        if not math.isfinite(history_decay) or not 0.0 <= history_decay < 1.0:
            raise ValueError("history_decay must be in [0, 1)")
        if isinstance(min_history_rounds, bool) or not isinstance(min_history_rounds, int) or min_history_rounds < 1:
            raise ValueError("min_history_rounds must be a positive integer")
        if not math.isfinite(history_cosine_threshold) or not -1.0 <= history_cosine_threshold <= 1.0:
            raise ValueError("history_cosine_threshold must be in [-1, 1]")
        if isinstance(min_survivors, bool) or not isinstance(min_survivors, int) or min_survivors < 2:
            raise ValueError("min_survivors must be an integer >= 2")

        self.max_malicious = max_malicious
        self.residual_z_threshold = residual_z_threshold
        self.min_cosine_similarity = min_cosine_similarity
        self.max_norm_ratio = max_norm_ratio
        self.min_survivors = min_survivors
        self.history_decay = history_decay
        self.min_history_rounds = min_history_rounds
        self.history_cosine_threshold = history_cosine_threshold
        # client_id -> (EMA of normalized accepted update directions, count)
        self.history: dict[int, tuple[torch.Tensor, int]] = {}

    def aggregate(
        self,
        client_states: Sequence[Mapping[str, torch.Tensor]],
        num_examples: Sequence[int],
        global_state: Mapping[str, torch.Tensor],
        *,
        client_ids: Sequence[int] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict]:
        _validate_states(client_states, num_examples, global_state)
        count = len(client_states)
        if self.max_malicious >= count:
            raise ValueError("max_malicious must be less than number of clients")
        if self.min_survivors > count:
            raise ValueError("min_survivors cannot exceed number of clients")

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
        scale = max(float(residual_mad.item()) * 1.4826, 1e-12)
        robust_z = (residuals - residual_median) / scale
        cohort_cosine = torch.nn.functional.cosine_similarity(
            vectors, centre.unsqueeze(0).expand_as(vectors), dim=1, eps=1e-12
        )
        norm_median = max(float(torch.median(norms).item()), 1e-12)
        norm_ratios = norms / norm_median

        reports = []
        flagged_indices: set[int] = set()
        historical_flags: set[int] = set()
        for index, client_id in enumerate(ids):
            residual_outlier = float(robust_z[index].item()) > self.residual_z_threshold
            direction_outlier = (
                float(centre_norm.item()) > 1e-12
                and float(cohort_cosine[index].item()) < self.min_cosine_similarity
            )
            magnitude_outlier = float(norm_ratios[index].item()) > self.max_norm_ratio

            previous = self.history.get(client_id)
            historical_rounds = previous[1] if previous else 0
            historical_cosine = None
            historical_inconsistent = False
            if previous and historical_rounds >= self.min_history_rounds:
                current_direction = torch.nn.functional.normalize(
                    vectors[index], dim=0, eps=1e-12
                )
                historical_cosine = float(torch.dot(
                    current_direction, previous[0]
                ).clamp(-1.0, 1.0).item())
                historical_inconsistent = (
                    historical_cosine < self.history_cosine_threshold
                )

            flagged = (
                residual_outlier or direction_outlier
                or magnitude_outlier or historical_inconsistent
            )
            if flagged:
                flagged_indices.add(index)
            if historical_inconsistent:
                historical_flags.add(index)

            reasons = [
                name for condition, name in (
                    (residual_outlier, "median_residual_outlier"),
                    (direction_outlier, "opposed_cohort_direction"),
                    (magnitude_outlier, "excessive_update_norm"),
                    (historical_inconsistent, "historical_contribution_inconsistent"),
                ) if condition
            ]
            reports.append({
                "client_id": client_id,
                "update_norm": float(norms[index].item()),
                "distance_from_median": float(residuals[index].item()),
                "robust_z": float(robust_z[index].item()),
                "cohort_cosine_similarity": float(cohort_cosine[index].item()),
                "norm_ratio": float(norm_ratios[index].item()),
                "historical_cosine_similarity": historical_cosine,
                "historical_observations": historical_rounds,
                "historical_inconsistent": bool(historical_inconsistent),
                "flagged": bool(flagged),
                "reasons": reasons,
            })

        ranked = sorted(
            flagged_indices,
            key=lambda index: (
                1 if index in historical_flags else 0,
                float(robust_z[index].item()),
                1.0 - float(cohort_cosine[index].item()),
                float(norm_ratios[index].item()),
            ),
            reverse=True,
        )
        rejected = set(ranked[:self.max_malicious])
        survivors = [index for index in range(count) if index not in rejected]

        fallback = len(survivors) < self.min_survivors
        if fallback:
            aggregated = _coordinate_median_fallback(client_states, global_state)
            # A fallback does not declare an individual client trustworthy.
            rule = "coordinate_median_fallback"
        else:
            aggregated = weighted_fedavg(
                [client_states[index] for index in survivors],
                [num_examples[index] for index in survivors],
                global_state,
            )
            rule = "screened_sample_weighted_fedavg"

        rejected_ids = [ids[index] for index in sorted(rejected)]
        accepted_ids = [ids[index] for index in survivors]
        for item in reports:
            item["rejected"] = item["client_id"] in rejected_ids

        # Update reputation only from non-flagged clients in a successful
        # screened-FedAvg round. Rejected or suspicious updates never train
        # their own future reference profile.
        history_updated_ids = []
        if not fallback:
            for index in survivors:
                if index in flagged_indices:
                    continue
                client_id = ids[index]
                direction = torch.nn.functional.normalize(vectors[index], dim=0, eps=1e-12)
                previous = self.history.get(client_id)
                if previous is None:
                    ema, observations = direction, 1
                else:
                    old_ema, old_observations = previous
                    ema = torch.nn.functional.normalize(
                        self.history_decay * old_ema
                        + (1.0 - self.history_decay) * direction,
                        dim=0,
                        eps=1e-12,
                    )
                    observations = old_observations + 1
                self.history[client_id] = (ema.detach().clone(), observations)
                history_updated_ids.append(client_id)

        return aggregated, {
            "aggregation_rule": rule,
            "detected_client_ids": rejected_ids,
            "detected_client_count": len(rejected_ids),
            "flagged_client_ids": [ids[index] for index in sorted(flagged_indices)],
            "flagged_client_count": len(flagged_indices),
            "accepted_client_ids": accepted_ids,
            "accepted_client_count": len(accepted_ids),
            "history_updated_client_ids": history_updated_ids,
            "fallback_used": fallback,
            "client_scores": reports,
        }


def robust_filter_and_aggregate(
    client_states: Sequence[Mapping[str, torch.Tensor]],
    num_examples: Sequence[int],
    global_state: Mapping[str, torch.Tensor],
    **kwargs,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Stateless compatibility wrapper; use RobustUpdateDetector for FL rounds."""
    options = {
        key: kwargs.pop(key) for key in list(kwargs)
        if key in {
            "max_malicious", "residual_z_threshold", "min_cosine_similarity",
            "max_norm_ratio", "min_survivors", "history_decay",
            "min_history_rounds", "history_cosine_threshold",
        }
    }
    detector = RobustUpdateDetector(**options)
    return detector.aggregate(
        client_states, num_examples, global_state,
        client_ids=kwargs.pop("client_ids", None),
    )
