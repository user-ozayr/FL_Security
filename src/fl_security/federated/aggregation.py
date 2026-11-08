"""Weighted FedAvg and reconstruction of remotely summed secret shares."""
from __future__ import annotations
from typing import Mapping, Sequence
import numpy as np
import torch
from fl_security.smpc.additive_sharing import DEFAULT_FIELD_MODULUS, aggregate_shared_updates, validate_modulus
from fl_security.smpc.fixed_point import FixedPointEncoder

def weighted_fedavg(client_states: Sequence[Mapping[str, torch.Tensor]], num_examples: Sequence[int],
                    global_state: Mapping[str, torch.Tensor], *, encoder: FixedPointEncoder | None = None,
                    committees: Mapping[int, Sequence[int]] | None = None,
                    modulus: int = DEFAULT_FIELD_MODULUS) -> dict[str, torch.Tensor]:
    """Sample-count weighted FedAvg; integer buffers remain from global model."""
    if not client_states or len(client_states) != len(num_examples):
        raise ValueError("client_states and num_examples must be non-empty and aligned")
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in num_examples):
        raise ValueError("each client example count must be a positive integer")
    keys, total_examples = list(global_state), sum(num_examples)
    expected = set(keys)
    if any(set(state) != expected for state in client_states):
        raise ValueError("all client states must match global_state keys")
    secure = encoder is not None
    if secure:
        q = validate_modulus(modulus)
        if committees is None:
            raise ValueError("secure aggregation requires per-client committees")
        try:
            assignments = [list(committees[i]) for i in range(len(client_states))]
        except KeyError as exc:
            raise ValueError(f"missing committee assignment for client index {exc}") from exc

    aggregated = {}
    for key in keys:
        base = global_state[key]
        if not torch.is_floating_point(base):
            aggregated[key] = base.detach().clone()
            continue
        weighted = [state[key].detach().cpu().to(torch.float64) * n
                    for state, n in zip(client_states, num_examples)]
        if secure:
            bound = sum(float(torch.max(torch.abs(item)).item()) if item.numel() else 0.0 for item in weighted)
            if bound * encoder.scale >= q / 2:
                raise OverflowError(f"weighted sum for {key!r} may wrap in the configured field")
            encoded = [encoder.encode(item).numpy() for item in weighted]
            field_sum = aggregate_shared_updates(encoded, assignments, q)
            average = encoder.decode_modular(field_sum, q) / total_examples
        else:
            summed = torch.zeros_like(weighted[0], dtype=torch.float64)
            for item in weighted:
                summed += item
            average = summed / total_examples
        aggregated[key] = average.to(device=base.device, dtype=base.dtype)
    return aggregated


def reconstruct_weighted_average(
    party_partial_sums: Sequence[Mapping[str, np.ndarray]],
    global_state: Mapping[str, torch.Tensor],
    total_examples: int,
    encoder: FixedPointEncoder,
    modulus: int = DEFAULT_FIELD_MODULUS,
) -> dict[str, torch.Tensor]:
    """Combine party aggregate shares and decode the sample-weighted global model."""
    q = validate_modulus(modulus)
    if len(party_partial_sums) < 2:
        raise ValueError("at least two party partial sums are required")
    if total_examples <= 0:
        raise ValueError("total_examples must be positive")
    keys = [key for key, tensor in global_state.items() if torch.is_floating_point(tensor)]
    expected = set(keys)
    total = {}
    for key in keys:
        arrays = [np.asarray(partial[key]) for partial in party_partial_sums]
        if any(array.dtype != np.int64 or array.shape != tuple(global_state[key].shape) for array in arrays):
            raise ValueError(f"invalid partial aggregate for tensor {key!r}")
        running = np.zeros(arrays[0].shape, dtype=np.int64)
        for array in arrays:
            if np.any(array < 0) or np.any(array >= q):
                raise ValueError("party partial sum contains an out-of-field residue")
            running = np.remainder(running + array, q).astype(np.int64)
        total[key] = running

    result = {}
    for key, base in global_state.items():
        if not torch.is_floating_point(base):
            result[key] = base.detach().clone()
            continue
        result[key] = (
            encoder.decode_modular(total[key], q) / total_examples
        ).to(device=base.device, dtype=base.dtype)
    return result
