import numpy as np
import pytest
import torch
from fl_security.federated.aggregation import weighted_fedavg
from fl_security.smpc.additive_sharing import DEFAULT_FIELD_MODULUS, aggregate_shared_updates, generate_additive_shares, reconstruct_additive_shares
from fl_security.smpc.fixed_point import FixedPointEncoder

def test_additive_shares_reconstruct_signed_values_modularly():
    secret = np.array([-17, 0, 23], dtype=np.int64)
    residues = reconstruct_additive_shares(generate_additive_shares(secret, 3))
    np.testing.assert_array_equal(residues, np.remainder(secret, DEFAULT_FIELD_MODULUS))

def test_aggregation_routes_client_shares_via_different_committees():
    updates = [np.array([11, -3]), np.array([7, 9]), np.array([-2, 4])]
    committees = [[0, 1], [1, 2], [0, 2]]
    actual = aggregate_shared_updates(updates, committees)
    np.testing.assert_array_equal(actual, np.remainder(sum(updates), DEFAULT_FIELD_MODULUS))

def test_secure_weighted_fedavg_matches_plain_weighted_fedavg():
    base = {"weight": torch.tensor([0.0, 0.0]), "counter": torch.tensor(4)}
    states = [
        {"weight": torch.tensor([1.25, -2.0]), "counter": torch.tensor(9)},
        {"weight": torch.tensor([4.0, 1.0]), "counter": torch.tensor(10)},
    ]
    plain = weighted_fedavg(states, [2, 6], base)
    secure = weighted_fedavg(states, [2, 6], base, encoder=FixedPointEncoder(1_000_000),
                             committees={0: [0, 1, 2], 1: [0, 2]})
    torch.testing.assert_close(secure["weight"], plain["weight"], atol=1e-6, rtol=0)
    assert secure["counter"].item() == base["counter"].item()

def test_aggregation_rejects_single_party_committee():
    with pytest.raises(ValueError):
        aggregate_shared_updates([np.array([1])], [[0]])

def test_fixed_point_modular_decode_handles_negative_values():
    encoder = FixedPointEncoder(10_000)
    value = torch.tensor([-1.125, 0.25, 2.5], dtype=torch.float64)
    residues = np.remainder(encoder.encode(value).numpy(), DEFAULT_FIELD_MODULUS)
    torch.testing.assert_close(encoder.decode_modular(residues), value, atol=1e-4, rtol=0)
