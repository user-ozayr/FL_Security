import pytest
import torch

from fl_security.federated.robust_aggregation import robust_filter_and_aggregate


def _states(values):
    return [{"weight": torch.tensor([float(value), float(value * 2)])} for value in values]


def test_median_residual_detector_rejects_sign_flipped_updates():
    global_state = {"weight": torch.zeros(2)}
    states = _states([1.0, 1.1, 0.9, -1.0, -1.1])
    aggregate, report = robust_filter_and_aggregate(
        states,
        [10, 10, 10, 10, 10],
        global_state,
        client_ids=[0, 1, 2, 3, 4],
        max_malicious=2,
        residual_z_threshold=3.5,
        min_cosine_similarity=0.0,
    )
    assert set(report["detected_client_ids"]) == {3, 4}
    assert report["aggregation_rule"] == "screened_sample_weighted_fedavg"
    assert report["accepted_client_count"] == 3
    torch.testing.assert_close(
        aggregate["weight"],
        torch.tensor([1.0, 2.0]),
        atol=1e-6,
        rtol=0,
    )


def test_coordinate_median_fallback_when_filtering_leaves_too_few():
    global_state = {"weight": torch.zeros(2)}
    states = _states([1.0, -1.0, -1.1])
    aggregate, report = robust_filter_and_aggregate(
        states,
        [1, 1, 1],
        global_state,
        max_malicious=2,
        min_survivors=3,
    )
    assert report["fallback_used"] is True
    assert report["aggregation_rule"] == "coordinate_median_fallback"
    assert torch.isfinite(aggregate["weight"]).all()


def test_detector_rejects_non_finite_updates():
    global_state = {"weight": torch.zeros(2)}
    states = [{"weight": torch.tensor([1.0, float("nan")])}, _states([1.0])[0]]
    with pytest.raises(ValueError, match="NaN or infinity"):
        robust_filter_and_aggregate(states, [1, 1], global_state)


def test_detector_validates_client_id_alignment():
    with pytest.raises(ValueError, match="unique and aligned"):
        robust_filter_and_aggregate(
            _states([1.0, 2.0]), [1, 1], {"weight": torch.zeros(2)},
            client_ids=[4, 4],
        )
