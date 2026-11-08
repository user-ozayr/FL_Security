import pytest
import torch

from fl_security.federated.robust_aggregation import (
    RobustUpdateDetector,
    robust_filter_and_aggregate,
)


def _states(values):
    return [{"weight": torch.tensor([float(value), float(value * 2)])} for value in values]


def test_median_residual_detector_rejects_sign_flipped_updates():
    global_state = {"weight": torch.zeros(2)}
    states = _states([1.0, 1.1, 0.9, -1.0, -1.1])
    aggregate, report = robust_filter_and_aggregate(
        states, [10, 10, 10, 10, 10], global_state,
        client_ids=[0, 1, 2, 3, 4], max_malicious=2,
        residual_z_threshold=3.5, min_cosine_similarity=0.0,
    )
    assert set(report["detected_client_ids"]) == {3, 4}
    assert report["aggregation_rule"] == "screened_sample_weighted_fedavg"
    assert report["accepted_client_count"] == 3
    torch.testing.assert_close(aggregate["weight"], torch.tensor([1.0, 2.0]), atol=1e-6, rtol=0)


def test_coordinate_median_fallback_when_filtering_leaves_too_few():
    global_state = {"weight": torch.zeros(2)}
    states = _states([1.0, -1.0, -1.1])
    aggregate, report = robust_filter_and_aggregate(
        states, [1, 1, 1], global_state, max_malicious=2, min_survivors=3,
    )
    assert report["fallback_used"] is True
    assert report["aggregation_rule"] == "coordinate_median_fallback"
    assert torch.isfinite(aggregate["weight"]).all()


def test_norm_ratio_flags_extreme_magnitude_update():
    global_state = {"weight": torch.zeros(2)}
    states = _states([1.0, 1.1, 0.9, 20.0])
    _, report = robust_filter_and_aggregate(
        states, [1, 1, 1, 1], global_state,
        client_ids=[0, 1, 2, 3], max_malicious=1, max_norm_ratio=5.0,
    )
    score = next(item for item in report["client_scores"] if item["client_id"] == 3)
    assert score["norm_ratio"] > 5.0
    assert "excessive_update_norm" in score["reasons"]
    assert 3 in report["detected_client_ids"]


def test_detector_rejects_non_finite_updates():
    global_state = {"weight": torch.zeros(2)}
    states = [{"weight": torch.tensor([1.0, float("nan")])}, _states([1.0])[0]]
    with pytest.raises(ValueError, match="NaN or infinity"):
        robust_filter_and_aggregate(states, [1, 1], global_state)


def test_detector_validates_client_id_alignment():
    with pytest.raises(ValueError, match="unique and aligned"):
        robust_filter_and_aggregate(
            _states([1.0, 2.0]), [1, 1], {"weight": torch.zeros(2)}, client_ids=[4, 4],
        )


def test_historical_contribution_consistency_flags_reversal_after_history():
    global_state = {"weight": torch.zeros(2)}
    detector = RobustUpdateDetector(
        max_malicious=1,
        min_survivors=2,
        min_history_rounds=2,
        history_cosine_threshold=-0.25,
    )
    honest = _states([1.0, 1.1, 0.9])
    # Two accepted rounds establish a positive update-direction profile for
    # each client. The small variation remains directionally consistent.
    detector.aggregate(honest, [1, 1, 1], global_state, client_ids=[0, 1, 2])
    detector.aggregate(honest, [1, 1, 1], global_state, client_ids=[0, 1, 2])

    changed = _states([1.0, 1.1, -0.9])
    _, report = detector.aggregate(changed, [1, 1, 1], global_state, client_ids=[0, 1, 2])
    score = next(item for item in report["client_scores"] if item["client_id"] == 2)
    assert score["historical_observations"] == 2
    assert score["historical_inconsistent"] is True
    assert "historical_contribution_inconsistent" in score["reasons"]
    assert 2 in report["detected_client_ids"]


def test_suspicious_updates_do_not_update_client_history():
    global_state = {"weight": torch.zeros(2)}
    detector = RobustUpdateDetector(max_malicious=1, min_survivors=2)
    _, report = detector.aggregate(
        _states([1.0, 1.1, 0.9, -1.0]),
        [1, 1, 1, 1], global_state, client_ids=[0, 1, 2, 3],
    )
    assert 3 in report["detected_client_ids"]
    assert 3 not in detector.history
    assert set(report["history_updated_client_ids"]) == {0, 1, 2}


def test_detector_validates_temporal_parameters():
    with pytest.raises(ValueError, match="history_decay"):
        RobustUpdateDetector(history_decay=1.0)
    with pytest.raises(ValueError, match="history_cosine_threshold"):
        RobustUpdateDetector(history_cosine_threshold=2.0)
