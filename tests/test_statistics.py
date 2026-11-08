import math

from fl_security.evaluation.statistics import summarize_values


def test_student_t_confidence_interval_uses_small_sample_critical_value():
    result = summarize_values([1, 2, 3, 4, 5])
    assert result["n"] == 5
    assert result["mean"] == 3.0
    assert math.isclose(result["std"], math.sqrt(2.5), rel_tol=1e-12)
    expected_margin = 2.776 * math.sqrt(2.5) / math.sqrt(5)
    assert math.isclose(result["ci95_lower"], 3.0 - expected_margin, rel_tol=1e-12)
    assert math.isclose(result["ci95_upper"], 3.0 + expected_margin, rel_tol=1e-12)


def test_single_seed_has_no_confidence_interval():
    result = summarize_values([0.75])
    assert result["n"] == 1
    assert result["mean"] == 0.75
    assert math.isnan(result["std"])
    assert math.isnan(result["ci95_lower"])
    assert math.isnan(result["ci95_upper"])


def test_empty_or_nonfinite_samples_are_ignored():
    result = summarize_values([float("nan"), float("inf"), 0.25, 0.75])
    assert result["n"] == 2
    assert result["mean"] == 0.5
