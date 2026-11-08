from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fl_security.evaluation.csv_safety import sanitize_csv_dataframe
from fl_security.evaluation.metrics import ExperimentLogger
from fl_security.federated.flower_simulation import _resolve_authenticated_client_ids
from fl_security.mtd.controller import MTDController
from fl_security.mtd.randomness import HMACCounterRNG
from fl_security.smpc import additive_sharing
from fl_security.config import validate_config


def test_hmac_counter_rng_is_reproducible_and_samples_without_replacement():
    key = b"unit-test-key-material-32-bytes!!"
    left = HMACCounterRNG(key)
    right = HMACCounterRNG(key)

    assert left.sample(range(100), 30) == right.sample(range(100), 30)

    rng = HMACCounterRNG(key)
    selected = rng.sample(range(20), 20)
    assert sorted(selected) == list(range(20))


def test_mtd_controller_uses_secret_keyed_schedule(monkeypatch):
    config = {
        "experiment": {"seed": 123, "rounds": 3},
        "federated_learning": {"num_clients": 10},
        "mtd": {"enabled": True, "selection_strategy": "random", "committee_size": 3},
    }
    monkeypatch.setenv("FL_MTD_SECRET", "s" * 32)
    first = MTDController(config)
    second = MTDController(config)

    first_round = first.generate_round_config(1, num_clients=10, fraction=0.6)
    second_round = second.generate_round_config(1, num_clients=10, fraction=0.6)

    assert first.secret_seeded is True
    assert first_round["secret_seeded"] is True
    assert first_round["selected_clients"] == second_round["selected_clients"]
    assert first_round["committees"] == second_round["committees"]


def test_mtd_controller_rejects_short_deployment_secret(monkeypatch):
    monkeypatch.setenv("FL_MTD_SECRET", "too-short")
    with pytest.raises(ValueError, match="at least 32 bytes"):
        MTDController({"experiment": {"seed": 1}, "mtd": {}})


def test_additive_shares_draw_randomness_from_os_backed_token_bytes(monkeypatch):
    calls = []

    def fake_token_bytes(size):
        calls.append(size)
        return b"\x01" * size

    monkeypatch.setattr(additive_sharing.secrets, "token_bytes", fake_token_bytes)
    secret = np.array([10, 20, 30], dtype=np.int64)
    shares = additive_sharing.generate_additive_shares(secret, num_parties=2)

    assert calls
    np.testing.assert_array_equal(
        additive_sharing.reconstruct_additive_shares(shares), secret
    )


def test_flower_result_identity_must_match_registered_proxy():
    results = [
        (SimpleNamespace(cid="proxy-a"), SimpleNamespace(metrics={"client_id": 0})),
        (SimpleNamespace(cid="proxy-b"), SimpleNamespace(metrics={"client_id": 1})),
    ]
    assert _resolve_authenticated_client_ids(
        results, {0: "proxy-a", 1: "proxy-b"}, [0, 1]
    ) == [0, 1]

    spoofed = [
        (SimpleNamespace(cid="proxy-a"), SimpleNamespace(metrics={"client_id": 1}))
    ]
    with pytest.raises(RuntimeError, match="does not match"):
        _resolve_authenticated_client_ids(spoofed, {0: "proxy-a"}, [0])


def test_flower_rejects_unregistered_or_unselected_proxy_results():
    result = [
        (SimpleNamespace(cid="unknown-proxy"), SimpleNamespace(metrics={"client_id": 0}))
    ]
    with pytest.raises(RuntimeError, match="unregistered proxy"):
        _resolve_authenticated_client_ids(result, {0: "proxy-a"}, [0])

    unselected = [
        (SimpleNamespace(cid="proxy-a"), SimpleNamespace(metrics={"client_id": 0}))
    ]
    with pytest.raises(RuntimeError, match="not selected"):
        _resolve_authenticated_client_ids(unselected, {0: "proxy-a"}, [1])


def test_csv_sanitizer_neutralizes_formula_like_text_but_preserves_numbers():
    frame = pd.DataFrame({
        "text": ["=1+1", "  +SUM(A1:A2)", "\t=HYPERLINK(\"x\")", "normal"],
        "number": [-1, 2, 3, 4],
    })
    safe = sanitize_csv_dataframe(frame)

    assert safe["text"].tolist() == [
        "'=1+1",
        "'  +SUM(A1:A2)",
        "'\t=HYPERLINK(\"x\")",
        "normal",
    ]
    assert safe["number"].tolist() == [-1, 2, 3, 4]


def test_experiment_logger_rejects_traversal_and_writes_plain_csv(tmp_path: Path):
    logger = ExperimentLogger(tmp_path)
    logger.log_round({
        "round": 1,
        "global_accuracy": 0.5,
        "label": "=HYPERLINK(\"https://example.invalid\")",
    })

    with pytest.raises(ValueError, match="single"):
        logger.save_to_csv("../outside.csv")
    with pytest.raises(ValueError, match="single"):
        logger.save_to_csv("nested/outside.csv")
    with pytest.raises(ValueError, match="single"):
        logger.save_to_csv("C:outside.csv")

    outside = tmp_path.parent / "symlink-target.csv"
    outside.write_text("do not overwrite", encoding="utf-8")
    (tmp_path / "linked.csv").symlink_to(outside)
    with pytest.raises(ValueError, match="output_dir"):
        logger.save_to_csv("linked.csv")

    logger.save_to_csv("safe.csv")
    with (tmp_path / "safe.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["label"].startswith("'=")
    assert not (tmp_path.parent / "outside.csv").exists()


def test_experiment_logger_rejects_traversal_plot_filename(tmp_path: Path):
    logger = ExperimentLogger(tmp_path)
    logger.log_round({"round": 1, "global_accuracy": 0.5})
    with pytest.raises(ValueError, match="single"):
        logger.plot_metrics("../outside.png")


def test_configuration_validation_uses_exceptions_for_invalid_input():
    with pytest.raises(ValueError, match="root"):
        validate_config(None)
