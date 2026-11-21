import numpy as np
import pandas as pd
import pytest

from signlight import evaluate as E
from signlight.evaluate import FAIL, accuracy_drop, bootstrap_ci, confusion, e2e_accuracy


def test_e2e_accuracy_counts_failures_as_wrong():
    assert e2e_accuracy([0, 1, 2, 3], [0, 1, FAIL, FAIL]) == 0.5


def test_e2e_accuracy_empty_is_nan():
    assert np.isnan(e2e_accuracy([], []))


def test_confusion_failure_column():
    cm = confusion([0, 1], [FAIL, 1], 2)
    assert cm.shape == (2, 3)
    assert cm[0, 2] == 1 and cm[1, 1] == 1 and cm.sum() == 2


def test_accuracy_drop_pp():
    s = pd.DataFrame({
        "model": ["rf", "rf", "cnn", "cnn"], "seed": [-1, -1, 42, 42], "testset": ["synth"] * 4,
        "condition": ["clean", "dark", "clean", "dark"], "acc": [0.9, 0.6, 0.8, 0.7],
    })
    d = accuracy_drop(s).set_index(["model", "condition"])["drop_pp"]
    assert d[("rf", "dark")] == pytest.approx(30.0)
    assert d[("cnn", "dark")] == pytest.approx(10.0)
    assert d[("rf", "clean")] == pytest.approx(0.0)


def test_save_and_summarize_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(E.C, "PREDS_DIR", tmp_path)
    p = E.save_preds("rf_geom", "synth", "dark_0.5", [0, 1, 1, 0], [0, 1, FAIL, 1])
    assert p.name == "rf_geom__synth__dark_0.5.npz"
    E.save_preds("cnn_plain", "synth", "clean", [0, 1], [0, 1], seed=43)
    s = E.summarize(tmp_path).set_index(["model", "condition"])
    assert list(s.columns) == ["seed", "testset", "n", "acc", "ci_lo", "ci_hi", "fail_rate"]
    assert s.loc[("rf_geom", "dark_0.5"), "acc"] == 0.5
    assert s.loc[("rf_geom", "dark_0.5"), "fail_rate"] == 0.25
    assert s.loc[("rf_geom", "dark_0.5"), "seed"] == -1
    assert s.loc[("cnn_plain", "clean"), "seed"] == 43


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 500)
    p = np.where(rng.random(500) < 0.7, y, (y + 1) % 5)
    lo, hi = bootstrap_ci(y, p)
    assert lo < e2e_accuracy(y, p) < hi and hi - lo < 0.15
