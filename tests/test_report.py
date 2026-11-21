import pandas as pd
import pytest

from signlight.report import seed_mean


def _summary():
    rows = []
    for s, acc in ((42, 0.6), (43, 0.7), (44, 0.8)):
        rows.append(("cnn_plain", s, "synth", "clean", 100, acc))
    rows.append(("rf_geom", -1, "synth", "clean", 100, 0.9))
    return pd.DataFrame(rows, columns=["model", "seed", "testset", "condition", "n", "acc"])


def test_seed_mean_aggregates_cnn_only():
    m = seed_mean(_summary()).set_index("model")
    assert len(m) == 2
    assert m.loc["cnn_plain", "acc"] == pytest.approx(0.7)
    assert m.loc["cnn_plain", "acc_std"] == pytest.approx(0.1)
    assert m.loc["cnn_plain", "n_seeds"] == 3
    assert m.loc["rf_geom", "acc"] == pytest.approx(0.9)
    assert m.loc["rf_geom", "n_seeds"] == 1
    assert pd.isna(m.loc["rf_geom", "acc_std"]) or m.loc["rf_geom", "acc_std"] == 0


def test_seed_mean_keeps_drop_column():
    s = _summary()
    s["drop_pp"] = [1.0, 2.0, 3.0, 4.0]
    m = seed_mean(s).set_index("model")
    assert m.loc["cnn_plain", "drop_pp"] == pytest.approx(2.0)


def test_main_writes_tables_and_figures(tmp_path, monkeypatch):
    import numpy as np

    from signlight import config as C
    from signlight import report as R
    from signlight.evaluate import save_preds

    for name in ("PREDS_DIR", "RESULTS_DIR", "FIGURES_DIR"):
        monkeypatch.setattr(C, name, tmp_path / name)
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(24), 5)
    for cond in C.SYNTH_CONDITIONS:
        save_preds("rf_geom", "synth", cond, y, np.where(rng.random(len(y)) < 0.8, y, -1))
        for s in C.CNN_SEEDS:
            save_preds("cnn_plain", "synth", cond, y, np.where(rng.random(len(y)) < 0.6, y, (y + 1) % 24), seed=s)
    R.main()
    s = pd.read_csv(C.RESULTS_DIR / "summary.csv")
    assert {"rf_geom", "cnn_plain"} == set(s["model"])
    d = pd.read_csv(C.RESULTS_DIR / "drop.csv")
    assert "drop_pp" in d.columns and (d["condition"] != "clean").all()
    figs = {p.name for p in C.FIGURES_DIR.glob("*.png")}
    assert {"acc_by_condition_synth.png", "drop_by_condition_synth.png",
            "confusion_rf_geom_synth.png", "confusion_cnn_plain_synth.png"} <= figs
