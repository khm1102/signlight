import numpy as np
from sklearn.tree import DecisionTreeClassifier

from signlight.evaluate import FAIL
from signlight.rf import predict_e2e, select_params


def test_predict_e2e_marks_nan_rows_as_fail():
    m = DecisionTreeClassifier().fit(np.eye(3), [0, 1, 2])
    X = np.vstack([np.eye(3)[:1], np.full((1, 3), np.nan)])
    assert predict_e2e(m, X).tolist() == [0, FAIL]


def test_predict_e2e_all_nan():
    m = DecisionTreeClassifier().fit(np.eye(3), [0, 1, 2])
    assert predict_e2e(m, np.full((2, 3), np.nan)).tolist() == [FAIL, FAIL]


def test_select_params_picks_best_val():
    rng = np.random.default_rng(0)
    X = rng.random((200, 4))
    y = (X[:, 0] > 0.5).astype(int)
    grid = [{"max_depth": 1}, {"max_depth": None}]
    best, scores = select_params(DecisionTreeClassifier, grid, X[:150], y[:150], X[150:], y[150:], random_state=0)
    assert best in grid and len(scores) == 2
    assert max(scores) >= 0.9


def _photos(root):
    from PIL import Image

    for cond, letter in (("normal", "A"), ("dark", "B")):
        d = root / cond / letter
        d.mkdir(parents=True)
        Image.fromarray(np.zeros((60, 80, 3), np.uint8)).save(d / "khm_1.jpg")


def test_eval_custom_saves_preds_per_condition(tmp_path, monkeypatch, need_model):
    import joblib

    from signlight import config as C
    from signlight.evaluate import load_preds
    from signlight.rf import eval_custom

    monkeypatch.setattr(C, "PREDS_DIR", tmp_path / "preds")
    monkeypatch.setattr(C, "MODELS_DIR", tmp_path / "models")
    C.MODELS_DIR.mkdir()
    joblib.dump(DecisionTreeClassifier().fit(np.eye(288), np.arange(288) % 24), C.MODELS_DIR / "rf_geom.joblib")
    _photos(tmp_path / "photos")
    eval_custom(["rf_geom"], root=tmp_path / "photos")
    yt, yp = load_preds(C.PREDS_DIR / "rf_geom__custom__normal.npz")
    assert yt.tolist() == [0] and yp.tolist() == [FAIL]  # 손이 없는 사진 → 검출 실패
    assert (C.PREDS_DIR / "rf_geom__custom__dark.npz").exists()


def test_synth_cache_invalidated_when_fingerprint_changes(tmp_path, monkeypatch):
    import pandas as pd

    from signlight import rf

    calls = []

    def fake_extract(images, n, **kw):
        list(images)
        calls.append(kw.get("preprocess"))
        return {"lm": np.zeros((n, 21, 3), np.float32), "det": np.ones(n, bool), "right": np.ones(n, bool)}

    monkeypatch.setattr(rf, "extract", fake_extract)
    monkeypatch.setattr(rf, "load_rgb", lambda p: np.zeros((8, 8, 3), np.uint8))
    sub = pd.DataFrame({"relpath": ["train/A/A1.jpg", "train/B/B1.jpg"], "y": [0, 1]}, index=[5, 9])
    cache = tmp_path / "c.npz"
    rf.synth_landmarks(["clean"], sub, cache=cache)
    rf.synth_landmarks(["clean"], sub, cache=cache)
    assert len(calls) == 1  # 같은 설정이면 캐시 재사용
    rf.synth_landmarks(["clean"], sub, cache=cache, preprocess="clahe")
    assert len(calls) == 2  # 전처리가 다르면 다시 추출
    monkeypatch.setattr(rf.C, "DET_CONF", 0.9)
    rf.synth_landmarks(["clean"], sub, cache=cache, preprocess="clahe")
    assert len(calls) == 3  # 검출 설정이 바뀌어도 다시 추출


def test_cli_rejects_unknown_testsets():
    import pytest

    from signlight.rf import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--testsets", "synth"])
    assert parse_args(["--testsets", "custom"]).testsets == "custom"
