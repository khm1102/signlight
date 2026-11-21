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
