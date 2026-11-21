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
