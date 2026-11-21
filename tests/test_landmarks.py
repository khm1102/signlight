import numpy as np

from signlight.landmarks import HandDetector, extract


def test_detect_blank_image_fails(need_model):
    with HandDetector() as d:
        assert d.detect(np.zeros((200, 200, 3), np.uint8)) == (None, None)


def test_detect_real_hand(hand_image):
    with HandDetector() as d:
        lm, right = d.detect(hand_image)
    assert lm.shape == (21, 3) and lm.dtype == np.float32
    assert np.isfinite(lm).all()
    assert isinstance(right, bool)


def test_pad_option_detects_same_hand(hand_image):
    with HandDetector(pad=0.3) as d:
        lm, _ = d.detect(hand_image)
    assert lm is not None
    # 여백을 넣은 이미지 기준 좌표라 손이 가운데 쪽으로 몰린다
    assert 0.2 < lm[:, 0].mean() < 0.8


def test_extract_marks_failures_nan(hand_image):
    blank = np.zeros_like(hand_image)
    out = extract([blank, hand_image], n=2)
    assert out["det"].tolist() == [False, True]
    assert np.isnan(out["lm"][0]).all() and np.isfinite(out["lm"][1]).all()
    assert out["right"].dtype == bool and not out["right"][0]


def test_save_and_load_landmarks_roundtrip(tmp_path, monkeypatch):
    from signlight import landmarks as L

    monkeypatch.setattr(L, "LANDMARKS_PATH", tmp_path / "lm.npz")
    out = {"lm": np.zeros((2, 21, 3), np.float32), "det": np.array([True, False]), "right": np.array([True, False])}
    L.save_landmarks(out)
    back = L.load_landmarks()
    assert set(back) == {"lm", "det", "right"}
    np.testing.assert_array_equal(back["det"], out["det"])


def test_detection_rate_table():
    import pandas as pd

    from signlight.landmarks import detection_rate_table

    idx = pd.DataFrame({"label": ["A", "A", "B", "B"], "split": ["train", "test", "train", "train"]})
    t = detection_rate_table(idx, np.array([True, False, True, True]))
    assert t.loc[("A", "train"), "det_rate"] == 1.0
    assert t.loc[("A", "test"), "det_rate"] == 0.0
    assert t.loc[("B", "train"), "n"] == 2
