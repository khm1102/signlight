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
