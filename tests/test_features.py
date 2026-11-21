import numpy as np
import pytest

from signlight.features import build, canonicalize, normalize


def _hand(seed=0):
    return np.random.default_rng(seed).random((1, 21, 3)).astype(np.float32)


def test_norm_invariant_to_translation_and_scale():
    lm = _hand()
    lm2 = lm * 2.5 + np.array([0.1, -0.2, 0.0], np.float32)
    np.testing.assert_allclose(build(lm, [True], "norm"), build(lm2, [True], "norm"), atol=1e-5)


def test_normalize_wrist_at_origin_and_unit_palm():
    nl = normalize(_hand())
    np.testing.assert_allclose(nl[0, 0], 0, atol=1e-7)
    assert np.linalg.norm(nl[0, 9, :2]) == pytest.approx(1.0)


def test_canonicalize_mirrors_left_hand():
    lm = _hand()
    flipped = lm.copy()
    flipped[..., 0] = 1 - flipped[..., 0]
    np.testing.assert_allclose(build(lm, [True]), build(flipped, [False]), atol=1e-5)


def test_canonicalize_keeps_nan_rows():
    lm = np.full((1, 21, 3), np.nan, np.float32)
    assert np.isnan(canonicalize(lm, np.array([False]))).all()


def test_geom_dim_and_nan_rows():
    f = build(np.full((2, 21, 3), np.nan, np.float32), [False, False])
    assert f.shape == (2, 288) and np.isnan(f).all()
    assert build(_hand(), [True], "raw").shape == (1, 63)
    assert build(_hand(), [True], "norm").shape == (1, 63)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        build(_hand(), [True], "pixels")


def test_real_image_mirror_consistency(hand_image):
    from signlight.landmarks import HandDetector

    with HandDetector() as d:
        a, ra = d.detect(hand_image)
        b, rb = d.detect(np.ascontiguousarray(hand_image[:, ::-1]))
    if a is None or b is None:
        pytest.skip("반전 이미지 검출 실패")
    fa = build(a[None], np.array([ra]))
    fb = build(b[None], np.array([rb]))
    assert ra != rb  # 반전하면 손 판정도 바뀌어야 정렬이 의미 있다
    assert np.linalg.norm(fa - fb) < 0.1 * np.linalg.norm(fa)
