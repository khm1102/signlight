import numpy as np
import pytest

from signlight import config as C
from signlight.perturb import CONDITIONS, apply


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    a = np.full((200, 200, 3), 160, np.uint8)
    a[60:140, 70:130] = rng.integers(80, 200, (80, 60, 3))  # 가운데 "손"
    return a


def test_condition_order_matches_config():
    assert list(CONDITIONS) == C.SYNTH_CONDITIONS


@pytest.mark.parametrize("name", C.SYNTH_CONDITIONS)
def test_apply_is_deterministic_and_shape_preserved(img, name):
    a, b = apply(name, img, key=7), apply(name, img, key=7)
    assert np.array_equal(a, b)
    assert a.shape == img.shape and a.dtype == np.uint8


def test_different_keys_differ_for_random_conditions(img):
    assert not np.array_equal(apply("clutter", img, 0), apply("clutter", img, 1))
    assert not np.array_equal(apply("dark_0.5", img, 0), apply("dark_0.5", img, 1))


def test_clean_is_identity(img):
    assert np.array_equal(apply("clean", img, 3), img)


def test_dark_lowers_mean(img):
    assert apply("dark_0.25", img, 0).mean() < img.mean() * 0.4
    assert apply("dark_0.5", img, 0).mean() < img.mean() * 0.6


def test_clutter_keeps_center(img):
    out = apply("clutter", img, 0)
    c0, c1 = int(200 * 0.22), int(200 * 0.78)
    assert np.array_equal(out[c0:c1, c0:c1], img[c0:c1, c0:c1])
    assert not np.array_equal(out, img)


def test_scale_small_shrinks_hand(img):
    out = apply("scale_small", img, 0)
    # 가장자리는 배경색(160)으로 채워지고, 원래 손 영역의 바깥쪽은 배경이 된다
    assert np.abs(out[:10].astype(int) - 160).max() <= 1
    assert np.abs(out[65:70, 72:76].astype(int) - 160).max() <= 1


def test_unknown_condition_raises(img):
    with pytest.raises(KeyError):
        apply("fog", img, 0)


def test_affine_fill_ignores_colored_frame():
    # Kaggle 사진처럼 바깥 2px가 파란 테두리인 이미지: 빈 공간은 테두리 색이 아니라 안쪽 배경색이어야 한다
    img = np.full((200, 200, 3), 150, np.uint8)
    img[:2, :] = img[-2:, :] = img[:, :2] = img[:, -2:] = [0, 0, 250]
    for name in ("shift", "scale_small"):
        out = apply(name, img, key=0)
        blank = out[100, 5] if name == "scale_small" else out[5, 100]
        corner = out[3, 3]
        for px in (blank, corner):
            assert abs(int(px[0]) - 150) <= 2 and abs(int(px[2]) - 150) <= 2, (name, px)
        blue = (out[..., 2].astype(int) - out[..., 0]) > 60
        assert blue.sum() == 0, (name, int(blue.sum()))  # 테두리가 안쪽으로 옮겨져 파란 선이 생기면 안 된다
