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
