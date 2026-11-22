import numpy as np
import pytest
from PIL import Image

from signlight.images import load_rgb, pad_to_square, resize, to_model_input


def test_load_rgb_applies_exif_orientation(tmp_path):
    # H=20, W=40, 왼쪽 10열만 빨강. Orientation=6 → 시계 방향 90° 회전해서 표시
    a = np.zeros((20, 40, 3), np.uint8)
    a[:, :10] = [255, 0, 0]
    exif = Image.Exif()
    exif[0x0112] = 6
    path = tmp_path / "rot.jpg"
    Image.fromarray(a).save(path, exif=exif, quality=100, subsampling=0)
    out = load_rgb(path)
    assert out.shape == (40, 20, 3)
    assert out[2, -1, 0] > 200  # 왼쪽 열 → 회전 후 위쪽 띠
    assert out[-3, 0, 0] < 60


def test_load_rgb_reads_heic(tmp_path):
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    path = tmp_path / "photo.HEIC"
    Image.fromarray(np.full((16, 24, 3), 120, np.uint8)).save(path, format="HEIF")
    out = load_rgb(path)
    assert out.shape == (16, 24, 3) and out.dtype == np.uint8


def test_load_rgb_drops_alpha(tmp_path):
    path = tmp_path / "a.png"
    Image.fromarray(np.zeros((5, 5, 4), np.uint8)).save(path)
    assert load_rgb(path).shape == (5, 5, 3)


def test_pad_to_square_keeps_whole_image():
    img = np.full((100, 60, 3), 50, np.uint8)
    sq = pad_to_square(img)
    assert sq.shape == (100, 100, 3)
    assert (sq == 50).all()


def test_pad_to_square_centers_original():
    img = np.zeros((10, 4, 3), np.uint8)
    img[:, :, 0] = 200
    img[0, 0] = [0, 0, 0]  # 가장자리 평균이 원본과 달라지도록 점 하나
    sq = pad_to_square(img)
    assert sq.shape == (10, 10, 3)
    np.testing.assert_array_equal(sq[:, 3:7], img)


def test_resize_and_model_input_shape():
    assert resize(np.zeros((200, 200, 3), np.uint8), 128).shape == (128, 128, 3)
    assert to_model_input(np.zeros((300, 200, 3), np.uint8), 128).shape == (128, 128, 3)
