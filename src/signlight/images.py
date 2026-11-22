"""이미지 입출력: 로드(EXIF 방향, HEIC), 정사각형 패딩, 리사이즈."""

from pathlib import Path

import cv2
import numpy as np
import pillow_heif
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()


def load_rgb(path: Path) -> np.ndarray:
    """(H,W,3) uint8 RGB. 휴대폰 사진의 EXIF 방향을 적용한다."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def edge_mean(img: np.ndarray) -> tuple[float, float, float]:
    """네 가장자리 픽셀의 평균색. 빈 영역을 채울 때 쓴다."""
    edge = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    return tuple(float(v) for v in edge.mean(0))


def pad_to_square(img: np.ndarray) -> np.ndarray:
    """짧은 변 양쪽에 가장자리 평균색 여백을 넣어 정사각형으로 만든다. 원본은 잘라 내지 않는다."""
    h, w = img.shape[:2]
    if h == w:
        return img
    d = abs(h - w)
    a, b = d // 2, d - d // 2
    top, bottom, left, right = (a, b, 0, 0) if h < w else (0, 0, a, b)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=edge_mean(img))


def resize(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def to_model_input(img: np.ndarray, size: int) -> np.ndarray:
    return resize(pad_to_square(img), size)
