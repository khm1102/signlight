"""결정적 합성 환경 변형(테스트용). 입력/출력: (H,W,3) uint8 RGB.

같은 (조건, 이미지, key)는 항상 같은 결과를 낸다. 로컬(RF)과 Colab(CNN)이 같은 변형 이미지를 보게 하기 위해서다.
key는 index.csv의 행 번호를 쓴다.
기획안 실험 대응: 실험1 = clean, 실험2 = dark_*, 실험3 = clutter, 실험4 = shift / scale_*
한계: 실제 촬영 환경이 아니라 시뮬레이션이며 손 마스크가 없다.
"""

from collections.abc import Callable

import cv2
import numpy as np

from . import config as C
from .images import edge_mean


def low_light(img: np.ndarray, factor: float, rng: np.random.Generator, noise_sigma: float = 6.0) -> np.ndarray:
    """어두운 조명: 선형 감광 + 센서 노이즈."""
    out = img.astype(np.float32) * factor + rng.normal(0, noise_sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def clutter(img: np.ndarray, rng: np.random.Generator, n: int = 10, border: float = 0.22) -> np.ndarray:
    """복잡한 배경 근사: 가장자리 띠(border 비율)에 색 도형 n개와 텍스처 잡음. 가운데는 원본 그대로."""
    out = img.copy()
    h, w = out.shape[:2]
    t = int(border * min(h, w))
    for _ in range(n):
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        side = int(rng.integers(0, 4))
        if side == 0:
            cx, cy = rng.integers(0, w), rng.integers(0, t)
        elif side == 1:
            cx, cy = rng.integers(0, w), rng.integers(h - t, h)
        elif side == 2:
            cx, cy = rng.integers(0, t), rng.integers(0, h)
        else:
            cx, cy = rng.integers(w - t, w), rng.integers(0, h)
        r = int(rng.integers(int(0.03 * min(h, w)), int(0.13 * min(h, w))))
        if rng.random() < 0.5:
            cv2.circle(out, (int(cx), int(cy)), r, color, -1)
        else:
            cv2.rectangle(out, (int(cx - r), int(cy - r)), (int(cx + r), int(cy + r)), color, -1)
    y0, y1, x0, x1 = int(h * border), int(h * (1 - border)), int(w * border), int(w * (1 - border))
    mask = np.ones((h, w, 1), np.float32)
    mask[y0:y1, x0:x1] = 0
    out = np.clip(out.astype(np.float32) + rng.normal(0, 12, out.shape) * mask, 0, 255).astype(np.uint8)
    out[y0:y1, x0:x1] = img[y0:y1, x0:x1]
    return out


FRAME = 3  # Kaggle 사진 바깥의 파란 테두리 두께(px). 1~2px가 순수 파랑, 3px부터 사진 내용


def unframe(img: np.ndarray, b: int = FRAME) -> np.ndarray:
    """바깥 b px를 안쪽 픽셀로 덮어 테두리를 지운다. 이동·축소 때 테두리가 사진 안으로 들어오거나 빈 곳이 파랗게 채워지지 않게."""
    return cv2.copyMakeBorder(img[b:-b, b:-b], b, b, b, b, cv2.BORDER_REPLICATE)


def affine(img: np.ndarray, scale: float = 1.0, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """scale < 1: 손이 작아짐. dx, dy: 이미지 크기 대비 이동 비율. 빈 곳은 (테두리를 지운) 가장자리 평균색
    (reflect는 손이 거울상으로 복제돼 가짜 손을 만들어서 쓰지 않음)."""
    img = unframe(img)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    M[0, 2] += dx * w
    M[1, 2] += dy * h
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=edge_mean(img))


CONDITIONS: dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "clean": lambda im, rng: im,
    "dark_0.5": lambda im, rng: low_light(im, 0.5, rng),
    "dark_0.25": lambda im, rng: low_light(im, 0.25, rng),
    "clutter": lambda im, rng: clutter(im, rng),
    "shift": lambda im, rng: affine(im, 1.0, float(rng.choice([-0.2, 0.2])), float(rng.choice([-0.2, 0.2]))),
    "scale_small": lambda im, rng: affine(im, 0.6),
    "scale_large": lambda im, rng: affine(im, 1.35),
}
_ORDER = {name: i for i, name in enumerate(CONDITIONS)}


def apply(name: str, img: np.ndarray, key: int) -> np.ndarray:
    rng = np.random.default_rng([C.SEED, _ORDER[name], int(key)])
    return CONDITIONS[name](img, rng)
