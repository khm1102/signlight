"""손 관절 → 특징. 입력 lm: (n,21,3) 이미지 비율 좌표, 검출 실패는 NaN 행.

- raw  : (x,y,z) 63차원. 이동/스케일 불변성 없음(대조군)
- norm : 손목(0) 기준 이동 + 손바닥 크기(0→9 2D 거리)로 스케일, 63차원
- geom : norm + 21점 쌍거리 210 + 손가락 마디 굴곡각 15 = 288차원
왼손은 좌우 반전해 오른손 기준으로 맞춘 뒤 계산한다(canonicalize).
"""

import itertools

import numpy as np

CHAINS = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
_PAIRS = np.array(list(itertools.combinations(range(21), 2)))
KINDS = ("raw", "norm", "geom")


def canonicalize(lm: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = np.array(lm, np.float32, copy=True)
    left = ~np.asarray(right, bool)
    out[left, :, 0] = 1.0 - out[left, :, 0]
    return out


def normalize(lm: np.ndarray) -> np.ndarray:
    c = lm - lm[:, :1, :]
    scale = np.linalg.norm(c[:, 9, :2], axis=1)
    scale = np.where(scale < 1e-6, np.nan, scale)
    return c / scale[:, None, None]


def _pair_dists(nl: np.ndarray) -> np.ndarray:
    return np.linalg.norm(nl[:, _PAIRS[:, 0]] - nl[:, _PAIRS[:, 1]], axis=2)


def _joint_angles(nl: np.ndarray) -> np.ndarray:
    out = []
    for ch in CHAINS:
        for a, b, c in zip(ch[:-2], ch[1:-1], ch[2:]):
            v1, v2 = nl[:, a] - nl[:, b], nl[:, c] - nl[:, b]
            cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
            out.append(np.arccos(np.clip(cos, -1, 1)))
    return np.stack(out, 1)


def build(lm: np.ndarray, right, kind: str = "geom") -> np.ndarray:
    if kind not in KINDS:
        raise ValueError(f"unknown feature kind: {kind}")
    lm = canonicalize(lm, np.asarray(right, bool))
    n = len(lm)
    if kind == "raw":
        return lm.reshape(n, -1)
    nl = normalize(lm)
    if kind == "norm":
        return nl.reshape(n, -1)
    return np.concatenate([nl.reshape(n, -1), _pair_dists(nl), _joint_angles(nl)], 1).astype(np.float32)
