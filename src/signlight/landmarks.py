"""MediaPipe HandLandmarker(tasks API)로 손 관절 21점 추출.

검출 실패는 NaN + det=False로 남긴다(기획안: 실패도 인식 실패로 집계).
python -m signlight.landmarks   # index.csv 전체 → data/processed/landmarks.npz
"""

import time
from collections.abc import Iterable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from . import config as C
from .images import edge_mean

LANDMARKS_PATH = C.PROCESSED_DIR / "landmarks.npz"


class HandDetector:
    def __init__(self, conf: float = C.DET_CONF, pad: float = C.DET_PAD):
        self.pad = pad
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(C.HAND_MODEL)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=conf,
            min_hand_presence_confidence=conf,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        self._lm.close()

    def _prepare(self, img: np.ndarray) -> np.ndarray:
        if self.pad <= 0:
            return np.ascontiguousarray(img)
        h, w = img.shape[:2]
        py, px = int(round(self.pad * h)), int(round(self.pad * w))
        return cv2.copyMakeBorder(img, py, py, px, px, cv2.BORDER_CONSTANT, value=edge_mean(img))

    def detect(self, img: np.ndarray) -> tuple[np.ndarray | None, bool | None]:
        """(lm (21,3) float32 이미지 비율 좌표, is_right). 실패하면 (None, None)."""
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self._prepare(img))
        r = self._lm.detect(image)
        if not r.hand_landmarks:
            return None, None
        lm = np.array([(p.x, p.y, p.z) for p in r.hand_landmarks[0]], np.float32)
        return lm, r.handedness[0][0].category_name == "Right"


def extract(
    images: Iterable[np.ndarray],
    n: int,
    conf: float = C.DET_CONF,
    pad: float = C.DET_PAD,
    progress: int = 0,
) -> dict[str, np.ndarray]:
    lm = np.full((n, C.NUM_LANDMARKS, 3), np.nan, np.float32)
    det = np.zeros(n, bool)
    right = np.zeros(n, bool)
    t = time.time()
    with HandDetector(conf, pad) as d:
        for i, img in enumerate(images):
            pts, is_right = d.detect(img)
            if pts is not None:
                lm[i], det[i], right[i] = pts, True, is_right
            if progress and (i + 1) % progress == 0:
                print(f"{i + 1}/{n} {time.time() - t:.0f}s det={det[: i + 1].mean():.3f}", flush=True)
    return {"lm": lm, "det": det, "right": right}


def save_landmarks(out: dict[str, np.ndarray]) -> None:
    LANDMARKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(LANDMARKS_PATH, **out)


def load_landmarks() -> dict[str, np.ndarray]:
    """index.csv와 같은 행 순서의 lm/det/right."""
    with np.load(LANDMARKS_PATH) as z:
        return {k: z[k] for k in ("lm", "det", "right")}


def detection_rate_table(index, det: np.ndarray):
    import pandas as pd

    df = pd.DataFrame({"label": index["label"].values, "split": index["split"].values, "det": det})
    return df.groupby(["label", "split"])["det"].agg(n="size", det_rate="mean")


def main() -> None:
    from .data import load_index
    from .images import load_rgb

    idx = load_index()
    images = (load_rgb(C.ASL_ROOT / p) for p in idx["relpath"])
    out = extract(images, len(idx), progress=2000)
    save_landmarks(out)
    table = detection_rate_table(idx, out["det"])
    table.to_csv(C.RESULTS_DIR / "detection_rate.csv")
    det = out["det"]
    print(f"detect rate {det.mean():.4f}, right among detected {out['right'][det].mean():.4f}")
    print(table.groupby(level="split")["det_rate"].mean().round(4).to_dict())


if __name__ == "__main__":
    main()
