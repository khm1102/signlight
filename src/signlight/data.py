"""데이터 인덱스: 근접 중복 군집(dHash + union-find)과 그룹 분할.

Kaggle ASL Alphabet은 영상 연속 프레임이라 거의 같은 사진이 많다. 같은 군집은 같은 split에 넣는다.
python -m signlight.data   # data/processed/index.csv, results/split_stats.json
"""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from . import config as C
from .images import load_rgb

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
INDEX_PATH = C.PROCESSED_DIR / "index.csv"


def dhash(img: np.ndarray, n: int = 16) -> np.ndarray:
    """n×n 차분 해시(bool, n*n개)."""
    g = Image.fromarray(img).convert("L").resize((n + 1, n), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).ravel()


def _dhash_file(path: Path) -> np.ndarray:
    return dhash(load_rgb(path))


def cluster(H: np.ndarray, thr: int) -> np.ndarray:
    """해밍 거리 <= thr 인 쌍을 union-find로 묶는다. 군집 id(0부터) 반환."""
    Hf = H.astype(np.float32)
    s = Hf.sum(1)
    D = s[:, None] + s[None, :] - 2.0 * (Hf @ Hf.T)
    parent = np.arange(len(H))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ii, jj = np.where(np.triu(D <= thr + 0.5, k=1))
    for i, j in zip(ii, jj):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    roots = np.array([find(i) for i in range(len(H))])
    return np.unique(roots, return_inverse=True)[1]


def build_index(asl_root: Path, classes: list[str], thr: int = 20, seed: int = C.SEED) -> pd.DataFrame:
    rows, groups = [], []
    offset = 0
    with ProcessPoolExecutor() as ex:
        for y, label in enumerate(classes):
            files = sorted(p for p in (asl_root / "train" / label).iterdir() if p.suffix.lower() in IMAGE_EXTS)
            H = np.stack(list(ex.map(_dhash_file, files, chunksize=64)))
            g = cluster(H, thr) + offset
            offset = int(g.max()) + 1
            groups.append(g)
            rows += [(p.relative_to(asl_root).as_posix(), label, y) for p in files]
    idx = pd.DataFrame(rows, columns=["relpath", "label", "y"])
    idx["group"] = np.concatenate(groups)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    split = np.full(len(idx), "train", dtype=object)
    for k, (_, te) in enumerate(folds.split(np.zeros(len(idx)), idx["y"], idx["group"])):
        if k == 0:
            split[te] = "test"
        elif k == 1:
            split[te] = "val"
    idx["split"] = split
    return idx


def load_index() -> pd.DataFrame:
    return pd.read_csv(INDEX_PATH)


def synth_subset(index: pd.DataFrame, per_class: int = 100, seed: int = C.SEED) -> pd.DataFrame:
    """test 행에서 클래스별 per_class개. 원래 index(= 행 번호, 변형 key)를 보존하고 (y, relpath) 순서로 정렬."""
    test = index[index["split"] == "test"]
    parts = [g.sample(n=min(per_class, len(g)), random_state=seed) for _, g in test.groupby("y")]
    return pd.concat(parts).sort_values(["y", "relpath"])


def split_stats(idx: pd.DataFrame) -> dict:
    per_class = {}
    for label, g in idx.groupby("label"):
        sizes = g["group"].value_counts()
        per_class[label] = {
            "n": int(len(g)),
            "n_clusters": int(len(sizes)),
            "max_cluster": int(sizes.max()),
            "split": {k: int(v) for k, v in g["split"].value_counts().items()},
        }
    return {
        "n": int(len(idx)),
        "split": {k: int(v) for k, v in idx["split"].value_counts().items()},
        "n_clusters": int(idx["group"].nunique()),
        "per_class": per_class,
    }


def main(thr: int = 20) -> None:
    idx = build_index(C.ASL_ROOT, C.TARGET_CLASSES, thr=thr)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    idx.to_csv(INDEX_PATH, index=False)
    stats = split_stats(idx) | {"dhash_thr": thr, "seed": C.SEED}
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.RESULTS_DIR / "split_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps({k: stats[k] for k in ("n", "split", "n_clusters")}, ensure_ascii=False))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=int, default=20, help="dHash(256bit) 해밍 임계값")
    main(ap.parse_args().thr)
