"""평가 지표와 예측 파일 형식.

예측 파일: PREDS_DIR / "{model}__{testset}__{condition}[__s{seed}].npz" (y_true, y_pred; FAIL = 검출 실패)
정확도는 e2e: 검출 실패도 오답이고 분모는 전체 샘플 수다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

FAIL = -1
NO_SEED = -1


def e2e_accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")


def bootstrap_ci(y_true, y_pred, n_boot: int = 1000, seed: int = C.SEED) -> tuple[float, float]:
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(np.float64)
    if not len(correct):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = correct[rng.integers(0, len(correct), (n_boot, len(correct)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    """(n_classes, n_classes+1). 마지막 열 = 검출 실패."""
    y_pred = np.where(np.asarray(y_pred) == FAIL, n_classes, y_pred)
    cm = np.zeros((n_classes, n_classes + 1), np.int64)
    np.add.at(cm, (np.asarray(y_true), y_pred), 1)
    return cm


def pred_path(model: str, testset: str, condition: str, seed: int | None = None) -> Path:
    name = f"{model}__{testset}__{condition}" + (f"__s{seed}" if seed is not None else "")
    return C.PREDS_DIR / f"{name}.npz"


def save_preds(model: str, testset: str, condition: str, y_true, y_pred, seed: int | None = None) -> Path:
    p = pred_path(model, testset, condition, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, y_true=np.asarray(y_true, np.int64), y_pred=np.asarray(y_pred, np.int64))
    return p


def load_preds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        return z["y_true"], z["y_pred"]


def _parse_name(stem: str) -> dict:
    parts = stem.split("__")
    seed = int(parts[3][1:]) if len(parts) == 4 else NO_SEED
    return {"model": parts[0], "seed": seed, "testset": parts[1], "condition": parts[2]}


def summarize(pred_dir: Path | None = None) -> pd.DataFrame:
    """seed 없는 모델(RF/DT)은 seed = -1."""
    rows = []
    for p in sorted(Path(pred_dir or C.PREDS_DIR).glob("*.npz")):
        yt, yp = load_preds(p)
        lo, hi = bootstrap_ci(yt, yp)
        rows.append(_parse_name(p.stem) | {
            "n": len(yt), "acc": e2e_accuracy(yt, yp), "ci_lo": lo, "ci_hi": hi,
            "fail_rate": float((yp == FAIL).mean()) if len(yp) else float("nan"),
        })
    cols = ["model", "seed", "testset", "condition", "n", "acc", "ci_lo", "ci_hi", "fail_rate"]
    return pd.DataFrame(rows, columns=cols)


def accuracy_drop(summary: pd.DataFrame, base: str = "clean") -> pd.DataFrame:
    """같은 (model, seed, testset)에서 base 대비 정확도 감소 폭(%p)."""
    keys = ["model", "seed", "testset"]
    base_acc = summary[summary["condition"] == base].set_index(keys)["acc"].rename("base_acc")
    out = summary.join(base_acc, on=keys)
    out["drop_pp"] = (out["base_acc"] - out["acc"]) * 100
    return out.dropna(subset=["base_acc"]).reset_index(drop=True)
