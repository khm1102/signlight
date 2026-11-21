"""RF / DT 학습과 평가 (손 관절 특징).

python -m signlight.rf                      # 학습 + kaggle_val, kaggle_test, synth(7조건) 예측 저장
검출 실패 행(특징이 NaN)은 예측 FAIL(-1) = 오답으로 집계한다.
"""

import argparse
import json
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .data import load_index, synth_subset
from .evaluate import FAIL, e2e_accuracy, save_preds
from .features import build
from .images import load_rgb
from .landmarks import extract, load_landmarks
from .perturb import apply

SYNTH_LM_PATH = C.PROCESSED_DIR / "synth_landmarks.npz"
RF_GRID = [{"max_depth": d, "min_samples_leaf": m} for d in (None, 20) for m in (1, 3)]
DT_GRID = [{"max_depth": d} for d in (None, 10, 20)]


def predict_e2e(model, feats: np.ndarray) -> np.ndarray:
    ok = np.isfinite(feats).all(1)
    out = np.full(len(feats), FAIL, np.int64)
    if ok.any():
        out[ok] = model.predict(feats[ok])
    return out


def select_params(cls, grid, Xtr, ytr, Xva, yva, **fixed):
    """val 정확도(검출된 행 기준)가 가장 높은 설정. (best_params, scores)"""
    scores = [cls(**p, **fixed).fit(Xtr, ytr).score(Xva, yva) for p in grid]
    return grid[int(np.argmax(scores))], scores


def synth_landmarks(conditions: list[str], subset, cache=SYNTH_LM_PATH, preprocess: str | None = None) -> dict[str, dict[str, np.ndarray]]:
    """synth_subset 행마다 변형(key = index 행 번호) 후 관절 추출. 캐시는 행 번호가 같을 때만 재사용."""
    keys = subset.index.to_numpy()
    cached = {}
    if cache.exists():
        with np.load(cache) as z:
            if np.array_equal(z["keys"], keys):
                cached = {k: z[k] for k in z.files}
    out, changed = {}, False
    for cond in conditions:
        if f"{cond}__lm" in cached:
            out[cond] = {k: cached[f"{cond}__{k}"] for k in ("lm", "det", "right")}
            continue
        t = time.time()
        images = (apply(cond, load_rgb(C.ASL_ROOT / p), key) for key, p in zip(keys, subset["relpath"]))
        out[cond] = extract(images, len(keys), preprocess=preprocess)
        changed = True
        print(f"synth {cond}: det={out[cond]['det'].mean():.3f} ({time.time() - t:.0f}s)", flush=True)
    if changed:
        flat = {f"{c}__{k}": v for c, d in out.items() for k, v in d.items()}
        np.savez_compressed(cache, keys=keys, **flat)
    return out


def _feats(lm: dict, kind: str, canon: bool = True) -> np.ndarray:
    right = lm["right"] if canon else np.ones(len(lm["det"]), bool)
    return build(lm["lm"], right, kind)


def main(per_class: int = 100) -> None:
    idx = load_index()
    lm_all = load_landmarks()
    split = idx["split"].to_numpy()
    y = idx["y"].to_numpy()
    sub = synth_subset(idx, per_class=per_class)
    synth = synth_landmarks(C.SYNTH_CONDITIONS, sub)
    y_sub = sub["y"].to_numpy()

    # (이름, 분류기, 그리드, 특징, 좌우 정렬)
    variants = [
        ("rf_geom", RandomForestClassifier, RF_GRID, "geom", True),
        ("dt_geom", DecisionTreeClassifier, DT_GRID, "geom", True),
        ("rf_norm", RandomForestClassifier, RF_GRID, "norm", True),
        ("rf_geom_nocanon", RandomForestClassifier, RF_GRID, "geom", False),
    ]
    log = {}
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, cls, grid, kind, canon in variants:
        t = time.time()
        X = _feats(lm_all, kind, canon)
        det = np.isfinite(X).all(1)
        tr, va = (split == "train") & det, (split == "val") & det
        fixed = {"random_state": C.SEED} | ({"n_estimators": 300, "n_jobs": -1} if cls is RandomForestClassifier else {})
        best, scores = select_params(cls, grid, X[tr], y[tr], X[va], y[va], **fixed)
        model = cls(**best, **fixed).fit(X[tr], y[tr])
        joblib.dump(model, C.MODELS_DIR / f"{name}.joblib", compress=3)
        res = {"best": best, "val_scores_detected": scores}
        for testset, mask in (("kaggle_val", split == "val"), ("kaggle_test", split == "test")):
            pred = predict_e2e(model, X[mask])
            save_preds(name, testset, "clean", y[mask], pred)
            res[testset] = e2e_accuracy(y[mask], pred)
        for cond in C.SYNTH_CONDITIONS:
            pred = predict_e2e(model, _feats(synth[cond], kind, canon))
            save_preds(name, "synth", cond, y_sub, pred)
            res[f"synth_{cond}"] = e2e_accuracy(y_sub, pred)
        res["sec"] = round(time.time() - t)
        log[name] = res
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    (C.RESULTS_DIR / "rf_train_log.json").write_text(json.dumps(log, indent=1, default=str))


# 저장된 모델 이름 → (특징 종류, 좌우 정렬)
MODEL_FEATS = {"rf_geom": ("geom", True), "dt_geom": ("geom", True), "rf_norm": ("norm", True),
               "rf_geom_nocanon": ("geom", False)}


def eval_custom(models: list[str] = ("rf_geom", "dt_geom"), root=None) -> None:
    """직접 촬영셋(data/custom_test) 평가. 조건별로 예측을 저장한다(testset = custom)."""
    from .custom_test import scan

    df = scan(root)
    if df.empty:
        print("직접 촬영 사진 없음: data/custom_test/ 를 채운 뒤 다시 실행")
        return
    lm = extract((load_rgb(p) for p in df["path"]), len(df), progress=100)
    print(f"custom: {len(df)}장, 검출률 {lm['det'].mean():.3f}", flush=True)
    for name in models:
        model = joblib.load(C.MODELS_DIR / f"{name}.joblib")
        kind, canon = MODEL_FEATS[name]
        pred = predict_e2e(model, _feats(lm, kind, canon))
        for cond in C.CONDITIONS:
            mask = (df["condition"] == cond).to_numpy()
            if mask.any():
                save_preds(name, "custom", cond, df["y"].to_numpy()[mask], pred[mask])
                print(name, cond, f"n={mask.sum()}", f"acc={e2e_accuracy(df['y'].to_numpy()[mask], pred[mask]):.4f}")


def eval_clahe(per_class: int = 100) -> None:
    """EXP-005: 학습된 rf_geom에 CLAHE 전처리 후 추출한 합성셋 관절을 넣어 평가(rf_geom_clahe)."""
    idx = load_index()
    sub = synth_subset(idx, per_class=per_class)
    synth = synth_landmarks(C.SYNTH_CONDITIONS, sub, cache=C.PROCESSED_DIR / "synth_landmarks_clahe.npz", preprocess="clahe")
    model = joblib.load(C.MODELS_DIR / "rf_geom.joblib")
    y_sub = sub["y"].to_numpy()
    for cond in C.SYNTH_CONDITIONS:
        pred = predict_e2e(model, _feats(synth[cond], "geom"))
        save_preds("rf_geom_clahe", "synth", cond, y_sub, pred)
        print("rf_geom_clahe", cond, f"det={synth[cond]['det'].mean():.3f}", f"acc={e2e_accuracy(y_sub, pred):.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--clahe", action="store_true", help="EXP-005: CLAHE 전처리 평가만 실행")
    ap.add_argument("--testsets", default=None, help="custom: 저장된 모델로 직접 촬영셋만 평가")
    a = ap.parse_args()
    if a.clahe:
        eval_clahe(a.per_class)
    elif a.testsets == "custom":
        eval_custom()
    else:
        main(a.per_class)
