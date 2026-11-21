"""CNN(이미지 픽셀) 모델, 입력 파이프라인, 학습·예측 CLI. GPU 학습은 Colab에서 돌린다(scripts/colab_cnn.sh).

python -m signlight.cnn train   --variant plain --seed 42 --epochs 15
python -m signlight.cnn predict --variant plain --seed 42 --testsets kaggle_val,kaggle_test,synth
라벨은 index.csv의 y(= config.TARGET_CLASSES 순서)를 쓴다.
"""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from . import config as C
from .data import load_index, synth_subset
from .evaluate import e2e_accuracy, save_preds
from .images import load_rgb, to_model_input
from .perturb import apply

keras = tf.keras


def build_model(n_classes: int = len(C.TARGET_CLASSES), size: int = C.CNN_SIZE, augment: bool = False) -> keras.Model:
    inp = keras.Input((size, size, 3))
    x = inp
    if augment:  # 학습 때만 작동
        for layer in (
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.05),
            keras.layers.RandomTranslation(0.15, 0.15),
            keras.layers.RandomZoom(0.2),
            keras.layers.RandomBrightness(0.4, value_range=(0, 255)),
            keras.layers.RandomContrast(0.3),
        ):
            x = layer(x)
    x = keras.layers.Rescaling(1 / 255)(x)
    for f in (32, 64, 128, 256):
        x = keras.layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name=f"cnn_{'aug' if augment else 'plain'}")


def make_dataset(
    index: pd.DataFrame,
    split: str,
    batch: int = 64,
    shuffle: bool = False,
    seed: int = C.SEED,
    size: int = C.CNN_SIZE,
    asl_root: Path | None = None,
) -> tf.data.Dataset:
    root = Path(asl_root or C.ASL_ROOT)
    rows = index[index["split"] == split]
    paths = [str(root / p) for p in rows["relpath"]]
    ys = rows["y"].to_numpy(np.int64)

    def _load(p):
        return to_model_input(load_rgb(Path(p.decode())), size)

    def _map(p, y):
        img = tf.numpy_function(_load, [p], tf.uint8)
        img.set_shape((size, size, 3))
        return img, y

    ds = tf.data.Dataset.from_tensor_slices((paths, ys)).map(_map, num_parallel_calls=tf.data.AUTOTUNE).cache()
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def predict_images(model: keras.Model, images: Iterable[np.ndarray], batch: int = 256, size: int = C.CNN_SIZE) -> np.ndarray:
    out, buf = [], []
    for img in images:
        buf.append(to_model_input(img, size))
        if len(buf) == batch:
            out.append(model.predict(np.stack(buf), verbose=0).argmax(1))
            buf = []
    if buf:
        out.append(model.predict(np.stack(buf), verbose=0).argmax(1))
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def subsample(index: pd.DataFrame, per_class: int) -> pd.DataFrame:
    """(y, split)마다 앞에서 per_class개. 스모크 테스트용."""
    return index.groupby(["y", "split"], sort=False).head(per_class)


def model_path(variant: str, seed: int) -> Path:
    return C.MODELS_DIR / f"cnn_{variant}_s{seed}.keras"


def train(variant: str, seed: int, epochs: int, per_class: int | None = None) -> None:
    keras.utils.set_random_seed(seed)
    idx = load_index()
    if per_class:  # 스모크 테스트용 축소
        idx = subsample(idx, per_class)
    tr = make_dataset(idx, "train", shuffle=True, seed=seed)
    va = make_dataset(idx, "val")
    model = build_model(augment=variant == "aug")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(
        tr, validation_data=va, epochs=epochs, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)],
    )
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(model_path(variant, seed))
    log_dir = C.RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"cnn_{variant}_s{seed}_history.json").write_text(
        json.dumps({k: [float(v) for v in vs] for k, vs in hist.history.items()}, indent=1))


def predict_custom(model: keras.Model, name: str, seed: int, root: Path | None = None, size: int = C.CNN_SIZE) -> None:
    """직접 촬영셋 조건별 예측 저장. 사진은 잘라 내지 않고 정사각형으로 패딩한 뒤 리사이즈한다."""
    from .custom_test import scan

    df = scan(root)
    for cond in C.CONDITIONS:
        rows = df[df["condition"] == cond]
        if rows.empty:
            continue
        pred = predict_images(model, (load_rgb(p) for p in rows["path"]), size=size)
        save_preds(name, "custom", cond, rows["y"], pred, seed=seed)
        print(name, seed, "custom", cond, len(rows), round(e2e_accuracy(rows["y"], pred), 4), flush=True)


def predict(variant: str, seed: int, testsets: list[str], per_class: int = 100) -> None:
    model = keras.models.load_model(model_path(variant, seed))
    idx = load_index()
    name = f"cnn_{variant}"
    for testset in testsets:
        if testset in ("kaggle_val", "kaggle_test"):
            rows = idx[idx["split"] == testset.removeprefix("kaggle_")]
            pred = predict_images(model, (load_rgb(C.ASL_ROOT / p) for p in rows["relpath"]))
            save_preds(name, testset, "clean", rows["y"], pred, seed=seed)
            print(name, seed, testset, round(e2e_accuracy(rows["y"], pred), 4), flush=True)
        elif testset == "synth":
            sub = synth_subset(idx, per_class=per_class)
            for cond in C.SYNTH_CONDITIONS:
                imgs = (apply(cond, load_rgb(C.ASL_ROOT / p), key) for key, p in zip(sub.index, sub["relpath"]))
                pred = predict_images(model, imgs)
                save_preds(name, "synth", cond, sub["y"], pred, seed=seed)
                print(name, seed, "synth", cond, round(e2e_accuracy(sub["y"], pred), 4), flush=True)
        elif testset == "custom":
            predict_custom(model, name, seed)
        else:
            raise ValueError(f"unknown testset: {testset}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict"])
    ap.add_argument("--variant", choices=["plain", "aug"], required=True)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--per-class", type=int, default=None, help="train: 스모크용 축소 / predict: synth 클래스당 장수")
    ap.add_argument("--testsets", default="kaggle_val,kaggle_test,synth")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.variant, a.seed, a.epochs, a.per_class)
    else:
        predict(a.variant, a.seed, a.testsets.split(","), a.per_class or 100)
