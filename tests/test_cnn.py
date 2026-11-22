import numpy as np
import pandas as pd
import pytest
from PIL import Image

tf = pytest.importorskip("tensorflow")
from signlight.cnn import build_model, make_dataset, predict_images


def test_build_model_output_shape():
    m = build_model(24, 64)
    assert m.output_shape == (None, 24)


def test_augment_layers_inactive_at_inference():
    m = build_model(24, 64, augment=True)
    x = np.full((1, 64, 64, 3), 128, np.uint8)
    np.testing.assert_allclose(m(x, training=False), m(x, training=False))


def _fake_index(root, n_per=4):
    rows = []
    for y, label in enumerate(["A", "B"]):
        d = root / "train" / label
        d.mkdir(parents=True)
        for i in range(n_per):
            Image.fromarray(np.full((40, 40, 3), 30 + 180 * y, np.uint8)).save(d / f"{label}{i}.png")
            rows.append((f"train/{label}/{label}{i}.png", label, y, y * 10 + i, "train"))
    return pd.DataFrame(rows, columns=["relpath", "label", "y", "group", "split"])


def test_make_dataset_labels_follow_index(tmp_path):
    idx = _fake_index(tmp_path)
    ds = make_dataset(idx, "train", batch=3, size=32, asl_root=tmp_path)
    xs, ys = zip(*[(x.numpy(), y.numpy()) for x, y in ds])
    x, y = np.concatenate(xs), np.concatenate(ys)
    assert x.shape == (8, 32, 32, 3) and x.dtype == np.uint8
    assert y.tolist() == idx["y"].tolist()
    assert x[0].mean() < 50 and x[-1].mean() > 180  # 라벨과 이미지가 짝이 맞음


def test_tiny_overfit(tmp_path):
    tf.keras.utils.set_random_seed(0)
    idx = _fake_index(tmp_path)
    ds = make_dataset(idx, "train", batch=8, size=32, asl_root=tmp_path, shuffle=True)
    m = build_model(2, 32)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = m.fit(ds, epochs=30, verbose=0)
    assert hist.history["accuracy"][-1] == 1.0


def test_predict_images_batches_and_order():
    m = build_model(2, 32)
    imgs = [np.full((40, 40, 3), v, np.uint8) for v in (30, 210, 30)]
    out = predict_images(m, imgs, batch=2, size=32)
    assert out.shape == (3,) and out[0] == out[2]
    assert predict_images(m, [], size=32).shape == (0,)


def test_subsample_keeps_columns_and_limits_per_class_split():
    from signlight.cnn import subsample

    idx = pd.DataFrame({
        "relpath": [f"p{i}" for i in range(12)], "label": ["A"] * 6 + ["B"] * 6,
        "y": [0] * 6 + [1] * 6, "group": range(12), "split": ["train", "train", "train", "val", "val", "val"] * 2,
    })
    s = subsample(idx, 2)
    assert list(s.columns) == list(idx.columns)
    assert s.groupby(["y", "split"]).size().tolist() == [2, 2, 2, 2]


def test_predict_custom_saves_preds(tmp_path, monkeypatch):
    from signlight import config as C
    from signlight.cnn import predict_custom
    from signlight.evaluate import load_preds

    monkeypatch.setattr(C, "PREDS_DIR", tmp_path / "preds")
    for cond, letter in (("normal", "A"), ("position", "C")):
        d = tmp_path / "photos" / cond / letter
        d.mkdir(parents=True)
        Image.fromarray(np.zeros((60, 80, 3), np.uint8)).save(d / "khm_1.jpg")
    predict_custom(build_model(24, 64), "cnn_plain", 42, root=tmp_path / "photos", size=64)
    yt, yp = load_preds(C.PREDS_DIR / "cnn_plain__custom__position__s42.npz")
    assert yt.tolist() == [2] and yp.shape == (1,) and 0 <= yp[0] < 24
