import numpy as np
import pandas as pd
from PIL import Image

from signlight.data import build_index, cluster, dhash, synth_subset


def _img(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)


def test_dhash_identical_images_distance_zero():
    a = dhash(_img(0))
    assert a.dtype == bool and a.shape == (256,)
    assert (a ^ dhash(_img(0).copy())).sum() == 0
    assert (a ^ dhash(_img(1))).sum() > 50


def test_cluster_merges_chain():
    H = np.zeros((4, 4), bool)
    H[1, 0] = True  # a-b 1
    H[2, :2] = True  # b-c 1, a-c 2
    H[3, :] = True  # d: 멀리
    g = cluster(H, 1)
    assert g[0] == g[1] == g[2] and g[3] != g[0]


def _make_asl(root, letters, groups=4, per_group=3):
    for li, l in enumerate(letters):
        d = root / "train" / l
        d.mkdir(parents=True)
        for gi in range(groups):
            base = _img(li * 100 + gi)
            for k in range(per_group):
                Image.fromarray(base).save(d / f"{l}{gi * per_group + k}.png")


def test_build_index_groups_do_not_cross_splits(tmp_path):
    letters = ["A", "B", "C"]
    _make_asl(tmp_path, letters, groups=5)
    idx = build_index(tmp_path, letters, thr=0, seed=0)
    assert len(idx) == 3 * 5 * 3
    assert list(idx.columns) == ["relpath", "label", "y", "group", "split"]
    assert idx.groupby("group")["split"].nunique().max() == 1
    assert idx["group"].nunique() == 15
    assert set(idx["split"]) == {"train", "val", "test"}
    assert idx["y"].tolist() == [letters.index(lbl) for lbl in idx["label"]]
    assert idx["relpath"].iloc[0].startswith("train/A/")


def test_synth_subset_is_deterministic_and_test_only():
    idx = pd.DataFrame({
        "relpath": [f"train/{l}/{l}{i}.jpg" for l in "AB" for i in range(10)],
        "label": [l for l in "AB" for _ in range(10)],
        "y": [y for y in (0, 1) for _ in range(10)],
        "group": list(range(20)),
        "split": (["test"] * 6 + ["train"] * 4) * 2,
    })
    s1 = synth_subset(idx, per_class=3, seed=1)
    s2 = synth_subset(idx, per_class=3, seed=1)
    assert s1.index.tolist() == s2.index.tolist()
    assert (s1["split"] == "test").all()
    assert s1.groupby("y").size().tolist() == [3, 3]
    assert s1["y"].is_monotonic_increasing
