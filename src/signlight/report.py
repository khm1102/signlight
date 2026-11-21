"""결과 요약 표와 그림.

python -m signlight.report   # results/summary.csv, results/drop.csv, reports/figures/*.png
"""

import pandas as pd

KEYS = ["model", "testset", "condition"]


def seed_mean(summary: pd.DataFrame) -> pd.DataFrame:
    """(model, testset, condition)별 seed 평균. acc_std는 seed 간 표준편차(seed 1개면 NaN)."""
    agg = {"acc": ["mean", "std", "size"]}
    extra = [c for c in ("drop_pp", "fail_rate", "ci_lo", "ci_hi", "n") if c in summary.columns]
    agg |= {c: "mean" for c in extra}
    g = summary.groupby(KEYS, sort=False).agg(agg)
    g.columns = ["acc", "acc_std", "n_seeds", *extra]
    return g.reset_index()


# ---- 그림 ---------------------------------------------------------------
# 모델별 색은 모든 그림에서 고정(dataviz 기본 팔레트 순서, 검증 통과: 인접 CVD ΔE ≥ 9.1).
# 노랑·분홍·청록은 밝은 배경 대비 3:1 미만이라 같은 수치를 표(summary.csv, REPORT.md)로도 제공한다.
MODEL_STYLE = {
    "rf_geom": ("#2a78d6", "RF (관절 geom)"),
    "cnn_plain": ("#eb6834", "CNN"),
    "cnn_aug": ("#1baf7a", "CNN + 증강"),
    "dt_geom": ("#eda100", "DT (관절 geom)"),
    "rf_geom_clahe": ("#e87ba4", "RF + CLAHE"),
    "rf_norm": ("#008300", "RF (관절 norm)"),
    "rf_geom_nocanon": ("#6250d6", "RF (좌우 정렬 없음)"),
}
MAIN_MODELS = ["rf_geom", "cnn_plain", "cnn_aug", "dt_geom"]
SUPP_MODELS = ["rf_geom", "rf_norm", "rf_geom_nocanon", "rf_geom_clahe"]
COND_LABEL = {
    "clean": "원본", "dark_0.5": "어둡게 0.5", "dark_0.25": "어둡게 0.25", "clutter": "복잡한 배경",
    "shift": "이동", "scale_small": "축소", "scale_large": "확대",
    "normal": "일반 조명", "dark": "어두운 조명", "complex_bg": "복잡한 배경", "position": "손 위치",
}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": ["AppleGothic", "NanumGothic", "DejaVu Sans"], "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    })
    return plt


def _conditions(testset: str) -> list[str]:
    from . import config as C

    return C.SYNTH_CONDITIONS if testset == "synth" else C.CONDITIONS


def _grouped_bars(df: pd.DataFrame, value: str, testset: str, models: list[str], title: str, ylabel: str,
                  path, err: tuple[str, str] | None = None, skip_base: bool = False):
    plt = _plt()
    conds = [c for c in _conditions(testset) if c in set(df["condition"])]
    if skip_base:
        conds = conds[1:]
    models = [m for m in models if m in set(df["model"])]
    fig, ax = plt.subplots(figsize=(1.3 * len(conds) + 2, 4.2))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        sub = df[df["model"] == m].set_index("condition").reindex(conds)
        xs = [k + (i - (len(models) - 1) / 2) * width for k in range(len(conds))]
        yerr = None
        if err:
            lo, hi = sub[err[0]], sub[err[1]]
            yerr = [(sub[value] - lo).clip(lower=0), (hi - sub[value]).clip(lower=0)]
        color, label = MODEL_STYLE.get(m, ("#888888", m))
        ax.bar(xs, sub[value], width=width - 0.02, color=color, label=label, edgecolor=SURFACE, linewidth=1,
               yerr=yerr, error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
    ax.set_xticks(range(len(conds)), [COND_LABEL.get(c, c) for c in conds])
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, ncol=min(len(models), 4), loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _confusion_plot(model: str, testset: str, condition: str, path):
    from . import config as C
    from .evaluate import confusion, load_preds

    paths = sorted(C.PREDS_DIR.glob(f"{model}__{testset}__{condition}*.npz"))
    if not paths:
        return False
    cm = sum(confusion(*load_preds(p), len(C.TARGET_CLASSES)) for p in paths)
    rate = cm / cm.sum(1, keepdims=True).clip(min=1)
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", [SURFACE, "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
    fig, ax = plt.subplots(figsize=(8.2, 7))
    im = ax.imshow(rate, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    labels = C.TARGET_CLASSES
    ax.set_xticks(range(len(labels) + 1), labels + ["검출\n실패"], fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("예측")
    ax.set_ylabel("정답")
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("정답별 비율")
    name = MODEL_STYLE.get(model, ("", model))[1]
    seeds = f" (시드 {len(paths)}개 합산)" if len(paths) > 1 else ""
    ax.set_title(f"혼동행렬: {name}, {testset} {COND_LABEL.get(condition, condition)}{seeds}", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    from . import config as C
    from .evaluate import accuracy_drop, summarize

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    s = summarize(C.PREDS_DIR)
    s.to_csv(C.RESULTS_DIR / "summary.csv", index=False)
    drops = []
    for testset, base in (("synth", "clean"), ("custom", "normal")):
        d = accuracy_drop(s[s["testset"] == testset], base=base)
        drops.append(d[d["condition"] != base])
    drop = pd.concat(drops, ignore_index=True)
    drop.to_csv(C.RESULTS_DIR / "drop.csv", index=False)
    mean = seed_mean(s)
    mean.to_csv(C.RESULTS_DIR / "summary_seed_mean.csv", index=False)
    dmean = seed_mean(drop) if len(drop) else drop
    for testset in ("synth", "custom"):
        if not (mean["testset"] == testset).any():
            print(f"{testset} 예측 없음: 그림 생략")
            continue
        m = mean[mean["testset"] == testset]
        title = "합성 변형별 정확도 (e2e, 95% CI)" if testset == "synth" else "직접 촬영 환경별 정확도 (e2e, 95% CI)"
        _grouped_bars(m, "acc", testset, MAIN_MODELS, title, "정확도", C.FIGURES_DIR / f"acc_by_condition_{testset}.png",
                      err=("ci_lo", "ci_hi"))
        dm = dmean[dmean["testset"] == testset]
        base = "원본" if testset == "synth" else "일반 조명"
        _grouped_bars(dm, "drop_pp", testset, MAIN_MODELS, f"Accuracy Drop ({base} 대비, %p)", "정확도 감소 (%p)",
                      C.FIGURES_DIR / f"drop_by_condition_{testset}.png")
        if testset == "synth":
            _grouped_bars(m, "acc", testset, SUPP_MODELS, "RF 변형 비교: 특징·좌우 정렬·CLAHE", "정확도",
                          C.FIGURES_DIR / "acc_rf_variants_synth.png", err=("ci_lo", "ci_hi"))
            _grouped_bars(m.assign(det=1 - m["fail_rate"]), "det", testset, ["rf_geom", "rf_geom_clahe"],
                          "MediaPipe 손 검출률", "검출률", C.FIGURES_DIR / "detection_rate_synth.png")
        base_cond = "clean" if testset == "synth" else "normal"
        for model in ("rf_geom", "cnn_plain", "cnn_aug"):
            _confusion_plot(model, testset, base_cond, C.FIGURES_DIR / f"confusion_{model}_{testset}.png")
    print("figures:", sorted(p.name for p in C.FIGURES_DIR.glob("*.png")))


if __name__ == "__main__":
    main()
