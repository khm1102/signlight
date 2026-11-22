"""notebooks/*.ipynb 생성기. 노트북 내용을 코드로 관리해서 리뷰와 재생성을 쉽게 한다.

uv run python scripts/make_notebooks.py
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell


def results_notebook() -> nbf.NotebookNode:
    cells = [
        md("# 01. 결과 정리: 수어 알파벳 인식, CNN vs 손 관절 RF\n\n"
           "이미 계산된 결과(`results/`)와 데이터로 실험 과정을 셀 단위로 따라간다. 무거운 학습은 하지 않는다.\n\n"
           "- 실행 순서와 명령: `README.md`\n- 근거 문서: `docs/REPORT.md`, `docs/experiments/EXP-001…005`"),
        code("import json\n"
             "import matplotlib.pyplot as plt\n"
             "import numpy as np\n"
             "import pandas as pd\n"
             "from IPython.display import Image, display\n\n"
             "from signlight import config as C\n"
             "from signlight.data import load_index, synth_subset\n"
             "from signlight.images import load_rgb\n"
             "from signlight.perturb import apply\n\n"
             "plt.rcParams['font.family'] = ['AppleGothic', 'NanumGothic', 'DejaVu Sans']\n"
             "plt.rcParams['axes.unicode_minus'] = False\n"
             "pd.set_option('display.precision', 3)\n"
             "print('대상 클래스', len(C.TARGET_CLASSES), ''.join(C.TARGET_CLASSES))"),

        md("## 1. 데이터와 분할 (EXP-001)\n"
           "Kaggle ASL Alphabet은 영상 연속 프레임이라 거의 같은 사진이 많다. 비슷한 사진끼리 군집으로 묶어, 군집 단위로 train/val/test를 나눴다."),
        code("idx = load_index()\n"
             "stats = json.loads((C.RESULTS_DIR / 'split_stats.json').read_text())\n"
             "print('전체', stats['n'], '| split', stats['split'], '| 군집 수', stats['n_clusters'])\n"
             "pd.DataFrame({k: {'군집 수': v['n_clusters'], '최대 군집': v['max_cluster'], **v['split']}\n"
             "              for k, v in stats['per_class'].items()}).T.head(8)"),
        code("# 클래스별 예시 사진 1장씩\n"
             "fig, axes = plt.subplots(3, 8, figsize=(16, 6.5))\n"
             "for ax, label in zip(axes.ravel(), C.TARGET_CLASSES):\n"
             "    p = idx[idx['label'] == label]['relpath'].iloc[0]\n"
             "    ax.imshow(load_rgb(C.ASL_ROOT / p)); ax.set_title(label); ax.axis('off')\n"
             "plt.tight_layout()"),

        md("## 2. 합성 환경 변형\n"
           "기획안의 실험 1~4를 합성 변형으로 근사했다. 같은 (조건, 사진, key)는 항상 같은 결과를 내므로, RF(로컬)와 CNN(Colab)이 같은 사진으로 평가받는다."),
        code("sub = synth_subset(idx)\n"
             "ex = sub[sub['label'] == 'F']  # 검출이 잘 되는 클래스(EXP-002: F 20/20)를 예시로\n"
             "key, rel = ex.index[0], ex['relpath'].iloc[0]\n"
             "img = load_rgb(C.ASL_ROOT / rel)\n"
             "fig, axes = plt.subplots(1, len(C.SYNTH_CONDITIONS), figsize=(18, 3))\n"
             "for ax, cond in zip(axes, C.SYNTH_CONDITIONS):\n"
             "    ax.imshow(apply(cond, img, key)); ax.set_title(cond); ax.axis('off')\n"
             "plt.tight_layout()"),

        md("## 3. MediaPipe 손 관절 검출 (EXP-002)\n"
           "RF는 MediaPipe가 찾은 관절 21점으로 분류한다. 손을 못 찾으면 오답으로 센다."),
        code("from signlight.landmarks import HandDetector\n\n"
             "CHAINS = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]\n"
             "fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))\n"
             "with HandDetector() as det:\n"
             "    for ax, cond in zip(axes, ['clean', 'dark_0.5', 'clutter', 'scale_small']):\n"
             "        im = apply(cond, img, key)\n"
             "        lm, right = det.detect(im)\n"
             "        ax.imshow(im); ax.axis('off')\n"
             "        if lm is None:\n"
             "            ax.set_title(f'{cond}: 검출 실패')\n"
             "            continue\n"
             "        xy = lm[:, :2] * [im.shape[1], im.shape[0]]\n"
             "        for ch in CHAINS:\n"
             "            ax.plot(xy[list(ch), 0], xy[list(ch), 1], '-o', ms=3, lw=1.5, color='#2a78d6')\n"
             "        ax.set_title(f\"{cond}: {'오른손' if right else '왼손'}\")\n"
             "plt.tight_layout()"),
        code("det = pd.read_csv(C.RESULTS_DIR / 'detection_rate.csv')\n"
             "table = det.pivot(index='label', columns='split', values='det_rate')[['train', 'val', 'test']]\n"
             "print('전체 검출률(가중):', round((det['n'] * det['det_rate']).sum() / det['n'].sum(), 3))\n"
             "table.sort_values('test').head(6)  # test 검출률이 낮은 클래스: 촬영 회차 효과(EXP-005)"),

        md("## 4. 환경별 정확도 (EXP-003, 004)\n"
           "e2e 정확도다(검출 실패 = 오답). CNN은 시드 3개 평균이다."),
        code("mean = pd.read_csv(C.RESULTS_DIR / 'summary_seed_mean.csv')\n"
             "synth = mean[mean['testset'] == 'synth'].pivot(index='condition', columns='model', values='acc')\n"
             "synth.loc[C.SYNTH_CONDITIONS, ['rf_geom', 'dt_geom', 'cnn_plain', 'cnn_aug', 'rf_geom_clahe']]"),
        code("drop = pd.read_csv(C.RESULTS_DIR / 'drop.csv')\n"
             "d = drop[drop['testset'] == 'synth'].groupby(['condition', 'model'])['drop_pp'].mean().unstack()\n"
             "d.loc[C.SYNTH_CONDITIONS[1:], ['rf_geom', 'dt_geom', 'cnn_plain', 'cnn_aug']].round(1)  # 원본 대비 감소 폭(%p)"),
        code("for name in ['acc_by_condition_synth', 'drop_by_condition_synth', 'detection_rate_synth']:\n"
             "    display(Image(filename=str(C.FIGURES_DIR / f'{name}.png'), width=900))"),

        md("## 5. 혼동행렬과 오류 분석 (EXP-005)"),
        code("from signlight.evaluate import confusion, load_preds\n\n"
             "def top_confusions(files, k=6):\n"
             "    cm = sum(confusion(*load_preds(C.PREDS_DIR / f), 24) for f in files)\n"
             "    off = cm[:, :24].copy(); np.fill_diagonal(off, 0)\n"
             "    rate = off / cm.sum(1, keepdims=True)\n"
             "    pairs = sorted(((rate[i, j], C.TARGET_CLASSES[i], C.TARGET_CLASSES[j]) for i in range(24) for j in range(24) if off[i, j]), reverse=True)[:k]\n"
             "    return [(f'{a}→{b}', round(float(r), 2)) for r, a, b in pairs], round(float(cm[:, 24].sum() / cm.sum()), 3)\n\n"
             "print('RF  ', *top_confusions(['rf_geom__synth__clean.npz']))\n"
             "print('CNN+aug', *top_confusions([f'cnn_aug__synth__clean__s{s}.npz' for s in C.CNN_SEEDS]))"),
        code("for name in ['confusion_rf_geom_synth', 'confusion_cnn_aug_synth']:\n"
             "    display(Image(filename=str(C.FIGURES_DIR / f'{name}.png'), width=650))"),

        md("## 6. 직접 촬영 테스트셋 상태\n"
           "팀원 사진이 `data/custom_test/`에 들어오면 `docs/REPORT.md` 6절의 명령으로 평가한다."),
        code("from signlight.custom_test import report, scan\n\n"
             "df = scan()\n"
             "print('직접 촬영 사진:', len(df), '장')\n"
             "if len(df):\n"
             "    display(report(df))"),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def colab_notebook() -> nbf.NotebookNode:
    cells = [
        md("# 02. CNN 학습 (Colab GPU)\n\n"
           "`signlight` 패키지(wheel)와 `index.csv`를 이용해 CNN을 학습하고, Kaggle val/test와 합성 변형셋으로 예측한다. "
           "로컬 RF와 **같은 분할, 같은 변형 사진**으로 평가한다.\n\n"
           "**실행 방법**\n"
           "- Colab CLI: `scripts/colab_cnn.sh --notebook`이 wheel과 index.csv를 올리고 이 노트북을 실행한 뒤 결과를 받아 온다.\n"
           "- Colab 웹: 런타임을 T4 GPU로 바꾸고, 왼쪽 파일 패널에서 `/content/`에 `signlight-0.1.0-py3-none-any.whl`(`uv build --wheel`로 생성)과 "
           "`index.csv`(`data/processed/index.csv`)를 올린 뒤 위에서부터 실행한다."),
        code("# 설정: 학습할 변형과 시드. SMOKE=True면 클래스당 20장, 1 epoch로 동작만 확인한다.\n"
             "VARIANTS = ['plain', 'aug']\n"
             "SEEDS = [42, 43, 44]\n"
             "EPOCHS = 15\n"
             "SMOKE = False"),
        code("import glob, os, shutil, subprocess, sys, time\n"
             "from concurrent.futures import ThreadPoolExecutor\n\n"
             "ROOT = '/content/signlight'\n"
             "os.environ['SIGNLIGHT_ROOT'] = ROOT\n"
             "os.makedirs(f'{ROOT}/data/processed', exist_ok=True)\n"
             "if os.path.exists('/content/index.csv'):\n"
             "    shutil.copy('/content/index.csv', f'{ROOT}/data/processed/index.csv')\n"
             "wheel = sorted(glob.glob('/content/signlight-*.whl') + glob.glob(f'{ROOT}/dist/signlight-*.whl'))[-1]\n"
             "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--ignore-requires-python', wheel], check=True)\n"
             "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pillow-heif'], check=True)\n"
             "print('installed', os.path.basename(wheel))"),
        code("# 데이터: kagglehub(공개 데이터셋, 인증 불필요, Colab 캐시). /kaggle/input은 NFS라 대상 클래스만 로컬 디스크로 복사한다.\n"
             "import kagglehub\n"
             "from signlight import config as C\n\n"
             "src = kagglehub.dataset_download('grassknoted/asl-alphabet')\n"
             "t = time.time(); jobs = []\n"
             "for cls in C.TARGET_CLASSES:\n"
             "    s_dir = os.path.join(src, 'asl_alphabet_train', 'asl_alphabet_train', cls)\n"
             "    d_dir = C.ASL_TRAIN_DIR / cls\n"
             "    d_dir.mkdir(parents=True, exist_ok=True)\n"
             "    jobs += [(os.path.join(s_dir, f), d_dir / f) for f in os.listdir(s_dir) if not (d_dir / f).exists()]\n"
             "with ThreadPoolExecutor(32) as ex:\n"
             "    list(ex.map(lambda j: shutil.copyfile(*j), jobs))\n"
             "print('copied', len(jobs), 'files in', round(time.time() - t), 's')"),
        code("import tensorflow as tf\n"
             "from signlight.data import load_index\n\n"
             "idx = load_index()\n"
             "missing = [p for p in idx['relpath'] if not (C.ASL_ROOT / p).exists()]\n"
             "print('tf', tf.__version__, tf.config.list_physical_devices('GPU'))\n"
             "print('index', len(idx), 'missing', len(missing))\n"
             "assert not missing, missing[:5]"),
        md("## 학습과 예측\n각 (변형, 시드)마다 학습한 뒤 `kaggle_val`, `kaggle_test`, 합성 7조건 예측을 `results/preds/`에 저장한다."),
        code("from signlight.cnn import predict, train\n\n"
             "testsets = ['synth'] if SMOKE else ['kaggle_val', 'kaggle_test', 'synth']\n"
             "for variant in VARIANTS:\n"
             "    for seed in SEEDS:\n"
             "        t = time.time()\n"
             "        train(variant, seed, 1 if SMOKE else EPOCHS, per_class=20 if SMOKE else None)\n"
             "        predict(variant, seed, testsets, per_class=5 if SMOKE else 100)\n"
             "        print(f'== {variant} s{seed} done in {time.time() - t:.0f}s', flush=True)"),
        md("## 결과 확인"),
        code("import json\n"
             "import matplotlib.pyplot as plt\n"
             "from signlight.evaluate import summarize\n"
             "from signlight.report import seed_mean\n\n"
             "s = summarize()\n"
             "m = seed_mean(s[s['model'].str.startswith('cnn')])\n"
             "print(m.pivot_table(index=['testset', 'condition'], columns='model', values='acc').round(3))\n\n"
             "fig, ax = plt.subplots(figsize=(8, 4))\n"
             "for f in sorted((C.RESULTS_DIR / 'logs').glob('cnn_*_history.json')):\n"
             "    ax.plot(json.loads(f.read_text())['val_accuracy'], label=f.stem.replace('_history', ''))\n"
             "ax.set_xlabel('epoch'); ax.set_ylabel('val accuracy (그룹 분할)'); ax.legend(fontsize=8)\n"
             "plt.show()"),
        code("# 결과 묶기: Colab CLI는 이 파일을 받아 저장소에 푼다. Colab 웹에서는 파일 패널에서 내려받는다.\n"
             "subprocess.run(f'cd {ROOT} && tar czf out.tgz results/preds results/logs $(ls models/cnn_*.keras)', shell=True, check=True)\n"
             "print(os.path.getsize(f'{ROOT}/out.tgz') // 1024, 'KB ->', f'{ROOT}/out.tgz')"),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
    return nb


if __name__ == "__main__":
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    nbf.write(results_notebook(), out / "01_results.ipynb")
    nbf.write(colab_notebook(), out / "02_train_cnn_colab.ipynb")
    print("wrote", sorted(p.name for p in out.glob("*.ipynb")))
