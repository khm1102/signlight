#!/usr/bin/env bash
# Colab GPU에서 CNN 학습/예측을 돌리고 결과(예측 npz, 모델, 학습 로그)를 받아 온다.
# 성공이든 실패든 끝나면 세션을 끈다(colab stop).
#
#   scripts/colab_cnn.sh                     # plain·aug × seed 42 43 44, 15 epochs, kaggle_val·kaggle_test·synth 예측
#   scripts/colab_cnn.sh --dry               # 업로드~다운로드 동작 확인(plain, seed 42, 1 epoch, 축소 데이터)
#   scripts/colab_cnn.sh --predict-only --testsets custom   # 로컬 models/cnn_*.keras를 올려 예측만
#   scripts/colab_cnn.sh --notebook          # notebooks/02_train_cnn_colab.ipynb를 Colab에서 실행(--dry면 SMOKE 모드)
#
# 사전 준비: colab 인증(colab sessions), data/processed/index.csv
# 데이터는 Colab VM이 kagglehub로 직접 받는다(공개 데이터셋, 인증 불필요, Colab 캐시 사용).
# 1GB zip 업로드는 Broken pipe로 실패해서 쓰지 않는다(EXP-004).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SESSION=cnn GPU=T4 VARIANTS="plain aug" SEEDS="42 43 44" EPOCHS=15
TESTSETS="kaggle_val,kaggle_test,synth" DRY=0 PREDICT_ONLY=0 NOTEBOOK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry) DRY=1 ;;
    --predict-only) PREDICT_ONLY=1 ;;
    --notebook) NOTEBOOK=1 ;;
    --testsets) TESTSETS="$2"; shift ;;
    --variants) VARIANTS="$2"; shift ;;
    --seeds) SEEDS="$2"; shift ;;
    --epochs) EPOCHS="$2"; shift ;;
    --session) SESSION="$2"; shift ;;
    --gpu) GPU="$2"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
TRAIN_EXTRA="" PRED_EXTRA=""
if [[ $DRY == 1 ]]; then
  VARIANTS=plain SEEDS=42 EPOCHS=1 TESTSETS=synth TRAIN_EXTRA="--per-class 20" PRED_EXTRA="--per-class 5"
fi

R=/content/signlight
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p results/logs
log() { echo "[$(date +%H:%M:%S)] $*"; }
rbash() {  # 원격 bash 실행 (출력이 없어도 끊기지 않게 timeout 0)
  # colab exec는 셀이 실패해도 종료 코드 0을 돌려주므로 출력에서 실패를 찾아 멈춘다.
  # 성공 표시(마커)가 출력에 있어야만 통과시킨다(fail-closed): 커널 종료·연결 끊김도 실패로 본다.
  local out marker; out=$(mktemp); marker="__RBASH_OK_${RANDOM}${RANDOM}__"
  printf '%%%%bash\nset -euo pipefail\nexport SIGNLIGHT_ROOT=%s PYTHONUNBUFFERED=1\n%s\necho %s\n' "$R" "$1" "$marker" \
    | colab exec -s "$SESSION" --timeout 0 2>&1 | tee "$out"
  if grep -q "CalledProcessError" "$out" || ! grep -Eq "^${marker}[[:space:]]*$" "$out"; then
    rm -f "$out"; log "원격 명령 실패 — 중단"; exit 1
  fi
  rm -f "$out"
}

log "wheel 빌드"
rm -rf dist && uv build --wheel -q -o dist
WHEEL=$(ls dist/signlight-*.whl | head -1)

cleanup() {
  log "세션 로그 저장 후 종료"
  colab log -s "$SESSION" -o "results/logs/colab_${STAMP}.md" || true
  colab stop -s "$SESSION" || true
}
trap cleanup EXIT  # colab new 전에 걸어서, 생성 도중 실패·중단돼도 세션을 끈다
log "세션 생성: $SESSION ($GPU)"
colab new -s "$SESSION" --gpu "$GPU"

rbash "mkdir -p $R/dist $R/data/raw/_download $R/data/processed $R/models $R/results/preds $R/results/logs; nvidia-smi --query-gpu=name,memory.total --format=csv; python --version"

log "업로드: wheel, index.csv"
colab upload -s "$SESSION" "$WHEEL" "$R/dist/$(basename "$WHEEL")"
colab upload -s "$SESSION" data/processed/index.csv "$R/data/processed/index.csv"
if [[ $PREDICT_ONLY == 1 ]]; then
  for v in $VARIANTS; do for s in $SEEDS; do
    colab upload -s "$SESSION" "models/cnn_${v}_s${s}.keras" "$R/models/cnn_${v}_s${s}.keras"
  done; done
  if [[ ",$TESTSETS," == *",custom,"* ]]; then
    tar czf /tmp/signlight_custom.tgz -C data custom_test
    colab upload -s "$SESSION" /tmp/signlight_custom.tgz "$R/custom.tgz"
    rbash "tar xzf $R/custom.tgz -C $R/data"
  fi
fi

if [[ $NOTEBOOK == 1 ]]; then
  # 노트북이 설치·데이터 준비·학습·예측·결과 묶기(out.tgz)를 모두 한다. 마지막 셀이 "KB ->"를 출력해야 성공으로 본다.
  NB=notebooks/02_train_cnn_colab.ipynb
  if [[ $DRY == 1 ]]; then
    NB_DRY=$(mktemp -d)/02_smoke.ipynb
    uv run python - "$NB" "$NB_DRY" <<'PY'
import sys, nbformat
nb = nbformat.read(sys.argv[1], 4)
cell = next(c for c in nb.cells if c.cell_type == "code" and "SMOKE = False" in c.source)
cell.source = cell.source.replace("SMOKE = False", "SMOKE = True").replace("['plain', 'aug']", "['plain']").replace("[42, 43, 44]", "[42]")
nbformat.write(nb, sys.argv[2])
PY
    NB=$NB_DRY
  fi
  log "노트북 실행: $NB"
  out=$(mktemp)
  colab exec -s "$SESSION" --timeout 0 -f "$NB" 2>&1 | tee "$out"
  if grep -q -E "Traceback|CalledProcessError" "$out" || ! grep -q "KB ->" "$out"; then
    rm -f "$out"; log "노트북 실행 실패 — 중단"; exit 1
  fi
  rm -f "$out"
  tmp=$(mktemp -d)
  colab download -s "$SESSION" "$R/out.tgz" "$tmp/out.tgz"
  if [[ $DRY == 1 ]]; then tar tzf "$tmp/out.tgz"; else tar xzf "$tmp/out.tgz" -C "$ROOT"; fi
  rm -rf "$tmp"
  log "완료"
  exit 0
fi

log "원격 준비: 데이터(kagglehub), 패키지 설치"
rbash "pip install -q --no-deps --ignore-requires-python $R/dist/$(basename "$WHEEL") && pip install -q pillow-heif
python - <<'PY'
# /kaggle/input은 NFS라 파일당 ~12ms. 대상 클래스를 스레드 32개로 로컬 디스크에 한 번 복사한다.
import os, shutil, time, kagglehub
from concurrent.futures import ThreadPoolExecutor
from signlight import config as C
src = kagglehub.dataset_download('grassknoted/asl-alphabet').strip().splitlines()[-1]
t = time.time()
jobs = []
for cls in C.TARGET_CLASSES:
    s_dir = os.path.join(src, 'asl_alphabet_train', 'asl_alphabet_train', cls)
    d_dir = C.ASL_TRAIN_DIR / cls
    d_dir.mkdir(parents=True, exist_ok=True)
    jobs += [(os.path.join(s_dir, f), d_dir / f) for f in os.listdir(s_dir)]
with ThreadPoolExecutor(32) as ex:
    list(ex.map(lambda j: shutil.copyfile(*j), jobs))
shutil.copytree(os.path.join(src, 'asl_alphabet_test', 'asl_alphabet_test'), C.ASL_TEST_DIR, dirs_exist_ok=True)
print('copied', len(jobs), 'files in', round(time.time() - t), 's')
PY
python - <<'PY'
import tensorflow as tf
from signlight import config as C
from signlight.data import load_index
idx = load_index()
missing = [p for p in idx['relpath'] if not (C.ASL_ROOT / p).exists()]
print('tf', tf.__version__, tf.config.list_physical_devices('GPU'), 'index', len(idx), 'missing', len(missing))
assert not missing, missing[:5]
PY"

fetch() {  # 원격 결과(results/, models/cnn_*)를 받아 와서 저장소에 푼다(dry면 목록만)
  rbash "cd $R && tar czf out.tgz results/preds results/logs \$(ls models/cnn_*.keras 2>/dev/null)"
  local tmp; tmp=$(mktemp -d)
  colab download -s "$SESSION" "$R/out.tgz" "$tmp/out.tgz"
  if [[ $DRY == 1 ]]; then tar tzf "$tmp/out.tgz"; else tar xzf "$tmp/out.tgz" -C "$ROOT"; fi
  rm -rf "$tmp"
}

for v in $VARIANTS; do
  for s in $SEEDS; do
    if [[ $PREDICT_ONLY == 0 ]]; then
      log "학습: $v seed $s ($EPOCHS epochs)"
      rbash "cd $R && python -m signlight.cnn train --variant $v --seed $s --epochs $EPOCHS $TRAIN_EXTRA"
    fi
    log "예측: $v seed $s ($TESTSETS)"
    rbash "cd $R && python -m signlight.cnn predict --variant $v --seed $s --testsets $TESTSETS $PRED_EXTRA"
    log "결과 받기: $v seed $s"
    fetch
  done
done
log "완료"
