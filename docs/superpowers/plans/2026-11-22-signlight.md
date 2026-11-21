# signlight 전체 구현 플랜: 기획부터 최종 보고까지

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kaggle ASL 정적 알파벳 24개로 CNN(픽셀)과 RF/DT(MediaPipe 관절 좌표)를 학습한다. 같은 테스트 이미지에서 조명·배경·손 위치 변화에 따른 정확도와 Accuracy Drop을 비교하고, 그 결과로 보고서와 발표 자료를 만든다.

**Architecture:** `src/signlight/`에 이미지 입출력, 데이터 인덱스·분할, 관절 추출, 특징, 합성 변형, 평가를 모듈로 나눈다. RF/DT는 로컬(macOS)에서 학습한다. CNN은 같은 패키지를 wheel로 만들어 Colab GPU에 올려 학습한다. 두 모델의 예측은 같은 형식(`results/preds/*.npz`)으로 저장하고, `evaluate.py` 하나로 집계한다.

**Tech Stack:** Python 3.12, uv, TensorFlow/Keras 2.21(Colab에서는 런타임 기본 TF), MediaPipe 1.2.x tasks `HandLandmarker`, scikit-learn, OpenCV, Pillow(+pillow-heif), pandas, matplotlib/seaborn, pytest, Colab CLI(`colab`), Kaggle CLI(`kaggle`).

**Spec:** `docs/proposal.md` (원본 기획안 PPT를 옮긴 것. 클래스를 24개로 늘린 변경도 반영돼 있다)

## Global Constraints

- Python `>=3.12,<3.13`. 패키지 추가는 `uv add`, 실행은 `uv run`으로만 한다. `pip`는 쓰지 않는다(Colab VM 안은 예외).
- 대상 클래스는 `signlight.config.TARGET_CLASSES`(J·Z를 뺀 24개)다. **라벨 인덱스는 이 리스트의 순서**다. RF, DT, CNN, 직접 촬영셋 모두 이 순서를 쓴다.
- 전역 시드는 `config.SEED = 42`다. CNN 반복 시드는 `[42, 43, 44]`다.
- MediaPipe는 tasks API(`mediapipe.tasks.python.vision.HandLandmarker`)를 쓴다. 모델 파일은 `models/mediapipe/hand_landmarker.task`다. `mp.solutions`는 1.2.x에 없다.
- **검출 실패는 오답이다.** RF/DT의 예측값은 `-1`로 두고, 정확도 분모는 항상 전체 샘플 수로 한다.
- 두 모델은 **동일한 테스트 이미지**로 평가한다(같은 행, 같은 변형 시드).
- Kaggle 데이터는 연속 프레임이라 근접 중복이 매우 많다(사전 측정: 가장 가까운 이미지와의 dHash 해밍 거리가 25 이하인 비율 99.5%). Kaggle 내부 val/test 정확도는 **낙관적 수치**라고 항상 같이 적는다. 주 결론은 합성 변형과 직접 촬영셋에서 낸다.
- `data/`와 `models/`의 데이터·모델 파일은 git에 올리지 않는다. 결과 수치(`results/*.csv`, `results/preds/*.npz`)와 그림(`reports/figures/*.png`)은 올린다.
- Colab은 `--gpu T4`로 띄운다. 작업이 끝나거나 실패하면 **반드시 `colab stop`**으로 끈다(사용량 과금).
- 문서와 보고는 한국어로 쓴다. 코드 식별자는 영어로 쓴다.
- 실험 문서(`docs/experiments/EXP-NNN-*.md`)에는 git 커밋 해시, 시드, 설정값, 결과 파일 경로를 적는다. 결과가 예상과 달라도 지우지 않는다.
- 각 Task가 끝날 때마다 `docs/PROGRESS.md`를 갱신한다(완료 Task, 다음 Task, 막힌 점). 실행이 중간에 끊겨도 이 파일을 읽고 이어 간다.

## Review Focus

1. **휴대폰 사진의 EXIF 회전.** 세로로 찍은 사진이 옆으로 누운 채 로드되면 검출과 CNN이 모두 실패한다. `load_rgb`는 EXIF 방향대로 바로 세워야 한다(Task 2 `test_load_rgb_applies_exif_orientation`).
2. **iPhone HEIC, 대문자 확장자(`.JPG`, `.HEIC`), 정사각형이 아닌 사진.** 모두 로드돼야 하고, CNN 입력은 잘라내지 않고 여백을 채워 정사각형으로 만든다(Task 2 `test_load_rgb_reads_heic`, `test_pad_to_square_keeps_whole_image`, Task 14 `test_scan_accepts_uppercase_extensions`).
3. **왼손으로 찍은 사진.** 학습 데이터는 한 손 위주다. 왼손 관절은 좌우 반전해 오른손 기준으로 맞춘 뒤 특징을 만든다(Task 7 `test_canonicalize_mirrors_left_hand`).
4. **직접 촬영 폴더 이름 오타**(소문자 `a`, 대상이 아닌 `J`, 모르는 조건 폴더 `dark2`). 조용히 건너뛰면 결과가 틀린 줄 모르게 된다. 무엇이 문제인지 알려 주는 오류를 내야 한다(Task 14 `test_scan_rejects_unknown_folders`).
5. **검출 실패를 분모에서 빼 버리는 실수.** 이러면 RF 정확도가 부풀려진다. e2e 정확도의 분모는 전체여야 한다(Task 8 `test_e2e_accuracy_counts_failures_as_wrong`).

## 실행 규칙 (goal 실행 에이전트용)

- Task는 번호 순서대로 진행한다. 각 Task의 Interfaces 블록에 적힌 이름과 시그니처를 바꾸지 않는다.
- 🙋 표시가 붙은 Task는 **사람이 해야 하는 일**이다(사진 촬영, 확인). 준비가 안 됐으면 `PROGRESS.md`에 "대기"로 적고, 건너뛸 수 있는 다음 Task로 넘어간다. 대기 중인 Task가 끝나면 해당 결과를 반영해 Task 16~17을 다시 돌린다.
- 오래 걸리는 단계(전체 관절 추출, Colab 학습)는 백그라운드로 돌리고 로그 파일을 남긴다.
- 테스트는 `uv run pytest -q`로 돌린다. 실제 이미지나 모델 파일이 필요한 테스트는 파일이 없으면 `pytest.skip`으로 건너뛴다.

## 파일 구조

| 파일 | 책임 |
|---|---|
| `src/signlight/config.py` | 경로(환경변수 `SIGNLIGHT_ROOT`로 바꿀 수 있음), 클래스, 시드, 검출 설정, 조건 목록 |
| `src/signlight/images.py` | 이미지 로드(EXIF, HEIC), 정사각형 패딩, 리사이즈 |
| `src/signlight/data.py` | dHash, 근접 중복 군집, 인덱스 생성, 그룹 분할, 변형 평가용 부분집합 |
| `src/signlight/landmarks.py` | MediaPipe HandLandmarker 래퍼, 일괄 추출 |
| `src/signlight/features.py` | 좌우 정렬, 정규화, geom 특징 |
| `src/signlight/perturb.py` | 결정적 합성 환경 변형 |
| `src/signlight/evaluate.py` | e2e 정확도, 부트스트랩 CI, 혼동행렬, Accuracy Drop, 예측 파일 입출력, 집계 |
| `src/signlight/rf.py` | RF/DT 학습·평가 CLI |
| `src/signlight/cnn.py` | CNN 모델, 입력 파이프라인, 학습·예측 CLI |
| `src/signlight/custom_test.py` | 직접 촬영셋 스캔·검증, 평가 입력 생성 |
| `src/signlight/report.py` | 요약 표와 그림 생성 |
| `scripts/colab_cnn.sh` | Colab 세션 생성, 업로드, 학습, 다운로드, 종료 |
| `tests/test_*.py` | 모듈별 테스트 |
| `docs/PROGRESS.md`, `docs/experiments/`, `docs/PHOTO_PROTOCOL.md`, `docs/REPORT.md` | 진행 기록, 실험 기록, 촬영 안내, 최종 보고서 |

---

## Phase 0. 기반

### Task 1: 테스트 환경, 설정 보강, 진행 기록 시작

**Files:**
- Modify: `pyproject.toml` (dev에 `pytest` 추가, 의존성에 `pillow-heif` 추가)
- Modify: `src/signlight/config.py`
- Create: `tests/test_config.py`, `docs/PROGRESS.md`

**Interfaces:**
- Produces: `config.ROOT`(환경변수 `SIGNLIGHT_ROOT`가 있으면 그 경로, 없으면 저장소 루트), `config.ASL_ROOT = RAW_DIR / "asl_alphabet"`, `config.RESULTS_DIR = ROOT / "results"`, `config.PREDS_DIR = RESULTS_DIR / "preds"`, `config.CNN_SIZE = 128`, `config.CNN_SEEDS = [42, 43, 44]`, `config.DET_CONF = 0.3`, `config.DET_PAD = 0.0`(Task 5에서 확정), `config.HAND_MODEL = MODELS_DIR / "mediapipe" / "hand_landmarker.task"`, `config.SYNTH_CONDITIONS`(Task 9의 조건 이름 리스트)

- [ ] **Step 1: 의존성 추가.** `uv add pillow-heif` 후 `uv add --dev pytest`
- [ ] **Step 2: 실패하는 테스트 작성** (`tests/test_config.py`)

```python
def test_target_classes_are_24_static_letters():
    from signlight import config as C
    assert len(C.TARGET_CLASSES) == 24
    assert "J" not in C.TARGET_CLASSES and "Z" not in C.TARGET_CLASSES
    assert C.TARGET_CLASSES == sorted(C.TARGET_CLASSES)

def test_root_override(monkeypatch, tmp_path):
    import importlib
    monkeypatch.setenv("SIGNLIGHT_ROOT", str(tmp_path))
    from signlight import config
    C = importlib.reload(config)
    assert C.ROOT == tmp_path and C.DATA_DIR == tmp_path / "data"
    monkeypatch.delenv("SIGNLIGHT_ROOT"); importlib.reload(config)
```

- [ ] **Step 3: 실패 확인.** `uv run pytest tests/test_config.py -q` → `test_root_override` FAIL
- [ ] **Step 4: `config.py`에 위 Produces 항목 추가.** ROOT는 `Path(os.environ.get("SIGNLIGHT_ROOT", <기존 parents[2]>))`
- [ ] **Step 5: 통과 확인.** `uv run pytest -q` → PASS
- [ ] **Step 6: `docs/PROGRESS.md` 작성.** 섹션은 `## 현재 상태`, `## 완료`, `## 다음`, `## 막힌 점/대기`. 환경 설정 완료(uv, Colab CLI 인증, Kaggle CLI, 29개 클래스 압축 해제)와 Kaggle 사전 측정치(검출률 78%, 중복 99.5%)를 적는다.
- [ ] **Step 7: 첫 커밋.** 이 저장소의 첫 커밋이므로 전체를 올린다. `git add -A && git status`로 `data/`·`models/` 아래 데이터 파일이 빠졌는지 확인한 뒤 `git commit -m "chore: 프로젝트 초기 설정과 플랜"`

---

## Phase 1. 데이터

### Task 2: 이미지 입출력

**Files:**
- Create: `src/signlight/images.py`, `tests/test_images.py`

**Interfaces:**
- Produces:
  - `load_rgb(path: Path) -> np.ndarray`: (H,W,3) uint8 RGB. EXIF 방향을 적용하고 HEIC를 읽는다(`pillow_heif.register_heif_opener()`).
  - `pad_to_square(img: np.ndarray) -> np.ndarray`: 짧은 변 쪽에 가장자리 평균색 여백을 넣는다. 원본 픽셀은 전부 남긴다.
  - `resize(img: np.ndarray, size: int) -> np.ndarray`: `cv2.INTER_AREA`
  - `to_model_input(img: np.ndarray, size: int) -> np.ndarray`: `resize(pad_to_square(img), size)`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_load_rgb_applies_exif_orientation(tmp_path):
    # H=2, W=4 이미지(왼쪽 열 빨강)를 EXIF Orientation=6(시계 방향 90° 회전 필요)으로 저장
    out = load_rgb(path)
    assert out.shape == (4, 2, 3)          # 회전 후 H=4, W=2
    assert out[0, -1, 0] > 200             # 빨간 왼쪽 열이 회전 후 윗줄 오른쪽 끝으로 감

def test_load_rgb_reads_heic(tmp_path):   # pillow_heif로 저장한 .HEIC 를 읽어 (H,W,3) uint8
def test_load_rgb_drops_alpha(tmp_path):  # RGBA PNG → 3채널
def test_pad_to_square_keeps_whole_image():
    img = np.zeros((100, 60, 3), np.uint8); img[:, :] = 50
    sq = pad_to_square(img)
    assert sq.shape == (100, 100, 3) and (sq == 50).all()   # 원본을 잘라 내지 않음
def test_to_model_input_shape():
    assert to_model_input(np.zeros((300, 200, 3), np.uint8), 128).shape == (128, 128, 3)
```

(EXIF 테스트는 Pillow로 `Image.Exif()`에 `0x0112: 6`을 넣어 JPEG으로 저장해 만든다. 기대값은 위 두 assert로 고정한다.)

- [ ] **Step 2: 실패 확인.** `uv run pytest tests/test_images.py -q` → ImportError
- [ ] **Step 3: `images.py` 구현.** `ImageOps.exif_transpose` 다음 `.convert("RGB")`
- [ ] **Step 4: 통과 확인.** `uv run pytest tests/test_images.py -q` → 5 passed
- [ ] **Step 5: 커밋.** `git commit -m "feat: 이미지 로드(EXIF, HEIC)와 정사각형 패딩"`

### Task 3: 인덱스, 근접 중복 군집, 그룹 분할 (EXP-001)

**Files:**
- Create: `src/signlight/data.py`, `tests/test_data.py`, `docs/experiments/EXP-001-index-and-split.md`
- Output: `data/processed/index.csv`(git 제외), `results/split_stats.json`

**Interfaces:**
- Consumes: `images.load_rgb`
- Produces:
  - `dhash(img: np.ndarray, n: int = 16) -> np.ndarray`: bool (n*n,)
  - `cluster(H: np.ndarray, thr: int) -> np.ndarray`: 해밍 거리 `<= thr`인 쌍을 union-find로 묶은 군집 id (int, 0부터)
  - `build_index(asl_root: Path, classes: list[str], thr: int = 20, seed: int = 42) -> pd.DataFrame`: 컬럼 `relpath`(asl_root 기준, 예: `train/A/A1.jpg`), `label`, `y`(TARGET_CLASSES 인덱스), `group`(전체에서 유일), `split`(`train`/`val`/`test`)
  - `load_index() -> pd.DataFrame`: `PROCESSED_DIR / "index.csv"`
  - `synth_subset(index: pd.DataFrame, per_class: int = 100, seed: int = 42) -> pd.DataFrame`: test 행에서 클래스별 `per_class`개를 뽑는다. 순서는 고정한다(`y`, `relpath` 정렬).
  - CLI: `uv run python -m signlight.data`는 index.csv를 만들고 split_stats.json을 쓴다.
- 분할 방법: 군집은 클래스마다 따로 만들고 group id를 전체에서 유일하게 둔다. `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)`에서 fold 0을 test, fold 1을 val, 나머지를 train으로 한다(약 60/20/20).

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_dhash_identical_images_distance_zero()
def test_cluster_merges_chain():
    # H 3개: a-b 거리 1, b-c 거리 1, a-c 거리 2, thr=1 → 셋이 한 군집(체인 병합)
    assert len(set(cluster(H, 1))) == 1
def test_build_index_groups_do_not_cross_splits(tmp_path):
    # tmp asl_root/train/{A,B,C}/ 에 클래스당 12장(같은 이미지 3장씩 4묶음) 생성
    idx = build_index(tmp_path, ["A","B","C"], thr=0, seed=0)
    assert idx.groupby("group")["split"].nunique().max() == 1
    assert set(idx["split"]) == {"train", "val", "test"}
    assert idx["y"].tolist() == [["A","B","C"].index(l) for l in idx["label"]]
def test_synth_subset_is_deterministic_and_test_only()
```

- [ ] **Step 2: 실패 확인.** `uv run pytest tests/test_data.py -q` → FAIL
- [ ] **Step 3: `data.py` 구현.** 해밍 행렬은 `s[:,None]+s[None,:]-2*H@H.T`(float32)를 클래스별로(3,000×3,000) 계산한다. 이미지는 `ProcessPoolExecutor`로 병렬로 읽는다.
- [ ] **Step 4: 통과 확인.** `uv run pytest tests/test_data.py -q` → PASS
- [ ] **Step 5: 실제 실행.** `uv run python -m signlight.data`. 기대 결과는 72,000행, 클래스당 3,000, split 비율 약 0.6/0.2/0.2(±0.05)이고, `results/split_stats.json`에 클래스별 군집 수와 최대 군집 크기가 들어간다.
- [ ] **Step 6: EXP-001 작성.** 군집 수, 최대 군집 비율, split 크기, "Kaggle 내부 분할도 여전히 낙관적"이라는 한계를 적는다.
- [ ] **Step 7: 커밋.** `git commit -m "feat: 데이터 인덱스와 근접 중복 그룹 분할 (EXP-001)"`

---

## Phase 2. 손 관절

### Task 4: HandLandmarker 래퍼

**Files:**
- Create: `src/signlight/landmarks.py`, `tests/test_landmarks.py`

**Interfaces:**
- Consumes: `config.HAND_MODEL`, `config.DET_CONF`, `config.DET_PAD`, `images.load_rgb`
- Produces:
  - `class HandDetector(conf: float = DET_CONF, pad: float = DET_PAD)`: 컨텍스트 매니저. `detect(img: np.ndarray) -> tuple[np.ndarray | None, bool | None]`는 `(lm (21,3) float32 이미지 비율 좌표, is_right)`를 돌려주고, 실패하면 `(None, None)`이다. `pad > 0`이면 각 변에 `pad × 변 길이`만큼 가장자리 평균색 여백을 넣고 검출한다(좌표는 여백을 넣은 이미지 기준이다. 정규화하면 상관없다).
  - `extract(images: Iterable[np.ndarray], n: int, conf=..., pad=..., progress: int = 0) -> dict[str, np.ndarray]`: 키 `lm`(n,21,3; 실패하면 NaN), `det`(n,) bool, `right`(n,) bool(실패하면 False)
- 설정: `RunningMode.IMAGE`, `num_hands=1`, `min_hand_detection_confidence=conf`, `min_hand_presence_confidence=conf`

- [ ] **Step 1: 실패하는 테스트 작성** (모델 파일이나 Kaggle 이미지가 없으면 skip)

```python
def test_detect_blank_image_fails():
    with HandDetector() as d:
        assert d.detect(np.zeros((200, 200, 3), np.uint8)) == (None, None)
def test_detect_real_hand():  # ASL_ROOT/train/F/F1.jpg (사전 측정 F 검출 20/20)
    lm, right = d.detect(load_rgb(p)); assert lm.shape == (21, 3) and np.isfinite(lm).all()
def test_extract_marks_failures_nan():
    out = extract([blank, hand], n=2)
    assert out["det"].tolist() == [False, True] and np.isnan(out["lm"][0]).all()
```

- [ ] **Step 2: 실패 확인 → Step 3: 구현 → Step 4: `uv run pytest tests/test_landmarks.py -q` PASS**
- [ ] **Step 5: 커밋.** `git commit -m "feat: MediaPipe HandLandmarker 래퍼"`

### Task 5: 검출 실패 분석과 검출 설정 확정 (EXP-002)

**Files:**
- Create: `docs/experiments/EXP-002-detection.md`, `reports/figures/exp002_failures.png`
- Modify: `src/signlight/config.py` (`DET_CONF`, `DET_PAD` 확정값)

**Interfaces:**
- Consumes: `data.load_index`, `landmarks.HandDetector`
- Produces: 확정된 `config.DET_CONF`, `config.DET_PAD`. 이후 모든 추출(Kaggle, 변형, 직접 촬영)이 이 값을 쓴다.

- [ ] **Step 1: 탐색 스크립트 실행.** 일회성이라 scratchpad에 둔다. **train split**에서 클래스당 50장(총 1,200장)을 뽑아 `conf ∈ {0.3, 0.5}` × `pad ∈ {0.0, 0.15, 0.3}`의 클래스별 검출률을 표로 만든다.
- [ ] **Step 2: 실패 사례 갤러리.** 기본 설정에서 실패한 이미지 24장을 격자로 저장한다(`exp002_failures.png`). 원인을 눈으로 분류한다(손이 가장자리에서 잘림, 어두움, 흐림, 손목만 보임 등).
- [ ] **Step 3: 설정 결정.** 전체 검출률이 가장 높은 조합을 고른다. 차이가 1%p 이내면 단순한 쪽(`pad` 작은 쪽, `conf` 높은 쪽)을 고른다. `config.py`에 반영하고 근거를 EXP-002에 적는다.
- [ ] **Step 4: 테스트 재실행.** `uv run pytest -q` → PASS
- [ ] **Step 5: 커밋.** `git commit -m "exp: 검출 실패 분석과 검출 설정 확정 (EXP-002)"`

### Task 6: 전체 관절 추출

**Files:**
- Modify: `src/signlight/landmarks.py` (CLI 추가)
- Output: `data/processed/landmarks.npz`(git 제외), `results/detection_rate.csv`

**Interfaces:**
- Produces: `landmarks.npz`. 키는 `lm`(72000,21,3), `det`, `right`이고, 행 순서는 `index.csv`와 같다. `load_landmarks() -> dict[str, np.ndarray]`
- CLI: `uv run python -m signlight.landmarks`

- [ ] **Step 1: 구현.** CLI는 index.csv 순서대로 `load_rgb(ASL_ROOT / relpath)`를 읽어 `extract`에 넣고, `progress=2000`으로 진행을 출력한다.
- [ ] **Step 2: 실행.** 백그라운드로 돌리고 로그를 `results/logs/landmarks.log`에 쓴다. 장당 약 9ms이므로 15분 안팎으로 예상한다.
- [ ] **Step 3: 확인.** `det.mean()`이 EXP-002에서 고른 설정의 검출률과 ±3%p 안에 있어야 한다. 클래스·split별 검출률을 `results/detection_rate.csv`로 저장하고 `right` 비율도 기록한다(좌우 정렬 판단에 쓴다).
- [ ] **Step 4: 커밋.** `git commit -m "feat: 전체 손 관절 추출"`

---

## Phase 3. RF / DT

### Task 7: 특징

**Files:**
- Create: `src/signlight/features.py`, `tests/test_features.py`

**Interfaces:**
- Produces:
  - `canonicalize(lm: np.ndarray, right: np.ndarray) -> np.ndarray`: (n,21,3). 왼손(`right=False`)인 검출 행은 `x → 1 - x`로 뒤집어 오른손 기준으로 맞춘다. NaN 행은 그대로 둔다.
  - `normalize(lm) -> np.ndarray`: 손목(0번) 기준 이동, 손목→중지 MCP(9번) 2D 거리로 스케일
  - `build(lm: np.ndarray, right: np.ndarray, kind: str = "geom") -> np.ndarray`: `kind ∈ {"raw", "norm", "geom"}`. raw 63차원, norm 63차원, geom 288차원(norm 63 + 21점 쌍거리 210 + 손가락 마디 굴곡각 15). 실패 행은 전부 NaN이다.
- 참고: geom 정의는 `~/Documents/asl-robustness/src/aslrob/features.py`와 같다(이전 실험에서 norm보다 약 10%p 높았다).
- Task 6에서 `right` 비율이 한쪽으로 95% 이상 쏠리면 `canonicalize`를 그대로 쓴다. 그렇지 않으면 handedness 라벨이 불안정하다는 뜻이라 EXP-003에서 `canonicalize` 사용/미사용을 둘 다 비교한다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_norm_invariant_to_translation_and_scale():
    lm = rng.random((1, 21, 3)); lm2 = lm * 2.5 + [0.1, -0.2, 0]
    np.testing.assert_allclose(build(lm, [True], "norm"), build(lm2, [True], "norm"), atol=1e-5)
def test_canonicalize_mirrors_left_hand():
    flipped = lm.copy(); flipped[..., 0] = 1 - flipped[..., 0]
    np.testing.assert_allclose(build(lm, [True]), build(flipped, [False]), atol=1e-5)
def test_geom_dim_and_nan_rows():
    f = build(np.full((2, 21, 3), np.nan), [False, False]); assert f.shape == (2, 288) and np.isnan(f).all()
def test_real_image_mirror_consistency():  # 실제 이미지 F1과 좌우 반전본: 검출되면 geom 거리 < 0.1*norm (없으면 skip)
```

- [ ] **Step 2: 실패 확인 → Step 3: 구현 → Step 4: `uv run pytest tests/test_features.py -q` PASS**
- [ ] **Step 5: 커밋.** `git commit -m "feat: 관절 특징(raw/norm/geom)과 좌우 정렬"`

### Task 8: 평가 지표와 예측 파일 형식

**Files:**
- Create: `src/signlight/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Produces:
  - `FAIL = -1`
  - `e2e_accuracy(y_true, y_pred) -> float`: `FAIL`은 오답이고, 분모는 `len(y_true)`
  - `bootstrap_ci(y_true, y_pred, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]`: 95% 백분위 구간
  - `confusion(y_true, y_pred, n_classes: int) -> np.ndarray`: (n_classes, n_classes+1). 마지막 열은 검출 실패다.
  - `save_preds(model: str, testset: str, condition: str, y_true, y_pred, seed: int | None = None) -> Path`: `PREDS_DIR / f"{model}__{testset}__{condition}[__s{seed}].npz"`
  - `summarize(pred_dir: Path = PREDS_DIR) -> pd.DataFrame`: 컬럼 `model, seed, testset, condition, n, acc, ci_lo, ci_hi, fail_rate`
  - `accuracy_drop(summary: pd.DataFrame, base: str = "clean") -> pd.DataFrame`: 같은 (model, seed, testset)에서 `base - condition`을 %p로 낸다. 컬럼 `drop_pp`.
- 이름 규칙: model은 `rf_geom`, `dt_geom`, `rf_norm`, `cnn_plain`, `cnn_aug` 등이다. testset은 `kaggle_val`, `kaggle_test`, `synth`, `custom`이다. condition은 Kaggle 셋이면 `clean`, synth면 Task 9의 조건 이름, custom이면 `normal`/`dark`/`complex_bg`/`position`이다(custom의 Drop 기준은 `base="normal"`).

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_e2e_accuracy_counts_failures_as_wrong():
    assert e2e_accuracy([0, 1, 2, 3], [0, 1, FAIL, FAIL]) == 0.5
def test_confusion_failure_column():
    cm = confusion([0, 1], [FAIL, 1], 2); assert cm[0, 2] == 1 and cm[1, 1] == 1
def test_accuracy_drop_pp():   # clean 0.9, dark 0.6 → drop_pp == 30.0
def test_save_and_summarize_roundtrip(tmp_path)
def test_bootstrap_ci_contains_point_estimate()
```

- [ ] **Step 2: 실패 확인 → Step 3: 구현 → Step 4: `uv run pytest tests/test_evaluate.py -q` PASS**
- [ ] **Step 5: 커밋.** `git commit -m "feat: 평가 지표와 예측 파일 형식"`

### Task 9: 합성 환경 변형

**Files:**
- Create: `src/signlight/perturb.py`, `tests/test_perturb.py`

**Interfaces:**
- Produces:
  - `CONDITIONS: dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]]`. 키 순서는 `clean, dark_0.5, dark_0.25, clutter, shift, scale_small, scale_large`이고 `config.SYNTH_CONDITIONS`와 같아야 한다.
  - `apply(name: str, img: np.ndarray, key: int) -> np.ndarray`: rng는 `np.random.default_rng([config.SEED, CONDITIONS 순번, key])`. key는 index.csv의 행 번호다. **로컬(RF)과 Colab(CNN)이 같은 이미지를 만들어야 하므로 반드시 결정적이어야 한다.**
- 변형 정의는 `~/Documents/asl-robustness/src/aslrob/perturb.py`를 200×200 기준으로 옮긴다. 어두움은 감광에 센서 노이즈(σ=6)를 더한다. clutter는 가장자리 22% 띠에 도형 10개와 텍스처 노이즈를 넣는다. shift는 ±20% 이동, scale_small은 0.6배, scale_large는 1.35배이고, 빈 곳은 가장자리 평균색으로 채운다(reflect 금지).
- 기획안 실험과의 대응: 실험1=clean, 실험2=dark_*, 실험3=clutter, 실험4=shift·scale_*

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_apply_is_deterministic():  # 같은 (name, img, key) 두 번 → np.array_equal
def test_different_keys_differ_for_random_conditions():  # clutter key 0 vs 1 → 다름
def test_dark_lowers_mean():  # dark_0.25 평균 < clean 평균 * 0.4
def test_clean_is_identity_and_shapes_preserved()
def test_condition_order_matches_config()
```

- [ ] **Step 2: 실패 확인 → Step 3: 구현 → Step 4: `uv run pytest tests/test_perturb.py -q` PASS**
- [ ] **Step 5: 커밋.** `git commit -m "feat: 결정적 합성 환경 변형"`

### Task 10: RF/DT 학습과 평가 (EXP-003)

**Files:**
- Create: `src/signlight/rf.py`, `tests/test_rf.py`, `docs/experiments/EXP-003-rf-dt.md`
- Output: `models/rf_geom.joblib`, `models/dt_geom.joblib`(git 제외), `results/preds/{rf,dt}_*__*.npz`, `data/processed/synth_landmarks.npz`

**Interfaces:**
- Consumes: `data.load_index`, `data.synth_subset`, `landmarks.load_landmarks`, `landmarks.extract`, `features.build`, `perturb.apply`, `evaluate.save_preds`
- Produces:
  - `predict_e2e(model, feats: np.ndarray) -> np.ndarray`: NaN 행은 `FAIL`, 나머지는 `model.predict`
  - `synth_landmarks(conditions: list[str]) -> dict[str, dict]`: synth_subset 행마다 `apply(cond, load_rgb(...), key=row_index)`로 변형한 뒤 추출한다. 결과는 `data/processed/synth_landmarks.npz`에 캐시하고, Task 13(CLAHE 전후 비교)에서 기준값으로 재사용한다.
  - CLI: `uv run python -m signlight.rf`. train의 검출 행만으로 학습하고, `kaggle_val`, `kaggle_test`, `synth`×7 조건 예측을 저장한다.
- 모델: `RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)`. 하이퍼파라미터는 val로 고른다. `max_depth ∈ {None, 20}`, `min_samples_leaf ∈ {1, 3}`. DT는 `max_depth ∈ {None, 10, 20}`. 같은 설정으로 `rf_norm`도 만든다(특징 비교용).

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_predict_e2e_marks_nan_rows_as_fail():
    m = DecisionTreeClassifier().fit(np.eye(3), [0, 1, 2])
    X = np.vstack([np.eye(3)[:1], np.full((1, 3), np.nan)])
    assert predict_e2e(m, X).tolist() == [0, FAIL]
```

- [ ] **Step 2: 실패 확인 → Step 3: 구현 → Step 4: `uv run pytest tests/test_rf.py -q` PASS**
- [ ] **Step 5: 실행.** `uv run python -m signlight.rf` (synth 2,400장 × 7조건 추출은 약 3분)
- [ ] **Step 6: 확인.** `summarize()`에서 rf_geom/kaggle_test/clean의 acc가 검출률 이하여야 한다(e2e라서 상한은 검출률이다). 이를 어기면 분모 버그다.
- [ ] **Step 7: EXP-003 작성.** 조건별 acc, CI, fail_rate, Drop, RF와 DT의 차이, geom과 norm의 차이를 적는다. 검출 실패로 생긴 손실과 분류 오류로 생긴 손실을 나눠서 적는다.
- [ ] **Step 8: 커밋.** `git commit -m "exp: RF/DT 학습과 합성 변형 평가 (EXP-003)"`

---

## Phase 4. CNN (Colab)

### Task 11: CNN 모델과 입력 파이프라인 (로컬 스모크 테스트)

**Files:**
- Create: `src/signlight/cnn.py`, `tests/test_cnn.py`

**Interfaces:**
- Consumes: `config.CNN_SIZE`, `config.TARGET_CLASSES`, `data.load_index`, `data.synth_subset`, `images.load_rgb`, `images.to_model_input`, `perturb.apply`, `evaluate.save_preds`
- Produces:
  - `build_model(n_classes: int = 24, size: int = 128, augment: bool = False) -> keras.Model`: 입력은 uint8 범위 (size,size,3). 내부에서 `Rescaling(1/255)`를 한다. 합성곱 블록 4개(32, 64, 128, 256 필터; Conv3×3-BN-ReLU-MaxPool) 뒤에 GAP, Dropout 0.3, Dense softmax를 둔다. `augment=True`면 맨 앞에 `RandomFlip("horizontal")`, `RandomRotation(0.05)`, `RandomTranslation(0.15, 0.15)`, `RandomZoom(0.2)`, `RandomBrightness(0.4)`, `RandomContrast(0.3)`을 넣는다(학습 때만 작동한다).
  - `make_dataset(index: pd.DataFrame, split: str, batch: int = 64, shuffle: bool = False, seed: int = 42) -> tf.data.Dataset`: `(image uint8, y)`. 라벨은 index의 `y`(= TARGET_CLASSES 순서)를 쓴다.
  - `predict_images(model, images: Iterable[np.ndarray], batch: int = 256) -> np.ndarray`: argmax 라벨
  - CLI: `python -m signlight.cnn train --variant {plain,aug} --seed S --epochs 15` 그리고 `python -m signlight.cnn predict --variant V --seed S --testsets kaggle_val,kaggle_test,synth[,custom]`
  - synth 입력은 RF와 같은 이미지여야 한다. 행마다 `to_model_input(apply(cond, load_rgb(ASL_ROOT / relpath), key=row_index), CNN_SIZE)`로 만든다. Kaggle 셋은 `to_model_input(load_rgb(...), CNN_SIZE)`로 만든다.
- 학습: Adam 1e-3, `sparse_categorical_crossentropy`, `EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)`. 모델은 `models/cnn_{variant}_s{seed}.keras`에 저장한다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_build_model_output_shape():
    m = build_model(24, 64); assert m.output_shape == (None, 24)
def test_augment_layers_inactive_at_inference():
    m = build_model(24, 64, augment=True); x = np.full((1, 64, 64, 3), 128, np.uint8)
    np.testing.assert_allclose(m(x, training=False), m(x, training=False))
def test_make_dataset_labels_follow_index(tmp_path)  # 작은 가짜 index + 이미지 → 라벨 == index.y
def test_tiny_overfit():  # 2클래스 8장, 30 step 학습 → train acc 1.0 (CPU, size=32)
```

- [ ] **Step 2: 실패 확인 → Step 3: 구현 → Step 4: `uv run pytest tests/test_cnn.py -q` PASS**
- [ ] **Step 5: 로컬 스모크.** 클래스당 20장으로 `train --epochs 1`을 돌려 끝까지 실행되는지만 본다(CPU). `predict`가 `results/preds/cnn_plain__kaggle_val__clean__s42.npz`를 만드는지 확인하고, 스모크 결과물은 지운다.
- [ ] **Step 6: 커밋.** `git commit -m "feat: CNN 모델과 입력 파이프라인"`

### Task 12: Colab GPU 학습과 예측 (EXP-004)

**Files:**
- Create: `scripts/colab_cnn.sh`, `docs/experiments/EXP-004-cnn.md`
- Output: `results/preds/cnn_*__*.npz`, `models/cnn_*.keras`(git 제외), `results/logs/colab_*.md`

**Interfaces:**
- Consumes: Task 11 CLI, `uv build --wheel`로 만든 `dist/signlight-*.whl`, `data/raw/_download/asl-alphabet.zip`, `data/processed/index.csv`
- Produces: CNN 예측 파일. 이름 규칙은 Task 8과 같고, seed 3개 × variant 2개다.
- Colab 쪽 경로: `SIGNLIGHT_ROOT=/content/signlight`. zip을 `/content/signlight/data/raw/`에서 풀어 로컬과 같은 `asl_alphabet/train/...` 구조를 만든다. index.csv는 `/content/signlight/data/processed/`에 둔다.

- [ ] **Step 1: `scripts/colab_cnn.sh` 작성.** 순서는 다음과 같다.
  1. `uv build --wheel`
  2. `colab new -s cnn --gpu T4`
  3. `colab upload`로 wheel, zip, index.csv를 올린다.
  4. `colab ssh -s cnn -- ...`로 압축을 풀고 `pip install --no-deps <wheel>`을 한다(TF·OpenCV는 Colab 기본 패키지를 쓴다).
  5. variant(plain, aug) × seed(42, 43, 44)마다 `train`과 `predict`를 실행한다.
  6. `colab download`로 `results/preds/*.npz`와 `models/*.keras`를 받는다.
  7. `colab log -s cnn -o results/logs/colab_<날짜>.md`
  8. `trap`으로 **성공이든 실패든 `colab stop -s cnn`**을 실행한다.
- [ ] **Step 2: 짧게 시험.** 스크립트에 `--dry` 옵션을 둔다. plain, seed 42, `--epochs 1`로 한 번 돌려 업로드부터 다운로드까지 동작하는지 확인한다. 업로드 시간을 EXP-004에 적는다. 업로드가 너무 오래 걸리면 대안(`colab drivemount`로 Drive에 캐시)을 적고 사용자에게 묻는다.
- [ ] **Step 3: 본 실행.** 백그라운드로 돌리고 로그를 `results/logs/colab_cnn.log`에 쓴다. 끝나면 `colab sessions`로 남은 세션이 없는지 확인한다.
- [ ] **Step 4: 확인.** `summarize()`에 `cnn_plain`, `cnn_aug`가 seed 3개씩 있고, synth 7개 조건이 모두 있어야 한다.
- [ ] **Step 5: EXP-004 작성.** variant별로 seed 평균±표준편차를 적고, Kaggle val/test가 낙관적이라는 점과 synth에서의 Drop, aug가 clean 정확도에 준 비용을 적는다.
- [ ] **Step 6: 커밋.** `git commit -m "exp: CNN Colab 학습과 합성 변형 평가 (EXP-004)"`

### Task 13: 오류 분석과 개선 (EXP-005)

기획안 3주 차에 해당한다.

**Files:**
- Create: `docs/experiments/EXP-005-improvement.md`
- Modify: `src/signlight/landmarks.py` (선택: `preprocess="clahe"` 옵션)

**Interfaces:**
- Produces: `HandDetector(..., preprocess: str | None = None)`. `"clahe"`면 LAB 색공간의 L 채널에 `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))`을 적용한 뒤 검출한다. 예측 이름은 `rf_geom_clahe`다.

- [ ] **Step 1: 혼동행렬 분석.** rf_geom과 cnn_aug(seed 42)의 synth clean 혼동행렬에서 가장 많이 헷갈린 쌍 5개를 뽑는다(예: M/N, U/V 예상).
- [ ] **Step 2: CLAHE 테스트 추가 후 구현.** `test_clahe_preserves_shape_and_brightens_dark`: dark 이미지의 평균 밝기가 올라가야 한다.
- [ ] **Step 3: 실행.** dark_0.5, dark_0.25 조건에서 CLAHE를 쓰기 전과 후의 RF 검출률과 정확도를 비교한다.
- [ ] **Step 4: 최종 모델 선정.** 보고서 본문에 쓸 대표 모델을 고른다. RF 쪽은 rf_geom, 그리고 CLAHE가 dark Drop을 5%p 이상 줄이면 rf_geom_clahe도 같이 쓴다. CNN 쪽은 synth 평균 정확도가 높은 variant를 쓴다. 근거를 EXP-005에 적는다.
- [ ] **Step 5: 커밋.** `git commit -m "exp: 오류 분석과 CLAHE 개선 실험 (EXP-005)"`

---

## Phase 5. 직접 촬영 테스트셋

### Task 14: 촬영 안내와 촬영셋 로더

**Files:**
- Create: `src/signlight/custom_test.py`, `tests/test_custom_test.py`, `docs/PHOTO_PROTOCOL.md`

**Interfaces:**
- Consumes: `config.CUSTOM_TEST_DIR`, `config.CONDITIONS`(`normal`, `dark`, `complex_bg`, `position`), `config.TARGET_CLASSES`, `images.load_rgb`
- Produces:
  - `scan(root: Path = CUSTOM_TEST_DIR) -> pd.DataFrame`: 컬럼 `path, condition, label, y, person`. 구조는 `root/{condition}/{LETTER}/{person}_{n}.{jpg,jpeg,png,heic}`이다(확장자 대소문자는 구분하지 않는다). 모르는 조건 폴더나 글자 폴더(소문자, J, Z 포함)가 있으면 `ValueError`를 내고, 메시지에 문제 경로를 모두 적는다. `.DS_Store` 같은 이미지가 아닌 파일은 건너뛴다.
  - `report(df) -> pd.DataFrame`: 조건 × 글자별 장수. 빈 칸을 찾는 데 쓴다.
  - CLI: `uv run python -m signlight.custom_test check`는 scan과 report를 출력하고, 장수가 목표에 못 미친 칸을 표시한다.
- 촬영 목표(`PHOTO_PROTOCOL.md`): 4개 조건 × 24글자 × 팀원 4명 × 2장 = **768장**. 파일명은 `{이니셜}_{번호}`(예: `khm_1.jpg`)로 한다. 조건별 기준은 아래와 같다.
  - normal: 밝은 실내, 단색 벽
  - dark: 조명 끄고 화면 밝기 정도
  - complex_bg: 물건이 많은 책상이나 책장
  - position: 손을 화면 구석이나 멀리 두기
- 이 밖에 안내할 것: 오른손 기본(왼손이면 파일명에 `_L`), 손 전체가 보이게, 정적인 자세, 세로·가로 상관없음.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_scan_accepts_uppercase_extensions(tmp_path)   # normal/A/khm_1.JPG, dark/B/khm_1.HEIC → 2행
def test_scan_rejects_unknown_folders(tmp_path):
    (tmp_path/"dark2"/"A").mkdir(parents=True); (tmp_path/"normal"/"a").mkdir(parents=True)
    with pytest.raises(ValueError) as e: scan(tmp_path)
    assert "dark2" in str(e.value) and "normal/a" in str(e.value)
def test_scan_rejects_excluded_letters(tmp_path)        # normal/J → ValueError
def test_scan_ignores_non_images(tmp_path)              # .DS_Store, notes.txt 무시
def test_person_parsed_from_filename(tmp_path)          # khm_3_L.jpg → person == "khm"
```

- [ ] **Step 2: 실패 확인 → Step 3: 구현 → Step 4: `uv run pytest tests/test_custom_test.py -q` PASS**
- [ ] **Step 5: `PHOTO_PROTOCOL.md` 작성.** 팀원에게 공유할 수 있는 한 쪽 분량으로, 폴더 구조, 조건별 예시 설명, 파일명 규칙, 업로드 방법(`data/custom_test/`에 넣기)을 적는다.
- [ ] **Step 6: 커밋.** `git commit -m "feat: 직접 촬영셋 로더와 촬영 안내"`

### Task 15: 🙋 직접 촬영셋 수집과 평가 (EXP-006)

**사람이 할 일:** 팀원들이 `PHOTO_PROTOCOL.md`대로 찍어서 `data/custom_test/`에 넣는다. 아직 안 됐으면 이 Task는 `PROGRESS.md`에 "대기"로 두고 Task 16으로 넘어간다.

**Files:**
- Create: `docs/experiments/EXP-006-custom-test.md`
- Modify: `src/signlight/rf.py`, `src/signlight/cnn.py` (`custom` testset 지원)

**Interfaces:**
- Consumes: `custom_test.scan`, 최종 모델(Task 13), `images.to_model_input`(CNN), `HandDetector`(RF)
- Produces: `results/preds/*__custom__{normal,dark,complex_bg,position}[__sS].npz`

- [ ] **Step 1: 검증.** `uv run python -m signlight.custom_test check`. 오류가 나면 사용자에게 폴더를 고쳐 달라고 요청한다. 빈 칸이 있으면 목록을 알린다(평가는 있는 것만으로 진행한다).
- [ ] **Step 2: 눈으로 확인.** 조건마다 몇 장을 격자로 저장해 회전이나 잘림 문제가 없는지 본다(`reports/figures/custom_samples.png`).
- [ ] **Step 3: RF 평가(로컬).** `uv run python -m signlight.rf --testsets custom`
- [ ] **Step 4: CNN 평가(Colab).** `scripts/colab_cnn.sh --predict-only --testsets custom`. 사진 폴더와 `models/cnn_*.keras`를 올려서 예측만 돌리고, 끝나면 `colab stop`을 한다.
- [ ] **Step 5: EXP-006 작성.** 조건별 정확도, CI, fail_rate, Drop(`base="normal"`), 사람별 정확도, synth 결과와 같은 방향인지 다른지를 적는다.
- [ ] **Step 6: 커밋.** `git commit -m "exp: 직접 촬영셋 평가 (EXP-006)"` (사진 자체는 커밋하지 않는다)

---

## Phase 6. 분석과 보고

### Task 16: 요약 표와 그림

**Files:**
- Create: `src/signlight/report.py`, `tests/test_report.py`
- Output: `results/summary.csv`, `results/drop.csv`, `reports/figures/{acc_by_condition,drop_by_condition,confusion_rf,confusion_cnn,detection_rate}_{synth,custom}.png`

**Interfaces:**
- Consumes: `evaluate.summarize`, `evaluate.accuracy_drop`, `evaluate.confusion`
- Produces:
  - `seed_mean(summary) -> pd.DataFrame`: (model, testset, condition)별 seed 평균과 표준편차. RF/DT는 seed가 없으니 그대로 둔다.
  - CLI: `uv run python -m signlight.report`. custom 예측이 없으면 synth만 만들고 그 사실을 출력한다.
- 그림 규칙(dataviz 스킬을 따른다): 모델별 색은 모든 그림에서 같게 쓰고, 조건 순서는 `SYNTH_CONDITIONS`와 `CONDITIONS` 순서를 따른다. 막대에는 CI 오차막대를 넣는다. 혼동행렬에는 검출 실패 열을 넣는다.

- [ ] **Step 1: 실패하는 테스트 작성.** `test_seed_mean_aggregates_cnn_only`: 가짜 summary에서 cnn은 seed 3개가 평균 1행으로 합쳐지고 rf는 그대로여야 한다.
- [ ] **Step 2: 실패 확인 → Step 3: 구현 → Step 4: PASS**
- [ ] **Step 5: 실행하고 그림 확인.** 그림을 하나씩 열어 축, 범례, 잘림을 확인한다.
- [ ] **Step 6: 커밋.** `git commit -m "feat: 결과 요약 표와 그림"`

### Task 17: 최종 보고서

**Files:**
- Create: `docs/REPORT.md`
- Modify: `README.md` (결과 요약 표, 재현 명령)

- [ ] **Step 1: `REPORT.md` 작성.** 구성은 다음과 같다. 모든 수치는 `results/*.csv`에서 가져오고 출처를 적는다.
  1. 한 줄 결론
  2. 연구 질문과 설정(데이터, 분할, 모델, 조건)
  3. 핵심 결과 표(synth, custom)
  4. Accuracy Drop
  5. 혼동행렬 해석
  6. 검출 실패 분석
  7. 개선 실험
  8. 한계: Kaggle 근접 중복, 합성 변형의 한계, RF 우위는 사전학습된 MediaPipe 덕일 수 있음, 촬영자 수가 적음
  9. 기대 효과 대비 결론
  10. 재현 방법
- [ ] **Step 2: 기획안과 비교.** `docs/proposal.md`의 목적 3개, 실험 4개, 지표 3개가 보고서에 모두 있는지 확인하고, 클래스 변경(6→24)을 명시한다.
- [ ] **Step 3: README 갱신 후 커밋.** `git commit -m "docs: 최종 보고서"`

### Task 18: 🙋 발표 자료

**Files:**
- Create: `reports/final_presentation.pptx`

- [ ] **Step 1: 사용자에게 확인한다.** 기획안 PPT의 디자인을 이어 쓸지, 새로 만들지, 분량(발표 시간)은 어떻게 할지 묻는다.
- [ ] **Step 2: 생성.** pptx 스킬로 만든다. 구성은 REPORT.md의 한 줄 결론, 실험 설계, 결과 그림, Drop, 한계, 결론이다. 그림은 `reports/figures/`의 것을 쓴다.
- [ ] **Step 3: QA.** 스킬의 validate와 렌더링 확인을 거친다.
- [ ] **Step 4: 커밋.** `git commit -m "docs: 최종 발표 자료"`

---

## 일정 대응 (기획안 5주)

| 기획안 주차 | Task |
|---|---|
| 1주: 데이터·전처리·관절 추출 | 1–6 |
| 2주: CNN·RF·DT 구현, 기본 성능 | 7–12 |
| 3주: 오류 분석·개선·최적 모델 | 13 |
| 4주: 직접 촬영·환경별 실험 | 14–15 (14는 미리 해 두고 촬영을 일찍 요청) |
| 5주: 정리·보고서·발표 | 16–18 |

Task 14(촬영 안내)는 사람이 시간을 들여야 하는 일이라, **Phase 3이 끝나면 바로 팀에 공유**하는 것을 권장한다.
