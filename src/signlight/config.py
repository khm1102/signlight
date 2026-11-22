"""프로젝트 공통 경로와 실험 설정."""

import os
from pathlib import Path

# 저장소 루트. Colab처럼 패키지를 wheel로 설치해 쓰는 곳에서는 SIGNLIGHT_ROOT로 지정한다.
ROOT = Path(os.environ.get("SIGNLIGHT_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # Kaggle ASL Alphabet 원본
ASL_ROOT = RAW_DIR / "asl_alphabet"
ASL_TRAIN_DIR = ASL_ROOT / "train"  # 29개 클래스 폴더, 각 3000장
ASL_TEST_DIR = ASL_ROOT / "test"  # 클래스당 1장 ({C}_test.jpg, del 없음)
PROCESSED_DIR = DATA_DIR / "processed"  # 인덱스, 관절 좌표
CUSTOM_TEST_DIR = DATA_DIR / "custom_test"  # 팀원 직접 촬영 테스트셋
MODELS_DIR = ROOT / "models"
HAND_MODEL = MODELS_DIR / "mediapipe" / "hand_landmarker.task"
RESULTS_DIR = ROOT / "results"
PREDS_DIR = RESULTS_DIR / "preds"
FIGURES_DIR = ROOT / "reports" / "figures"

KAGGLE_DATASET = "grassknoted/asl-alphabet"

# 실험 대상: 정적 손 모양 알파벳 24개
# - J, Z: 움직임이 필요해 정지 이미지로 판별 불가 → 제외
# - del, nothing, space: 알파벳이 아님 (nothing은 손이 없어 MediaPipe 검출 불가) → 제외
# 순서 = 라벨 인덱스 (혼동행렬 축, 모델 출력 순서)
EXCLUDED_LETTERS = {"J", "Z"}
TARGET_CLASSES = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in EXCLUDED_LETTERS]

IMAGE_SIZE = 200  # 원본 이미지 크기 (200x200)
CNN_SIZE = 128  # CNN 입력 크기
NUM_LANDMARKS = 21  # MediaPipe 손 관절 개수

# MediaPipe 검출 설정 (EXP-002에서 확정)
DET_CONF = 0.3
DET_PAD = 0.0

# 직접 촬영 실험 조건 (custom_test 하위 폴더명)
CONDITIONS = ["normal", "dark", "complex_bg", "position"]
# 합성 변형 조건 (perturb.CONDITIONS와 같은 순서)
SYNTH_CONDITIONS = ["clean", "dark_0.5", "dark_0.25", "clutter", "shift", "scale_small", "scale_large"]

SEED = 42
CNN_SEEDS = [42, 43, 44]
