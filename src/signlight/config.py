"""프로젝트 공통 경로와 실험 설정."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # Kaggle ASL Alphabet 원본
ASL_TRAIN_DIR = RAW_DIR / "asl_alphabet" / "train"  # 29개 클래스 폴더, 각 3000장
ASL_TEST_DIR = RAW_DIR / "asl_alphabet" / "test"  # 클래스당 1장 ({C}_test.jpg, del 없음)
PROCESSED_DIR = DATA_DIR / "processed"  # CNN 전처리 이미지, MediaPipe 좌표 CSV
CUSTOM_TEST_DIR = DATA_DIR / "custom_test"  # 팀원 직접 촬영 테스트셋
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

KAGGLE_DATASET = "grassknoted/asl-alphabet"

# 실험 대상: 정적 손 모양 알파벳 24개
# - J, Z: 움직임이 필요해 정지 이미지로 판별 불가 → 제외
# - del, nothing, space: 알파벳이 아님 (nothing은 손이 없어 MediaPipe 검출 불가) → 제외
# 순서 = 라벨 인덱스 (혼동행렬 축, 모델 출력 순서)
EXCLUDED_LETTERS = {"J", "Z"}
TARGET_CLASSES = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in EXCLUDED_LETTERS]

IMAGE_SIZE = 200  # 원본 이미지 크기 (200x200)
NUM_LANDMARKS = 21  # MediaPipe 손 관절 개수

# 촬영 환경별 실험 (custom_test 하위 폴더명)
CONDITIONS = ["normal", "dark", "complex_bg", "position"]

SEED = 42
