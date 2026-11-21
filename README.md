# signlight — 수어 알파벳 인식 모델 성능 비교

CNN(이미지 픽셀)과 Random Forest(MediaPipe 손 관절 좌표)가 촬영 환경 변화(조명·배경·손 위치)에 얼마나 강한지 비교한다.

- 데이터: [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- 대상 클래스: 정적 알파벳 24개 (A~Z 중 움직임이 필요한 J, Z 제외)
- 평가: Accuracy, Confusion Matrix, Accuracy Drop

## 환경 설정

```bash
uv sync                 # .venv 생성 + 의존성 설치 (Python 3.12)
uv run jupyter lab      # 노트북
```

## 데이터 받기

```bash
uv tool install --with six kaggle   # Kaggle CLI (six 누락 버그 우회)
kaggle auth login                   # 브라우저 로그인 (최초 1회)
./scripts/download_data.sh          # 29개 클래스 전체를 data/raw/asl_alphabet/ 에 해제
```

## 디렉토리

```
data/raw/          Kaggle 원본 (git 제외)
data/processed/    전처리 이미지 / 관절 좌표 CSV
data/custom_test/  직접 촬영 테스트셋 (normal, dark, complex_bg, position)
notebooks/         실험 노트북
models/            학습된 모델 (git 제외)
reports/figures/   결과 그래프
src/signlight/     공통 코드 (config.py 등)
```

## Colab (GPU 학습)

```bash
colab new -s train --gpu T4
colab install -s train mediapipe
colab exec -s train -f notebooks/<notebook>.ipynb
colab download -s train /content/models/cnn.keras models/cnn.keras
colab stop -s train
```
