# signlight — 수어 알파벳 인식 모델 성능 비교

CNN(이미지 픽셀)과 Random Forest(MediaPipe 손 관절 좌표)가 촬영 환경 변화(조명·배경·손 위치)에 얼마나 강한지 비교한다. 인공지능 기말 프로젝트(팀 손전등).

- 데이터: [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- 대상 클래스: 정적 알파벳 24개(A~Z 중 움직임이 필요한 J, Z 제외)
- 평가: Accuracy(e2e: 손 검출 실패 = 오답), Confusion Matrix, Accuracy Drop
- 문서: 기획안 `docs/proposal.md` · **최종 보고서 `docs/REPORT.md`** · 실험 기록 `docs/experiments/` · 진행 상황 `docs/PROGRESS.md` · 촬영 안내 `docs/PHOTO_PROTOCOL.md`

## 결과 요약 (합성 환경 변형, 정확도)
| 조건 | RF (관절) | CNN | CNN + 증강 |
|---|---|---|---|
| 원본 | **0.717** | 0.632 | 0.576 |
| 어둡게 0.5 | 0.193 | 0.213 | **0.383** |
| 복잡한 배경 | **0.421** | 0.187 | 0.199 |
| 이동 | 0.518 | 0.072 | **0.539** |
| 축소 | **0.703** | 0.047 | 0.458 |

RF는 손 검출이 되면 강하지만 어두우면 검출부터 실패한다. CNN은 학습 사진의 구도에 과적합한다. 직접 촬영 검증은 대기 중이다. 자세한 내용과 한계는 `docs/REPORT.md`에 있다.

## 환경 설정
```bash
uv sync                 # .venv 생성 + 의존성 설치 (Python 3.12)
uv run pytest -q        # 테스트
uv run jupyter lab      # 노트북: notebooks/01_results.ipynb(결과 정리), 02_train_cnn_colab.ipynb(Colab 학습)
```

## 데이터 받기
```bash
uv tool install --with six kaggle   # Kaggle CLI (six 누락 버그 우회)
kaggle auth login                   # 브라우저 로그인 (최초 1회)
./scripts/download_data.sh          # 29개 클래스 전체를 data/raw/asl_alphabet/ 에 해제
curl -sSfL -o models/mediapipe/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## 재현
```bash
uv run python -m signlight.data            # 인덱스·근접 중복 그룹 분할 (EXP-001)
uv run python -m signlight.landmarks       # MediaPipe 관절 추출 (~10분)
uv run python -m signlight.rf              # RF/DT 학습 + 합성 변형 평가 (EXP-003)
scripts/colab_cnn.sh                       # CNN 학습·예측 (Colab T4, ~70분, 끝나면 세션 자동 종료)
uv run python -m signlight.report          # results/*.csv, reports/figures/*.png
```
Colab CLI(`uv tool install google-colab-cli`) 인증이 필요하다. Colab VM은 데이터를 `kagglehub`로 직접 받는다.

## 직접 촬영 테스트셋
`docs/PHOTO_PROTOCOL.md`대로 `data/custom_test/{normal,dark,complex_bg,position}/{글자}/`에 사진을 넣은 뒤 실행한다.
```bash
uv run python -m signlight.custom_test check
uv run python -m signlight.rf --testsets custom
scripts/colab_cnn.sh --predict-only --testsets custom
uv run python -m signlight.report
```

## 디렉토리
```
data/raw/          Kaggle 원본 (git 제외)
data/processed/    index.csv, 관절 좌표 npz (git 제외)
data/custom_test/  직접 촬영 테스트셋 (git 제외)
models/            학습된 모델, MediaPipe 모델 파일 (git 제외)
results/           예측(preds/*.npz), 요약 csv, 로그
reports/figures/   결과 그림
src/signlight/     config, images, data, landmarks, features, perturb, evaluate, rf, cnn, custom_test, report
scripts/           download_data.sh, colab_cnn.sh
tests/             pytest
```
