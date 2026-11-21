# 진행 상황 (실행이 끊기면 여기부터 읽기)

플랜: `docs/superpowers/plans/2026-11-22-signlight.md` · 기획안: `docs/proposal.md`
브랜치: `feat/pipeline` (master = 초기 설정 커밋)

## 현재 상태
- Phase 0 진행 중

## 완료
- 환경: uv(Python 3.12), Colab CLI 인증, Kaggle CLI(`uv tool install --with six kaggle`) 로그인
- 데이터: Kaggle ASL Alphabet 29개 클래스 전체를 `data/raw/asl_alphabet/{train,test}`에 해제(원본 zip은 `data/raw/_download/`)
- MediaPipe 손 모델: `models/mediapipe/hand_landmarker.task`
- 사전 측정(2026-11-22)
  - 검출률: 24개 클래스 × 20장, conf 0.3에서 78.3%(장당 9ms). M·N이 10/20으로 가장 낮다.
  - 근접 중복: 가장 가까운 이미지와의 dHash(256bit) 해밍 거리가 25 이하인 비율 99.5%. A 클래스를 thr 20으로 군집하면 69개가 남는다.
- Task 1: pytest, 설정 보강(SIGNLIGHT_ROOT, 경로, 시드, 조건)

## 다음
- Task 2: 이미지 입출력

## 막힌 점/대기
- 없음
