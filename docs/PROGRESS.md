# 진행 상황 (실행이 끊기면 여기부터 읽기)

플랜: `docs/superpowers/plans/2026-11-22-signlight.md` · 기획안: `docs/proposal.md` · 보고서: `docs/REPORT.md`
브랜치: `feat/pipeline` (master = 초기 설정 커밋). 원격 푸시는 하지 않았다.

## 현재 상태 (2026-11-22)
- Task 1~14, 16, 17 완료. **Task 15(직접 촬영)와 Task 18(발표 자료)은 사람 확인이 필요해 대기 중**이다.
- 합성 환경 변형 기준 결과와 보고서 초안을 완성했다. 직접 촬영 결과가 들어오면 보고서 1절과 6절을 갱신한다.

## 완료
- 환경: uv(Python 3.12), Colab CLI 인증, Kaggle CLI(`uv tool install --with six kaggle`) 로그인
- 데이터: Kaggle 29개 클래스 해제, 대상 24개, 72,000장
- Task 1 설정 · 2 이미지 입출력 · 3 그룹 분할(EXP-001) · 4 HandDetector · 5 검출 설정(EXP-002) · 6 관절 추출(검출률 80.2%)
- Task 7 특징 · 8 평가 · 9 합성 변형 · 10 RF/DT(EXP-003) · 11 CNN · 12 Colab 학습(EXP-004, 6회, 약 70분)
- Task 13 오류 분석·CLAHE(EXP-005, 기각) · 14 촬영 안내·로더 · 16 표·그림 · 17 보고서
- Task 15 코드 준비: `rf --testsets custom`, `colab_cnn.sh --predict-only --testsets custom`

## 다음
1. 🙋 팀원 촬영(`docs/PHOTO_PROTOCOL.md`, 목표 768장) → Task 15 실행 명령은 `docs/REPORT.md` 6절에 있다.
2. 발표 자료(Task 18) 초안: 비공개 Slides 아티팩트 https://claude.ai/artifact/1V4KsdXRLbNd7XazKrVWGL (7장, 발표자 노트 포함, .pptx/PDF로 내려받기 가능). 구성안은 `docs/PRESENTATION_OUTLINE.md`. 6번 슬라이드(직접 촬영 결과)는 비어 있고, 발표일은 [발표일]로 남겨 둠. 기획안 PPT 디자인으로 다시 만들려면 LibreOffice 설치가 필요(권한 거절됨).

## 막힌 점/대기
- 직접 촬영 사진 0장. 사용자가 나중에 하기로 함(2026-11-22). 사진이 들어오면 REPORT.md 6절 명령 → 보고서 1·6절과 발표 6번 슬라이드 갱신
- 발표 자료: 형식은 Slides 초안으로 확정(2026-11-22 사용자 결정). 6번 슬라이드는 촬영 결과 대기

## 운영 메모
- Colab: `colab exec`는 원격 셀이 실패해도 종료 코드 0을 돌려준다. 그래서 스크립트가 출력에서 `CalledProcessError`를 찾아 멈춘다. 1GB 업로드는 실패하므로 데이터는 `kagglehub`로 받는다(`/kaggle/input`은 NFS라 로컬 디스크로 복사).
- MediaPipe 1.2.x에는 `mp.solutions`가 없다. tasks `HandLandmarker`와 `models/mediapipe/hand_landmarker.task`를 쓴다.
