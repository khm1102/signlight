import pytest

from signlight import config as C


@pytest.fixture
def hand_image():
    """검출이 잘 되는 실제 Kaggle 이미지(사전 측정 F 20/20). 없으면 skip."""
    p = C.ASL_TRAIN_DIR / "F" / "F1.jpg"
    if not p.exists() or not C.HAND_MODEL.exists():
        pytest.skip("Kaggle 이미지 또는 MediaPipe 모델 파일 없음")
    from signlight.images import load_rgb

    return load_rgb(p)


@pytest.fixture
def need_model():
    if not C.HAND_MODEL.exists():
        pytest.skip("MediaPipe 모델 파일 없음")
