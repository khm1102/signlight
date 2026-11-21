import importlib


def test_target_classes_are_24_static_letters():
    from signlight import config as C

    assert len(C.TARGET_CLASSES) == 24
    assert "J" not in C.TARGET_CLASSES and "Z" not in C.TARGET_CLASSES
    assert C.TARGET_CLASSES == sorted(C.TARGET_CLASSES)


def test_root_override(monkeypatch, tmp_path):
    from signlight import config

    monkeypatch.setenv("SIGNLIGHT_ROOT", str(tmp_path))
    C = importlib.reload(config)
    try:
        assert C.ROOT == tmp_path
        assert C.DATA_DIR == tmp_path / "data"
        assert C.PREDS_DIR == tmp_path / "results" / "preds"
    finally:
        monkeypatch.delenv("SIGNLIGHT_ROOT")
        importlib.reload(config)


def test_synth_conditions_start_with_clean():
    from signlight import config as C

    assert C.SYNTH_CONDITIONS[0] == "clean"
    assert len(C.SYNTH_CONDITIONS) == 7
