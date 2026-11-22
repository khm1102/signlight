import pytest

from signlight.custom_test import report, scan


def _touch(root, rel):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return p


def test_scan_accepts_uppercase_extensions(tmp_path):
    _touch(tmp_path, "normal/A/khm_1.JPG")
    _touch(tmp_path, "dark/B/khm_1.HEIC")
    _touch(tmp_path, "dark/B/lhn_2.jpeg")
    df = scan(tmp_path)
    assert len(df) == 3
    assert list(df.columns) == ["path", "condition", "label", "y", "person"]
    assert sorted(df["label"]) == ["A", "B", "B"]
    assert df.set_index("label").loc["A", "y"] == 0


def test_scan_rejects_unknown_folders(tmp_path):
    (tmp_path / "dark2" / "A").mkdir(parents=True)
    (tmp_path / "normal" / "a").mkdir(parents=True)
    with pytest.raises(ValueError) as e:
        scan(tmp_path)
    assert "dark2" in str(e.value) and "normal/a" in str(e.value)


def test_scan_rejects_excluded_letters(tmp_path):
    _touch(tmp_path, "normal/J/khm_1.jpg")
    with pytest.raises(ValueError, match="normal/J"):
        scan(tmp_path)


def test_scan_ignores_non_images(tmp_path):
    _touch(tmp_path, "normal/A/.DS_Store")
    _touch(tmp_path, "normal/A/notes.txt")
    _touch(tmp_path, ".DS_Store")
    _touch(tmp_path, "normal/A/khm_1.png")
    assert len(scan(tmp_path)) == 1


def test_person_parsed_from_filename(tmp_path):
    _touch(tmp_path, "position/V/khm_3_L.jpg")
    assert scan(tmp_path)["person"].tolist() == ["khm"]


def test_scan_empty_root_returns_empty(tmp_path):
    assert len(scan(tmp_path)) == 0


def test_report_counts_condition_by_letter(tmp_path):
    _touch(tmp_path, "normal/A/khm_1.jpg")
    _touch(tmp_path, "normal/A/khm_2.jpg")
    _touch(tmp_path, "dark/C/khm_1.jpg")
    r = report(scan(tmp_path))
    assert r.loc["A", "normal"] == 2 and r.loc["A", "dark"] == 0
    assert list(r.columns) == ["normal", "dark", "complex_bg", "position"]
    assert len(r) == 24


def test_scan_rejects_misplaced_or_unknown_files(tmp_path):
    _touch(tmp_path, "normal/A/khm_1.jpg")
    _touch(tmp_path, "normal/khm_2.jpg")          # 글자 폴더 없이 조건 폴더에 바로
    _touch(tmp_path, "normal/B/sub/khm_3.jpg")    # 글자 폴더 안의 하위 폴더
    _touch(tmp_path, "dark/C/khm_4.webp")         # 지원하지 않는 형식
    _touch(tmp_path, "khm_6.jpg")                 # 루트에 바로
    with pytest.raises(ValueError) as e:
        scan(tmp_path)
    msg = str(e.value)
    for part in ("normal/khm_2.jpg", "normal/B/sub", "dark/C/khm_4.webp", "khm_6.jpg"):
        assert part in msg
