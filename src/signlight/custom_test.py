"""팀원 직접 촬영 테스트셋: 폴더 스캔과 검증.

구조: data/custom_test/{condition}/{LETTER}/{person}_{n}[_L].{jpg,jpeg,png,heic}
- condition: normal, dark, complex_bg, position (config.CONDITIONS)
- LETTER: config.TARGET_CLASSES (대문자, J·Z 제외)
폴더 이름이 틀리면 조용히 건너뛰지 않고 ValueError로 모두 알린다.

python -m signlight.custom_test check
"""

from pathlib import Path

import pandas as pd

from . import config as C

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
COLUMNS = ["path", "condition", "label", "y", "person"]


def scan(root: Path | None = None) -> pd.DataFrame:
    root = Path(root or C.CUSTOM_TEST_DIR)
    rows, bad = [], []
    if not root.exists():
        return pd.DataFrame(columns=COLUMNS)
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        if cdir.name not in C.CONDITIONS:
            bad.append(cdir.name)
            continue
        for ldir in sorted(p for p in cdir.iterdir() if p.is_dir()):
            if ldir.name not in C.TARGET_CLASSES:
                bad.append(f"{cdir.name}/{ldir.name}")
                continue
            for f in sorted(ldir.iterdir()):
                if f.is_file() and f.suffix.lower() in PHOTO_EXTS and not f.name.startswith("."):
                    person = f.stem.split("_")[0]
                    rows.append((f, cdir.name, ldir.name, C.TARGET_CLASSES.index(ldir.name), person))
    if bad:
        raise ValueError(
            "알 수 없는 폴더: " + ", ".join(bad)
            + f"\n조건은 {C.CONDITIONS}, 글자는 대문자 {''.join(C.TARGET_CLASSES)} 중 하나여야 합니다."
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def report(df: pd.DataFrame) -> pd.DataFrame:
    """글자 × 조건별 장수(빈 칸 = 0)."""
    t = pd.crosstab(df["label"], df["condition"]) if len(df) else pd.DataFrame()
    return t.reindex(index=C.TARGET_CLASSES, columns=C.CONDITIONS, fill_value=0).fillna(0).astype(int)


def main(target_per_cell: int = 8) -> None:
    df = scan()
    r = report(df)
    print(f"총 {len(df)}장, 촬영자 {sorted(df['person'].unique())}")
    print(r.to_string())
    short = [(lbl, c, int(r.loc[lbl, c])) for lbl in r.index for c in r.columns if r.loc[lbl, c] < target_per_cell]
    print(f"\n목표({target_per_cell}장) 미달 칸: {len(short)}개")
    for lbl, c, n in short[:50]:
        print(f"  {c}/{lbl}: {n}장")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check"])
    ap.add_argument("--target", type=int, default=8, help="칸(조건×글자)당 목표 장수")
    main(ap.parse_args().target)
