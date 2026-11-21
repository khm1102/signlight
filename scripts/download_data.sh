#!/usr/bin/env bash
# Kaggle ASL Alphabet 데이터셋(29개 클래스 전체)을 받아 data/raw/asl_alphabet/{train,test} 에 푼다.
# 실험 대상 클래스는 src/signlight/config.py 의 TARGET_CLASSES 로 고른다.
# 사전 준비: uv tool install --with six kaggle && kaggle auth login
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
ZIP="$RAW/_download/asl-alphabet.zip"
OUT="$RAW/asl_alphabet"

if [[ ! -f "$ZIP" ]]; then
  mkdir -p "$RAW/_download"
  kaggle datasets download grassknoted/asl-alphabet -p "$RAW/_download"
fi

rm -rf "$RAW/_extract" "$OUT"
mkdir -p "$RAW/_extract" "$OUT"
unzip -q "$ZIP" -d "$RAW/_extract"
mv "$RAW/_extract/asl_alphabet_train/asl_alphabet_train" "$OUT/train"
mv "$RAW/_extract/asl_alphabet_test/asl_alphabet_test" "$OUT/test"
rm -rf "$RAW/_extract"

echo "train: $(find "$OUT/train" -name '*.jpg' | wc -l | tr -d ' ') images, $(ls "$OUT/train" | wc -l | tr -d ' ') classes"
echo "test:  $(find "$OUT/test" -name '*.jpg' | wc -l | tr -d ' ') images"
