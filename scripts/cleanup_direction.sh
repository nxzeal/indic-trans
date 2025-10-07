#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

STAMP="$(date +"%Y%m%d_%H%M%S")"
TRASH_ROOT=".trash/$STAMP"
mkdir -p "$TRASH_ROOT"

MOVE_LIST=()

# Models
MOVE_LIST+=("models/indictrans2-en-indic-1B")

# Outputs (old runs)
MOVE_LIST+=(outputs/hi_en_r8)
for dir in outputs/*_r16; do
  MOVE_LIST+=("$dir")
done
for dir in outputs/*_r16; do
  MOVE_LIST+=("$dir")
done

# Data clean dirs
MOVE_LIST+=("data/clean/ta_en" "data/clean/te_en" "data/clean/ml_en")

# Data raw files
MOVE_LIST+=("data/raw/ta_en.tsv" "data/raw/te_en.tsv" "data/raw/ml_en.tsv")

moved=0
for src in "${MOVE_LIST[@]}"; do
  if [[ -e "$src" ]]; then
    dest="$TRASH_ROOT/$src"
    mkdir -p "$(dirname "$dest")"
    if mv -v "$src" "$dest"; then
      ((moved++))
    fi
  fi
done

echo "Cleanup complete. Items moved: $moved"
echo "Trash directory: $TRASH_ROOT"
