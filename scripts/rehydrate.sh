#!/usr/bin/env bash
set -euo pipefail

# --- 0) Bootstrap venv & deps ---
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"

# --- 1) Download base model (local, resumable, no accelerated transfer) ---
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p models
hf download ai4bharat/indictrans2-en-indic-1B \
  --local-dir models/indictrans2-en-indic-1B \
  --include model.safetensors \
  --include config.json tokenizer_config.json special_tokens_map.json generation_config.json \
  --include tokenization_indictrans.py configuration_indictrans.py modeling_indictrans.py \
  --include dict.SRC.json dict.TGT.json model.SRC model.TGT

# --- 2) Fetch corpora (adjust max_per_pair to taste) ---
python scripts/fetch_corpora.py --pairs hi-en ta-en --max_per_pair 80000 --stream yes --export_flores no

# --- 3) Clean & split ---
python scripts/data_prep.py --manifest data/manifests/review2.yaml
python scripts/make_splits.py --pair hi-en --in data/clean/hi_en --out data/clean/hi_en --seed 42
python scripts/make_splits.py --pair ta-en --in data/clean/ta_en --out data/clean/ta_en --seed 42

# --- 4) Repair splits (ensure pair column + direction) ---
python scripts/fix_splits_direction.py --dir data/clean/hi_en --pair hi-en --detect_lang yes --backup yes
python scripts/fix_splits_direction.py --dir data/clean/ta_en --pair ta-en --detect_lang yes --backup yes

# --- 5) Augment controls (teach style/simplify on EN targets) ---
python scripts/augment_controls.py \
  --in data/clean/hi_en/train.tsv \
  --out data/clean/hi_en/train_aug.tsv \
  --lang_pair hi-en \
  --sample_ratio 0.25 \
  --seed 42 \
  --modes informal,formal,simplify \
  --max_src_len 256 --max_tgt_len 192

python scripts/augment_controls.py \
  --in data/clean/ta_en/train.tsv \
  --out data/clean/ta_en/train_aug.tsv \
  --lang_pair ta-en \
  --sample_ratio 0.25 \
  --seed 42 \
  --modes informal,formal,simplify \
  --max_src_len 256 --max_tgt_len 192

echo
echo "Rehydrate complete ✅"
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  export TOKENIZERS_PARALLELISM=false"
echo "  accelerate launch scripts/train_lora.py --config configs/qlora_hi_en.yaml"
