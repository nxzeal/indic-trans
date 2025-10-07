#!/usr/bin/env bash
set -euo pipefail

# --- 0) Install runtime dependencies ---
python -m pip install -U pip
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"
pip install -U indictranstoolkit

# --- 1) Download base model (Indic → English) ---
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p models
# (toggle accelerator if needed)
export HF_HUB_ENABLE_HF_TRANSFER=1

hf download ai4bharat/indictrans2-indic-en-1B \
  --local-dir models/indictrans2-indic-en-1B \
  --local-dir-use-symlinks False \
  --include "model*.safetensors" \
  --include "pytorch_model*.bin" \
  --include "model*.index.json" \
  --include "config.json" \
  --include "generation_config.json" \
  --include "tokenizer*.json" \
  --include "*.model" \
  --include "special_tokens_map.json" \
  --include "tokenization_*.py" \
  --include "configuration_*.py" \
  --include "modeling_*.py" \
  --include "*.txt"

# --- 2) Prepare Hindi-English data if absent ---
if [[ ! -f data/raw/hi_en.tsv ]]; then
  python scripts/fetch_corpora.py --pairs hi-en --max_per_pair 80000 --stream yes --export_flores no
  python scripts/data_prep.py --manifest data/manifests/review2.yaml
  python scripts/make_splits.py --pair hi-en --in data/clean/hi_en --out data/clean/hi_en --seed 42
  python scripts/augment_controls.py \
    --in data/clean/hi_en/train.tsv \
    --out data/clean/hi_en/train_aug.tsv \
    --lang_pair hi-en \
    --sample_ratio 0.25 \
    --seed 42 \
    --modes informal,formal,simplify \
    --max_src_len 256 --max_tgt_len 192
fi

echo
echo "Rehydrate complete ✅"
echo "Next steps:"
echo "  export TOKENIZERS_PARALLELISM=false"
echo "  accelerate launch scripts/train_lora.py --config configs/qlora_hi_en.yaml"
echo
echo "Example inference commands:"
echo "  python scripts/translate_base_ip.py --text \"कृपया दरवाज़ा बंद करें।\" --src_lang hi --tgt_lang en --num_beams 4 --use_cache off --quant off"
echo "  python scripts/translate_adapter_ip.py --adapter outputs/hi_en_r8_v2/checkpoint-1500 --text \"कृपया दरवाज़ा बंद करें।\" --src_lang hi --tgt_lang en --style informal --simplify yes"
