#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL=${BASE_MODEL_DIR:-models/indictrans2-indic-en-1B}
ADAPTER_ROOT="outputs/hi_en_r8_v2"
SAMPLES_FILE="artifacts/review2/samples_hi_en.txt"

mkdir -p "$(dirname "$SAMPLES_FILE")"

echo "[eval] Base model quick checks"
python scripts/translate_base_ip.py --base "$BASE_MODEL" \
  --text "कृपया दरवाज़ा बंद करें।" --src_lang hi --tgt_lang en --num_beams 4 --use_cache off --quant off
python scripts/translate_base_ip.py --base "$BASE_MODEL" \
  --text "उन्होंने कहा कि परियोजना कल पूरी होगी।" --src_lang hi --tgt_lang en --num_beams 4 --use_cache off --quant off
python scripts/translate_base_ip.py --base "$BASE_MODEL" \
  --text "यह पहल भारतीय बाजारों में निवेश आकर्षित करने के लिए बनाई गई है, जिसमें कई चरणों में सुधार शामिल हैं।" \
  --src_lang hi --tgt_lang en --num_beams 4 --use_cache off --quant off

if compgen -G "$ADAPTER_ROOT/checkpoint-*" > /dev/null; then
  LATEST_ADAPTER=$(ls -d "$ADAPTER_ROOT"/checkpoint-* | sort | tail -n 1)
  echo "[eval] Adapter quick checks (writing to $SAMPLES_FILE)"
  {
    echo "Adapter: $LATEST_ADAPTER"
    python scripts/translate_adapter_ip.py --base "$BASE_MODEL" --adapter "$LATEST_ADAPTER" \
      --text "कृपया दरवाज़ा बंद करें।" --src_lang hi --tgt_lang en --style formal --simplify no \
      --num_beams 4 --use_cache off --quant off
    python scripts/translate_adapter_ip.py --base "$BASE_MODEL" --adapter "$LATEST_ADAPTER" \
      --text "कृपया दरवाज़ा बंद करें।" --src_lang hi --tgt_lang en --style informal --simplify yes \
      --num_beams 4 --use_cache off --quant off
  } | tee "$SAMPLES_FILE"
else
  echo "[eval] No adapter checkpoints found under $ADAPTER_ROOT; skipping adapter samples."
fi
