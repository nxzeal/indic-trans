# IndicTrans-LoRA for Cross-Lingual Formality & Readability Simplification

IndicTrans-LoRA fine-tunes IndicTrans2 checkpoints with QLoRA adapters for controllable style and simplification across Indic?English language pairs. Full scope pairs: hi?en, ta?en, te?en, ml?en. Review-2 shows only hi?en and ta?en while the other pairs stay staged off-line.

## Project Highlights
- Base model: ai4bharat/indictrans2-indic-en-1B, adapted with PEFT QLoRA (r=8) and bitsandbytes 4-bit quantization when available.
- Task control tags for style (formal/informal) and simplify (yes/no) baked into a prompt template.
- Deterministic workflows driven by YAML configs under configs/ (training and evaluation).
- Experiment tracking with MLflow (./mlruns) and artifacts captured under rtifacts/.
- Demo clients: CLI inference and a lightweight FastAPI web app (only hi?en / ta?en exposed for Review-2).

### Direction Correction
We initially pulled the en→indic base; project scope requires hi→en. As of 2025-10-06, the repo defaults to ai4bharat/indictrans2-indic-en-1B and the data/training/inference paths have been updated accordingly. Old artifacts are parked under `.trash/`.

## Setup
1. Create and activate a virtual environment (Python 3.10+ recommended):
   `ash
   python -m venv .venv
   # Linux/macOS
   source .venv/bin/activate
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   `
2. Install dependencies:
   `ash
   pip install -r requirements.txt
   `
   Install 	orch separately with the wheel that matches your platform/CUDA stack (see https://pytorch.org/get-started/locally/).
3. (Optional) Configure environment variables via .env (copy from .env.example).
4. Run ccelerate config once before launching distributed/QLoRA training.

## Data Expectations
- Place raw TSV files under data/raw/ (not versioned). Expected columns: src, 	gt, optional style_src, style_tgt, and simplify (yes/
o).
- Entries should already align with the language direction in the filename (e.g., hi_en.tsv for hi?en). During training we mirror examples to cover the reverse direction specified in configs.
- JSONL is acceptable if converted to TSV before ingestion.
- Cleaned splits are written to data/clean/<pair>/[train|val|test].tsv with deterministic seeds.

## Core Workflow
All commands assume the project root (indictrans-lora/). Adjust paths for other pairs as needed.

### 1. Data Cleaning (Review-2 manifest)
`ash
python scripts/data_prep.py --manifest data/manifests/review2.yaml
`

### 2. Train/val/test splits (deterministic seed=42)
`\bash
python scripts/make_splits.py --pair hi-en --in data/clean/hi_en --out data/clean/hi_en
`
(Repeat with additional pairs when expanding scope.)

### 2b. Optional control augmentation (teaches informal/simplify toggles)
`\bash
python scripts/augment_controls.py \
  --in data/clean/hi_en/train.tsv --out data/clean/hi_en/train_aug.tsv \
  --lang_pair hi-en --sample_ratio 0.25 --seed 42 \
  --modes informal,formal,simplify --max_src_len 256 --max_tgt_len 192
`
### 2c. Repair split direction/pair metadata (if needed)
`\bash
python scripts/fix_splits_direction.py --dir data/clean/hi_en --pair hi-en --detect_lang yes --backup yes
`

### 3. QLoRA Training (Review-2 focus pairs)
`\bash
accelerate launch scripts/train_lora.py --config configs/qlora_hi_en.yaml
`
Outputs land in outputs/hi_en_r8_v2/ with adapters, tokenizer snapshots, and validation predictions. MLflow logs the run in ./mlruns.

### 4. Evaluation
`\bash
python scripts/eval_metrics.py --config configs/eval.yaml --pair hi-en \
  --refs data/clean/hi_en/test.tsv --hyps outputs/hi_en_r8_v2/preds_test.txt \
  --out artifacts/review2/metrics_hi_en.json
`
Each run writes JSON + TSV metrics and examples_<pair>.tsv under artifacts/review2/.

### 5. Quick CLI Inference
`\bash
python scripts/translate_base_ip.py --text "कृपया दरवाज़ा बंद करें।" --src_lang hi --tgt_lang en \
  --num_beams 4 --use_cache off --quant off
`
The script automatically falls back to FP16/CPU when bitsandbytes or CUDA are unavailable.

### 6. FastAPI Demo (Hindi → English)
`ash
uvicorn demo.app:app --reload
`
Open http://127.0.0.1:8000/ to interact with the adapters. The UI checks that adapters exist before generating.

### 7. Docker CPU Inference
`ash
docker compose -f docker/compose.yaml up --build
`
Builds a CPU-only image for serving the FastAPI demo; mount outputs/ so adapters are accessible at runtime.

## Artifacts & Tracking
- MLflow tracking URI defaults to the local ile://./mlruns store. Update via .env if needed.
- Review-2 bundle lives under rtifacts/review2/ (metrics, qualitative examples, manifest copies, slide draft stub, and placeholders for screenshots).
- Full-scope artifacts are staged under rtifacts/full/ (ignored by default; add files as the project expands beyond Review-2).

## 70% Cut Checklist
| Scope | Status |
|-------|--------|
| hi→en | **Show** — cleaned data, adapter, metrics, demo |
| Other pairs | Off-stage — configs ready, data pipeline shared |

Launch Review-2 using the commands above; bring additional language pairs back when the scope expands.
