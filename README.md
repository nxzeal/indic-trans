# IndicTrans-LoRA for Cross-Lingual Formality & Readability Simplification

IndicTrans-LoRA fine-tunes IndicTrans2 checkpoints with QLoRA adapters for controllable style and simplification across Indic?English language pairs. Full scope pairs: hi?en, ta?en, te?en, ml?en. Review-2 shows only hi?en and ta?en while the other pairs stay staged off-line.

## Project Highlights
- Base model: i4bharat/indictrans2-en-indic-1B, adapted with PEFT QLoRA (r=16) and bitsandbytes 4-bit quantization when available.
- Task control tags for style (formal/informal) and simplify (yes/no) baked into a prompt template.
- Deterministic workflows driven by YAML configs under configs/ (training and evaluation).
- Experiment tracking with MLflow (./mlruns) and artifacts captured under rtifacts/.
- Demo clients: CLI inference and a lightweight FastAPI web app (only hi?en / ta?en exposed for Review-2).

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
`ash
python scripts/make_splits.py --pair hi-en --in data/clean/hi_en --out data/clean/hi_en
python scripts/make_splits.py --pair ta-en --in data/clean/ta_en --out data/clean/ta_en
`
(Repeat with 	e-en, ml-en when preparing the full scope.)

### 2b. Optional control augmentation (teaches informal/simplify toggles)
`\bash
python scripts/augment_controls.py \
  --in data/clean/hi_en/train.tsv --out data/clean/hi_en/train_aug.tsv \
  --lang_pair hi-en --sample_ratio 0.25 --seed 42 \
  --modes informal,formal,simplify --max_src_len 256 --max_tgt_len 192
`
Run the same command with `ta-en` paths to cover Tamil→English before retraining. The script preserves a `train.orig.tsv` backup and patches `train.tsv` in-place.

### 2c. Repair split direction/pair metadata (if needed)
`\bash
python scripts/fix_splits_direction.py --dir data/clean/hi_en --pair hi-en --detect_lang yes --backup yes
python scripts/fix_splits_direction.py --dir data/clean/ta_en --pair ta-en --detect_lang yes --backup yes
`

### 3. QLoRA Training (Review-2 focus pairs)
`ash
accelerate launch scripts/train_lora.py --config configs/qlora_hi_en.yaml
accelerate launch scripts/train_lora.py --config configs/qlora_ta_en.yaml
`
Outputs land in outputs/<pair>_r16/ with adapters, tokenizer snapshots, and validation predictions. MLflow logs the run in ./mlruns.

### 4. Evaluation
`ash
python scripts/eval_metrics.py --config configs/eval.yaml --pair hi-en \
  --refs data/clean/hi_en/test.tsv --hyps outputs/hi_en_r16/preds_test.txt \
  --out artifacts/review2/metrics_hi_en.json

python scripts/eval_metrics.py --config configs/eval.yaml --pair ta-en \
  --refs data/clean/ta_en/test.tsv --hyps outputs/ta_en_r16/preds_test.txt \
  --out artifacts/review2/metrics_ta_en.json
`
Each run writes JSON + TSV metrics and examples_<pair>.tsv under rtifacts/review2/.

### 5. Quick CLI Inference
`ash
python scripts/infer.py --model outputs/hi_en_r16 --src_lang hi --tgt_lang en \
  --style formal --simplify yes --text "?? ?? ??? ????? ???"
`
The script automatically falls back to FP16/CPU when bitsandbytes or CUDA are unavailable.

### 6. FastAPI Demo (70% cut: hi?en, ta?en)
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
| hi?en | **Show**  cleaned data, adapter, metrics, demo |
| ta?en | **Show**  cleaned data, adapter, metrics, demo |
| te?en | Off-stage  configs ready, data pipeline shared |
| ml?en | Off-stage  configs ready, data pipeline shared |

Launch Review-2 using the commands above; keep te?en and ml?en prepared but unpublished until the full review phase.
