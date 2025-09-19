# IndicTrans-LoRA Technical Brief

## 1. Background
- **Objective**: simplify and control formality for Indic?English translation using lightweight adapters on IndicTrans2.
- **Motivation**: editorial + customer support teams need faster domain adaptation without full model retraining.
- **Scope**: hi?en, ta?en in spotlight for Review-2; te?en, ml?en prepared but hidden.

## 2. Data Pipeline
- **Sources**: curated bilingual corpora with optional style tags; synthetic expansions capped at 5% (future work).
- **Manifest-driven ingestion**: data/manifests/*.yaml controls normalization, dedupe, length filters.
- **Cleaning outputs**: data/clean/<pair>/all.tsv + stats.json, deterministic splits via scripts/make_splits.py (seed=42).
- **Quality guardrails**: Unicode NFC normalization, whitespace compaction, style defaulting (formal), simplify default (no).

## 3. Modeling Approach
- **Base checkpoint**: i4bharat/indictrans2-en-indic-1B loaded in 4-bit when bitsandbytes is available, otherwise FP16/CPU fallback.
- **Adapters**: PEFT LoRA (r=16, alpha=32, dropout=0.05) targeting attention projection layers.
- **Prompting**: <SRC_LANG> <TGT_LANG> <STYLE> <SIMPLIFY> ||| <TEXT> ensures controllable style/simplify behaviors.
- **Bidirectional coverage**: datasets mirrored (src/tgt swap) when config lists both directions.

## 4. Training Runtime
- **Script**: scripts/train_lora.py orchestrates dataset prep, tokenization, and Seq2SeqTrainer loop.
- **Hyperparameters**: LR 2e-4, batch 64, grad accumulation 4, max steps 12k, warmup ratio 0.03, eval every 500 steps.
- **Infrastructure**: accelerate launch multi-GPU aware; gradient checkpointing toggled via config.
- **Tracking**: MLflow (./mlruns) logs params, metrics, predictions (preds_val.txt), and saved adapters.

## 5. Evaluation
- **Script**: scripts/eval_metrics.py outputs BLEU (sacrebleu), SARI (custom implementation), FKGL (textstat).
- **Artifacts**: JSON + TSV metrics per pair under rtifacts/review2/, plus examples_<pair>.tsv with 10 qualitative samples.
- **Gap**: need human eval rubric for stylistic accuracy; plan to add COMET-Full + MIRROR style coverage in Review-3.

## 6. Inference & Demo
- **CLI**: scripts/infer.py loads adapters with PEFT, supports CPU/CUDA auto device.
- **Web**: demo/app.py FastAPI app exposes hi?en and ta?en only; lazy-loads adapters, form UI under demo/templates/.
- **Packaging**: docker/Dockerfile.infer builds CPU-only container; docker/compose.yaml publishes port 8080 with outputs mounted read-only.

## 7. Risks & Mitigations
- **Data sparsity** (te?en, ml?en): allocate crawler + annotation sprint; integrate synthetic paraphrases once quality guardrails ready.
- **Style drift**: experiment with classifier-in-the-loop validation; add constraint decoding for over-simplification.
- **Deployment**: validate adapter compatibility with GPU inference stacks (Triton/Inferentia) and monitor memory headroom when stacking adapters.

## 8. Next Steps
1. Complete Review-2 data prep + training runs; publish metrics + qualitative samples.
2. Expand evaluation to include human review and COMET/chrF.
3. Build automation scripts for batch inference + report generation.
4. Advance off-stage pairs into Review-3 once data QA completes.
