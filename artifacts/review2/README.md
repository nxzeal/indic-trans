# Review-2 Bundle

This directory hosts the trimmed deliverables for the Review-2 checkpoint. Keep hi<->en and ta<->en front-and-center; other pairs stay off-stage.

## Contents
- metrics_*.json / metrics_*.tsv – aggregate BLEU, SARI, FKGL for the showcased adapters.
- examples_*.tsv – ten qualitative examples per pair with space for reviewer notes.
- data_prep_summary.json – manifest provenance and cleaning stats.
- screenshots/ – drop UI captures (placeholder folder until assets are produced).
- ../../reports/review2_slides.md – slide outline (20–25 frames) for the live walkthrough.

## Data Snapshot (populate after scripts/data_prep.py runs)
| Pair | Clean rows | Deduped | Notes |
|------|------------|---------|-------|
| hi<->en | _TBD_ | _TBD_ | Formal default, Hindi source |
| ta<->en | _TBD_ | _TBD_ | Tamil source |

Values populate automatically from data/clean/*/stats.json after processing.

## Metrics Table Placeholder
| Pair | BLEU | SARI | FKGL |
|------|------|------|------|
| hi<->en | _pending_ | _pending_ | _pending_ |
| ta<->en | _pending_ | _pending_ | _pending_ |

Fill via scripts/eval_metrics.py once predictions exist (outputs/<pair>_r16/preds_test.txt).

## CLI Trace (copy/paste-ready)
`
python scripts/data_prep.py --manifest data/manifests/review2.yaml
python scripts/make_splits.py --pair hi-en --in data/clean/hi_en --out data/clean/hi_en
python scripts/make_splits.py --pair ta-en --in data/clean/ta_en --out data/clean/ta_en
accelerate launch scripts/train_lora.py --config configs/qlora_hi_en.yaml
accelerate launch scripts/train_lora.py --config configs/qlora_ta_en.yaml
python scripts/eval_metrics.py --config configs/eval.yaml --pair hi-en --refs data/clean/hi_en/test.tsv --hyps outputs/hi_en_r16/preds_test.txt --out artifacts/review2/metrics_hi_en.json
python scripts/eval_metrics.py --config configs/eval.yaml --pair ta-en --refs data/clean/ta_en/test.tsv --hyps outputs/ta_en_r16/preds_test.txt --out artifacts/review2/metrics_ta_en.json
`

## Screenshots Placeholder
- screenshots/hi-en-demo.png – FastAPI UI showing Hindi->English generation.
- screenshots/ta-en-demo.png – FastAPI UI showing Tamil->English generation.
- Add up to three more for CLI traces or MLflow dashboards.

## Risks & Next Steps
- **Data coverage**: source additional style-annotated corpora for te<->en and ml<->en before the full launch.
- **Evaluation**: add human eval rubric for formality control; integrate COMET or BLEURT for deeper quality checks.
- **Deployment**: validate adapter loading on GPU serving stack and measure quantized latency.
