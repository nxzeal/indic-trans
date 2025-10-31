# Scripts Directory

Clean, organized scripts for training, inference, and evaluation.

---

## 🎯 Essential Scripts (Active)

### Training
- **`train_lora.py`** - Main training script for QLoRA fine-tuning
  ```bash
  python scripts/train_lora.py --config configs/qlora_hi_en_full.yaml
  ```

### Inference
- **`translate_adapter_ip.py`** - Translate with LoRA adapter
  ```bash
  python scripts/translate_adapter_ip.py \
      --adapter outputs/hi_en_r8_v5_full/checkpoint-10000 \
      --text "नमस्ते" --style formal --simplify yes
  ```

### Evaluation
- **`eval_control_adherence.py`** - Evaluate control quality
  ```bash
  python scripts/eval_control_adherence.py \
      --checkpoint_dir outputs/hi_en_r8_v5_full/checkpoint-10000 \
      --num_samples 500
  ```

- **`dataset_sanity_suite.py`** - Data quality checks
  ```bash
  python scripts/dataset_sanity_suite.py data/clean/hi_en/train_v2.tsv
  ```

### Utilities
- **`infer.py`** - Helper functions for model loading and inference
- **`utils_io.py`** - I/O utilities (read TSV, write JSON)
- **`utils_text.py`** - Text processing utilities

---

## 📦 Archived Scripts

**Location:** `scripts/archive/`

These scripts were used during initial setup or data preparation. They're archived but still accessible if needed.

### One-Time Setup (Already Executed)
- `add_special_tokens_to_tokenizer.py` - Add control tokens (already done)
- `migrate_to_control_tokens.py` - Convert data to v2 format (already done)
- `build_train_from_all.py` - Build training splits (already done)
- `normalize_v4_to_legacy.py` - Normalize data format (already done)
- `patch_dedupe_contradictions.py` - Remove duplicates (already done)
- `patch_enforce_simplify_margin.py` - Enforce length ratios (already done)

### Data Preparation (Data Already Prepared)
- `data_prep.py` - Initial data cleaning
- `make_splits.py` - Create train/val/test splits
- `augment_controls.py` - Augment control combinations
- `fetch_corpora.py` - Download raw data sources
- `fix_splits_direction.py` - Fix language direction metadata
- `quick_data_audit.py` - Quick dataset statistics

### Optional/Superseded
- `translate_base_ip.py` - Translate without adapter (base model only)
- `eval_metrics.py` - Old evaluation script (superseded by eval_control_adherence.py)
- `watch_checkpoints.py` - Monitor checkpoint creation (nice-to-have)

**To use archived scripts:**
```bash
python scripts/archive/script_name.py [arguments]
```

---

## 🚀 Quick Reference

### Most Common Commands

**Training:**
```bash
# Full training (10,000 steps)
python scripts/train_lora.py --config configs/qlora_hi_en_full.yaml

# Quick validation (3,600 steps)
python scripts/train_lora.py --config configs/qlora_hi_en.yaml
```

**Translation:**
```bash
# Formal + No simplification
python scripts/translate_adapter_ip.py \
    --adapter outputs/CHECKPOINT \
    --text "कृपया दरवाज़ा बंद करें" \
    --style formal --simplify no

# Informal + Simplification
python scripts/translate_adapter_ip.py \
    --adapter outputs/CHECKPOINT \
    --text "कृपया दरवाज़ा बंद करें" \
    --style informal --simplify yes

# Batch translation from file
python scripts/translate_adapter_ip.py \
    --adapter outputs/CHECKPOINT \
    --file input.txt \
    --style formal --simplify no
```

**Evaluation:**
```bash
# Control adherence metrics
python scripts/eval_control_adherence.py \
    --checkpoint_dir outputs/CHECKPOINT \
    --num_samples 500

# Data quality checks
python scripts/dataset_sanity_suite.py data/clean/hi_en/train_v2.tsv
```

---

## 📚 Full Documentation

- **Training Guide:** [../SAFE_TRAINING_GUIDE.md](../SAFE_TRAINING_GUIDE.md)
- **Quick Reference:** [../QUICK_TRAINING_REFERENCE.md](../QUICK_TRAINING_REFERENCE.md)
- **Training Strategy:** [../TRAINING_STRATEGY.md](../TRAINING_STRATEGY.md)
- **Technical Guide:** [../TECHNICAL_GUIDE.md](../TECHNICAL_GUIDE.md)

---

## 🔧 Development Notes

**Adding New Scripts:**
- Keep active/frequently-used scripts in `scripts/`
- Archive one-time/setup scripts in `scripts/archive/`
- Document purpose at top of script with docstring
- Add usage example in this README

**Script Naming Convention:**
- `train_*.py` - Training related
- `translate_*.py` - Translation/inference
- `eval_*.py` - Evaluation/metrics
- `*_prep.py` - Data preparation
- `utils_*.py` - Utility functions

---

**Last Updated:** 2025-10-31
