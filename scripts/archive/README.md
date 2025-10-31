# Archived Scripts

These scripts were used during initial setup, data preparation, or have been superseded by newer versions. They're kept here for reference but are not needed for normal training/inference workflows.

---

## 📦 Contents

### One-Time Setup Scripts (Already Executed)

**`add_special_tokens_to_tokenizer.py`**
- Purpose: Add `<FORMAL>`, `<INFORMAL>`, `<SIMPL_Y>`, `<SIMPL_N>` to tokenizer
- Status: ✅ Already executed, tokenizer updated
- Kept for: Reference if need to setup new model

**`migrate_to_control_tokens.py`**
- Purpose: Convert legacy TSV files to special token format (v2)
- Status: ✅ Already executed, `*_v2.tsv` files created
- Kept for: Reference if need to migrate new data

**`build_train_from_all.py`**
- Purpose: Build training splits from all.tsv source
- Status: ✅ Already executed, data prepared
- Kept for: Rebuilding splits if needed

**`normalize_v4_to_legacy.py`**
- Purpose: Normalize various dataset formats to consistent schema
- Status: ✅ Already executed, data normalized
- Kept for: Reference for data schema

**`patch_dedupe_contradictions.py`**
- Purpose: Remove exact duplicates and contradictory pairs
- Status: ✅ Already executed, data cleaned
- Kept for: Reference for deduplication logic

**`patch_enforce_simplify_margin.py`**
- Purpose: Enforce minimum length difference between simplify=yes/no
- Status: ✅ Already executed, data validated
- Kept for: Reference for quality constraints

---

### Data Preparation Scripts

**`data_prep.py`**
- Purpose: Initial data cleaning from raw sources
- Status: Data already cleaned and in `data/clean/`
- Kept for: Processing new raw data sources

**`make_splits.py`**
- Purpose: Create train/val/test splits with deterministic seed
- Status: Splits already created
- Kept for: Creating new splits with different ratios

**`augment_controls.py`**
- Purpose: Augment dataset with control tag combinations
- Status: Data already has balanced controls
- Kept for: Augmenting new datasets

**`fetch_corpora.py`**
- Purpose: Download raw parallel corpora from sources
- Status: Data already downloaded and processed
- Kept for: Fetching additional language pairs or updates

**`fix_splits_direction.py`**
- Purpose: Fix language direction metadata in splits
- Status: Metadata already corrected
- Kept for: Fixing direction issues in new data

**`quick_data_audit.py`**
- Purpose: Quick statistics on dataset
- Status: Superseded by `dataset_sanity_suite.py` (more comprehensive)
- Kept for: Quick lightweight checks

---

### Optional/Superseded Scripts

**`translate_base_ip.py`**
- Purpose: Translate with base model only (no adapter)
- Status: Not needed for current workflow (we use adapters)
- Kept for: Comparing base vs fine-tuned output

**`eval_metrics.py`**
- Purpose: Old evaluation script
- Status: ✅ Superseded by `eval_control_adherence.py` (better metrics)
- Kept for: Reference implementation

**`watch_checkpoints.py`**
- Purpose: Monitor checkpoint creation during training
- Status: Nice-to-have utility
- Kept for: Convenience tool if needed

---

## 🔄 When to Use These

**During New Setup:**
- Use `add_special_tokens_to_tokenizer.py` for new base models
- Use `migrate_to_control_tokens.py` for new datasets

**During Data Expansion:**
- Use `fetch_corpora.py` to download new language pairs
- Use `data_prep.py` to clean new data
- Use `make_splits.py` to create splits

**For Debugging:**
- Use `translate_base_ip.py` to test base model
- Use `quick_data_audit.py` for fast stats

---

## 📝 Usage Examples

```bash
# Re-run tokenizer setup (if needed)
python scripts/archive/add_special_tokens_to_tokenizer.py

# Convert new data to v2 format
python scripts/archive/migrate_to_control_tokens.py

# Test base model translation
python scripts/archive/translate_base_ip.py \
    --text "नमस्ते" --src_lang hi --tgt_lang en

# Quick dataset audit
python scripts/archive/quick_data_audit.py data/clean/hi_en/train_v2.tsv
```

---

## 🗂️ Why Archived?

These scripts cluttered the main `scripts/` directory but are still valuable for reference. Archiving them:

- ✅ Keeps main directory clean (8 scripts vs 23)
- ✅ Makes essential scripts easy to find
- ✅ Preserves scripts for future reference
- ✅ Maintains working copies if needed later

---

**See:** [../README.md](../README.md) for active scripts and common workflows.
