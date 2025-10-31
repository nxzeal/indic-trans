# Migration to Special Control Tokens

## Overview

This project has migrated from plain-text control tokens to special control tokens for stronger style and simplification signals.

### Why Special Tokens?

**Problem with legacy format:**
- Plain text controls like `formal`, `informal`, `yes`, `no` were ambiguous
- These words could appear naturally in the source text or translations
- Model struggled to distinguish control signals from actual content

**Benefits of special token format:**
- Unambiguous control signals using special tokens: `<FORMAL>`, `<INFORMAL>`, `<SIMPL_Y>`, `<SIMPL_N>`
- Stronger attention to control parameters during training
- Clear separation between control metadata and translation content

## Format Comparison

### Legacy Format (v1)
```
hin_Deva eng_Latn formal yes ||| यह एक उदाहरण है।
```

### Special Token Format (v2)
```
hin_Deva eng_Latn <FORMAL> <SIMPL_Y> ||| यह एक उदाहरण है।
```

## File Versions

### Training Data

| Version | Format | Files |
|---------|--------|-------|
| v1 (legacy) | Plain text controls | `data/clean/hi_en/train.tsv`<br>`data/clean/hi_en/val.tsv`<br>`data/clean/hi_en/test.tsv` |
| v2 (special) | Special tokens | `data/clean/hi_en/train_v2.tsv`<br>`data/clean/hi_en/val_v2.tsv`<br>`data/clean/hi_en/test_v2.tsv` |

### Checkpoints

| Checkpoint | Format | Notes |
|------------|--------|-------|
| `outputs/hi_en_r8_v2/*` | Legacy | Trained on plain text controls |
| `outputs/hi_en_r8_v5/*` | Legacy | Trained on plain text controls |
| Future checkpoints | Special | Will use special token format |

## Migration Steps

### 1. Add Special Tokens to Tokenizer

**IMPORTANT:** This must be done before training with v2 data.

```bash
# Add special tokens to the tokenizer configuration
python scripts/add_special_tokens_to_tokenizer.py
```

This script:
- Adds `<FORMAL>`, `<INFORMAL>`, `<SIMPL_Y>`, `<SIMPL_N>` to tokenizer
- Does NOT resize model embeddings (IndicTrans2 constraint)
- Tokens will be tokenized as subword units
- Updates `models/indictrans2-indic-en-1B/special_tokens_map.json`

### 2. Convert Training Data

```bash
# Convert train.tsv, val.tsv, test.tsv to v2 format
python scripts/migrate_to_control_tokens.py
```

This creates:
- `data/clean/hi_en/train_v2.tsv`
- `data/clean/hi_en/val_v2.tsv`
- `data/clean/hi_en/test_v2.tsv`

Original files are preserved unchanged.

### 3. Update Training Configuration

The config file `configs/qlora_hi_en.yaml` has been updated with comments indicating v2 format usage. To train with v2 data, ensure your training script uses `train_v2.tsv`.

## Usage

### Translation Script

The translation script supports both formats via `--token_format` argument:

```bash
# Special token format (default)
python scripts/translate_adapter_ip.py \
    --adapter outputs/hi_en_r8_NEW/checkpoint-3600 \
    --text "नमस्ते" \
    --style formal \
    --simplify yes \
    --token_format special

# Legacy format (for old checkpoints)
python scripts/translate_adapter_ip.py \
    --adapter outputs/hi_en_r8_v5/checkpoint-3600 \
    --text "नमस्ते" \
    --style formal \
    --simplify yes \
    --token_format legacy
```

### Evaluation Script

```bash
# Evaluate checkpoint trained with special tokens
python scripts/eval_control_adherence.py \
    --checkpoint_dir outputs/hi_en_NEW/checkpoint-3600 \
    --num_samples 500 \
    --token_format special

# Evaluate old checkpoint with legacy format
python scripts/eval_control_adherence.py \
    --checkpoint_dir outputs/hi_en_r8_v5/checkpoint-3600 \
    --num_samples 500 \
    --token_format legacy
```

### Demo App

The demo app (`demo/app.py`) has been updated to use special token format by default. It automatically uses `<FORMAL>`, `<INFORMAL>`, `<SIMPL_Y>`, `<SIMPL_N>` when an adapter is loaded.

## Token Mapping

| Control | Legacy | Special Token |
|---------|--------|---------------|
| Formal style | `formal` | `<FORMAL>` |
| Informal style | `informal` | `<INFORMAL>` |
| Simplification ON | `yes` | `<SIMPL_Y>` |
| Simplification OFF | `no` | `<SIMPL_N>` |

## Backward Compatibility

All scripts support both legacy and special token formats:

- **Default behavior:** Scripts now default to `special` format
- **Legacy support:** Use `--token_format legacy` for old checkpoints
- **Original data:** All original TSV files are preserved unchanged

## Training New Models

To train a new model with special tokens:

```bash
# 1. Ensure tokenizer has special tokens
python scripts/add_special_tokens_to_tokenizer.py

# 2. Convert data to v2 format
python scripts/migrate_to_control_tokens.py

# 3. Update your training script to use train_v2.tsv
# (Modify data loading to point to *_v2.tsv files)

# 4. Train as usual
python scripts/train_lora.py --config configs/qlora_hi_en.yaml
```

## Important Notes

### IndicTrans2 Constraint

⚠️ **We cannot resize token embeddings** due to IndicTrans2 model constraints.

This means:
- Special tokens are added to tokenizer only
- They are tokenized as subword units (e.g., `<`, `FORMAL`, `>`)
- Model learns to associate these subword patterns with control behavior
- No modification to model architecture required

### Example Tokenization

```python
# Input prompt
"hin_Deva eng_Latn <FORMAL> <SIMPL_Y> ||| नमस्ते"

# Tokenized (approximate, subword units)
['hin', '_De', 'va', 'eng', '_La', 'tn', '<', 'FOR', 'MAL', '>', '<', 'SIM', 'PL', '_Y', '>', '|||', ...]
```

The model learns that the pattern `< FOR MAL >` (or similar) represents formal style control.

## Questions?

- For issues or questions, open a GitHub issue
- Check script help: `python scripts/<script_name>.py --help`
