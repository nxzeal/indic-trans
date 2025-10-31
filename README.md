# IndicTrans2 LoRA: Controllable Style & Simplification

> Fine-tune IndicTrans2 with QLoRA for controllable formality and simplification in Hindi-English translation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 Overview

This repository fine-tunes [AI4Bharat's IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) (1B parameter model) using **QLoRA** (4-bit quantization) to add controllable style and simplification to Hindi→English translation.

### Key Features

- **🎨 Style Control:** Generate formal vs informal translations
- **📝 Simplification Control:** Toggle between standard and simplified output
- **⚡ Efficient Training:** QLoRA with 4-bit quantization (~8GB VRAM)
- **🔤 Special Tokens:** Unambiguous control signals (`<FORMAL>`, `<INFORMAL>`, `<SIMPL_Y>`, `<SIMPL_N>`)
- **📊 Quality Metrics:** Automated evaluation of control adherence
- **🌐 Demo App:** FastAPI web interface for interactive translation

### Control Examples

| Source (Hindi) | Style | Simplify | Output (English) |
|---------------|-------|----------|------------------|
| क्या आप मुझे बता सकते हैं? | Formal | No | Could you please tell me? |
| क्या आप मुझे बता सकते हैं? | Informal | No | Can you tell me? |
| यह एक जटिल प्रक्रिया है। | Formal | Yes | This is a complex process. |
| यह एक जटिल प्रक्रिया है। | Informal | Yes | It's a tough process. |

---

## 📂 Repository Structure

```
├── configs/
│   └── qlora_hi_en.yaml         # Training configuration
├── data/clean/hi_en/
│   ├── train_v2.tsv             # 318K training samples (special tokens)
│   ├── val_v2.tsv               # 1.6K validation samples
│   └── test_v2.tsv              # Test set
├── scripts/
│   ├── train_lora.py            # QLoRA training
│   ├── translate_adapter_ip.py  # Inference script
│   ├── eval_control_adherence.py # Control quality evaluation
│   └── ...                      # Utilities (see TECHNICAL_GUIDE.md)
├── demo/
│   └── app.py                   # FastAPI demo application
├── models/
│   └── indictrans2-indic-en-1B/ # Base model (download separately)
└── outputs/
    └── hi_en_r8_v5/             # Training checkpoints
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/nxzeal/indic-trans.git
cd indic-trans

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Base Model

Download the IndicTrans2 Indic-English model and place it in `models/indictrans2-indic-en-1B/`:

```bash
# Using Hugging Face CLI
huggingface-cli download ai4bharat/indictrans2-indic-en-1B --local-dir models/indictrans2-indic-en-1B
```

Or download manually from: https://huggingface.co/ai4bharat/indictrans2-indic-en-1B

### 3. Setup Special Tokens (One-time)

```bash
# Add control tokens to tokenizer
python scripts/add_special_tokens_to_tokenizer.py

# Convert training data to special token format
python scripts/migrate_to_control_tokens.py
```

This adds 4 control tokens:
- `<FORMAL>` (ID: 122706) - Formal style
- `<INFORMAL>` (ID: 122707) - Informal style
- `<SIMPL_Y>` (ID: 122708) - Simplification ON
- `<SIMPL_N>` (ID: 122709) - Simplification OFF

### 4. Train

```bash
python scripts/train_lora.py --config configs/qlora_hi_en.yaml
```

**Training Progress:**
- Checkpoints saved every 600 steps to `outputs/hi_en_r8_v5/checkpoint-{step}/`
- MLflow tracking enabled (view with `mlflow ui`)
- Expected duration: ~6-12 hours on single GPU (depends on hardware)

---

## 💡 Usage

### Command-Line Inference

```bash
# Translate with controls
python scripts/translate_adapter_ip.py \
    --adapter outputs/hi_en_r8_v5/checkpoint-3600 \
    --text "क्या आप मुझे बता सकते हैं कि यह कैसे काम करता है?" \
    --style formal \
    --simplify yes
```

**Output:**
```
Could you please tell me how this works?
```

### Batch Translation

```bash
# Create input file
cat > input.txt <<EOF
नमस्ते, आप कैसे हैं?
मुझे यह समझ में नहीं आ रहा है।
कृपया मेरी मदद करें।
EOF

# Translate batch
python scripts/translate_adapter_ip.py \
    --adapter outputs/hi_en_r8_v5/checkpoint-3600 \
    --file input.txt \
    --style informal \
    --simplify no
```

### Web Demo

```bash
# Set environment variables
export ADAPTER_DIR=outputs/hi_en_r8_v5/checkpoint-3600
export BASE_MODEL_DIR=models/indictrans2-indic-en-1B

# Launch demo
cd demo
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

---

## 📊 Evaluation

### Control Adherence Metrics

Evaluate how well the model follows style and simplification controls:

```bash
python scripts/eval_control_adherence.py \
    --checkpoint_dir outputs/hi_en_r8_v5/checkpoint-3600 \
    --num_samples 500
```

**Metrics:**
- **Formal Adherence:** % without contractions + % with formal markers
- **Informal Adherence:** % with contractions + % without excessive formality
- **Simplify Adherence:** Length ratio (target: 0.85-0.92)

**Output Files:**
- `{checkpoint}_adherence_report.json` - Full metrics
- `{checkpoint}_adherence_summary.txt` - Human-readable summary
- `{checkpoint}_samples.tsv` - Generated samples

### Data Quality Checks

Validate training data quality:

```bash
python scripts/dataset_sanity_suite.py data/clean/hi_en/train_v2.tsv
```

---

## 📖 Documentation

- **[TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md)** - Comprehensive technical documentation
  - Complete repository walkthrough
  - All scripts explained in detail
  - Training workflow and configurations
  - Troubleshooting guide

- **[MIGRATION_TO_SPECIAL_TOKENS.md](MIGRATION_TO_SPECIAL_TOKENS.md)** - Special token migration guide
  - Why we migrated from legacy format
  - Token mapping and usage
  - Backward compatibility

---

## 🔧 Configuration

### Training Config (`configs/qlora_hi_en.yaml`)

Key parameters:

```yaml
lora:
  r: 8                    # LoRA rank
  alpha: 16               # Scaling factor
  dropout: 0.05

train:
  lr: 2.0e-4              # Learning rate
  batch_size: 4           # Per-device batch size
  grad_accum: 16          # Gradient accumulation steps
  max_steps: 3600         # Total training steps
  save_every: 600         # Checkpoint frequency

quantization:
  load_in_4bit: true      # 4-bit quantization
  bnb_4bit_quant_type: "nf4"
```

**Memory Requirements:**
- **Training:** ~8GB VRAM (4-bit quantization + gradient checkpointing)
- **Inference:** ~4GB VRAM (4-bit quantization)

---

## 📦 Dataset

### Current Dataset (v2)

- **Format:** Special tokens (`<FORMAL>`, `<INFORMAL>`, `<SIMPL_Y>`, `<SIMPL_N>`)
- **Size:** 318,800 training samples
- **Language Pair:** Hindi (Devanagari) → English (Latin)
- **Quality:** ✅ All validation checks passed

### Control Distribution

Balanced 4-way split:

| Control Combination | Count | Percentage |
|---------------------|--------|-----------|
| formal + no         | 79,744 | 25.0% |
| formal + yes        | 79,699 | 25.0% |
| informal + no       | 79,744 | 25.0% |
| informal + yes      | 79,613 | 25.0% |

### Dataset Schema

TSV format with columns:

```
src | tgt | style_src | style_tgt | simplify | pair
```

**Example:**
```tsv
क्या आप बता सकते हैं?	Could you please tell me?	<FORMAL>	<FORMAL>	<SIMPL_N>	hi-en
```

---

## 🛠️ Scripts Reference

| Script | Purpose |
|--------|---------|
| `train_lora.py` | Train QLoRA adapter on IndicTrans2 |
| `translate_adapter_ip.py` | Inference with control tokens |
| `eval_control_adherence.py` | Evaluate control following quality |
| `migrate_to_control_tokens.py` | Convert legacy to v2 format |
| `add_special_tokens_to_tokenizer.py` | Setup tokenizer with control tokens |
| `dataset_sanity_suite.py` | Comprehensive data quality checks |
| `quick_data_audit.py` | Quick dataset statistics |

See **[TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md)** for detailed documentation of all scripts.

---

## 🔄 Special Token Migration

This repository uses **special control tokens** (v2 format) instead of plain text controls.

### Why?

**Legacy (v1):**
```
hin_Deva eng_Latn formal yes ||| नमस्ते
```
- Ambiguous: "formal" could be confused with content

**Special Tokens (v2):**
```
hin_Deva eng_Latn <FORMAL> <SIMPL_Y> ||| नमस्ते
```
- Unambiguous: Clear control signals
- Stronger attention during training

### Backward Compatibility

All scripts support both formats via `--token_format` flag:

```bash
# New checkpoints (default)
python scripts/translate_adapter_ip.py --adapter outputs/NEW/checkpoint-3600 ...

# Old checkpoints
python scripts/translate_adapter_ip.py --adapter outputs/OLD/checkpoint-1200 ... --token_format legacy
```

---

## 🧪 Model Details

### Base Model

- **Name:** ai4bharat/indictrans2-indic-en-1B
- **Architecture:** Transformer (encoder-decoder)
- **Parameters:** 1 billion
- **Training:** Pre-trained on large-scale Indic-English parallel data

### LoRA Adapter

- **Method:** QLoRA (4-bit quantization)
- **Rank:** 8
- **Target Modules:** Q, K, V, O projections in attention
- **Trainable Parameters:** ~4.7M (0.47% of base model)

### Generation

- **Beam Search:** 4 beams
- **Max Length:** 192 tokens
- **No Sampling:** Deterministic output

---

## 📈 Performance

### Quality Metrics (Checkpoint 3600)

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Formal Adherence | 0.84 | ≥0.70 | ✅ PASS |
| Informal Adherence | 0.76 | ≥0.70 | ✅ PASS |
| Simplify Length Ratio | 0.88 | 0.85-0.92 | ✅ PASS |

### Dataset Quality

- ✅ Zero duplicates
- ✅ Zero contradictions
- ✅ 99.7% correct scripts (Devanagari → Latin)
- ✅ Balanced control distribution

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Additional language pairs (Tamil, Telugu, Malayalam)
- [ ] Better control adherence metrics
- [ ] Reinforcement learning from human feedback (RLHF)
- [ ] Model distillation for faster inference
- [ ] Mobile deployment

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@software{indictrans_lora_2025,
  title = {IndicTrans2 LoRA: Controllable Style and Simplification},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/nxzeal/indic-trans}
}
```

Also cite the original IndicTrans2 paper:

```bibtex
@article{gala2023indictrans,
  title={IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages},
  author={Gala, Jay and others},
  journal={Transactions on Machine Learning Research},
  year={2023}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The base IndicTrans2 model is licensed under MIT by AI4Bharat.

---

## 🙏 Acknowledgments

- **AI4Bharat** for the IndicTrans2 base model
- **Hugging Face** for transformers and PEFT libraries
- **bitsandbytes** for efficient quantization

---

## 📞 Contact

For questions or issues, please:
- Open an issue on [GitHub](https://github.com/nxzeal/indic-trans/issues)
- See [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md) for troubleshooting

---

**Status:** ✅ Production-ready | **Last Updated:** 2025-10-31
