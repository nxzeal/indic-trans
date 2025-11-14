# IndicTrans2 LoRA: Style & Simplification Control

Fine-tuning [AI4Bharat's IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) (1B parameters) with QLoRA to add controllable style (formal/informal) and simplification to Hindi→English translation.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 🎯 Overview

This project demonstrates parameter-efficient fine-tuning of neural machine translation models using LoRA adapters with style and simplification controls.

**Key Achievements:**
- ✅ QLoRA training pipeline with 4-bit quantization (~8GB VRAM)
- ✅ Production-ready web demo with modern UI
- ✅ BLEU retention: 102.5% (translation quality preserved)
- ✅ Rule-based post-processing for reliable style control
- ❌ Model-based style learning: LoRA r=8 insufficient for style shifts

---

## 🚀 Quick Start

### 1. Setup

```bash
# Clone and setup environment
git clone https://github.com/nxzeal/indictrans-lora.git
cd indictrans-lora
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download base model
huggingface-cli download ai4bharat/indictrans2-indic-en-1B --local-dir models/indictrans2-indic-en-1B
```

### 2. Run Demo

```bash
./demo/start_demo.sh
# Open http://localhost:8000
```

The demo uses:
- **Model:** checkpoint-8000 (final trained adapter)
- **Post-processing:** Rule-based formalization/informalization/simplification
- **Features:** Animated gradient UI, auto-scroll results, style badges

---

## 💡 Usage

### Command-Line Translation

```bash
# With adapter (formal style, no simplification)
python scripts/translate_adapter_ip.py \
  --adapter outputs/hi_en_r8_v5_full/checkpoint-8000 \
  --src hi --tgt en \
  --style formal --simplify no \
  --text "कृपया दरवाज़ा बंद कर दें"

# Output: Could you please close the door.
```

```bash
# Base model only (no adapter)
python scripts/translate_base_ip.py \
  --base models/indictrans2-indic-en-1B \
  --src hi --tgt en \
  --num_beams 4 \
  --text "नमस्ते"
```

### Control Parameters

- `--style`: `formal` or `informal`
- `--simplify`: `yes` or `no`
- `--enforce_controls`: `on` (default) applies post-processing rules

---

## 🔬 Training

### Dataset (v3)

- **Training samples:** 110MB (~318K samples)
- **Format:** Vocabulary control tokens (formal, casual, simple, detailed)
- **Location:** `data/clean/hi_en/train_v3.tsv`

### Train LoRA Adapter

```bash
python scripts/train_lora.py --config configs/qlora_hi_en_v3.yaml
```

**Configuration:**
- LoRA rank: 8
- Learning rate: 2.0e-4
- Batch size: 4 (effective: 64 with gradient accumulation)
- Steps: 10,000 (checkpoint-8000 used for demo)

**Hardware:** RTX 3050 Laptop (6GB VRAM) with 4-bit quantization

---

## 📊 Results & Findings

### Translation Quality

**BLEU Retention Test (val50 dataset):**
- Base model: 37.54
- Adapter model (checkpoint-8000): 38.48
- **Retention: 102.5%** ✅

Core translation capability preserved after LoRA fine-tuning.

### Style Control

**Key Finding:** LoRA with r=8 **cannot learn** style shifts (formal ↔ informal) on a 1B parameter model.

**Evidence:**
- All 4 control combinations produce identical outputs
- Formal/informal adherence: ~48-52% (random chance)
- Model ignores control tokens during generation

**Solution:** Rule-based post-processing (`_formalize()`, `_informalize()`, `_simplify_yes()`)
- Reliable style transformation
- Handles edge cases (duplicate "please", capitalization, contractions)
- 82% test pass rate on comprehensive test suite

### Post-Processing Features

**Formalization:**
- Converts imperatives → "Could you please..." scaffolding
- Expands contractions (don't → do not)
- Formal vocabulary (but → however, so → therefore)
- Proper capitalization after sentence boundaries

**Informalization:**
- Adds contractions (do not → don't)
- Removes "please"
- Casual vocabulary (however → but)
- Direct tone

**Simplification:**
- Vocabulary replacement (30+ complex→simple mappings)
- Intensifier removal (very, really, extremely)
- Parenthetical removal
- Complex phrase simplification

---

## 📁 Repository Structure

```
indictrans-lora/
├── models/indictrans2-indic-en-1B/    [4.2GB]  Base model
├── outputs/hi_en_r8_v5_full/
│   └── checkpoint-8000/               [20MB]   Demo checkpoint
├── data/clean/hi_en/
│   ├── train_v3.tsv                   [110MB]  Training data
│   ├── val_v3.tsv                     [588KB]  Validation data
│   ├── test_v3.tsv                    [575KB]  Test data
│   └── splits_research_v3/            [112MB]  Split structure
│       └── splits_synthetic_casual/   [3.4MB]  10K synthetic samples
├── scripts/
│   ├── train_lora.py                           Training script
│   ├── translate_adapter_ip.py                 Inference with post-processing
│   ├── infer.py                                Clean inference
│   ├── eval_control_adherence.py               Evaluation
│   └── generate_synthetic_gemini.py            Synthetic data generation
├── configs/
│   ├── qlora_hi_en_v3.yaml                     Current training config
│   └── eval.yaml                               Evaluation config
├── demo/
│   ├── app.py                                  FastAPI application
│   ├── start_demo.sh                           Startup script
│   └── templates/index.html                    Modern gradient UI
├── artifacts/
│   ├── bleu_retention/                         BLEU test results
│   └── eval_reports/                           Checkpoint-8000 evaluation
└── README.md                                   This file
```

---

## 🛠️ Key Scripts

| Script | Purpose |
|--------|---------|
| `train_lora.py` | QLoRA training with MLflow tracking |
| `translate_adapter_ip.py` | Inference with post-processing rules |
| `infer.py` | Clean inference without post-processing |
| `eval_control_adherence.py` | Evaluate control following |
| `generate_synthetic_gemini.py` | Generate synthetic training data |
| `rebuild_dataset_vocab_tokens.py` | Build v3 dataset |

---

## 📖 Documentation

- **[FINAL_REVIEW_TECHNICAL_SUMMARY.md](FINAL_REVIEW_TECHNICAL_SUMMARY.md)** - Comprehensive project review
- **[CLEANUP_RECOMMENDATIONS.md](CLEANUP_RECOMMENDATIONS.md)** - Repository cleanup plan

---

## 🔍 Evaluation

### Control Adherence (checkpoint-8000)

```bash
python scripts/eval_control_adherence.py \
  --checkpoint_dir outputs/hi_en_r8_v5_full/checkpoint-8000 \
  --num_samples 200
```

**Results:**
- Formal adherence: 48.0% (FAIL - random chance)
- Informal adherence: 52.0% (FAIL - random chance)
- Model produces identical outputs regardless of control tokens

### BLEU Retention

```bash
# Results in artifacts/bleu_retention/
python scripts/translate_base_ip.py --file artifacts/review2/val50.src > base_output.txt
python scripts/translate_adapter_ip.py --file artifacts/review2/val50.src > adapter_output.txt
# Compare with sacrebleu
```

---

## 💡 Lessons Learned

1. **LoRA Limitations:** Small rank (r=8) insufficient for learning style transformations on 1B models
   - Translation quality preserved ✅
   - Style control failed ❌
   - Future: Try r=32-64 or full fine-tuning

2. **Post-Processing Works:** Rule-based transformations provide reliable style control
   - Deterministic and debuggable
   - Handles edge cases well
   - Production-ready for demo

3. **Synthetic Data:** Gemini-generated 10K samples useful for demo diversity
   - Stored in `data/clean/hi_en/splits_synthetic_casual/`

4. **Efficient Training:** 4-bit quantization enables training on consumer GPUs
   - RTX 3050 (6GB) sufficient
   - ~10 hours for 10K steps

---

## 🧪 Technical Details

**Base Model:**
- ai4bharat/indictrans2-indic-en-1B
- 1 billion parameters
- Transformer encoder-decoder

**LoRA Configuration:**
- Rank: 8
- Alpha: 16
- Dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj
- Trainable parameters: ~4.7M (0.47% of base)

**Quantization:**
- 4-bit NF4 quantization (bitsandbytes)
- Compute dtype: float16
- Memory: ~8GB during training, ~4GB during inference

**Generation:**
- Beam search: 1-4 beams
- Max tokens: 128
- No sampling (deterministic)

---

## 🌐 Demo Features

- Modern animated gradient background
- Plus Jakarta Sans font
- Glassmorphism effects
- Auto-scroll to results
- Style/simplify toggle controls
- Professional globe translation icon
- Mobile-responsive design

---

## 📝 Citation

```bibtex
@software{indictrans_lora_2025,
  title = {IndicTrans2 LoRA: Style and Simplification Control},
  author = {Nazeel},
  year = {2025},
  url = {https://github.com/nxzeal/indictrans-lora}
}
```

**Original IndicTrans2:**
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

MIT License - See [LICENSE](LICENSE) file

Base IndicTrans2 model: MIT License by AI4Bharat

---

## 🙏 Acknowledgments

- **AI4Bharat** for IndicTrans2
- **Hugging Face** for transformers and PEFT
- **bitsandbytes** for efficient quantization

---

**Status:** ✅ Production Demo Ready | **Last Updated:** 2025-11-14
