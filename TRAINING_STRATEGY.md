# Training Strategy: Quick Validation vs Full Training

> Understanding why 3,600 steps isn't enough and what to do about it

---

## 📊 The Math

**Your dataset:**
- Training samples: 318,800
- Effective batch size: 64 (4 per-device × 16 grad_accum)
- Steps per epoch: 4,981

**Current config (3,600 steps):**
- Epochs: **0.72** (72% of data seen once)
- Training time: ~7-18 hours (GPU dependent)
- Result: Weak control following

**Recommended (10,000 steps):**
- Epochs: **2.0** (full dataset seen twice)
- Training time: ~20-50 hours (GPU dependent)
- Result: Strong control following

---

## 🎯 Two-Phase Approach

### Phase 1: Quick Validation ✅ START HERE

**Config:** `configs/qlora_hi_en.yaml` (3,600 steps)

**Purpose:**
1. ✅ Verify setup works (tokenizer, data, model loading)
2. ✅ Test training completes without errors
3. ✅ Get baseline checkpoint to play with
4. ✅ See initial control behavior

**Command:**
```bash
python scripts/train_lora.py --config configs/qlora_hi_en.yaml
```

**Time Investment:**
- ~7-18 hours total (split into 2-4 hour sessions)
- Low risk: shorter training = less heat exposure
- Quick feedback on whether approach works

**Expected Results:**
- ✅ Model translates Hindi→English well
- ⚠️ Controls are inconsistent/weak
- ⚠️ Need post-processing to enforce controls
- ⚠️ Formal vs informal not clearly different

**Testing after Phase 1:**
```bash
# WITHOUT post-processing (raw model output)
python scripts/translate_adapter_ip.py \
    --adapter outputs/hi_en_r8_v5/checkpoint-3600 \
    --text "क्या आप बता सकते हैं?" \
    --style formal --simplify no \
    --enforce_controls off

# WITH post-processing (forced controls)
python scripts/translate_adapter_ip.py \
    --adapter outputs/hi_en_r8_v5/checkpoint-3600 \
    --text "क्या आप बता सकते हैं?" \
    --style formal --simplify no \
    --enforce_controls on
```

**You'll probably see:**
- Raw output: "Can you tell me?" (informal, even though you asked for formal)
- Post-processed: "Could you please tell me?" (forced to formal)

**This tells you:** Model hasn't learned controls yet, just translation.

---

### Phase 2: Full Training 🚀 DO THIS AFTER PHASE 1

**Config:** `configs/qlora_hi_en_full.yaml` (10,000 steps)

**Purpose:**
1. Train long enough for model to learn control distinctions
2. Get model that naturally follows controls without post-processing
3. Achieve publication-quality results

**Command:**
```bash
python scripts/train_lora.py --config configs/qlora_hi_en_full.yaml
```

**Key Differences:**
```yaml
max_steps: 10000           # vs 3600 (2.8× longer)
output_dir: "outputs/hi_en_r8_v5_full"  # Different folder
save_every: 1000           # vs 600 (fewer checkpoints)
save_total_limit: 3        # vs 2 (keep more checkpoints)
```

**Time Investment:**
- ~20-50 hours total (GPU dependent)
- Split into 2-4 hour sessions over 1-2 weeks
- Higher heat exposure: follow cooling protocols strictly

**Expected Results:**
- ✅ Model translates well AND follows controls
- ✅ Formal outputs actually sound formal
- ✅ Informal outputs use contractions naturally
- ✅ Simplification reduces length appropriately
- ✅ Can use `--enforce_controls off` and still get good results

**Testing after Phase 2:**
```bash
# Raw output should now respect controls
python scripts/translate_adapter_ip.py \
    --adapter outputs/hi_en_r8_v5_full/checkpoint-10000 \
    --text "क्या आप बता सकते हैं?" \
    --style formal --simplify no \
    --enforce_controls off
```

**You should see:**
- Raw output: "Could you please tell me?" (naturally formal!)
- Post-processing not needed anymore

---

## 📈 Training Progression

### What Happens During Training

**Steps 0-1000 (0-0.2 epochs):**
- Model learns basic translation
- Control tokens mostly ignored
- High loss, rapid improvement

**Steps 1000-3000 (0.2-0.6 epochs):**
- Translation quality improves
- Some control awareness emerges
- Loss plateaus

**Steps 3000-5000 (0.6-1.0 epochs):**
- First full pass through data
- Controls start influencing output
- But still inconsistent

**Steps 5000-7500 (1.0-1.5 epochs):**
- Second pass reinforces patterns
- Control distinctions become clearer
- Formal vs informal diverge

**Steps 7500-10000 (1.5-2.0 epochs):**
- Strong control adherence
- Natural-sounding outputs
- Minimal post-processing needed

---

## 🔬 Evaluation at Each Phase

### After Phase 1 (3,600 steps)

```bash
# Run control adherence evaluation
python scripts/eval_control_adherence.py \
    --checkpoint_dir outputs/hi_en_r8_v5/checkpoint-3600 \
    --num_samples 500
```

**Expected metrics:**
- Formal adherence: ~0.50-0.60 (weak)
- Informal adherence: ~0.40-0.50 (weak)
- Simplify ratio: ~0.95 (almost no effect)

**Interpretation:** Model hasn't learned controls yet.

### After Phase 2 (10,000 steps)

```bash
# Run control adherence evaluation
python scripts/eval_control_adherence.py \
    --checkpoint_dir outputs/hi_en_r8_v5_full/checkpoint-10000 \
    --num_samples 500
```

**Expected metrics:**
- Formal adherence: **0.75-0.85** (strong) ✅
- Informal adherence: **0.70-0.80** (strong) ✅
- Simplify ratio: **0.86-0.90** (in target range) ✅

**Interpretation:** Model has learned controls!

---

## 💡 Why Two Phases?

### Benefits of Phase 1 First

**Risk Mitigation:**
- ✅ Shorter training = less overheating risk
- ✅ Quick validation of setup
- ✅ Catch errors early (6 hours vs 40 hours wasted)
- ✅ Test inference pipeline with real checkpoint

**Learning:**
- ✅ Understand training speed on your GPU
- ✅ See control behavior (even if weak)
- ✅ Calibrate expectations
- ✅ Decide if Phase 2 is worth it

**Practical:**
- ✅ Can demo basic translation immediately
- ✅ Have checkpoint to test evaluation scripts
- ✅ Confidence to commit to longer training

### When to Skip Phase 1

**Skip directly to Phase 2 if:**
- ✅ You've trained LoRA adapters before
- ✅ Very confident in your setup
- ✅ Need strong control adherence immediately
- ✅ Have good cooling solution
- ✅ Can commit to 1-2 weeks of training sessions

**Most people should do Phase 1 first!**

---

## 🎯 Recommended Path

### For You (First Time Training)

**Week 1: Phase 1 Validation**
```bash
# Days 1-2: Train 3600 steps
python scripts/train_lora.py --config configs/qlora_hi_en.yaml

# Day 3: Evaluate
python scripts/eval_control_adherence.py \
    --checkpoint_dir outputs/hi_en_r8_v5/checkpoint-3600 \
    --num_samples 500

# Day 3: Test inference
python scripts/translate_adapter_ip.py \
    --adapter outputs/hi_en_r8_v5/checkpoint-3600 \
    --text "परीक्षण वाक्य" --style formal --simplify yes
```

**Week 2+: Decide**

**Option A - Good Enough:**
- Phase 1 checkpoint works well with post-processing
- Use `--enforce_controls on` in production
- Stop here, save time

**Option B - Need Better Control:**
- Phase 1 shows weak control adherence
- Want model to naturally follow controls
- Continue to Phase 2

**Week 2-3: Phase 2 Full Training** (if needed)
```bash
# Train 10000 steps over 1-2 weeks
python scripts/train_lora.py --config configs/qlora_hi_en_full.yaml

# Evaluate
python scripts/eval_control_adherence.py \
    --checkpoint_dir outputs/hi_en_r8_v5_full/checkpoint-10000 \
    --num_samples 500
```

---

## 📊 Time Estimates by GPU

### Phase 1 (3,600 steps)

| GPU | Time | Sessions (3h each) | Days (2 sessions/day) |
|-----|------|-------------------|---------------------|
| RTX 4090 | 7h | 3 | 2 days |
| RTX 3090 | 10h | 4 | 2 days |
| RTX 3080 | 12h | 4 | 2 days |
| RTX 3070 | 14h | 5 | 3 days |
| RTX 3060 | 18h | 6 | 3 days |

### Phase 2 (10,000 steps)

| GPU | Time | Sessions (3h each) | Days (2 sessions/day) |
|-----|------|-------------------|---------------------|
| RTX 4090 | 20h | 7 | 4 days |
| RTX 3090 | 28h | 10 | 5 days |
| RTX 3080 | 33h | 11 | 6 days |
| RTX 3070 | 39h | 13 | 7 days |
| RTX 3060 | 50h | 17 | 9 days |

---

## 🔄 Comparison Table

| Aspect | Phase 1 (3,600 steps) | Phase 2 (10,000 steps) |
|--------|----------------------|------------------------|
| **Epochs** | 0.72 | 2.0 |
| **Time** | ~7-18h | ~20-50h |
| **Heat Risk** | Lower | Higher |
| **Control Quality** | Weak (needs post-processing) | Strong (natural) |
| **Translation Quality** | Good | Good |
| **Use Case** | Quick demo, validation | Production, research |
| **Post-processing** | Required | Optional |
| **Formal Adherence** | ~0.50-0.60 | ~0.75-0.85 |
| **Informal Adherence** | ~0.40-0.50 | ~0.70-0.80 |

---

## 🎯 Final Recommendation

### START HERE:

```bash
# Phase 1: Quick validation (do this first!)
python scripts/train_lora.py --config configs/qlora_hi_en.yaml
```

### THEN EVALUATE:

```bash
# Test control adherence
python scripts/eval_control_adherence.py \
    --checkpoint_dir outputs/hi_en_r8_v5/checkpoint-3600 \
    --num_samples 500

# Try translations
python scripts/translate_adapter_ip.py \
    --adapter outputs/hi_en_r8_v5/checkpoint-3600 \
    --text "कृपया दरवाज़ा बंद करें" \
    --style formal --simplify no \
    --enforce_controls off  # See raw model output
```

### THEN DECIDE:

**If satisfied with Phase 1 + post-processing:**
- ✅ Stop here
- Use `--enforce_controls on` for demos
- Save 2 weeks of training time

**If need natural control following:**
- 🚀 Continue to Phase 2
- Train full 10,000 steps
- Get publication-quality results

---

**TL;DR:**
- 3,600 steps = only 0.72 epochs = weak controls
- Do Phase 1 first (validate setup, 1-2 days)
- Evaluate results
- Then decide: stop or continue to Phase 2 (1-2 weeks)

**You don't have to commit to full training upfront!** ✅
