# Quick Training Reference Card

> Keep this handy during training sessions

---

## 🚀 The One Command You Need

```bash
python scripts/train_lora.py --config configs/qlora_hi_en.yaml
```

**That's it!** Run the same command every time. It auto-resumes from last checkpoint.

---

## ✅ Before Each Session (5 min checklist)

```bash
# 1. Check temperatures are at baseline
nvidia-smi  # GPU should be ~40°C

# 2. Start monitoring (separate terminal)
watch -n 1 nvidia-smi

# 3. Activate environment
source .venv/bin/activate

# 4. Start training
python scripts/train_lora.py --config configs/qlora_hi_en.yaml
```

---

## 🛑 How to Stop Safely

**WAIT for this message:**
```
Saving model checkpoint to outputs/hi_en_r8_v5/checkpoint-XXXX
```

**THEN press:**
```
Ctrl+C
```

**That's it!** Your progress is saved.

---

## 🌡️ Temperature Thresholds

| Status | GPU Temp | Action |
|--------|----------|--------|
| ✅ Good | <75°C | Keep training |
| ⚠️ Warning | 75-82°C | Consider stopping |
| 🛑 STOP | >82°C | Stop immediately, cool down |

---

## ⏰ Recommended Schedule

**Session Length:** 2-4 hours each
**Cool-down Between:** 30-60 minutes

**Example:**
- Morning (8am-11am): Train 3 hours
- **Cool down 1 hour**
- Afternoon (12pm-3pm): Train 3 hours
- **Done for the day!**

---

## 📊 Check Your Progress

```bash
# See current progress
ls -lh outputs/hi_en_r8_v5/

# Shows:
checkpoint-600/
checkpoint-1200/
checkpoint-1800/  ← Your current progress
```

**Math:**
- Total steps needed: 3,600
- Current step: 1,800
- Progress: 50% complete
- Remaining: ~3-6 more hours (GPU dependent)

---

## ❄️ Quick Cooling Tips

**Physical:**
- Open case panel
- Point external fan at GPU
- Elevate laptop on cooling pad

**Software:**
```bash
# Limit GPU power (if needed)
sudo nvidia-smi -pl 200
```

**Timing:**
- Best: Morning (6-10am) or Late night (10pm-2am)
- Avoid: Hot afternoons

---

## 🆘 Quick Troubleshooting

### Training starts from 0 instead of resuming?

```bash
# Check checkpoints exist
ls outputs/hi_en_r8_v5/checkpoint-*

# Should show numbered checkpoints
```

### GPU too hot?

```bash
# Stop training (Ctrl+C at checkpoint)
# Wait 30 minutes
# Add more fans
# Try again during cooler hours
```

### Out of memory?

```bash
# Reduce batch size in configs/qlora_hi_en.yaml:
batch_size: 2  # (was 4)
grad_accum: 32  # (was 16)
```

---

## 📈 Expected Timeline

**Your training will take:**
- Fast GPU (RTX 4090): ~7 hours total → 2-3 sessions
- Mid GPU (RTX 3080): ~12 hours total → 4-6 sessions
- Slower GPU (RTX 3060): ~18 hours total → 6-9 sessions

**Spread over 3-7 days with 2-3 hour sessions.**

---

## ✅ You'll Know It's Working When You See:

```
[trainer] Loading train split from v2 format: data/clean/hi_en/train_v2.tsv
[trainer] Loading val split from v2 format: data/clean/hi_en/val_v2.tsv

# On resume:
[trainer] Resuming from checkpoint: outputs/hi_en_r8_v5/checkpoint-1200

# During training:
{'loss': 1.234, 'learning_rate': 0.0001, 'epoch': 0.5}
Saving model checkpoint to outputs/hi_en_r8_v5/checkpoint-1800
```

---

## 🎯 Complete Training Workflow

```bash
# Day 1 - Morning
python scripts/train_lora.py --config configs/qlora_hi_en.yaml
# Train 3 hours → checkpoint-600, checkpoint-1200
# Ctrl+C at checkpoint save

# Day 1 - Evening
python scripts/train_lora.py --config configs/qlora_hi_en.yaml
# Resumes from checkpoint-1200 automatically
# Train 3 hours → checkpoint-1800, checkpoint-2400
# Ctrl+C at checkpoint save

# Day 2 - Morning
python scripts/train_lora.py --config configs/qlora_hi_en.yaml
# Resumes from checkpoint-2400 automatically
# Train 3 hours → checkpoint-3000, checkpoint-3600
# ✅ DONE!
```

---

## 📞 Need Help?

**See full details:** [SAFE_TRAINING_GUIDE.md](SAFE_TRAINING_GUIDE.md)

**Common files:**
- Config: `configs/qlora_hi_en.yaml`
- Checkpoints: `outputs/hi_en_r8_v5/checkpoint-*/`
- Logs: MLflow UI (`mlflow ui`)

---

**TL;DR:**
1. Run same command each time
2. Monitor temps (<75°C)
3. Stop at checkpoint (Ctrl+C)
4. Cool down 30-60 min
5. Repeat until done

**You got this!** 🚀
