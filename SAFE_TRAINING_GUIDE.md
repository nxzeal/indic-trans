# Safe Incremental Training Guide

> How to train safely in short sessions to prevent overheating

---

## 🔥 The Problem

Long continuous training (20+ hours) causes:
- GPU/CPU overheating
- Potential hardware damage
- Display/system instability
- Thermal throttling (slower training)

**Solution:** Train in shorter sessions (2-4 hours) spread over multiple days.

---

## ✅ Checkpoint Resume (Now Implemented)

The training script now **automatically resumes** from the latest checkpoint:

```python
# When you run this command:
python scripts/train_lora.py --config configs/qlora_hi_en.yaml

# The script will:
# 1. Check outputs/hi_en_r8_v5/ for existing checkpoints
# 2. If found, automatically resume from latest checkpoint
# 3. Print: "[trainer] Resuming from checkpoint: checkpoint-XXXX"
```

**You'll see this message:**
```
[trainer] Resuming from checkpoint: outputs/hi_en_r8_v5/checkpoint-1800
```

---

## 📋 Safe Training Schedule

### Recommended Session Length

**Target:** 2-4 hours per session
**Why:**
- Keeps temperatures manageable
- Allows hardware to cool between sessions
- Reduces risk of thermal damage

### Example Weekly Schedule

**Configuration:**
- Total steps: 3,600
- Checkpoint every: 600 steps
- Steps per session: ~600-1200 (2-4 hours)

| Day | Session | Duration | Steps | Checkpoints | Cumulative |
|-----|---------|----------|-------|-------------|------------|
| Mon | Morning | 2-3h | 0→600 | checkpoint-600 | 600/3600 |
| Mon | Evening | 2-3h | 600→1200 | checkpoint-1200 | 1200/3600 |
| Tue | Morning | 2-3h | 1200→1800 | checkpoint-1800 | 1800/3600 |
| Tue | Evening | 2-3h | 1800→2400 | checkpoint-2400 | 2400/3600 |
| Wed | Morning | 2-3h | 2400→3000 | checkpoint-3000 | 3000/3600 |
| Wed | Evening | 2-3h | 3000→3600 | checkpoint-3600 | ✅ DONE |

---

## 🛡️ How to Safely Stop Training

### Method 1: Ctrl+C (Recommended)

**Safest way to stop:**

```bash
# Press Ctrl+C once when you see a checkpoint being saved
# You'll see:
Saving model checkpoint to outputs/hi_en_r8_v5/checkpoint-1200
{'loss': 1.234, 'learning_rate': 0.0001, 'epoch': 0.5}
# NOW press Ctrl+C
^C
```

**Why wait for checkpoint save?**
- Ensures latest progress is saved
- Preserves optimizer state
- Clean resume on next run

### Method 2: Wait for Checkpoint Completion

Monitor the output and wait until you see:
```
Saving model checkpoint to outputs/hi_en_r8_v5/checkpoint-XXXX
```

Then press `Ctrl+C`.

### Method 3: Let It Run to Next Checkpoint

If you want to stop after exactly N hours, calculate:
- Each checkpoint = ~2 hours (depends on GPU)
- Stop after 2 checkpoints = 4 hours
- Stop after 3 checkpoints = 6 hours

---

## 🌡️ Temperature Monitoring

### Before Starting Each Session

**Check baseline temperatures:**

```bash
# Install monitoring tools (one-time)
sudo apt-get install lm-sensors  # Linux
# or
brew install lm-sensors  # macOS

# Monitor GPU temperature
watch -n 1 nvidia-smi

# Monitor CPU temperature
sensors
```

**Baseline (idle):**
- GPU: 30-40°C
- CPU: 40-50°C

### During Training

**Create a monitoring script:**

```bash
# Save as monitor_temps.sh
#!/bin/bash
while true; do
    clear
    date
    echo "=== GPU ==="
    nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw --format=csv,noheader
    echo ""
    echo "=== CPU ==="
    sensors | grep -E "Core|Package"
    sleep 2
done
```

**Run in separate terminal:**
```bash
chmod +x monitor_temps.sh
./monitor_temps.sh
```

### Safe Temperature Ranges

| Component | Safe | Warning | STOP |
|-----------|------|---------|------|
| GPU | <75°C | 75-82°C | >82°C |
| CPU | <70°C | 70-80°C | >80°C |

**⚠️ If temperatures exceed WARNING:**
1. Press `Ctrl+C` to stop training
2. Let hardware cool for 30 minutes
3. Improve cooling (see below)

---

## ❄️ Cooling Strategies

### 1. Physical Setup

**Do:**
- ✅ Open case side panel (if desktop)
- ✅ Use external fans pointed at GPU/CPU
- ✅ Elevate laptop for airflow (use cooling pad)
- ✅ Clean dust from fans/heatsinks
- ✅ Ensure room has good ventilation
- ✅ Run AC or open windows

**Don't:**
- ❌ Place laptop on soft surfaces (bed, couch)
- ❌ Block air vents
- ❌ Train in small enclosed spaces
- ❌ Stack other devices nearby

### 2. Software Optimizations

**Reduce power/heat:**

```bash
# Limit GPU power (NVIDIA)
sudo nvidia-smi -pl 200  # Limit to 200W (adjust for your card)

# Set fan speed manually
nvidia-settings -a "[gpu:0]/GPUFanControlState=1" -a "[fan:0]/GPUTargetFanSpeed=80"
```

**Reduce batch size (if needed):**

Edit `configs/qlora_hi_en.yaml`:
```yaml
train:
  batch_size: 2  # Reduce from 4 to 2
  grad_accum: 32  # Double to maintain effective batch size
```

### 3. Training Schedule

**Best times to train:**
- ☀️ **Morning (6am-10am):** Coolest ambient temperature
- 🌙 **Late night (10pm-2am):** Room temperature drops
- ❄️ **Winter months:** Lower ambient temps

**Avoid:**
- 🔥 **Afternoon (12pm-6pm):** Hottest time of day
- ☀️ **Summer days:** High ambient temperature

---

## 📊 Monitoring Training Progress

### Check Current Status

```bash
# See all checkpoints
ls -lh outputs/hi_en_r8_v5/

# Output:
checkpoint-600/
checkpoint-1200/
checkpoint-1800/  ← Latest (will resume from here)
```

### Calculate Progress

```bash
# Check trainer_state.json in latest checkpoint
cat outputs/hi_en_r8_v5/checkpoint-1800/trainer_state.json | grep -E "global_step|max_steps"
```

**Example output:**
```json
"global_step": 1800,
"max_steps": 3600
```

**Progress:** 1800/3600 = 50% complete

### Estimate Time Remaining

**Formula:**
```
Time remaining = (remaining_steps / steps_per_hour) hours
```

**Example:**
- Current step: 1800
- Remaining steps: 3600 - 1800 = 1800
- Your speed: ~300 steps/hour (check your GPU)
- Time remaining: 1800 / 300 = 6 hours
- Sessions needed: 6 hours / 3 hours per session = 2 sessions

---

## 🚀 Training Workflow

### Session 1: Start Training

```bash
# Activate environment
source .venv/bin/activate

# Start monitoring in separate terminal
./monitor_temps.sh

# Start training
python scripts/train_lora.py --config configs/qlora_hi_en.yaml

# You'll see:
[trainer] Loading train split from v2 format: data/clean/hi_en/train_v2.tsv
[trainer] Starting from step 0
```

**Watch for:**
- ✅ "Loading ... from v2 format" (confirms special tokens)
- ✅ GPU utilization ~95-100%
- ✅ Temperatures in safe range

**Train for 2-3 hours, then:**
- Wait for checkpoint save message
- Press `Ctrl+C`
- Let GPU cool for 30 minutes

### Session 2+: Resume Training

```bash
# Same command as before
python scripts/train_lora.py --config configs/qlora_hi_en.yaml

# You'll see:
[trainer] Resuming from checkpoint: outputs/hi_en_r8_v5/checkpoint-1200
```

**The script will:**
1. ✅ Load model from checkpoint
2. ✅ Restore optimizer state
3. ✅ Continue from step 1200
4. ✅ Save next checkpoint at step 1800

**No data loss!** Everything is preserved.

---

## 🔍 Verification After Each Session

### 1. Check Checkpoint Exists

```bash
ls -lh outputs/hi_en_r8_v5/checkpoint-XXXX/

# Should contain:
adapter_config.json
adapter_model.safetensors  ← LoRA weights
trainer_state.json         ← Training state
```

### 2. Quick Test (Optional)

Test the latest checkpoint works:

```bash
python scripts/translate_adapter_ip.py \
    --adapter outputs/hi_en_r8_v5/checkpoint-1200 \
    --text "परीक्षण" \
    --style formal --simplify no
```

Should output English translation.

### 3. Check MLflow

```bash
mlflow ui

# Open: http://localhost:5000
# See: Training curves, loss, steps completed
```

---

## ⚠️ Troubleshooting

### Problem: Script doesn't resume

**Symptom:** Training starts from step 0 instead of resuming

**Solution:**
```bash
# Check checkpoints exist
ls outputs/hi_en_r8_v5/

# If no checkpoints, check output_dir in config
cat configs/qlora_hi_en.yaml | grep output_dir
```

### Problem: Out of memory after resume

**Symptom:** CUDA OOM error on resume

**Solution:**
```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Or restart the session
# (Ctrl+C, wait 1 minute, restart)
```

### Problem: High temperature persists

**Symptoms:** GPU >80°C, fan noise loud

**Solutions:**
1. Reduce power limit (see Cooling Strategies)
2. Reduce batch size
3. Add more cooling
4. Train during cooler hours

### Problem: Checkpoint corrupted

**Symptom:** Error loading checkpoint

**Solution:**
```bash
# Delete corrupted checkpoint
rm -rf outputs/hi_en_r8_v5/checkpoint-XXXX

# Script will resume from previous valid checkpoint
```

---

## 📈 Expected Performance

### Training Speed (Reference)

**Hardware benchmarks:**

| GPU | Steps/hour | Time per 600 steps | Total time (3600 steps) |
|-----|------------|-------------------|------------------------|
| RTX 4090 | ~500 | ~1.2h | ~7h |
| RTX 3090 | ~350 | ~1.7h | ~10h |
| RTX 3080 | ~300 | ~2h | ~12h |
| RTX 3070 | ~250 | ~2.4h | ~14h |
| RTX 3060 | ~200 | ~3h | ~18h |

**Your GPU will vary!** Check your actual speed after first checkpoint.

### Session Planning

**Calculate your sessions:**

```python
# After first checkpoint, note the time
time_per_600_steps = X  # hours (measure this)
total_sessions = 6  # (3600 / 600 checkpoints)
total_time = time_per_600_steps * 6
sessions_per_day = 2  # morning + evening
days_needed = total_sessions / sessions_per_day
```

---

## ✅ Final Checklist

Before each training session:

- [ ] Check GPU temperature (should be at idle baseline)
- [ ] Start monitoring script in separate terminal
- [ ] Verify last checkpoint exists (if resuming)
- [ ] Ensure good ventilation/cooling
- [ ] Set timer for 2-3 hours
- [ ] Have Ctrl+C ready to stop safely

During training:

- [ ] Monitor temperatures every 30 minutes
- [ ] Check training output for errors
- [ ] Watch for checkpoint save messages
- [ ] Take breaks to let hardware cool

After each session:

- [ ] Verify checkpoint was saved
- [ ] Check temperatures return to baseline
- [ ] Note current progress (step count)
- [ ] Log any issues or observations

---

## 🎯 Summary

**Key Points:**

1. ✅ **Auto-resume works:** Same command resumes automatically
2. 🛑 **Stop safely:** Wait for checkpoint, then Ctrl+C
3. 🌡️ **Monitor temps:** Keep GPU <75°C, CPU <70°C
4. ❄️ **Cool between sessions:** 30-60 minutes rest
5. ⏰ **Train in short bursts:** 2-4 hours per session
6. 📊 **Track progress:** Check checkpoint folders

**Training Command (always the same):**
```bash
python scripts/train_lora.py --config configs/qlora_hi_en.yaml
```

**You're ready to train safely!** 🚀

---

**Last Updated:** 2025-10-31
