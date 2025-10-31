#!/usr/bin/env python3
"""
Evaluate how well the LoRA adapter follows style and simplification controls.

Example usage:
# Evaluate single checkpoint
python scripts/eval_control_adherence.py --checkpoint_dir outputs/hi_en_r8_v5/checkpoint-3600

# Compare multiple checkpoints (run separately, reports will be timestamped)
python scripts/eval_control_adherence.py --checkpoint_dir outputs/hi_en_r8_v5/checkpoint-1800
python scripts/eval_control_adherence.py --checkpoint_dir outputs/hi_en_r8_v5/checkpoint-2400

# Custom validation set and sample size
python scripts/eval_control_adherence.py \
    --checkpoint_dir outputs/hi_en_r8_v5/checkpoint-3600 \
    --val_file data/clean/hi_en/val.tsv \
    --num_samples 100
"""

import os
import sys
import warnings
import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers.utils import logging as hf_logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from IndicTransToolkit.processor import IndicProcessor

# Suppress warnings
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.pop("TRANSFORMERS_CACHE", None)
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")

IT2_TAG = {"en": "eng_Latn", "hi": "hin_Deva", "ta": "tam_Taml", "te": "tel_Telu", "ml": "mal_Mlym"}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate control adherence for LoRA adapter")
    p.add_argument("--checkpoint_dir", required=True, help="Path to adapter checkpoint (e.g., outputs/hi_en_r8_v5/checkpoint-3600)")
    p.add_argument("--val_file", default="data/clean/hi_en/val.tsv", help="Path to validation TSV")
    p.add_argument("--num_samples", type=int, default=500, help="Number of samples to evaluate")
    p.add_argument("--output_dir", default="artifacts/eval_reports/", help="Where to save reports")
    p.add_argument("--base_model_dir", default="models/indictrans2-indic-en-1B", help="Path to base model")
    p.add_argument("--token_format", choices=["legacy", "special"], default="special",
        help="Control token format: 'legacy' uses text (formal/informal/yes/no), "
             "'special' uses tokens (<FORMAL>/<INFORMAL>/<SIMPL_Y>/<SIMPL_N>)")
    return p.parse_args()


def resolve_tag(code: str) -> str:
    return IT2_TAG.get(code.lower(), code)


def load_model(base: str, adapter: str, device: str):
    """Load base model with LoRA adapter using 4-bit quantization on CUDA."""
    if device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes as _bnb  # noqa
            q = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                base,
                trust_remote_code=True,
                quantization_config=q,
                device_map={"": 0}
            )
        except Exception:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                base, trust_remote_code=True, torch_dtype=torch.float16
            ).to(device)
    else:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base, trust_remote_code=True, torch_dtype=torch.float32
        ).to(device)

    model = PeftModel.from_pretrained(base_model, adapter, is_trainable=False)

    # Tie weights if needed
    bm = getattr(model, "base_model", model)
    if hasattr(bm, "tie_weights"):
        try:
            bm.tie_weights()
        except Exception:
            pass

    return model


def detect_contractions(text: str) -> bool:
    """Returns True if text contains contractions."""
    contraction_patterns = [
        r"\bn't\b",  # don't, can't, won't, etc.
        r"\b're\b",  # you're, we're, they're
        r"\b've\b",  # I've, we've, they've
        r"\b'll\b",  # I'll, you'll, we'll
        r"\b'd\b",   # I'd, he'd, they'd
        r"\b'm\b",   # I'm
        r"\b's\b",   # it's, that's (when contraction, not possessive)
        r"\bwanna\b",
        r"\bgonna\b",
        r"\bgotta\b",
    ]
    for pattern in contraction_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def detect_formal_markers(text: str) -> bool:
    """Returns True if text contains formal request markers."""
    formal_patterns = [
        r"\bcould you\b",
        r"\bwould you\b",
        r"\bplease\b",
        r"\bkindly\b",
        r"\bi would appreciate\b",
        r"\bmay i\b",
        r"\bwould it be possible\b",
    ]
    for pattern in formal_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def compute_length_ratio(yes_texts: list, no_texts: list) -> float:
    """Compute mean_length(simplify=yes) / mean_length(simplify=no)."""
    if not yes_texts or not no_texts:
        return 1.0

    yes_lengths = [len(t.split()) for t in yes_texts]
    no_lengths = [len(t.split()) for t in no_texts]

    mean_yes = sum(yes_lengths) / len(yes_lengths)
    mean_no = sum(no_lengths) / len(no_lengths)

    return mean_yes / mean_no if mean_no > 0 else 1.0


def load_validation_data(val_file: str, num_samples: int):
    """Load validation samples from TSV file."""
    if not os.path.exists(val_file):
        raise FileNotFoundError(f"Validation file not found: {val_file}")

    samples = []
    with open(val_file, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")

        # Expected columns: src, tgt, style_src, style_tgt, simplify, pair
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) >= 6:
                samples.append({
                    "src": parts[0],
                    "tgt": parts[1],
                    "style_src": parts[2],
                    "style_tgt": parts[3],
                    "simplify": parts[4],
                    "pair": parts[5],
                })

            if len(samples) >= num_samples:
                break

    return samples


def generate_translations(model, tokenizer, ip, sources, src_lang, tgt_lang, style, simplify, device, token_format="special"):
    """Generate translations for a batch of sources with given control settings."""
    # Preprocess with IndicProcessor
    pres = ip.preprocess_batch(sources, src_lang=src_lang, tgt_lang=tgt_lang)

    # Insert controls before separator
    # Format depends on token_format argument
    if token_format == "special":
        style_token = f"<{style.upper()}>"
        simplify_token = f"<SIMPL_{simplify.upper()}>"
    else:  # legacy
        style_token = style
        simplify_token = simplify

    with_ctrls = []
    for pre in pres:
        if " ||| " in pre:
            pre = pre.replace(" ||| ", f" {style_token} {simplify_token} ||| ", 1)
        with_ctrls.append(pre)

    # Tokenize
    enc = tokenizer(with_ctrls, padding=True, truncation=True, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_length=192,
            num_beams=4,
            do_sample=False,
            use_cache=False
        )

    # Decode
    raw = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    post = ip.postprocess_batch(raw, lang=tgt_lang)

    return post


def evaluate_checkpoint(args):
    """Main evaluation logic."""

    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint_dir}")

    checkpoint_name = checkpoint_path.name

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load validation data
    print(f"Loading validation data from {args.val_file}...")
    samples = load_validation_data(args.val_file, args.num_samples)
    print(f"Loaded {len(samples)} validation samples")

    if len(samples) == 0:
        raise ValueError("No validation samples loaded. Check your val_file.")

    # Extract source language from first sample's pair field
    src_lang = samples[0]["pair"].split("-")[0]  # e.g., "hi-en" -> "hi"
    tgt_lang = samples[0]["pair"].split("-")[1]  # e.g., "hi-en" -> "en"
    src_tag = resolve_tag(src_lang)
    tgt_tag = resolve_tag(tgt_lang)

    # Load model
    print(f"Loading model from {args.base_model_dir} with adapter {args.checkpoint_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_dir,
        trust_remote_code=True,
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.base_model_dir, args.checkpoint_dir, device)
    model.config.use_cache = False
    model.generation_config.use_cache = False
    model = model.eval()

    # Initialize IndicProcessor
    ip = IndicProcessor(inference=True)

    # Generate translations for all control combinations
    print("Generating translations with all control combinations...")

    all_results = []
    sources = [s["src"] for s in samples]

    # Process in batches to avoid OOM
    batch_size = 8

    for i in tqdm(range(0, len(sources), batch_size), desc="Generating"):
        batch_sources = sources[i:i + batch_size]
        batch_samples = samples[i:i + batch_size]

        # Generate 4 variants
        formal_no = generate_translations(model, tokenizer, ip, batch_sources, src_tag, tgt_tag, "formal", "no", device, args.token_format)
        formal_yes = generate_translations(model, tokenizer, ip, batch_sources, src_tag, tgt_tag, "formal", "yes", device, args.token_format)
        informal_no = generate_translations(model, tokenizer, ip, batch_sources, src_tag, tgt_tag, "informal", "no", device, args.token_format)
        informal_yes = generate_translations(model, tokenizer, ip, batch_sources, src_tag, tgt_tag, "informal", "yes", device, args.token_format)

        # Store results
        for j, sample in enumerate(batch_samples):
            all_results.append({
                "source": sample["src"],
                "reference": sample["tgt"],
                "formal_no": formal_no[j],
                "formal_yes": formal_yes[j],
                "informal_no": informal_no[j],
                "informal_yes": informal_yes[j],
            })

    # Compute adherence metrics
    print("Computing adherence metrics...")
    metrics = compute_adherence_metrics(all_results)

    # Save reports
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{checkpoint_name}_{timestamp}"

    # Save JSON report
    json_path = output_dir / f"{base_name}_adherence_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": args.checkpoint_dir,
            "val_file": args.val_file,
            "num_samples": len(samples),
            "metrics": metrics,
            "sample_results": all_results[:10],  # First 10 for inspection
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON report: {json_path}")

    # Save human-readable summary
    summary_path = output_dir / f"{base_name}_adherence_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        write_summary(f, metrics, all_results, checkpoint_name)
    print(f"Saved summary: {summary_path}")

    # Save all samples TSV
    tsv_path = output_dir / f"{base_name}_samples.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("source\treference\tformal_no\tformal_yes\tinformal_no\tinformal_yes\n")
        for r in all_results:
            f.write(f"{r['source']}\t{r['reference']}\t{r['formal_no']}\t{r['formal_yes']}\t{r['informal_no']}\t{r['informal_yes']}\n")
    print(f"Saved samples: {tsv_path}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print_metrics_summary(metrics)


def compute_adherence_metrics(results):
    """Compute adherence metrics from generated results."""

    # Formal outputs: formal_no and formal_yes
    formal_outputs = [r["formal_no"] for r in results] + [r["formal_yes"] for r in results]

    # Informal outputs: informal_no and informal_yes
    informal_outputs = [r["informal_no"] for r in results] + [r["informal_yes"] for r in results]

    # Formal adherence
    formal_no_contractions = sum(1 for text in formal_outputs if not detect_contractions(text))
    formal_has_markers = sum(1 for text in formal_outputs if detect_formal_markers(text))

    formal_no_contractions_pct = formal_no_contractions / len(formal_outputs) * 100
    formal_has_markers_pct = formal_has_markers / len(formal_outputs) * 100
    formal_score = (formal_no_contractions_pct + formal_has_markers_pct) / 2 / 100  # Normalize to 0-1

    # Informal adherence
    informal_has_contractions = sum(1 for text in informal_outputs if detect_contractions(text))
    informal_not_overly_formal = sum(1 for text in informal_outputs if not re.search(r"\bcould you please\b", text, re.IGNORECASE))

    informal_has_contractions_pct = informal_has_contractions / len(informal_outputs) * 100
    informal_not_overly_formal_pct = informal_not_overly_formal / len(informal_outputs) * 100
    informal_score = (informal_has_contractions_pct + informal_not_overly_formal_pct) / 2 / 100  # Normalize to 0-1

    # Simplify adherence
    simplify_yes_texts = [r["formal_yes"] for r in results] + [r["informal_yes"] for r in results]
    simplify_no_texts = [r["formal_no"] for r in results] + [r["informal_no"] for r in results]

    length_ratio = compute_length_ratio(simplify_yes_texts, simplify_no_texts)

    # Score based on target range 0.85-0.92
    if 0.85 <= length_ratio <= 0.92:
        simplify_score = 1.0
    elif length_ratio < 0.85:
        simplify_score = max(0.0, 1.0 - (0.85 - length_ratio) / 0.15)  # Penalize
    else:  # length_ratio > 0.92
        simplify_score = max(0.0, 1.0 - (length_ratio - 0.92) / 0.15)  # Penalize

    # Average words per sentence
    avg_words_yes = sum(len(t.split()) for t in simplify_yes_texts) / len(simplify_yes_texts)
    avg_words_no = sum(len(t.split()) for t in simplify_no_texts) / len(simplify_no_texts)

    return {
        "formal": {
            "no_contractions_pct": formal_no_contractions_pct,
            "has_markers_pct": formal_has_markers_pct,
            "score": formal_score,
        },
        "informal": {
            "has_contractions_pct": informal_has_contractions_pct,
            "not_overly_formal_pct": informal_not_overly_formal_pct,
            "score": informal_score,
        },
        "simplify": {
            "length_ratio": length_ratio,
            "avg_words_yes": avg_words_yes,
            "avg_words_no": avg_words_no,
            "score": simplify_score,
        },
    }


def write_summary(f, metrics, results, checkpoint_name):
    """Write human-readable summary to file."""
    f.write(f"Control Adherence Evaluation Report\n")
    f.write(f"{'='*80}\n\n")
    f.write(f"Checkpoint: {checkpoint_name}\n")
    f.write(f"Evaluated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Samples: {len(results)}\n\n")

    # Overall scores
    f.write(f"Overall Adherence Scores\n")
    f.write(f"{'-'*80}\n")
    f.write(f"Formal Score:    {metrics['formal']['score']:.3f}\n")
    f.write(f"Informal Score:  {metrics['informal']['score']:.3f}\n")
    f.write(f"Simplify Score:  {metrics['simplify']['score']:.3f}\n\n")

    # Detailed metrics
    f.write(f"Formal Adherence Details\n")
    f.write(f"{'-'*80}\n")
    f.write(f"  No contractions:    {metrics['formal']['no_contractions_pct']:.1f}%\n")
    f.write(f"  Has formal markers: {metrics['formal']['has_markers_pct']:.1f}%\n\n")

    f.write(f"Informal Adherence Details\n")
    f.write(f"{'-'*80}\n")
    f.write(f"  Has contractions:      {metrics['informal']['has_contractions_pct']:.1f}%\n")
    f.write(f"  Not overly formal:     {metrics['informal']['not_overly_formal_pct']:.1f}%\n\n")

    f.write(f"Simplify Adherence Details\n")
    f.write(f"{'-'*80}\n")
    f.write(f"  Length ratio (yes/no): {metrics['simplify']['length_ratio']:.3f}\n")
    f.write(f"  Target range:          0.850 - 0.920\n")
    f.write(f"  Avg words (yes):       {metrics['simplify']['avg_words_yes']:.1f}\n")
    f.write(f"  Avg words (no):        {metrics['simplify']['avg_words_no']:.1f}\n\n")

    # Pass/fail
    f.write(f"Pass/Fail Against Targets\n")
    f.write(f"{'-'*80}\n")
    formal_pass = "PASS" if metrics['formal']['score'] >= 0.70 else "FAIL"
    informal_pass = "PASS" if metrics['informal']['score'] >= 0.70 else "FAIL"
    simplify_pass = "PASS" if 0.85 <= metrics['simplify']['length_ratio'] <= 0.92 else "FAIL"

    f.write(f"  Formal (≥0.70):         {formal_pass}\n")
    f.write(f"  Informal (≥0.70):       {informal_pass}\n")
    f.write(f"  Simplify (0.85-0.92):   {simplify_pass}\n\n")

    # Example outputs
    f.write(f"Example Outputs (5 samples)\n")
    f.write(f"{'='*80}\n\n")

    for i in range(min(5, len(results))):
        r = results[i]
        f.write(f"Sample {i+1}\n")
        f.write(f"{'-'*80}\n")
        f.write(f"Source:          {r['source']}\n")
        f.write(f"Reference:       {r['reference']}\n\n")
        f.write(f"formal+no:       {r['formal_no']}\n")
        f.write(f"formal+yes:      {r['formal_yes']}\n")
        f.write(f"informal+no:     {r['informal_no']}\n")
        f.write(f"informal+yes:    {r['informal_yes']}\n\n")


def print_metrics_summary(metrics):
    """Print metrics summary to console."""
    print(f"\nFormal Score:    {metrics['formal']['score']:.3f} ({'PASS' if metrics['formal']['score'] >= 0.70 else 'FAIL'})")
    print(f"Informal Score:  {metrics['informal']['score']:.3f} ({'PASS' if metrics['informal']['score'] >= 0.70 else 'FAIL'})")
    print(f"Simplify Score:  {metrics['simplify']['score']:.3f}")
    print(f"Length Ratio:    {metrics['simplify']['length_ratio']:.3f} ({'PASS' if 0.85 <= metrics['simplify']['length_ratio'] <= 0.92 else 'FAIL'})")


def main():
    args = parse_args()
    evaluate_checkpoint(args)


if __name__ == "__main__":
    main()
