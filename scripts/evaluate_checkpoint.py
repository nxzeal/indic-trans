#!/usr/bin/env python3
"""
Evaluate adapter checkpoint for style learning and translation quality.
"""

from pathlib import Path
import sys
import csv
import re
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from IndicTransToolkit.processor import IndicProcessor

random.seed(42)


def has_contraction(text: str) -> bool:
    """Check if text contains any contractions."""
    patterns = [
        r"\bdon't\b", r"\bdoesn't\b", r"\bdidn't\b", r"\bisn't\b", r"\baren't\b",
        r"\bwasn't\b", r"\bweren't\b", r"\bcan't\b", r"\bwon't\b", r"\bwouldn't\b",
        r"\bshouldn't\b", r"\bcouldn't\b", r"\bhaven't\b", r"\bhasn't\b", r"\bhadn't\b",
        r"\bit's\b", r"\bthat's\b", r"\bthere's\b", r"\bi'm\b", r"\bwe'll\b",
        r"\byou'll\b", r"\bthey'll\b", r"\bhe's\b", r"\bshe's\b", r"\bwe're\b",
        r"\bthey're\b", r"\bi've\b", r"\bwe've\b", r"\bthey've\b",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def has_formal_vocab(text: str) -> bool:
    """Check if text contains formal vocabulary."""
    patterns = [
        r"\bhowever\b", r"\btherefore\b", r"\bconsequently\b", r"\bfurthermore\b",
        r"\bmoreover\b", r"\bnevertheless\b", r"\badditionally\b", r"\bthus\b",
        r"\bhence\b", r"\bregarding\b", r"\butilize\b", r"\bdemonstrate\b",
        r"\bfacilitate\b", r"\bcommence\b", r"\bterminate\b", r"\bobtain\b",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def load_test_samples(test_file: Path, n: int = 20):
    """Load random test samples."""
    samples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        all_samples = list(reader)

    random.shuffle(all_samples)
    return all_samples[:n]


def translate_batch(model, tokenizer, ip, texts, device):
    """Translate a batch of texts."""
    # Preprocess
    processed = ip.preprocess_batch(texts, src_lang="hin_Deva", tgt_lang="eng_Latn")

    # Tokenize
    inputs = tokenizer(processed, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

    # Generate
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=4,
            do_sample=False,
            use_cache=False
        )

    # Decode
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Postprocess
    translations = ip.postprocess_batch(decoded, lang="eng_Latn")

    return translations


def main():
    print("="*70)
    print("ADAPTER CHECKPOINT-2000 EVALUATION")
    print("="*70)

    # Config
    base_model_dir = PROJECT_ROOT / "models" / "indictrans2-indic-en-1B"
    adapter_dir = PROJECT_ROOT / "outputs" / "adapter_formal_detailed" / "checkpoint-2000"
    test_file = PROJECT_ROOT / "data" / "clean" / "hi_en_formal_detailed" / "test_v3.tsv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load processor
    print("\nLoading IndicProcessor...")
    ip = IndicProcessor(inference=True)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print("Loading base model (quantized)...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        str(base_model_dir),
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True
    )

    # Load adapter model
    print(f"Loading adapter from {adapter_dir}...")
    adapter_model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    # Load test samples
    print(f"\nLoading test samples from {test_file.name}...")
    samples = load_test_samples(test_file, n=20)
    print(f"Loaded {len(samples)} test samples\n")

    # Translate
    print("="*70)
    print("TRANSLATING WITH BASE MODEL")
    print("="*70)
    hindi_texts = [s['src'] for s in samples]
    base_translations = translate_batch(base_model, tokenizer, ip, hindi_texts, device)

    print("\n" + "="*70)
    print("TRANSLATING WITH ADAPTER (FORMAL DETAILED)")
    print("="*70)
    adapter_translations = translate_batch(adapter_model, tokenizer, ip, hindi_texts, device)

    # Analyze
    print("\n" + "="*70)
    print("STYLE ANALYSIS")
    print("="*70)

    base_contractions = sum(has_contraction(t) for t in base_translations)
    adapter_contractions = sum(has_contraction(t) for t in adapter_translations)

    base_formal = sum(has_formal_vocab(t) for t in base_translations)
    adapter_formal = sum(has_formal_vocab(t) for t in adapter_translations)

    base_avg_len = sum(len(t.split()) for t in base_translations) / len(base_translations)
    adapter_avg_len = sum(len(t.split()) for t in adapter_translations) / len(adapter_translations)

    print(f"\nBase Model:")
    print(f"  Contractions: {base_contractions}/{len(base_translations)} ({100*base_contractions/len(base_translations):.1f}%)")
    print(f"  Formal vocab: {base_formal}/{len(base_translations)} ({100*base_formal/len(base_translations):.1f}%)")
    print(f"  Avg length: {base_avg_len:.1f} words")

    print(f"\nAdapter (Formal Detailed):")
    print(f"  Contractions: {adapter_contractions}/{len(adapter_translations)} ({100*adapter_contractions/len(adapter_translations):.1f}%)")
    print(f"  Formal vocab: {adapter_formal}/{len(adapter_translations)} ({100*adapter_formal/len(adapter_translations):.1f}%)")
    print(f"  Avg length: {adapter_avg_len:.1f} words")

    print(f"\nDifference:")
    print(f"  Contractions: {base_contractions - adapter_contractions} fewer (target: more negative = better)")
    print(f"  Formal vocab: +{adapter_formal - base_formal} more (target: positive = better)")
    print(f"  Avg length: +{adapter_avg_len - base_avg_len:.1f} words")

    # Sample outputs
    print("\n" + "="*70)
    print("SAMPLE COMPARISONS (First 5)")
    print("="*70)

    for i in range(min(5, len(samples))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Hindi: {hindi_texts[i][:100]}...")
        print(f"\nBase:    {base_translations[i]}")
        print(f"Adapter: {adapter_translations[i]}")

        # Highlight differences
        if base_translations[i] != adapter_translations[i]:
            print("✓ DIFFERENT")
        else:
            print("✗ IDENTICAL")

    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    different_count = sum(1 for b, a in zip(base_translations, adapter_translations) if b != a)

    print(f"\nDifferent outputs: {different_count}/{len(samples)} ({100*different_count/len(samples):.1f}%)")

    if adapter_contractions < base_contractions:
        print("✓ Adapter has fewer contractions (GOOD for formal)")
    else:
        print("✗ Adapter has same or more contractions (BAD)")

    if adapter_formal > base_formal:
        print("✓ Adapter has more formal vocabulary (GOOD)")
    else:
        print("✗ Adapter has same or less formal vocabulary (BAD)")

    if different_count >= len(samples) * 0.3:  # 30% threshold
        print("\n✓ STYLE LEARNING DETECTED: Adapter produces different outputs")
    else:
        print("\n✗ NO STYLE LEARNING: Adapter mostly copies base model")

    if adapter_contractions <= base_contractions and adapter_formal >= base_formal and different_count >= len(samples) * 0.3:
        print("\n" + "="*70)
        print("SUCCESS: Adapter shows formal style learning! 🎉")
        print("Recommendation: Continue training to 10k steps")
        print("="*70)
    elif different_count >= len(samples) * 0.1:
        print("\n" + "="*70)
        print("PARTIAL SUCCESS: Some style learning detected")
        print("Recommendation: Continue to 5k steps and re-evaluate")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("FAILURE: No meaningful style learning")
        print("Recommendation: Check data quality or approach")
        print("="*70)


if __name__ == "__main__":
    main()
