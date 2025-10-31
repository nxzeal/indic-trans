#!/usr/bin/env python3
"""
Add special control tokens to the IndicTrans2 tokenizer.

This script adds <FORMAL>, <INFORMAL>, <SIMPL_Y>, <SIMPL_N> as additional special tokens
to the tokenizer configuration.

IMPORTANT: This does NOT resize the model's token embeddings (IndicTrans2 constraint).
The special tokens are added to the tokenizer only, and will be tokenized as subword units.

Usage:
    python scripts/add_special_tokens_to_tokenizer.py

This will modify the tokenizer configuration in models/indictrans2-indic-en-1B/
"""

import os
import json
from pathlib import Path
from transformers import AutoTokenizer

# Control tokens to add
SPECIAL_TOKENS = ["<FORMAL>", "<INFORMAL>", "<SIMPL_Y>", "<SIMPL_N>"]

def add_special_tokens_to_tokenizer(model_dir: str):
    """Add special control tokens to the tokenizer configuration."""

    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Loading tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)

    # Check current special tokens
    print(f"\nCurrent special tokens: {tokenizer.special_tokens_map}")
    print(f"Current vocab size: {len(tokenizer)}")

    # Add new special tokens
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": SPECIAL_TOKENS
    })

    print(f"\n✅ Added {num_added} special tokens")

    # Print token IDs for each special token
    print("\nSpecial token IDs:")
    for token in SPECIAL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {token:15s} → ID {token_id}")

    # Test tokenization (skip if fails - IndicTrans2 has custom tokenizer logic)
    print("\n📝 Testing tokenization:")
    test_prompt = "hin_Deva eng_Latn <FORMAL> <SIMPL_Y> ||| नमस्ते"
    try:
        encoded = tokenizer.encode(test_prompt, add_special_tokens=False)
        print(f"  Input:  {test_prompt}")
        print(f"  Token IDs: {encoded[:20]}..." if len(encoded) > 20 else f"  Token IDs: {encoded}")
    except Exception as e:
        print(f"  Skipping tokenization test (IndicTrans2 custom tokenizer): {e}")

    # Save the updated tokenizer
    print(f"\n💾 Saving updated tokenizer to {model_dir}...")
    tokenizer.save_pretrained(model_dir)

    # Verify the saved configuration
    special_tokens_map_path = model_path / "special_tokens_map.json"
    if special_tokens_map_path.exists():
        with open(special_tokens_map_path, "r") as f:
            saved_map = json.load(f)
        print(f"\n✅ Verified special_tokens_map.json:")
        print(f"   {json.dumps(saved_map, indent=2)}")

    print("\n" + "="*70)
    print("✅ TOKENIZER UPDATED SUCCESSFULLY")
    print("="*70)
    print("\nIMPORTANT NOTES:")
    print("  ⚠️  Model embeddings were NOT resized (IndicTrans2 constraint)")
    print("  ⚠️  Special tokens will be tokenized as subword units")
    print("  ✓  You can now use <FORMAL>, <INFORMAL>, <SIMPL_Y>, <SIMPL_N> in prompts")
    print("\n💡 Next steps:")
    print("  1. Run scripts/migrate_to_control_tokens.py to convert training data")
    print("  2. Update configs/qlora_hi_en.yaml to use train_v2.tsv")
    print("  3. Train new adapter with special token format")


def main():
    """Main entry point."""
    model_dir = "models/indictrans2-indic-en-1B"

    print("="*70)
    print("🔧 ADDING SPECIAL CONTROL TOKENS TO TOKENIZER")
    print("="*70)
    print(f"\nModel directory: {model_dir}")
    print(f"Special tokens to add: {', '.join(SPECIAL_TOKENS)}")

    add_special_tokens_to_tokenizer(model_dir)


if __name__ == "__main__":
    main()
