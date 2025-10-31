#!/usr/bin/env python3
"""
Migrate training data from legacy text controls to special control tokens.

Legacy format: "hin_Deva eng_Latn formal yes ||| <text>"
New format:    "hin_Deva eng_Latn <FORMAL> <SIMPL_Y> ||| <text>"

This script converts:
- "formal" → <FORMAL>
- "informal" → <INFORMAL>
- "yes" → <SIMPL_Y>
- "no" → <SIMPL_N>

Usage:
    python scripts/migrate_to_control_tokens.py

This will create:
    - data/clean/hi_en/train_v2.tsv
    - data/clean/hi_en/val_v2.tsv
    - data/clean/hi_en/test_v2.tsv

Original files are preserved unchanged.
"""

import os
from pathlib import Path
from collections import Counter

# Control token mapping
STYLE_MAP = {
    "formal": "<FORMAL>",
    "informal": "<INFORMAL>"
}

SIMPLIFY_MAP = {
    "yes": "<SIMPL_Y>",
    "no": "<SIMPL_N>"
}


def convert_file(input_path: str, output_path: str):
    """Convert a TSV file from legacy to special token format."""

    if not os.path.exists(input_path):
        print(f"⚠️  Skipping {input_path} (not found)")
        return None

    print(f"\n📄 Converting {input_path} → {output_path}")

    stats = Counter()
    converted_lines = []

    with open(input_path, "r", encoding="utf-8") as f:
        # Read header
        header = f.readline().strip()
        converted_lines.append(header)

        # Process data rows
        for line_num, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 6:
                print(f"⚠️  Line {line_num}: Expected 6 columns, got {len(parts)}. Skipping.")
                continue

            src, tgt, style_src, style_tgt, simplify, pair = parts[:6]

            # Count control combinations
            combo = f"{style_tgt}+{simplify}"
            stats[combo] += 1

            # Convert to special tokens
            style_src_token = STYLE_MAP.get(style_src, style_src)
            style_tgt_token = STYLE_MAP.get(style_tgt, style_tgt)
            simplify_token = SIMPLIFY_MAP.get(simplify, simplify)

            # Rebuild line with special tokens
            new_parts = [src, tgt, style_src_token, style_tgt_token, simplify_token, pair]
            converted_lines.append("\t".join(new_parts))

    # Write output file
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(converted_lines))
        if converted_lines:  # Add final newline
            f.write("\n")

    num_converted = len(converted_lines) - 1  # Exclude header
    print(f"✅ Converted {num_converted:,} lines")

    return stats


def print_stats(filename: str, stats: Counter):
    """Print conversion statistics."""
    if stats is None:
        return

    print(f"\n📊 Control combination statistics for {filename}:")
    print(f"{'='*50}")

    total = sum(stats.values())
    for combo, count in sorted(stats.items()):
        pct = count / total * 100 if total > 0 else 0
        print(f"  {combo:20s}: {count:6,} ({pct:5.1f}%)")

    print(f"{'='*50}")
    print(f"  {'TOTAL':20s}: {total:6,}")


def main():
    """Convert all TSV files from legacy to special token format."""

    print("="*70)
    print("🔄 MIGRATING TO SPECIAL CONTROL TOKENS")
    print("="*70)
    print("\nControl token mappings:")
    print("  formal   → <FORMAL>")
    print("  informal → <INFORMAL>")
    print("  yes      → <SIMPL_Y>")
    print("  no       → <SIMPL_N>")

    base_dir = "data/clean/hi_en"
    files_to_convert = [
        ("train.tsv", "train_v2.tsv"),
        ("val.tsv", "val_v2.tsv"),
        ("test.tsv", "test_v2.tsv"),
    ]

    all_stats = {}

    for input_file, output_file in files_to_convert:
        input_path = os.path.join(base_dir, input_file)
        output_path = os.path.join(base_dir, output_file)

        stats = convert_file(input_path, output_path)
        if stats:
            all_stats[input_file] = stats

    # Print summary statistics
    print("\n" + "="*70)
    print("📈 SUMMARY STATISTICS")
    print("="*70)

    for filename, stats in all_stats.items():
        print_stats(filename, stats)

    print("\n" + "="*70)
    print("✅ MIGRATION COMPLETE")
    print("="*70)
    print("\nNew files created:")
    for _, output_file in files_to_convert:
        output_path = os.path.join(base_dir, output_file)
        if os.path.exists(output_path):
            print(f"  ✓ {output_path}")

    print("\n⚠️  Original files preserved (train.tsv, val.tsv, test.tsv)")
    print("\n💡 Next steps:")
    print("  1. Run scripts/add_special_tokens_to_tokenizer.py")
    print("  2. Update configs/qlora_hi_en.yaml to use train_v2.tsv")
    print("  3. Use --token_format=special for inference")


if __name__ == "__main__":
    main()
