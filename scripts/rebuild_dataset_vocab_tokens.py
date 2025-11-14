#!/usr/bin/env python3
"""
Rebuild dataset with vocabulary-native control tokens instead of special tokens.

This script converts:
  <FORMAL> → formal
  <INFORMAL> → casual
  <SIMPL_Y> → simple
  <SIMPL_N> → detailed

These are all single-token words from the existing vocabulary, which should
be much easier for the model to learn as control signals.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict

# Control token mappings (old → new)
CONTROL_MAPPING = {
    '<FORMAL>': 'formal',
    '<INFORMAL>': 'casual',
    '<SIMPL_Y>': 'simple',
    '<SIMPL_N>': 'detailed',
}


def convert_tsv(input_file: Path, output_file: Path) -> Dict[str, int]:
    """Convert TSV file from special tokens to vocab tokens."""
    print(f"Converting {input_file} → {output_file}")

    stats = {'total': 0, 'converted': 0}

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8', newline='') as fout:

        reader = csv.DictReader(fin, delimiter='\t')
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter='\t')
        writer.writeheader()

        for row in reader:
            stats['total'] += 1

            # Convert style_src and simplify columns
            if row['style_src'] in CONTROL_MAPPING:
                row['style_src'] = CONTROL_MAPPING[row['style_src']]
                stats['converted'] += 1

            if row['simplify'] in CONTROL_MAPPING:
                row['simplify'] = CONTROL_MAPPING[row['simplify']]

            # Also convert style_tgt if it exists
            if 'style_tgt' in row and row['style_tgt'] in CONTROL_MAPPING:
                row['style_tgt'] = CONTROL_MAPPING[row['style_tgt']]

            writer.writerow(row)

    print(f"  ✓ Converted {stats['converted']:,} rows")
    return stats


def main():
    parser = argparse.ArgumentParser(description='Rebuild dataset with vocab tokens')
    parser.add_argument('--input-dir', default='data/clean/hi_en',
                       help='Input directory with v2 TSV files')
    parser.add_argument('--output-dir', default='data/clean/hi_en',
                       help='Output directory for v3 TSV files')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DATASET REBUILD: Special Tokens → Vocabulary Tokens")
    print("=" * 80)
    print()
    print("Control Token Mappings:")
    for old, new in CONTROL_MAPPING.items():
        print(f"  {old:15s} → {new}")
    print()

    # Convert all dataset files
    files_to_convert = {
        'train_v2.tsv': 'train_v3.tsv',
        'val_v2.tsv': 'val_v3.tsv',
        'test_v2.tsv': 'test_v3.tsv',
    }

    total_stats = {'total': 0, 'converted': 0}

    for input_name, output_name in files_to_convert.items():
        input_file = input_dir / input_name
        output_file = output_dir / output_name

        if not input_file.exists():
            print(f"⚠️  Skipping {input_name} (not found)")
            continue

        stats = convert_tsv(input_file, output_file)
        total_stats['total'] += stats['total']
        total_stats['converted'] += stats['converted']
        print()

    print("=" * 80)
    print(f"CONVERSION COMPLETE")
    print(f"  Total rows:      {total_stats['total']:,}")
    print(f"  Converted rows:  {total_stats['converted']:,}")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Verify new dataset: head data/clean/hi_en/train_v3.tsv")
    print("  2. Update config to use v3 files")
    print("  3. Start training!")
    print()


if __name__ == '__main__':
    main()
