#!/usr/bin/env python3
"""Comprehensive verification of v3 dataset quality."""

import csv
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer

print('=' * 80)
print('V3 DATASET COMPREHENSIVE VERIFICATION')
print('=' * 80)
print()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('models/indictrans2-indic-en-1B', trust_remote_code=True, use_fast=False)

# Verify control tokens are single-token
print('1. CONTROL TOKEN VERIFICATION')
print('-' * 80)
control_tokens = ['formal', 'casual', 'simple', 'detailed']
all_single = True
for token in control_tokens:
    ids = tokenizer.encode(token, add_special_tokens=False)
    token_str = 'token' if len(ids) == 1 else 'tokens'
    status = '✓' if len(ids) == 1 else '✗'
    id_str = str(ids[0]) if len(ids) == 1 else str(ids)
    print(f'{status} {token:12s} → Token ID: {id_str} ({len(ids)} {token_str})')
    if len(ids) != 1:
        all_single = False
print()
if all_single:
    print('✓ All control tokens are SINGLE tokens from vocabulary')
else:
    print('✗ ERROR: Some control tokens are multi-token!')
    exit(1)
print()

# Verify each dataset file
for split in ['train', 'val', 'test']:
    filepath = Path(f'data/clean/hi_en/{split}_v3.tsv')
    if not filepath.exists():
        print(f'✗ {split}_v3.tsv NOT FOUND')
        continue

    print(f'2. {split.upper()} DATASET VERIFICATION')
    print('-' * 80)

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames

        # Verify columns
        required_cols = ['src', 'tgt', 'style_src', 'style_tgt', 'simplify']
        print(f'Columns: {fieldnames}')
        missing = [col for col in required_cols if col not in fieldnames]
        if missing:
            print(f'✗ Missing columns: {missing}')
            exit(1)
        print(f'✓ All required columns present')
        print()

        # Analyze data
        rows = list(reader)
        style_src_dist = Counter()
        style_tgt_dist = Counter()
        simplify_dist = Counter()
        empty_src = 0
        empty_tgt = 0
        invalid_controls = []

        for i, row in enumerate(rows):
            # Check for empty fields
            if not row['src'].strip():
                empty_src += 1
            if not row['tgt'].strip():
                empty_tgt += 1

            # Track control token distribution
            style_src_dist[row['style_src']] += 1
            style_tgt_dist[row['style_tgt']] += 1
            simplify_dist[row['simplify']] += 1

            # Check for invalid control values
            if row['style_src'] not in control_tokens[:2]:  # formal, casual
                invalid_controls.append(f'Row {i+1}: style_src={row["style_src"]}')
            if row['style_tgt'] not in control_tokens[:2]:
                invalid_controls.append(f'Row {i+1}: style_tgt={row["style_tgt"]}')
            if row['simplify'] not in control_tokens[2:]:  # simple, detailed
                invalid_controls.append(f'Row {i+1}: simplify={row["simplify"]}')

        # Print statistics
        print(f'Total rows: {len(rows):,}')
        print()

        print('Control Token Distribution:')
        print(f'  style_src:  {dict(style_src_dist)}')
        print(f'  style_tgt:  {dict(style_tgt_dist)}')
        print(f'  simplify:   {dict(simplify_dist)}')
        print()

        # Quality checks
        issues = []
        if empty_src > 0:
            issues.append(f'{empty_src} rows with empty source text')
        if empty_tgt > 0:
            issues.append(f'{empty_tgt} rows with empty target text')
        if invalid_controls:
            issues.append(f'{len(invalid_controls)} rows with invalid control tokens')
            for err in invalid_controls[:3]:  # Show first 3
                print(f'  ✗ {err}')

        if issues:
            print('✗ QUALITY ISSUES FOUND:')
            for issue in issues:
                print(f'  - {issue}')
            print()
        else:
            print('✓ No quality issues found')
        print()

        # Show sample rows
        print(f'Sample rows from {split}:')
        print('-' * 80)
        for i in range(min(3, len(rows))):
            row = rows[i]
            print(f'Row {i+1}:')
            print(f'  SRC: {row["src"][:80]}...' if len(row["src"]) > 80 else f'  SRC: {row["src"]}')
            print(f'  TGT: {row["tgt"][:80]}...' if len(row["tgt"]) > 80 else f'  TGT: {row["tgt"]}')
            print(f'  Controls: style_src={row["style_src"]}, style_tgt={row["style_tgt"]}, simplify={row["simplify"]}')
            print()

print('=' * 80)
print('VERIFICATION COMPLETE')
print('=' * 80)
