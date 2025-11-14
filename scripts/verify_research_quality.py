#!/usr/bin/env python3
"""
Verify research-grade dataset quality by measuring style signal strength.
"""

from pathlib import Path
import csv
import re

def has_contraction(text: str) -> bool:
    """Check if text contains any contractions."""
    contraction_patterns = [
        r"\bdon't\b", r"\bdoesn't\b", r"\bdidn't\b", r"\bisn't\b", r"\baren't\b",
        r"\bwasn't\b", r"\bweren't\b", r"\bcan't\b", r"\bwon't\b", r"\bwouldn't\b",
        r"\bshouldn't\b", r"\bcouldn't\b", r"\bhaven't\b", r"\bhasn't\b", r"\bhadn't\b",
        r"\bit's\b", r"\bthat's\b", r"\bthere's\b", r"\bi'm\b", r"\bwe'll\b",
        r"\byou'll\b", r"\bthey'll\b", r"\bhe's\b", r"\bshe's\b", r"\bwe're\b",
        r"\bthey're\b", r"\bi've\b", r"\bwe've\b", r"\bthey've\b", r"\bi'd\b",
        r"\byou'd\b", r"\bhe'd\b", r"\bshe'd\b", r"\bwe'd\b", r"\bthey'd\b",
        r"\blet's\b", r"\bwho's\b", r"\bwhat's\b", r"\bwhere's\b", r"\byou're\b",
        r"\bthere're\b", r"\bwho're\b", r"\bwhen's\b", r"\bwhy's\b", r"\bhow's\b",
        r"\bi'll\b", r"\bthat'll\b",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in contraction_patterns)


def has_formal_vocab(text: str) -> bool:
    """Check if text contains formal vocabulary."""
    formal_words = [
        r"\bhowever\b", r"\btherefore\b", r"\bconsequently\b", r"\bfurthermore\b",
        r"\bmoreover\b", r"\bnevertheless\b", r"\badditionally\b", r"\bthus\b",
        r"\bhence\b", r"\bregarding\b", r"\bconcerning\b", r"\butilize\b",
        r"\bdemonstrate\b", r"\bfacilitate\b", r"\bcommence\b", r"\bterminate\b",
        r"\bobtain\b", r"\bpurchase\b", r"\bassist(?:ance)?\b", r"\binquire\b",
        r"\bensure\b", r"\bindicate\b", r"\brequire\b", r"\bnumerous\b",
        r"\bsubstantial\b", r"\bsufficient\b", r"\bimplement\b",
        r"\bcomprehensive\b", r"\bendeavor\b",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in formal_words)


def has_casual_vocab(text: str) -> bool:
    """Check if text contains casual vocabulary."""
    casual_patterns = [
        r"\bbut\b", r"\bso\b", r"\balso\b", r"\bokay\b", r"\bok\b",
        r"\bguys\b", r"\bstuff\b", r"\bgonna\b", r"\bwanna\b",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in casual_patterns)


def get_word_count(text: str) -> int:
    """Get word count of text."""
    return len(text.split())


def analyze_file(file_path: Path) -> dict:
    """Analyze a single TSV file for style characteristics."""

    if not file_path.exists():
        return {"error": "File not found"}

    total = 0
    with_contractions = 0
    with_formal_vocab = 0
    with_casual_vocab = 0
    word_counts = []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            total += 1
            text = row['tgt']

            if has_contraction(text):
                with_contractions += 1
            if has_formal_vocab(text):
                with_formal_vocab += 1
            if has_casual_vocab(text):
                with_casual_vocab += 1

            word_counts.append(get_word_count(text))

    if total == 0:
        return {"error": "Empty file"}

    avg_words = sum(word_counts) / len(word_counts)

    return {
        "total": total,
        "contraction_pct": 100 * with_contractions / total,
        "formal_vocab_pct": 100 * with_formal_vocab / total,
        "casual_vocab_pct": 100 * with_casual_vocab / total,
        "avg_words": avg_words,
    }


def main():
    project_root = Path(__file__).resolve().parents[1]
    research_dir = project_root / "data" / "clean" / "hi_en" / "splits_research_v3"

    if not research_dir.exists():
        print(f"ERROR: {research_dir} not found")
        return 1

    print("=" * 70)
    print("RESEARCH-GRADE DATASET QUALITY VERIFICATION")
    print("=" * 70)

    combinations = [
        ("formal", "detailed"),
        ("formal", "simple"),
        ("casual", "detailed"),
        ("casual", "simple"),
    ]

    for split in ["train", "val", "test"]:
        print(f"\n{'='*70}")
        print(f"{split.upper()} SPLIT")
        print('='*70)

        for style, simplify in combinations:
            combo_name = f"{style}_{simplify}"
            file_path = research_dir / split / f"{combo_name}.tsv"

            print(f"\n{combo_name}:")
            stats = analyze_file(file_path)

            if "error" in stats:
                print(f"  ERROR: {stats['error']}")
                continue

            print(f"  Total samples: {stats['total']:,}")
            print(f"  Contractions: {stats['contraction_pct']:.1f}%")
            print(f"  Formal vocab: {stats['formal_vocab_pct']:.1f}%")
            print(f"  Casual vocab: {stats['casual_vocab_pct']:.1f}%")
            print(f"  Avg words: {stats['avg_words']:.1f}")

            # Quality assessment
            if style == "formal":
                if stats['contraction_pct'] < 5.0 and stats['formal_vocab_pct'] > 30.0:
                    print(f"  ✓ QUALITY: EXCELLENT (research-grade formal)")
                elif stats['contraction_pct'] < 10.0 and stats['formal_vocab_pct'] > 20.0:
                    print(f"  ✓ QUALITY: GOOD (acceptable formal)")
                else:
                    print(f"  ✗ QUALITY: WEAK (needs more formalization)")

            elif style == "casual":
                if stats['contraction_pct'] > 60.0 and stats['casual_vocab_pct'] > 40.0:
                    print(f"  ✓ QUALITY: EXCELLENT (research-grade casual)")
                elif stats['contraction_pct'] > 40.0 and stats['casual_vocab_pct'] > 20.0:
                    print(f"  ✓ QUALITY: GOOD (acceptable casual)")
                else:
                    print(f"  ✗ QUALITY: WEAK (needs more casualization)")

    print(f"\n{'='*70}")
    print("SIGNAL STRENGTH COMPARISON")
    print('='*70)

    # Compare formal_detailed vs casual_simple (maximum contrast)
    formal_file = research_dir / "train" / "formal_detailed.tsv"
    casual_file = research_dir / "train" / "casual_simple.tsv"

    formal_stats = analyze_file(formal_file)
    casual_stats = analyze_file(casual_file)

    if "error" not in formal_stats and "error" not in casual_stats:
        contraction_diff = abs(formal_stats['contraction_pct'] - casual_stats['contraction_pct'])
        vocab_diff = abs(formal_stats['formal_vocab_pct'] - casual_stats['casual_vocab_pct'])

        print(f"\nFormal (detailed) vs Casual (simple):")
        print(f"  Contraction difference: {contraction_diff:.1f}%")
        print(f"  Vocabulary difference: {vocab_diff:.1f}%")
        print(f"  Total signal strength: {contraction_diff + vocab_diff:.1f}%")

        total_signal = contraction_diff + vocab_diff
        if total_signal >= 100:
            print(f"  ✓ RESEARCH-GRADE: Excellent (≥100% signal)")
        elif total_signal >= 70:
            print(f"  ✓ RESEARCH-GRADE: Good (≥70% signal)")
        elif total_signal >= 50:
            print(f"  ~ ACCEPTABLE: Moderate (≥50% signal)")
        else:
            print(f"  ✗ WEAK: Insufficient (<50% signal)")

    print("=" * 70)


if __name__ == "__main__":
    exit(main())
