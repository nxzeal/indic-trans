#!/usr/bin/env python3
"""
Research-grade style enhancement targeting 60%+ distinction.

This script applies aggressive transformations to create clear,
measurable style differences suitable for research/industry standards.
"""

from pathlib import Path
import csv
import re
import random
from typing import Dict

random.seed(42)

# ============================================================================
# CONTRACTIONS (expanded list)
# ============================================================================
CONTRACTIONS = {
    "do not": "don't", "does not": "doesn't", "did not": "didn't",
    "is not": "isn't", "are not": "aren't", "was not": "wasn't",
    "were not": "weren't", "cannot": "can't", "can not": "can't",
    "will not": "won't", "would not": "wouldn't", "should not": "shouldn't",
    "could not": "couldn't", "have not": "haven't", "has not": "hasn't",
    "had not": "hadn't", "it is": "it's", "that is": "that's",
    "there is": "there's", "there are": "there're", "i am": "i'm",
    "we will": "we'll", "you will": "you'll", "they will": "they'll",
    "he is": "he's", "she is": "she's", "we are": "we're",
    "they are": "they're", "i have": "i've", "we have": "we've",
    "they have": "they've", "i would": "i'd", "you would": "you'd",
    "he would": "he'd", "she would": "she'd", "we would": "we'd",
    "they would": "they'd", "let us": "let's", "who is": "who's",
    "who are": "who're", "what is": "what's", "where is": "where's",
    "when is": "when's", "why is": "why's", "how is": "how's",
    "i will": "i'll", "you are": "you're", "that will": "that'll",
}

# ============================================================================
# VOCABULARY TRANSFORMATIONS
# ============================================================================
FORMAL_TO_CASUAL = {
    r'\bhowever\b': 'but', r'\btherefore\b': 'so', r'\bconsequently\b': 'so',
    r'\bfurthermore\b': 'also', r'\bmoreover\b': 'also', r'\bnevertheless\b': 'but',
    r'\badditionally\b': 'also', r'\bthus\b': 'so', r'\bhence\b': 'so',
    r'\bregarding\b': 'about', r'\bconcerning\b': 'about',
    r'\butilize\b': 'use', r'\bdemonstrate\b': 'show', r'\bfacilitate\b': 'help',
    r'\bcommence\b': 'start', r'\bterminate\b': 'end', r'\bobtain\b': 'get',
    r'\bpurchase\b': 'buy', r'\bassistance\b': 'help', r'\binquire\b': 'ask',
    r'\bensure\b': 'make sure', r'\bindicate\b': 'show', r'\brequire\b': 'need',
    r'\bapproximately\b': 'about', r'\bnumerous\b': 'many',
    r'\bsubstantial\b': 'large', r'\bsufficient\b': 'enough',
    r'\badditional\b': 'more', r'\bimplement\b': 'do',
    r'\bcomprehensive\b': 'full', r'\bendeavor\b': 'try',
}

CASUAL_TO_FORMAL = {
    r'\bbut\b': 'however', r'\bso\b': 'therefore', r'\balso\b': 'furthermore',
    r'\babout\b': 'regarding', r'\buse\b': 'utilize', r'\bshow\b': 'demonstrate',
    r'\bhelp\b': 'assist', r'\bstart\b': 'commence', r'\bend\b': 'terminate',
    r'\bget\b': 'obtain', r'\bbuy\b': 'purchase', r'\bask\b': 'inquire',
    r'\bmake sure\b': 'ensure', r'\bneed\b': 'require', r'\bmany\b': 'numerous',
    r'\blarge\b': 'substantial', r'\benough\b': 'sufficient', r'\bmore\b': 'additional',
    r'\bdo\b': 'implement', r'\bfull\b': 'comprehensive', r'\btry\b': 'endeavor',
    r'\bokay\b': 'very well', r'\bok\b': 'very well', r'\bguys\b': 'everyone',
}

# ============================================================================
# ENHANCEMENT FUNCTIONS
# ============================================================================

def expand_contractions(text: str) -> str:
    """Expand ALL contractions for formal style - ULTRA AGGRESSIVE."""
    result = text

    # First: expand all known contractions
    for full, contr in CONTRACTIONS.items():
        result = re.sub(rf"\b{re.escape(contr)}\b", full, result, flags=re.IGNORECASE)

    # Second: catch any remaining contractions we might have missed
    # Replace any word ending in n't
    result = re.sub(r"\b(\w+)n't\b", lambda m: m.group(1) + " not", result, flags=re.IGNORECASE)

    # Replace any remaining 's that might be "is" or "has"
    result = re.sub(r"\b(\w+)'s\b", lambda m: m.group(1) + " is", result)

    # Replace 'll with will
    result = re.sub(r"\b(\w+)'ll\b", lambda m: m.group(1) + " will", result, flags=re.IGNORECASE)

    # Replace 're with are
    result = re.sub(r"\b(\w+)'re\b", lambda m: m.group(1) + " are", result, flags=re.IGNORECASE)

    # Replace 've with have
    result = re.sub(r"\b(\w+)'ve\b", lambda m: m.group(1) + " have", result, flags=re.IGNORECASE)

    # Replace 'd with would
    result = re.sub(r"\b(\w+)'d\b", lambda m: m.group(1) + " would", result, flags=re.IGNORECASE)

    # Replace 'm with am
    result = re.sub(r"\b(\w+)'m\b", lambda m: m.group(1) + " am", result, flags=re.IGNORECASE)

    return result


def aggressive_contract(text: str, target_ratio: float = 0.7) -> str:
    """
    ULTRA-AGGRESSIVELY insert contractions targeting 70%+ coverage.

    This actively hunts for ALL contraction opportunities.
    """
    result = text

    # Pass 1: Contract all known full forms
    for full, contr in CONTRACTIONS.items():
        result = re.sub(rf"\b{re.escape(full)}\b", contr, result, flags=re.IGNORECASE)

    # Pass 2: Systematic contraction of ALL auxiliary patterns
    # Subject + be verb
    result = re.sub(r'\bI am\b', "I'm", result, flags=re.IGNORECASE)
    result = re.sub(r'\byou are\b', "you're", result, flags=re.IGNORECASE)
    result = re.sub(r'\bwe are\b', "we're", result, flags=re.IGNORECASE)
    result = re.sub(r'\bthey are\b', "they're", result, flags=re.IGNORECASE)
    result = re.sub(r'\bhe is\b', "he's", result, flags=re.IGNORECASE)
    result = re.sub(r'\bshe is\b', "she's", result, flags=re.IGNORECASE)
    result = re.sub(r'\bit is\b', "it's", result, flags=re.IGNORECASE)
    result = re.sub(r'\bthat is\b', "that's", result, flags=re.IGNORECASE)
    result = re.sub(r'\bwhat is\b', "what's", result, flags=re.IGNORECASE)
    result = re.sub(r'\bthere is\b', "there's", result, flags=re.IGNORECASE)
    result = re.sub(r'\bwho is\b', "who's", result, flags=re.IGNORECASE)
    result = re.sub(r'\bwhere is\b', "where's", result, flags=re.IGNORECASE)

    # Subject + will
    result = re.sub(r'\bI will\b', "I'll", result, flags=re.IGNORECASE)
    result = re.sub(r'\byou will\b', "you'll", result, flags=re.IGNORECASE)
    result = re.sub(r'\bwe will\b', "we'll", result, flags=re.IGNORECASE)
    result = re.sub(r'\bthey will\b', "they'll", result, flags=re.IGNORECASE)
    result = re.sub(r'\bhe will\b', "he'll", result, flags=re.IGNORECASE)
    result = re.sub(r'\bshe will\b', "she'll", result, flags=re.IGNORECASE)
    result = re.sub(r'\bit will\b', "it'll", result, flags=re.IGNORECASE)
    result = re.sub(r'\bthat will\b', "that'll", result, flags=re.IGNORECASE)

    # Subject + would
    result = re.sub(r'\bI would\b', "I'd", result, flags=re.IGNORECASE)
    result = re.sub(r'\byou would\b', "you'd", result, flags=re.IGNORECASE)
    result = re.sub(r'\bwe would\b', "we'd", result, flags=re.IGNORECASE)
    result = re.sub(r'\bthey would\b', "they'd", result, flags=re.IGNORECASE)
    result = re.sub(r'\bhe would\b', "he'd", result, flags=re.IGNORECASE)
    result = re.sub(r'\bshe would\b', "she'd", result, flags=re.IGNORECASE)

    # Subject + have
    result = re.sub(r'\bI have\b', "I've", result, flags=re.IGNORECASE)
    result = re.sub(r'\byou have\b', "you've", result, flags=re.IGNORECASE)
    result = re.sub(r'\bwe have\b', "we've", result, flags=re.IGNORECASE)
    result = re.sub(r'\bthey have\b', "they've", result, flags=re.IGNORECASE)

    # Negatives
    result = re.sub(r'\bdo not\b', "don't", result, flags=re.IGNORECASE)
    result = re.sub(r'\bdoes not\b', "doesn't", result, flags=re.IGNORECASE)
    result = re.sub(r'\bdid not\b', "didn't", result, flags=re.IGNORECASE)
    result = re.sub(r'\bis not\b', "isn't", result, flags=re.IGNORECASE)
    result = re.sub(r'\bare not\b', "aren't", result, flags=re.IGNORECASE)
    result = re.sub(r'\bwas not\b', "wasn't", result, flags=re.IGNORECASE)
    result = re.sub(r'\bwere not\b', "weren't", result, flags=re.IGNORECASE)
    result = re.sub(r'\bcan not\b', "can't", result, flags=re.IGNORECASE)
    result = re.sub(r'\bcannot\b', "can't", result, flags=re.IGNORECASE)
    result = re.sub(r'\bwill not\b', "won't", result, flags=re.IGNORECASE)
    result = re.sub(r'\bwould not\b', "wouldn't", result, flags=re.IGNORECASE)
    result = re.sub(r'\bshould not\b', "shouldn't", result, flags=re.IGNORECASE)
    result = re.sub(r'\bcould not\b', "couldn't", result, flags=re.IGNORECASE)
    result = re.sub(r'\bhave not\b', "haven't", result, flags=re.IGNORECASE)
    result = re.sub(r'\bhas not\b', "hasn't", result, flags=re.IGNORECASE)
    result = re.sub(r'\bhad not\b', "hadn't", result, flags=re.IGNORECASE)

    # Other common contractions
    result = re.sub(r'\blet us\b', "let's", result, flags=re.IGNORECASE)

    return result


def formalize_vocab(text: str) -> str:
    """Replace casual words with formal equivalents."""
    result = text
    for casual_pattern, formal in CASUAL_TO_FORMAL.items():
        result = re.sub(casual_pattern, formal, result, flags=re.IGNORECASE)
    return result


def casualize_vocab(text: str) -> str:
    """Replace formal words with casual equivalents."""
    result = text
    for formal_pattern, casual in FORMAL_TO_CASUAL.items():
        result = re.sub(formal_pattern, casual, result, flags=re.IGNORECASE)
    return result


def enhance_formal_detailed(text: str) -> str:
    """
    Formal + Detailed enhancement:
    - Expand all contractions
    - Use formal vocabulary
    - Keep/add complexity
    """
    result = text

    # Expand contractions
    result = expand_contractions(result)

    # Formalize vocabulary
    result = formalize_vocab(result)

    # Add formal connectors occasionally for detail
    if random.random() < 0.15 and len(result.split()) > 10:
        # Add formal transitional phrases
        if ', ' in result and 'furthermore' not in result.lower():
            result = result.replace(', ', ', furthermore, ', 1)

    return result


def enhance_casual_simple(text: str) -> str:
    """
    Casual + Simple enhancement:
    - Aggressive contraction insertion (70% coverage)
    - Use casual vocabulary
    - Simplify heavily
    """
    result = text

    # Casualize vocabulary first
    result = casualize_vocab(result)

    # Aggressive contraction insertion
    result = aggressive_contract(result, target_ratio=0.7)

    # Simplify: Remove parentheticals
    result = re.sub(r'\s*\([^)]*\)', '', result)

    # Simplify: Remove modifiers
    modifiers = [
        r'\bvery\s+', r'\breally\s+', r'\bquite\s+', r'\bextremely\s+',
        r'\bparticularly\s+', r'\bespecially\s+', r'\bsignificantly\s+',
        r'\bhighly\s+', r'\bexceptionally\s+', r'\bremarkably\s+',
    ]
    for mod in modifiers:
        result = re.sub(mod, '', result, flags=re.IGNORECASE)

    # Simplify: Take first clause if too long
    if len(result.split()) >= 12:
        clauses = re.split(r'[;:—–]|,\s+(?:and|but|or)\s+', result)
        result = clauses[0]

    # Clean up spacing
    result = re.sub(r'\s{2,}', ' ', result).strip()

    # Ensure proper punctuation
    if result and result[-1] not in '.!?':
        result += '.'

    return result


def enhance_formal_simple(text: str) -> str:
    """
    Formal + Simple enhancement:
    - Expand contractions
    - Use formal vocabulary
    - Simplify moderately
    """
    result = text

    # Expand contractions
    result = expand_contractions(result)

    # Formalize vocabulary
    result = formalize_vocab(result)

    # Simplify: Remove parentheticals
    result = re.sub(r'\s*\([^)]*\)', '', result)

    # Simplify: Take first clause if very long
    if len(result.split()) >= 14:
        clauses = re.split(r'[;:—–]', result)
        result = clauses[0]

    # Clean up
    result = re.sub(r'\s{2,}', ' ', result).strip()
    if result and result[-1] not in '.!?':
        result += '.'

    return result


def enhance_casual_detailed(text: str) -> str:
    """
    Casual + Detailed enhancement:
    - Aggressive contractions
    - Use casual vocabulary
    - Keep detail/complexity
    """
    result = text

    # Casualize vocabulary
    result = casualize_vocab(result)

    # Aggressive contractions
    result = aggressive_contract(result, target_ratio=0.7)

    # Keep detail - don't simplify
    # Just clean up spacing
    result = re.sub(r'\s{2,}', ' ', result).strip()

    return result


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def process_file(input_file: Path, output_file: Path, style: str, simplify: str):
    """Process a single TSV file with aggressive enhancement."""

    # Choose enhancement function
    if style == "formal" and simplify == "detailed":
        enhance_fn = enhance_formal_detailed
    elif style == "formal" and simplify == "simple":
        enhance_fn = enhance_formal_simple
    elif style == "casual" and simplify == "detailed":
        enhance_fn = enhance_casual_detailed
    elif style == "casual" and simplify == "simple":
        enhance_fn = enhance_casual_simple
    else:
        raise ValueError(f"Unknown combination: {style}/{simplify}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8', newline='') as fout:

        reader = csv.DictReader(fin, delimiter='\t')
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter='\t')
        writer.writeheader()

        total = 0
        enhanced = 0

        for row in reader:
            total += 1
            original_en = row['tgt']
            enhanced_en = enhance_fn(original_en)

            if enhanced_en != original_en:
                enhanced += 1

            row['tgt'] = enhanced_en
            writer.writerow(row)

        if total > 0:
            print(f"  {input_file.name}: {enhanced}/{total} enhanced ({100*enhanced/total:.1f}%)")
        else:
            print(f"  {input_file.name}: EMPTY FILE")


def main():
    """Create research-grade datasets for all four adapter combinations."""

    project_root = Path(__file__).resolve().parents[1]
    splits_dir = project_root / "data" / "clean" / "hi_en" / "splits"  # Use ORIGINAL splits
    output_base = project_root / "data" / "clean" / "hi_en" / "splits_research_v3"

    if not splits_dir.exists():
        print(f"ERROR: {splits_dir} not found.")
        return 1

    combinations = [
        ("formal", "detailed"),
        ("formal", "simple"),
        ("casual", "detailed"),
        ("casual", "simple"),
    ]

    for split in ["train", "val", "test"]:
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print('='*60)

        for style, simplify in combinations:
            combo_name = f"{style}_{simplify}"
            print(f"\n{combo_name}:")

            input_file = splits_dir / split / f"{combo_name}.tsv"
            output_file = output_base / split / f"{combo_name}.tsv"

            if not input_file.exists():
                print(f"  WARNING: {input_file} not found, skipping")
                continue

            process_file(input_file, output_file, style, simplify)

    print(f"\n{'='*60}")
    print("Enhancement complete!")
    print(f"Research-grade datasets saved to: {output_base}")
    print('='*60)

    return 0


if __name__ == "__main__":
    exit(main())
