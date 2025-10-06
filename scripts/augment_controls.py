import argparse
import csv
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Iterable, List, Sequence, Tuple

FIELDNAMES = ["src", "tgt", "style_src", "style_tgt", "simplify", "pair"]
COMMENT_MARK = "# augmented_by=augment_controls.py"
SIMPLIFY_MAX_TOKENS = 25
CUT_WORDS = {"which", "that", "where", "when"}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize controllable targets for Indic-English TSV datasets.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input TSV path (train split).")
    parser.add_argument("--out", dest="output_path", required=True, help="Augmented TSV output path.")
    parser.add_argument("--lang_pair", required=True, help="Language pair code (e.g., hi-en).")
    parser.add_argument("--sample_ratio", type=float, default=0.25, help="Fraction of rows to sample per mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    parser.add_argument("--modes", default="informal,formal,simplify", help="Comma-separated list of augmentation modes.")
    parser.add_argument("--max_src_len", type=int, default=256, help="Max tokens allowed for source sentences.")
    parser.add_argument("--max_tgt_len", type=int, default=192, help="Max tokens allowed for target sentences.")
    return parser.parse_args(argv)


def preserve_case(source: str, replacement: str) -> str:
    if not source:
        return replacement
    if source.isupper():
        return replacement.upper()
    if len(source) > 1 and source[0].isupper() and source[1:].islower():
        return replacement.capitalize()
    return replacement


def replace_case_insensitive(text: str, pattern: str, replacement: str) -> str:
    regex = re.compile(pattern, flags=re.IGNORECASE)

    def _sub(match: re.Match[str]) -> str:
        return preserve_case(match.group(0), replacement)

    return regex.sub(_sub, text)


INFORMAL_CONTRACTIONS: Sequence[Tuple[str, str]] = (
    (r"\bdo not\b", "don't"),
    (r"\bcannot\b", "can't"),
    (r"\bi am\b", "I'm"),
    (r"\bit is\b", "it's"),
    (r"\byou are\b", "you're"),
    (r"\bwe are\b", "we're"),
    (r"\bthey are\b", "they're"),
    (r"\bis not\b", "isn't"),
    (r"\bare not\b", "aren't"),
    (r"\bwas not\b", "wasn't"),
    (r"\bwere not\b", "weren't"),
    (r"\bwill not\b", "won't"),
    (r"\bwould not\b", "wouldn't"),
    (r"\bshould not\b", "shouldn't"),
    (r"\bcould not\b", "couldn't"),
    (r"\bhave not\b", "haven't"),
    (r"\bhas not\b", "hasn't"),
    (r"\bhad not\b", "hadn't"),
    (r"\blet us\b", "let's"),
)

INFORMAL_LEXICON: Sequence[Tuple[str, str]] = (
    (r"\bassist\b", "help"),
    (r"\brequest\b", "ask"),
    (r"\bpurchase\b", "buy"),
    (r"\butilize\b", "use"),
    (r"\bobtain\b", "get"),
    (r"\binform me\b", "tell me"),
    (r"\binform\b", "tell"),
    (r"\bregarding\b", "about"),
    (r"\brequire\b", "need"),
    (r"\battempt\b", "try"),
    (r"\bhowever\b", "but"),
)

FORMAL_EXPANSIONS: Sequence[Tuple[str, str]] = (
    (r"\bdon't\b", "do not"),
    (r"\bcan't\b", "cannot"),
    (r"\bI'm\b", "I am"),
    (r"\bit's\b", "it is"),
    (r"\byou're\b", "you are"),
    (r"\bwe're\b", "we are"),
    (r"\bthey're\b", "they are"),
    (r"\bisn't\b", "is not"),
    (r"\baren't\b", "are not"),
    (r"\bwasn't\b", "was not"),
    (r"\bweren't\b", "were not"),
    (r"\bwon't\b", "will not"),
    (r"\bwouldn't\b", "would not"),
    (r"\bshouldn't\b", "should not"),
    (r"\bcouldn't\b", "could not"),
    (r"\bhaven't\b", "have not"),
    (r"\bhasn't\b", "has not"),
    (r"\bhadn't\b", "had not"),
    (r"\blet's\b", "let us"),
)

FORMAL_LEXICON: Sequence[Tuple[str, str]] = (
    (r"\bhelp\b", "assist"),
    (r"\bask\b", "request"),
    (r"\bbuy\b", "purchase"),
    (r"\buse\b", "utilize"),
    (r"\bget\b", "obtain"),
    (r"\btell\b", "inform"),
    (r"\bneed\b", "require"),
    (r"\btry\b", "attempt"),
    (r"\bbut\b", "however"),
)

SIMPLIFY_LEXICON: Sequence[Tuple[str, str]] = (
    (r"\bcommence\b", "start"),
    (r"\bterminate\b", "end"),
    (r"\bapproximately\b", "about"),
    (r"\bpurchase\b", "buy"),
    (r"\bassistance\b", "help"),
    (r"\bendeavor\b", "try"),
    (r"\binform\b", "tell"),
)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def to_informal_en(text: str) -> str:
    result = text
    # Drop leading "please"
    result = re.sub(r"^(\s*)please\s+", r"\1", result, flags=re.IGNORECASE)
    for pattern, repl in INFORMAL_CONTRACTIONS:
        result = replace_case_insensitive(result, pattern, repl)
    for pattern, repl in INFORMAL_LEXICON:
        result = replace_case_insensitive(result, pattern, repl)
    return normalize_spaces(result)


def to_formal_en(text: str) -> str:
    result = text
    for pattern, repl in FORMAL_EXPANSIONS:
        result = replace_case_insensitive(result, pattern, repl)
    for pattern, repl in FORMAL_LEXICON:
        result = replace_case_insensitive(result, pattern, repl)
    result = ensure_please(result)
    return normalize_spaces(result)


def ensure_please(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return text
    lowered = stripped.lower()
    if "please" in lowered:
        return text
    leading_ws = len(text) - len(text.lstrip())
    prefix = " " * leading_ws
    if lowered.startswith("could you") or lowered.startswith("would you"):
        return f"{prefix}Please {stripped}"
    first = stripped.split()[0]
    if first.lower() not in {"i", "we", "you", "they", "he", "she", "it", "the", "this", "that"}:
        return f"{prefix}Please {stripped}"
    return text


def simplify_en(text: str) -> str:
    result = re.sub(r"\([^)]*\)", "", text)
    for pattern, repl in SIMPLIFY_LEXICON:
        result = replace_case_insensitive(result, pattern, repl)
    result = normalize_spaces(result)
    if not result:
        return result
    parts = [p.strip() for p in re.split(r"\s*(?:,|;|\band\b)\s*", result, flags=re.IGNORECASE) if p.strip()]
    if not parts:
        parts = [result]
    if len(parts) > 2:
        parts = parts[:2]
    simplified = ". ".join(parts)
    tokens = simplified.split()
    if len(tokens) > SIMPLIFY_MAX_TOKENS:
        for idx, tok in enumerate(tokens):
            bare = re.sub(r"[^a-zA-Z]", "", tok).lower()
            if bare in CUT_WORDS:
                tokens = tokens[:idx]
                break
        if len(tokens) > SIMPLIFY_MAX_TOKENS:
            tokens = tokens[:SIMPLIFY_MAX_TOKENS]
        simplified = " ".join(tokens)
    return normalize_spaces(simplified)


def read_tsv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader((line for line in handle if not line.lstrip().startswith("#")), delimiter="\t")
        for row in reader:
            rows.append({key: row.get(key, "").strip() for key in FIELDNAMES})
    return rows


def write_tsv(path: Path, rows: Sequence[Dict[str, str]], include_comment: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(path.parent)) as tmp:
        if include_comment:
            tmp.write(COMMENT_MARK + "\n")
        writer = csv.DictWriter(tmp, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temp_path = Path(tmp.name)
    temp_path.replace(path)


def tokenize_len(text: str) -> int:
    return len(text.split())


def build_augmented_row(
    mode: str,
    base_row: Dict[str, str],
    max_src_len: int,
    max_tgt_len: int,
) -> Tuple[Dict[str, str] | None, str]:
    src = base_row["src"].strip()
    tgt = base_row["tgt"].strip()
    if tokenize_len(src) > max_src_len:
        return None, "src_exceeds"
    mode_lower = mode.lower()
    new_style = base_row["style_tgt"]
    new_simplify = base_row["simplify"]
    if mode_lower == "informal":
        new_tgt = to_informal_en(tgt)
        new_style = "informal"
        new_simplify = "no"
    elif mode_lower == "formal":
        new_tgt = to_formal_en(tgt)
        new_style = "formal"
        new_simplify = "no"
    elif mode_lower == "simplify":
        new_tgt = simplify_en(tgt)
        new_simplify = "yes"
    else:
        return None, "unknown_mode"

    if not new_tgt:
        return None, "empty_tgt"
    if tokenize_len(new_tgt) > max_tgt_len:
        return None, "tgt_exceeds"

    augmented = {
        "src": src,
        "tgt": new_tgt,
        "style_src": base_row["style_src"],
        "style_tgt": new_style,
        "simplify": new_simplify,
        "pair": base_row["pair"],
    }
    return augmented, "ok"


def select_indices(total: int, sample_ratio: float, modes: Sequence[str], seed: int) -> Dict[str, List[int]]:
    if total == 0:
        return {mode: [] for mode in modes}
    sample_size = max(0, round(sample_ratio * total))
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    assigned: Dict[str, List[int]] = {}
    cursor = 0
    for idx, mode in enumerate(modes):
        remaining = len(indices) - cursor
        if remaining <= 0:
            assigned[mode] = []
            continue
        needed = min(sample_size, remaining)
        # Ensure remaining modes can still get their share
        remaining_modes = len(modes) - idx - 1
        if remaining_modes > 0 and needed > remaining - remaining_modes * sample_size:
            needed = max(0, remaining - remaining_modes * sample_size)
        assigned[mode] = indices[cursor : cursor + needed]
        cursor += needed
    return assigned


def ensure_pair_english(lang_pair: str) -> bool:
    return lang_pair.lower().endswith("-en")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    lang_pair = args.lang_pair.lower().strip()

    if not ensure_pair_english(lang_pair):
        print(f"[loader] Pair {lang_pair} is not English-target; skipping augmentation.")
        return 0

    if not input_path.exists():
        print(f"Input file {input_path} not found.", file=sys.stderr)
        return 1

    original_rows = read_tsv(input_path)
    originals_count = len(original_rows)
    if originals_count == 0:
        print("No rows to augment; exiting.")
        return 0

    modes = [mode.strip().lower() for mode in args.modes.split(",") if mode.strip()]
    modes = [mode for mode in modes if mode in {"informal", "formal", "simplify"}]
    if not modes:
        print("No valid augmentation modes provided.", file=sys.stderr)
        return 1

    existing_keys = {
        (
            row["src"],
            row["tgt"],
            row["style_tgt"],
            row["simplify"],
        )
        for row in original_rows
    }

    augmented_rows: List[Dict[str, str]] = []
    per_mode_counts: Dict[str, int] = defaultdict(int)

    eligible_indices = [idx for idx, row in enumerate(original_rows) if row["pair"].lower() == lang_pair]
    if not eligible_indices:
        print(f"No eligible rows found for pair {lang_pair}; exiting.")
        return 0

    selection = select_indices(len(eligible_indices), args.sample_ratio, modes, args.seed)

    for mode, sub_indices in selection.items():
        for pos in sub_indices:
            base_idx = eligible_indices[pos]
            base_row = original_rows[base_idx]
            augmented, status = build_augmented_row(mode, base_row, args.max_src_len, args.max_tgt_len)
            if augmented is None:
                continue
            key = (
                augmented["src"],
                augmented["tgt"],
                augmented["style_tgt"],
                augmented["simplify"],
            )
            if key in existing_keys:
                continue
            existing_keys.add(key)
            augmented_rows.append(augmented)
            per_mode_counts[mode] += 1

    combined_rows = original_rows + augmented_rows

    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
            if first_line.startswith(COMMENT_MARK):
                print("[loader] Existing augmentation detected; overwriting output.")

    write_tsv(output_path, combined_rows, include_comment=True)

    backup_path = input_path.with_suffix(".orig.tsv")
    if not backup_path.exists():
        shutil.copy2(input_path, backup_path)
        print(f"[loader] Backed up original split to {backup_path}.")

    write_tsv(input_path, combined_rows, include_comment=False)
    print(f"[loader] Replaced {input_path.name} with augmented data.")

    informal_count = per_mode_counts.get("informal", 0)
    formal_count = per_mode_counts.get("formal", 0)
    simplify_count = per_mode_counts.get("simplify", 0)
    total_added = informal_count + formal_count + simplify_count
    total_final = originals_count + total_added

    print("Augmentation summary:")
    print(
        f"- {lang_pair}: +{informal_count} informal, +{formal_count} formal, +{simplify_count} simplified "
        f"(from {originals_count} originals) -> {total_final} total"
    )
    print("Next:")
    print(
        "  python scripts/augment_controls.py --in data/clean/hi_en/train.tsv --out data/clean/hi_en/train_aug.tsv "
        "--lang_pair hi-en --sample_ratio 0.25 --seed 42 --modes informal,formal,simplify "
        "--max_src_len 256 --max_tgt_len 192"
    )
    print(
        "  python scripts/augment_controls.py --in data/clean/ta_en/train.tsv --out data/clean/ta_en/train_aug.tsv "
        "--lang_pair ta-en --sample_ratio 0.25 --seed 42 --modes informal,formal,simplify "
        "--max_src_len 256 --max_tgt_len 192"
    )
    print("Then re-train:")
    print("  accelerate launch scripts/train_lora.py --config configs/qlora_hi_en.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
