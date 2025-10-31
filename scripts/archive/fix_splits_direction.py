import argparse
import csv
import shutil
import string
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Iterable, List, Sequence

ASCII_ALLOWED = set(string.ascii_letters + string.digits + string.punctuation + " ")
SCRIPT_RANGES: Dict[str, List[tuple[int, int]]] = {
    "hi": [(0x0900, 0x097F)],  # Devanagari
    "ta": [(0x0B80, 0x0BFF)],  # Tamil
}
DEFAULT_FIELDS = ["src", "tgt", "style_src", "style_tgt", "simplify", "pair"]
SPLITS = ["train.tsv", "val.tsv", "test.tsv"]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix split direction and add missing pair columns.")
    parser.add_argument("--dir", required=True, help="Directory containing split TSV files.")
    parser.add_argument("--pair", required=True, help="Expected language pair (e.g., hi-en).")
    parser.add_argument("--detect_lang", choices=["yes", "no"], default="yes")
    parser.add_argument("--backup", choices=["yes", "no"], default="yes")
    return parser.parse_args(argv)


def english_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = 0
    ascii_count = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if ord(ch) < 128 and ch in ASCII_ALLOWED:
            ascii_count += 1
    if total == 0:
        return 0.0
    return ascii_count / total


def has_script(text: str, base_lang: str) -> bool:
    ranges = SCRIPT_RANGES.get(base_lang)
    if not ranges:
        return False
    for ch in text:
        code = ord(ch)
        for start, end in ranges:
            if start <= code <= end:
                return True
    return False


def read_tsv(path: Path) -> tuple[List[Dict[str, str]], Sequence[str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        rows = [{key: (row.get(key, "") or "").strip() for key in fieldnames} for row in reader]
    return rows, fieldnames


def write_tsv(path: Path, fieldnames: Sequence[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(path.parent)) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    Path(tmp.name).replace(path)


def detect_swap(rows: List[Dict[str, str]], base_lang: str, detect_lang: bool) -> tuple[bool, Dict[str, float]]:
    sample = rows[:100]
    total = len(sample)
    stats = {
        "sample": total,
        "english_src": 0,
        "english_tgt": 0,
        "indic_src": 0,
        "indic_tgt": 0,
    }
    if total == 0 or not detect_lang:
        return False, stats

    for row in sample:
        src = row.get("src", "")
        tgt = row.get("tgt", "")
        if english_ratio(src) >= 0.8:
            stats["english_src"] += 1
        if english_ratio(tgt) >= 0.8:
            stats["english_tgt"] += 1
        if has_script(src, base_lang):
            stats["indic_src"] += 1
        if has_script(tgt, base_lang):
            stats["indic_tgt"] += 1

    english_src_ratio = stats["english_src"] / total if total else 0.0
    english_tgt_ratio = stats["english_tgt"] / total if total else 0.0
    indic_src_ratio = stats["indic_src"] / total if total else 0.0
    indic_tgt_ratio = stats["indic_tgt"] / total if total else 0.0

    should_swap = (
        english_src_ratio >= 0.6
        and english_tgt_ratio <= 0.4
        and indic_tgt_ratio >= 0.3
        and indic_src_ratio <= 0.5
    )
    return should_swap, stats


def process_file(path: Path, pair: str, detect_lang: bool, backup: bool) -> None:
    if not path.exists():
        print(f"[fix] {path.name}: file missing, skipping.")
        return

    rows, original_fields = read_tsv(path)
    fields = list(original_fields)
    pair_added = False
    if "pair" not in fields:
        fields.append("pair")
        pair_added = True
        for row in rows:
            row["pair"] = pair
    else:
        for row in rows:
            if not row.get("pair"):
                row["pair"] = pair

    base_lang = pair.split("-", 1)[0].lower()
    should_detect = detect_lang and pair.lower().endswith("-en")
    swapped = False
    stats = {
        "sample": 0,
        "english_src": 0,
        "english_tgt": 0,
        "indic_src": 0,
        "indic_tgt": 0,
    }
    if should_detect:
        swapped, stats = detect_swap(rows, base_lang, True)

    if swapped:
        for row in rows:
            row["src"], row["tgt"] = row.get("tgt", ""), row.get("src", "")
            row["style_src"], row["style_tgt"] = row.get("style_tgt", ""), row.get("style_src", "")

    if backup:
        backup_path = path.with_suffix(path.suffix + ".orig")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
            print(f"[fix] {path.name}: backup written to {backup_path.name}.")

    expected_order = [field for field in DEFAULT_FIELDS if field in fields]
    for field in fields:
        if field not in expected_order:
            expected_order.append(field)

    write_tsv(path, expected_order, rows)

    summary = (
        f"[fix] {path.name}: pair_column={'added' if pair_added else 'present'}, "
        f"swapped={'yes' if swapped else 'no'}, "
        f"sample={stats['sample']}, "
        f"english_src={stats['english_src']}, english_tgt={stats['english_tgt']}, "
        f"indic_src={stats['indic_src']}, indic_tgt={stats['indic_tgt']}"
    )
    print(summary)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    target_dir = Path(args.dir)
    if not target_dir.exists():
        print(f"Directory {target_dir} not found.", file=sys.stderr)
        return 1

    detect_lang = args.detect_lang.lower() == "yes"
    backup = args.backup.lower() == "yes"

    for split_name in SPLITS:
        process_file(target_dir / split_name, args.pair, detect_lang, backup)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
