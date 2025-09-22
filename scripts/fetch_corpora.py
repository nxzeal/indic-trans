import argparse
import csv
import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from datasets import get_dataset_config_names, load_dataset
from tqdm.auto import tqdm

try:  # `datasets` surfaces different translation feature classes across versions
    from datasets.features import Translation, TranslationVariableLanguages  # type: ignore

    TRANSLATION_TYPES = (Translation, TranslationVariableLanguages)
except (ImportError, AttributeError):  # pragma: no cover - defensive import
    TRANSLATION_TYPES = tuple()

try:
    import langid
except ImportError:  # pragma: no cover - optional dependency
    langid = None  # type: ignore

SUPPORTED_PAIRS = {"hi-en", "ta-en", "te-en", "ml-en"}
LANG_ALIASES: Dict[str, Sequence[str]] = {
    "hi": ["hi", "hin", "hindi", "hin_deva", "hin-devanagari", "indic_hi"],
    "ta": ["ta", "tam", "tamil", "tam_taml", "tam-tamil"],
    "te": ["te", "tel", "telugu", "tel_telu", "tel-telugu"],
    "ml": ["ml", "mal", "malayalam", "mal_mlym", "mal-malayalam"],
}
LANGID_CODES = {"hi": "hi", "ta": "ta", "te": "te", "ml": "ml"}
FLORES_CODES = {
    "hi-en": ("hin_Deva", "eng_Latn"),
    "ta-en": ("tam_Taml", "eng_Latn"),
    "te-en": ("tel_Telu", "eng_Latn"),
    "ml-en": ("mal_Mlym", "eng_Latn"),
}

DEFAULT_STYLE = {
    "style_src": "formal",
    "style_tgt": "formal",
    "simplify": "no",
}


@dataclass
class Mapping:
    kind: str
    indic: Optional[str] = None
    english: Optional[str] = None
    column: Optional[str] = None
    indic_key: Optional[str] = None
    english_key: Optional[str] = None


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        text = " ".join(str(piece) for piece in text if piece)
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _score_column(name: str, aliases: Sequence[str], extra_hits: Sequence[str] | None = None) -> int:
    lower = name.lower()
    score = 0
    for alias in aliases:
        alias_lower = alias.lower()
        if lower == alias_lower:
            score += 10
        if lower.endswith(f"_{alias_lower}") or lower.startswith(f"{alias_lower}_"):
            score += 8
        if alias_lower in lower:
            score += 6
    if extra_hits:
        for token in extra_hits:
            if token in lower:
                score += 2
    return score


def detect_language_columns(features: Dict[str, Any], lang_code: str, dataset_name: str) -> Mapping:
    for col_name, feature in features.items():
        if TRANSLATION_TYPES and isinstance(feature, TRANSLATION_TYPES):
            languages = getattr(feature, "languages", [])
            english_key = _match_language_key(languages, ["en", "eng", "english"])
            indic_key = _match_language_key(languages, LANG_ALIASES.get(lang_code, []))
            if english_key and indic_key:
                return Mapping(kind="translation", column=col_name, indic_key=indic_key, english_key=english_key)

    columns = list(features.keys())
    english_scores = {name: _score_column(name, ["en", "eng", "english"], extra_hits=["target", "tgt"]) for name in columns}
    indic_scores = {name: _score_column(name, LANG_ALIASES.get(lang_code, []), extra_hits=["source", "src"]) for name in columns}

    english_col = _best_scoring_column(english_scores)
    indic_col = _best_scoring_column(indic_scores)

    if english_col and indic_col and english_col != indic_col:
        return Mapping(kind="columns", indic=indic_col, english=english_col)

    if "src" in columns and "tgt" in columns:
        if "samanantar" in dataset_name:
            return Mapping(kind="columns", indic="tgt", english="src")
        return Mapping(kind="columns", indic="src", english="tgt")

    raise ValueError(f"Unable to auto-detect language columns in {dataset_name} for lang {lang_code}. Columns: {columns}")


def _match_language_key(candidates: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    lowered_aliases = [alias.lower() for alias in aliases]
    for alias in lowered_aliases:
        for candidate in candidates:
            cand_lower = candidate.lower()
            if cand_lower == alias or cand_lower.endswith(alias) or alias.endswith(cand_lower):
                return candidate
            if alias in cand_lower:
                return candidate
    return None


def _best_scoring_column(scores: Dict[str, int]) -> Optional[str]:
    if not scores:
        return None
    best_name = None
    best_score = -1
    for name, score in scores.items():
        if score > best_score:
            best_score = score
            best_name = name
    if best_score <= 0:
        return None
    return best_name


def build_extractor(mapping: Mapping) -> Callable[[Dict[str, Any]], Tuple[str, str]]:
    if mapping.kind == "translation":
        column = mapping.column or "translation"
        indic_key = mapping.indic_key or ""
        english_key = mapping.english_key or ""

        def extractor(row: Dict[str, Any]) -> Tuple[str, str]:
            data = row.get(column, {})
            indic = normalize_text(data.get(indic_key))
            english = normalize_text(data.get(english_key))
            return indic, english

        return extractor

    indic_col = mapping.indic or "src"
    english_col = mapping.english or "tgt"

    def extractor(row: Dict[str, Any]) -> Tuple[str, str]:
        indic = normalize_text(row.get(indic_col))
        english = normalize_text(row.get(english_col))
        return indic, english

    return extractor


def collect_dataset_pairs(
    dataset_name: str,
    dataset: Dict[str, Any],
    pair: str,
) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    lang_code, _ = pair.split("-", 1)
    pairs: List[Tuple[str, str]] = []
    summary: Dict[str, Any] = {"source": dataset_name, "by_split": {}, "total": 0}

    for split_name, split_ds in dataset.items():
        try:
            mapping = detect_language_columns(split_ds.features, lang_code, dataset_name)
        except Exception as exc:  # pragma: no cover - dataset schema fallback
            print(f"[{pair}] Warning: {exc}. Skipping split {split_name} in {dataset_name}.")
            continue
        extractor = build_extractor(mapping)
        total_rows = getattr(split_ds, "num_rows", None)
        split_count = 0
        for indic, english in tqdm(
            (extractor(row) for row in split_ds),
            total=total_rows,
            desc=f"{dataset_name}::{split_name}::{pair}",
            leave=False,
        ):
            pairs.append((indic, english))
            split_count += 1
        summary["by_split"][split_name] = split_count
        summary["total"] += split_count
    return pairs, summary


def load_samanantar(pair: str) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    lang_code, _ = pair.split("-", 1)
    dataset_name = "ai4bharat/samanantar"
    config_candidates = [f"{lang_code}-en", f"{lang_code}_en", lang_code, f"{lang_code}en"]

    dataset = None
    tried: List[str] = []
    for config in config_candidates:
        try:
            dataset = load_dataset(dataset_name, config)
            break
        except Exception:
            tried.append(config)

    if dataset is None:
        try:
            configs = get_dataset_config_names(dataset_name)
        except Exception:
            configs = []
        for config in configs:
            if lang_code not in config:
                continue
            if config in tried:
                continue
            try:
                dataset = load_dataset(dataset_name, config)
                break
            except Exception:
                tried.append(config)

    if dataset is None:
        tried_str = ", ".join(tried) if tried else "none"
        raise RuntimeError(f"Unable to load {dataset_name} for lang {lang_code}. Tried configs: {tried_str}")

    return collect_dataset_pairs(dataset_name, dataset, pair)


def load_iitb() -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    dataset_name = "cfilt/iitb-english-hindi"
    dataset = load_dataset(dataset_name)
    return collect_dataset_pairs(dataset_name, dataset, "hi-en")


def clean_pairs(
    raw_pairs: Sequence[Tuple[str, str]],
    pair: str,
    *,
    enforce_langid: bool,
    random_seed: int,
    max_per_pair: Optional[int],
    lang_identifier: Any,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    stats = Counter()
    cleaned: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    lang_code, _ = pair.split("-", 1)
    expected_lang = LANGID_CODES.get(lang_code)

    for indic, english in raw_pairs:
        stats["loaded"] += 1

        src = normalize_text(indic)
        tgt = normalize_text(english)

        if not src or not tgt:
            stats["dropped_empty"] += 1
            continue

        tokens_src = src.split()
        tokens_tgt = tgt.split()
        if len(tokens_src) < 3 or len(tokens_src) > 256 or len(tokens_tgt) < 3 or len(tokens_tgt) > 256:
            stats["dropped_length"] += 1
            continue

        ratio = len(tokens_tgt) / max(len(tokens_src), 1)
        if ratio < 0.5 or ratio > 2.0:
            stats["dropped_ratio"] += 1
            continue

        if enforce_langid and lang_identifier is not None and expected_lang:
            src_lang = lang_identifier.classify(src)[0]
            tgt_lang = lang_identifier.classify(tgt)[0]
            if src_lang != expected_lang or tgt_lang != "en":
                stats["dropped_langid"] += 1
                continue

        key = (src, tgt)
        if key in seen:
            stats["duplicates"] += 1
            continue
        seen.add(key)

        record = {
            "src": src,
            "tgt": tgt,
            "style_src": DEFAULT_STYLE["style_src"],
            "style_tgt": DEFAULT_STYLE["style_tgt"],
            "simplify": DEFAULT_STYLE["simplify"],
            "pair": pair,
        }
        cleaned.append(record)

    stats["kept"] = len(cleaned)

    if max_per_pair is not None and max_per_pair > 0 and len(cleaned) > max_per_pair:
        rng = random.Random(random_seed)
        indices = rng.sample(range(len(cleaned)), k=max_per_pair)
        indices.sort()
        cleaned = [cleaned[i] for i in indices]
        stats["downsampled_to"] = max_per_pair
    else:
        stats["downsampled_to"] = len(cleaned)

    return cleaned, dict(stats)


def write_tsv_rows(rows: Sequence[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["src", "tgt", "style_src", "style_tgt", "simplify", "pair"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_flores(pairs: Sequence[str]) -> Dict[str, Any]:
    dataset = None
    used_name = None
    for name in ("facebook/flores-200", "facebook/flores"):
        try:
            dataset = load_dataset(name, "all")
            used_name = name
            break
        except Exception:
            continue

    if dataset is None:
        print("Warning: FLORES-200 dataset not available; skipping export.")
        return {"status": "skipped"}

    artifacts_root = Path("artifacts/review2/flores")
    summary: Dict[str, Any] = {"status": "exported", "dataset": used_name, "pairs": {}}

    for pair in pairs:
        if pair not in FLORES_CODES:
            continue
        indic_code, english_code = FLORES_CODES[pair]
        pair_dir = artifacts_root / pair
        pair_dir.mkdir(parents=True, exist_ok=True)
        pair_summary: Dict[str, Any] = {}

        for split_name in ("dev", "devtest"):
            if split_name not in dataset:
                continue
            split_ds = dataset[split_name]
            try:
                mapping = detect_language_columns(
                    split_ds.features,
                    pair.split("-", 1)[0],
                    f"{used_name}::{split_name}",
                )
                extractor = build_extractor(mapping)
                indic_column = _resolve_flores_column(split_ds.features, indic_code)
                english_column = _resolve_flores_column(split_ds.features, english_code)
            except Exception:
                indic_column = _resolve_flores_column(split_ds.features, indic_code)
                english_column = _resolve_flores_column(split_ds.features, english_code)
                extractor = None

            src_path = pair_dir / f"{split_name}.src"
            tgt_path = pair_dir / f"{split_name}.ref"
            count = 0
            with src_path.open("w", encoding="utf-8", newline="") as src_handle, tgt_path.open(
                "w", encoding="utf-8", newline=""
            ) as tgt_handle:
                for row in split_ds:
                    if extractor:
                        indic_text, english_text = extractor(row)
                        if not indic_text or not english_text:
                            continue
                    else:
                        indic_text = normalize_text(row.get(indic_column))
                        english_text = normalize_text(row.get(english_column))
                        if not indic_text or not english_text:
                            continue
                    src_handle.write(indic_text + "\n")
                    tgt_handle.write(english_text + "\n")
                    count += 1
            pair_summary[split_name] = count
        summary["pairs"][pair] = pair_summary
    return summary


def _resolve_flores_column(features: Dict[str, Any], lang_code: str) -> str:
    canonical = [f"sentence_{lang_code}", lang_code]
    available = list(features.keys())
    for candidate in canonical:
        if candidate in features:
            return candidate
    lowered = lang_code.lower()
    for name in available:
        if lowered in name.lower():
            return name
    raise KeyError(f"Could not locate FLORES column for {lang_code}. Available: {available}")


def run(args: argparse.Namespace) -> None:
    pairs = args.pairs
    max_per_pair = args.max_per_pair
    lang_check = args.lang_check.lower() == "yes"
    export_flores_flag = args.export_flores.lower() == "yes"

    if lang_check and langid is None:
        print("Warning: lang_check requested but langid is not installed. Skipping language identification.")
        lang_check = False

    if lang_check and langid is not None:
        langid.set_languages(["en"] + [LANGID_CODES[pair.split("-", 1)[0]] for pair in pairs if pair.split("-", 1)[0] in LANGID_CODES])

    summary: Dict[str, Any] = {}
    data_root = Path("data/raw")
    data_root.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        if pair not in SUPPORTED_PAIRS:
            print(f"Skipping unsupported pair {pair}.")
            continue

        print(f"[{pair}] Starting dataset fetch...")
        samanantar_pairs, samanantar_stats = load_samanantar(pair)
        all_pairs: List[Tuple[str, str]] = list(samanantar_pairs)
        source_breakdown = [samanantar_stats]

        if pair == "hi-en":
            iitb_pairs, iitb_stats = load_iitb()
            all_pairs.extend(iitb_pairs)
            source_breakdown.append(iitb_stats)

        sources_label = ", ".join(f"{src['source']}={src['total']}" for src in source_breakdown)
        print(f"[{pair}] Loaded {len(all_pairs)} rows before cleaning (sources: {sources_label})")

        cleaned_rows, stats = clean_pairs(
            all_pairs,
            pair,
            enforce_langid=lang_check,
            random_seed=args.seed,
            max_per_pair=max_per_pair,
            lang_identifier=langid,
        )
        kept = stats.get("kept", 0)
        written_count = stats.get("downsampled_to", kept)
        print(
            f"[{pair}] After cleaning: kept={kept}, removed_empty={stats.get('dropped_empty', 0)}, "
            f"removed_length={stats.get('dropped_length', 0)}, removed_ratio={stats.get('dropped_ratio', 0)}, "
            f"removed_langid={stats.get('dropped_langid', 0)}, deduped={stats.get('duplicates', 0)}"
        )
        if written_count != kept:
            print(f"[{pair}] Downsampled to {written_count} due to max_per_pair={max_per_pair}.")

        output_path = data_root / f"{pair.replace('-', '_')}.tsv"
        write_tsv_rows(cleaned_rows, output_path)
        print(f"[{pair}] Wrote {written_count} rows to {output_path}.")

        summary[pair] = {
            "sources": source_breakdown,
            "raw_total": len(all_pairs),
            "stats": stats,
            "written": written_count,
        }

    flores_summary = None
    if export_flores_flag:
        flores_summary = export_flores(pairs)
        summary["flores"] = flores_summary

    summary_path = Path("artifacts/review2/fetch_stats.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("\nFetch summary written to", summary_path)
    print("\nChecklist:")
    for pair in pairs:
        if pair in summary:
            written = summary[pair].get("written", 0)
            print(f"- {pair}: wrote {written} cleaned sentence pairs to data/raw/{pair.replace('-', '_')}.tsv")
    if export_flores_flag and flores_summary:
        status = flores_summary.get("status", "unknown")
        print(f"- FLORES export: {status}")

    print("\nNext commands:")
    print("python scripts/fetch_corpora.py --pairs hi-en ta-en te-en ml-en --max_per_pair 300000")
    print("python scripts/data_prep.py --manifest data/manifests/review2.yaml")
    print("python scripts/make_splits.py --pair hi-en --in data/clean/hi_en --out data/clean/hi_en --seed 42")
    print("python scripts/make_splits.py --pair ta-en --in data/clean/ta_en --out data/clean/ta_en --seed 42")
    print("\nNote: switch --manifest to data/manifests/full.yaml when preparing te/ml clean splits later.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and clean Indic-English corpora into TSV format.")
    parser.add_argument("--pairs", nargs="+", required=True, help="Space-separated list of language pairs (e.g., hi-en ta-en).")
    parser.add_argument("--max_per_pair", type=int, default=None, help="Optional maximum pairs per language after cleaning.")
    parser.add_argument("--export_flores", choices=["yes", "no"], default="no", help="Export FLORES-200 dev/devtest files.")
    parser.add_argument("--lang_check", choices=["yes", "no"], default="no", help="Enable langid-based sanity check.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for downsampling.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
