import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover - defensive import error
    raise SystemExit("PyYAML is required to parse manifests. Please install pyyaml.") from exc

from utils_io import ensure_dir, read_tsv, write_json, write_tsv
from utils_text import normalize


def _normalize_style(value: str | None) -> str:
    if not value:
        return "formal"
    value = value.strip().lower()
    if value not in {"formal", "informal"}:
        return "formal"
    return value


def _normalize_simplify(value: str | None) -> str:
    if not value:
        return "no"
    value = value.strip().lower()
    return "yes" if value == "yes" else "no"


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest at {path} must be a mapping.")
    return data


def length_filter(text: str, *, minimum: int | None, maximum: int | None) -> bool:
    tokens = text.split()
    if minimum is not None and len(tokens) < minimum:
        return False
    if maximum is not None and len(tokens) > maximum:
        return False
    return True


def process_pair(pair_cfg: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    pair_name = pair_cfg["pair"]
    raw_path = Path(pair_cfg["src"]).expanduser()
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found for {pair_name}: {raw_path}")

    clean_dir = ensure_dir(Path("data/clean") / pair_name.replace("-", "_"))
    dest_path = clean_dir / "all.tsv"
    stats: Dict[str, Any] = {
        "pair": pair_name,
        "raw_rows": 0,
        "kept_rows": 0,
        "dropped_missing": 0,
        "dropped_length": 0,
        "deduped": 0,
        "note": "Synthetic rewrite generation capped at 5% (not run; placeholder).",
    }

    minimum = options.get("min_len")
    maximum = options.get("max_len")
    seen: set[str] = set()
    cleaned_rows: list[Dict[str, str]] = []

    for row in read_tsv(raw_path):
        stats["raw_rows"] += 1
        src = normalize(row.get("src", ""))
        tgt = normalize(row.get("tgt", ""))
        if not src or not tgt:
            stats["dropped_missing"] += 1
            continue

        if not length_filter(src, minimum=minimum, maximum=maximum) or not length_filter(
            tgt, minimum=minimum, maximum=maximum
        ):
            stats["dropped_length"] += 1
            continue

        record_key = hashlib.sha1(f"{src}\t{tgt}".encode("utf-8")).hexdigest()
        if record_key in seen:
            stats["deduped"] += 1
            continue
        seen.add(record_key)

        style_src = _normalize_style(row.get("style_src"))
        style_tgt = _normalize_style(row.get("style_tgt") or style_src)
        simplify = _normalize_simplify(row.get("simplify"))

        cleaned_rows.append(
            {
                "src": src,
                "tgt": tgt,
                "style_src": style_src,
                "style_tgt": style_tgt,
                "simplify": simplify,
            }
        )

    if not cleaned_rows:
        raise ValueError(f"No usable rows produced for {pair_name} from {raw_path}")

    stats["kept_rows"] = len(cleaned_rows)

    write_tsv(cleaned_rows, dest_path, fieldnames=["src", "tgt", "style_src", "style_tgt", "simplify"])
    write_json(stats, clean_dir / "stats.json")

    return stats


def run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    options = manifest.get("options", {})

    all_stats: list[Dict[str, Any]] = []
    for pair_cfg in manifest.get("pairs", []):
        stats = process_pair(pair_cfg, options)
        all_stats.append(stats)
        print(f"Processed {pair_cfg['pair']}: {stats['kept_rows']} rows")

    summary_path = Path("artifacts/review2") / "data_prep_summary.json"
    write_json({"manifest": str(manifest_path), "pairs": all_stats}, summary_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean and normalize IndicTrans manifest datasets.")
    parser.add_argument("--manifest", required=True, help="Path to manifest YAML file.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
