import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from utils_io import ensure_dir, read_tsv, write_tsv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create train/val/test splits for a language pair.")
    parser.add_argument("--pair", required=True, help="Language pair identifier, e.g. hi-en.")
    parser.add_argument("--in", dest="input_dir", required=True, help="Directory containing all.tsv.")
    parser.add_argument("--out", dest="output_dir", required=True, help="Directory to write split TSV files.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val", type=float, default=0.02, help="Validation fraction.")
    parser.add_argument("--test", type=float, default=0.02, help="Test fraction.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    data_path = input_dir / "all.tsv"
    if not data_path.exists():
        raise FileNotFoundError(f"Expected {data_path} to exist.")

    rows = list(read_tsv(data_path))
    if not rows:
        raise ValueError(f"No rows found in {data_path}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    total = len(rows)
    test_count = max(1, int(total * args.test)) if args.test > 0 else 0
    val_count = max(1, int(total * args.val)) if args.val > 0 else 0
    train_count = total - val_count - test_count
    if train_count <= 0:
        raise ValueError("Not enough rows to create train split with the provided fractions.")

    splits = {
        "train": rows[:train_count],
        "val": rows[train_count : train_count + val_count],
        "test": rows[train_count + val_count : train_count + val_count + test_count],
    }

    output_dir = ensure_dir(Path(args.output_dir))
    fieldnames = ["src", "tgt", "style_src", "style_tgt", "simplify"]

    for split_name, split_rows in splits.items():
        if not split_rows:
            continue
        write_tsv(split_rows, output_dir / f"{split_name}.tsv", fieldnames=fieldnames)

    stats = {
        "pair": args.pair,
        "total": total,
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "seed": args.seed,
    }
    with (output_dir / "split_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(f"Split {args.pair}: train={stats['train']} val={stats['val']} test={stats['test']}")


if __name__ == "__main__":
    main()
