from pathlib import Path
import json
import csv


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_tsv(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            yield row


def write_tsv(rows, path: str | Path, fieldnames):
    path = Path(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for record in rows:
            writer.writerow(record)


def write_json(obj, path: str | Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
