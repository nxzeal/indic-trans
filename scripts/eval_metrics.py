import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Sequence

import sacrebleu
from textstat import textstat

from utils_io import ensure_dir, read_tsv, write_json, write_tsv

try:
    import yaml
except ImportError as exc:  # pragma: no cover - defensive import error
    raise SystemExit("PyYAML is required to parse configs. Please install pyyaml.") from exc


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Eval config at {path} must be a mapping.")
    return cfg


def get_ngrams(tokens: List[str], n: int) -> set[str]:
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def f1(precision: float, recall: float) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def sari_sentence(src: str, pred: str, refs: Sequence[str]) -> float:
    add_scores: List[float] = []
    keep_scores: List[float] = []
    del_scores: List[float] = []

    src_tokens = src.split()
    pred_tokens = pred.split()
    refs_tokens = [ref.split() for ref in refs]

    for n in range(1, 5):
        src_ngrams = get_ngrams(src_tokens, n)
        pred_ngrams = get_ngrams(pred_tokens, n)
        ref_sets = [get_ngrams(ref_tok, n) for ref_tok in refs_tokens]
        ref_union = set().union(*ref_sets) if ref_sets else set()

        ref_add = ref_union - src_ngrams
        ref_keep = ref_union & src_ngrams
        ref_del = src_ngrams - ref_union

        added = pred_ngrams - src_ngrams
        kept = pred_ngrams & src_ngrams
        deleted = src_ngrams - pred_ngrams

        add_precision = len(added & ref_add) / len(added) if added else 1.0
        add_recall = len(added & ref_add) / len(ref_add) if ref_add else 1.0
        add_scores.append(f1(add_precision, add_recall))

        keep_precision = len(kept & ref_keep) / len(kept) if kept else 1.0
        keep_recall = len(kept & ref_keep) / len(ref_keep) if ref_keep else 1.0
        keep_scores.append(f1(keep_precision, keep_recall))

        del_precision = len(deleted & ref_del) / len(deleted) if deleted else 1.0
        del_recall = len(deleted & ref_del) / len(ref_del) if ref_del else 1.0
        del_scores.append(f1(del_precision, del_recall))

    return (mean(add_scores) + mean(keep_scores) + mean(del_scores)) / 3.0


def sari_corpus(sources: List[str], predictions: List[str], references: List[Sequence[str]]) -> float:
    scores = [
        sari_sentence(src, pred, refs)
        for src, pred, refs in zip(sources, predictions, references)
    ]
    return mean(scores) * 100.0


def compute_metrics(
    pair: str,
    predictions: List[str],
    sources: List[str],
    references: List[Sequence[str]],
    config_path: Path,
) -> dict:
    if not predictions:
        raise ValueError("No predictions provided for evaluation.")

    refs_for_bleu = [list(ref_group) for ref_group in zip(*references)]
    bleu = sacrebleu.corpus_bleu(predictions, refs_for_bleu).score
    sari = sari_corpus(sources, predictions, references)
    fkgl_scores = [textstat.flesch_kincaid_grade(pred) for pred in predictions if pred.strip()]
    fkgl = mean(fkgl_scores) if fkgl_scores else 0.0

    return {
        "pair": pair,
        "bleu": bleu,
        "sari": sari,
        "fkgl": fkgl,
        "samples": len(predictions),
        "config": str(config_path),
    }


def write_table(metrics: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["metric", "value"])
        writer.writerow(["pair", metrics["pair"]])
        writer.writerow(["BLEU", f"{metrics['bleu']:.2f}"])
        writer.writerow(["SARI", f"{metrics['sari']:.2f}"])
        writer.writerow(["FKGL", f"{metrics['fkgl']:.2f}"])
        writer.writerow(["samples", metrics["samples"]])


def write_examples(
    sources: List[str], predictions: List[str], references: List[Sequence[str]], path: Path
) -> None:
    rows = []
    for src, pred, refs in list(zip(sources, predictions, references))[:10]:
        rows.append({
            "src": src,
            "hyp": pred,
            "ref": refs[0] if refs else "",
            "notes": "",
        })
    write_tsv(rows, path, fieldnames=["src", "hyp", "ref", "notes"])


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute BLEU/SARI/FKGL metrics for model outputs.")
    parser.add_argument("--config", required=True, help="Path to eval config YAML.")
    parser.add_argument("--pair", required=True, help="Language pair identifier, e.g. hi-en.")
    parser.add_argument("--refs", required=True, help="Reference TSV file (expects src/tgt columns).")
    parser.add_argument("--hyps", required=True, help="Hypothesis text file (one per line).")
    parser.add_argument("--out", required=True, help="Output JSON path for aggregated metrics.")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    eval_cfg = load_config(config_path)
    if args.pair not in eval_cfg.get("pairs", []):
        print(f"Warning: pair {args.pair} not listed in {config_path}.")

    refs_rows = list(read_tsv(args.refs))
    predictions = [line.strip() for line in Path(args.hyps).read_text(encoding="utf-8").splitlines()]
    if len(predictions) != len(refs_rows):
        raise ValueError(
            f"Predictions count ({len(predictions)}) does not match references ({len(refs_rows)})."
        )

    sources = [row["src"] for row in refs_rows]
    references = [[row["tgt"]] for row in refs_rows]

    metrics = compute_metrics(args.pair, predictions, sources, references, config_path)

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    write_json(metrics, out_path)

    table_path = out_path.with_suffix(".tsv")
    write_table(metrics, table_path)

    examples_path = out_path.parent / f"examples_{args.pair.replace('-', '_')}.tsv"
    write_examples(sources, predictions, references, examples_path)

    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
