#!/usr/bin/env python3
import csv, sys, re
from pathlib import Path
from collections import defaultdict

def pick(row, keys, default=""):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k].strip()
    return default

def score_tgt(tgt, style, simplify):
    # higher is better
    t = tgt.strip()
    tokens = len(t.split())
    has_pls = "please" in t.lower()
    has_contr = bool(re.search(r"\b\w+'\w+\b", t))
    bad = 0
    if "  " in t or ".." in t or "<" in t or ">" in t: bad -= 2

    s = 0
    if style == "formal":
        if t.lstrip().startswith("Could you please "): s += 5
        if has_pls: s += 1
        if not has_contr: s += 1
    else:  # informal
        if not has_pls: s += 3
        if has_contr: s += 2

    if simplify == "yes":
        s += max(0, 80 - tokens) / 100.0  # prefer shorter
    else:
        s += min(tokens, 80) / 200.0      # mild bias to fuller

    return s + bad, tokens

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/patch_dedupe_contradictions.py <in.tsv> <out.tsv>")
        sys.exit(1)
    inp, outp = sys.argv[1], sys.argv[2]
    Path(outp).parent.mkdir(parents=True, exist_ok=True)
    with open(inp, encoding="utf-8") as fi:
        r = csv.DictReader(fi, delimiter="\t")
        fn = r.fieldnames
        assert "src" in fn and "tgt" in fn, "need src,tgt"
        style_col = "style" if "style" in fn else ("style_tgt" if "style_tgt" in fn else "style_src")
        simpl_col = "simplify" if "simplify" in fn else ("simplify_tgt" if "simplify_tgt" in fn else "simplify_src")
        pair_col = "pair" if "pair" in fn else None

        buckets = defaultdict(list)
        for row in r:
            sty = pick(row, [style_col], "formal").lower()
            simp = pick(row, [simpl_col], "no").lower()
            key = (row["src"], sty, simp)
            buckets[key].append(row)

    kept = 0
    with open(outp, "w", encoding="utf-8", newline="") as fo:
        w = csv.DictWriter(fo, delimiter="\t", fieldnames=fn)
        w.writeheader()
        for key, rows in buckets.items():
            if len(rows) == 1:
                w.writerow(rows[0]); kept += 1
                continue
            # choose best by score; tie-breaker: shorter if simplify=yes else longer
            sty = pick(rows[0], ["style","style_tgt","style_src"], "formal").lower()
            simp = pick(rows[0], ["simplify","simplify_tgt","simplify_src"], "no").lower()
            scored = []
            for row in rows:
                sc, toks = score_tgt(row["tgt"], sty, simp)
                scored.append((sc, toks, row))

            # sort ONLY by numeric fields to avoid comparing dicts
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

            top_score = scored[0][0]
            best = [x for x in scored if x[0] == top_score]

            # secondary tie-breaker on length
            if len(best) > 1:
                if simp == "yes":
                    best.sort(key=lambda x: x[1])    # prefer shorter
                else:
                    best.sort(key=lambda x: -x[1])   # prefer longer

            chosen_row = best[0][2]
            w.writerow(chosen_row); kept += 1
    print(f"Wrote {kept} rows to {outp}")

if __name__ == "__main__":
    main()
