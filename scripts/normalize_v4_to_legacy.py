#!/usr/bin/env python3
import csv, sys, os
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/normalize_v4_to_legacy.py <in.tsv> <out.tsv>")
        sys.exit(1)

    inp, outp = sys.argv[1], sys.argv[2]
    Path(outp).parent.mkdir(parents=True, exist_ok=True)

    with open(inp, encoding="utf-8") as fi, open(outp, "w", encoding="utf-8", newline="") as fo:
        r = csv.DictReader(fi, delimiter="\t")
        need = {"src","tgt","style","simplify"}
        missing = need - set(r.fieldnames or [])
        if missing:
            raise SystemExit(f"Input missing columns: {missing}; got {r.fieldnames}")

        fieldnames = ["src","tgt","style_src","style_tgt","simplify","pair"]
        w = csv.DictWriter(fo, delimiter="\t", fieldnames=fieldnames)
        w.writeheader()

        n = 0
        for row in r:
            style = (row.get("style") or "").strip().lower()
            simplify = (row.get("simplify") or "").strip().lower()
            out = {
                "src": row["src"],
                "tgt": row["tgt"],
                "style_src": style,     # mirror style for compatibility
                "style_tgt": style,
                "simplify": simplify,
                "pair": "hi-en",
            }
            w.writerow(out); n += 1
    print(f"Wrote {n} rows -> {outp}")

if __name__ == "__main__":
    main()
