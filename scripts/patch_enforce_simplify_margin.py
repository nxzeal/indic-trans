#!/usr/bin/env python3
import csv, sys, re
from pathlib import Path
from collections import defaultdict

EASY_REPL = {
    "approximately":"about","purchase":"buy","assistance":"help","commence":"start",
    "terminate":"end","endeavor":"try","inform":"tell","obtain":"get","require":"need",
    "inquire":"ask","therefore":"so","however":"but","consequently":"so","nevertheless":"but",
    "assure":"promise","ensure":"make sure","indicate":"show","regarding":"about",
}
def replace_wordwise(text, mapping):
    def repl(m):
        w = m.group(0); lw = w.lower()
        rep = mapping.get(lw)
        if not rep: return w
        if w and w[0].isupper(): rep = rep[0].upper() + rep[1:]
        return rep
    return re.sub(r"[A-Za-z']+", repl, text)

def extra_simplify(s):
    t = s
    # drop parentheticals
    t = re.sub(r"\s*\([^)]*\)", "", t)
    # cut relative clauses (that/which/who ... up to next comma or period)
    t = re.sub(r"\b(that|which|who|whom|whose)\b[^.,;:]*", "", t, flags=re.I)
    # cut after first main clause if long
    if len(t.split()) >= 12:
        t = re.split(r"[;:—–]|, and |, but ", t)[0]
    # simplify vocab and whitespace
    t = replace_wordwise(t, EASY_REPL)
    t = re.sub(r"\s{2,}", " ", t).strip()
    if t and t[-1] not in ".!?": t += "."
    return t

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/patch_enforce_simplify_margin.py <in.tsv> <out.tsv> [target_ratio]")
        sys.exit(1)
    inp, outp = sys.argv[1], sys.argv[2]
    target_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9

    with open(inp, encoding="utf-8") as fi:
        r = csv.DictReader(fi, delimiter="\t")
        fn = r.fieldnames
        style_col = "style" if "style" in fn else ("style_tgt" if "style_tgt" in fn else "style_src")
        simpl_col = "simplify" if "simplify" in fn else ("simplify_tgt" if "simplify_tgt" in fn else "simplify_src")

        by_src_style = defaultdict(dict)  # (src,style) -> {"no":[t1..], "yes":[t2..]}
        rows = []
        for row in r:
            sty = (row.get(style_col) or "").strip().lower()
            simp = (row.get(simpl_col) or "").strip().lower()
            by_src_style[(row["src"], sty)].setdefault(simp, []).append(row["tgt"])
            rows.append(row)

    # build reference lengths for "no"
    ref_len = {}
    for (src,sty), d in by_src_style.items():
        if "no" in d and d["no"]:
            ref_len[(src,sty)] = sum(len(t.split()) for t in d["no"]) / len(d["no"])

    fixed = 0
    with open(outp, "w", encoding="utf-8", newline="") as fo:
        w = csv.DictWriter(fo, delimiter="\t", fieldnames=fn)
        w.writeheader()
        for row in rows:
            sty = (row.get(style_col) or "").strip().lower()
            simp = (row.get(simpl_col) or "").strip().lower()
            tgt = row["tgt"]
            if simp == "yes" and (row["src"], sty) in ref_len:
                ref = ref_len[(row["src"], sty)]
                if ref > 0 and len(tgt.split()) > target_ratio * ref:
                    # try to simplify further
                    new_tgt = extra_simplify(tgt)
                    if len(new_tgt.split()) > target_ratio * ref:
                        # last resort: truncate to target length
                        keep = max(1, int(target_ratio * ref))
                        toks = new_tgt.split()
                        if len(toks) > keep:
                            new_tgt = " ".join(toks[:keep])
                            if new_tgt and new_tgt[-1] not in ".!?": new_tgt += "."
                    if new_tgt != tgt:
                        row["tgt"] = new_tgt
                        fixed += 1
            w.writerow(row)
    print(f"Wrote {outp} (strengthened simplify for {fixed} rows, target_ratio={target_ratio})")

if __name__ == "__main__":
    main()

