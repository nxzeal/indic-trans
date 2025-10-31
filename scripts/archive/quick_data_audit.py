# scripts/quick_data_audit.py
import csv, sys, collections

if len(sys.argv) < 2:
    print("Usage: python scripts/quick_data_audit.py <tsv>")
    sys.exit(1)

path = sys.argv[1]
# Try to be robust to column names
STYLE_KEYS = ["style_tgt", "style", "style_src"]
SIMPL_KEYS  = ["simplify", "simplify_tgt", "simplify_src"]

def pick(row, keys, default=""):
    for k in keys:
        if k in row: return row[k].strip()
    return default

by_src = collections.defaultdict(list)
with open(path, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f, delimiter="\t")
    assert "src" in r.fieldnames and "tgt" in r.fieldnames, f"Missing src/tgt in {r.fieldnames}"
    for row in r:
        sty = pick(row, STYLE_KEYS)
        simp = pick(row, SIMPL_KEYS)
        by_src[row["src"].strip()].append((sty, simp, row["tgt"].strip()))

collisions = 0
total = 0
for k, lst in by_src.items():
    total += 1
    uniq_tgts = {t for (_,_,t) in lst}
    uniq_ctrl = {(s,y) for (s,y,_) in lst}
    if len(uniq_ctrl) > 1 and len(uniq_tgts) == 1:
        collisions += 1

print(f"unique_src: {total}")
print(f"same_tgt_across_multiple_controls: {collisions}")
print(f"ratio: {collisions/total:.3f}")
