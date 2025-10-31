#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv, sys, os, re, json, random
from pathlib import Path
from collections import Counter, defaultdict

OUTDIR = Path("artifacts/dataset_checks")

# ---------- schema helpers ----------
def pick(row, keys, default=""):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k].strip()
    return default

def normalize_row(row):
    """Return (src, tgt, style, simplify, pair) from either schema."""
    src = pick(row, ["src","source"])
    tgt = pick(row, ["tgt","target"])

    # Handle both legacy format and special token format
    style_raw = pick(row, ["style","style_tgt","style_src"])
    simplify_raw = pick(row, ["simplify","simplify_tgt","simplify_src"])

    # Normalize special tokens to legacy format for consistency in checks
    style_map = {"<FORMAL>": "formal", "<INFORMAL>": "informal"}
    simplify_map = {"<SIMPL_Y>": "yes", "<SIMPL_N>": "no"}

    style = style_map.get(style_raw, style_raw.lower() if style_raw else "formal")
    simplify = simplify_map.get(simplify_raw, simplify_raw.lower() if simplify_raw else "no")

    pair = pick(row, ["pair"], "hi-en")
    return src, tgt, style, simplify, pair

# ---------- language / script ----------
DEV_RANGE = (0x0900, 0x097F)  # Devanagari
def frac_deva(s):
    total = sum(ch.isalpha() for ch in s)
    if total == 0: return 0.0
    deva = sum(1 for ch in s if ch.isalpha() and DEV_RANGE[0] <= ord(ch) <= DEV_RANGE[1])
    return deva / total

def frac_latin(s):
    total = sum(ch.isalpha() for ch in s)
    if total == 0: return 0.0
    latin = sum(1 for ch in s if ch.isalpha() and (u'A' <= ch <= u'Z' or u'a' <= ch <= u'z'))
    return latin / total

# ---------- heuristics ----------
CONTRACTION_RX = re.compile(r"\b\w+'\w+\b")
def has_contraction(s): return bool(CONTRACTION_RX.search(s))
def starts_with_please(s): return s.lstrip().lower().startswith("please ")
def starts_with_could_you_please(s): return s.lstrip().startswith("Could you please ")
REQUEST_RX = re.compile(r"^(can|could|will|would)\s+you\b", re.I)
VERB_SET = {"close","open","check","tell","give","send","provide","share","explain",
            "call","confirm","submit","review","attach","reply","respond","turn",
            "start","stop","report","pay","bring","take","show","help","deliver"}
def looks_like_request_en(en: str) -> bool:
    s = en.strip()
    if s.endswith("?"): return True
    if REQUEST_RX.search(s): return True
    m = re.match(r"^[Pp]lease\s+([A-Za-z']+)", s)
    if m and m.group(1).lower() in VERB_SET: return True
    m2 = re.match(r"^\s*([A-Za-z']+)\b", s)
    return bool(m2 and m2.group(1).lower() in VERB_SET)

def is_entity_list(s: str) -> bool:
    caps = sum(1 for w in re.findall(r"\b[A-Z][a-z]+\b", s))
    return s.count(",") >= 4 or caps >= 8

def is_short_or_numeric(s: str) -> bool:
    txt = re.sub(r"[^A-Za-z]+", " ", s).strip()
    return len(txt.split()) < 4

# ---------- main checks ----------
def run_checks(path_train, path_val=None, path_test=None, max_src_len=256, max_tgt_len=192):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    report = {"train": {}, "val": {}, "test": {}}

    def load(path):
        with open(path, encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter="\t")
            rows = []
            for row in r:
                src, tgt, style, simplify, pair = normalize_row(row)
                if not src or not tgt: continue
                rows.append((src, tgt, style, simplify, pair))
            return rows

    train = load(path_train)

    # 1) Schema & integrity
    total = len(train)
    nulls = sum(1 for src,tgt,_,_,_ in train if not src or not tgt)
    pairs_ok = sum(1 for *_, pair in train if pair == "hi-en")
    report["train"]["rows"] = total
    report["train"]["null_rows"] = nulls
    report["train"]["pair_hi_en_ratio"] = round(pairs_ok/total, 4) if total else 0.0

    # 2) Distribution of controls
    c_style = Counter([s for _,_,s,_,_ in train])
    c_simpl = Counter([y for _,_,_,y,_ in train])
    c_combo = Counter([(s, y) for _,_,s,y,_ in train])

    # cast to plain dicts; flatten tuple keys for JSON
    report["train"]["style_counts"] = dict(c_style)
    report["train"]["simplify_counts"] = dict(c_simpl)
    report["train"]["combo_counts"] = {f"{s}|{y}": n for (s, y), n in c_combo.items()}

    # 3) Language/script sanity
    src_deva_hi = sum(1 for src,_,_,_,_ in train if frac_deva(src) >= 0.5)
    tgt_latin_hi = sum(1 for _,tgt,_,_,_ in train if frac_latin(tgt) >= 0.5)
    report["train"]["src_mostly_devanagari_ratio"] = round(src_deva_hi/total, 4) if total else 0.0
    report["train"]["tgt_mostly_latin_ratio"] = round(tgt_latin_hi/total, 4) if total else 0.0

    # 4) Lengths & trunc risk
    def toks(x): return len(x.split())
    src_over = sum(1 for src,_,_,_,_ in train if toks(src) > max_src_len)
    tgt_over = sum(1 for _,tgt,_,_,_ in train if toks(tgt) > max_tgt_len)
    report["train"]["src_over_max"] = src_over
    report["train"]["tgt_over_max"] = tgt_over

    # 5) Artifacts/contamination
    starts_please = sum(1 for _,tgt,_,_,_ in train if starts_with_please(tgt))
    cup = sum(1 for _,tgt,_,_,_ in train if starts_with_could_you_please(tgt))
    doublespace = sum(1 for _,tgt,_,_,_ in train if "  " in tgt)
    ellipses = sum(1 for _,tgt,_,_,_ in train if ".." in tgt)
    html_like = sum(1 for _,tgt,_,_,_ in train if "<" in tgt or ">" in tgt)
    report["train"]["starts_with_Please_ratio"] = round(starts_please/total, 4) if total else 0.0
    report["train"]["starts_with_CouldYouPlease_ratio"] = round(cup/total, 4) if total else 0.0
    report["train"]["double_space_ratio"] = round(doublespace/total, 4) if total else 0.0
    report["train"]["ellipsis_ratio"] = round(ellipses/total, 4) if total else 0.0
    report["train"]["html_like_ratio"] = round(html_like/total, 4) if total else 0.0

    # 6) Duplicates & contradictions
    seen = set()
    dup_exact = 0
    map_src_combo_to_tgts = defaultdict(set)
    for src,tgt,s,y,_ in train:
        k = (src,tgt,s,y)
        if k in seen: dup_exact += 1
        seen.add(k)
        map_src_combo_to_tgts[(src,s,y)].add(tgt)
    contradictions = sum(1 for tgts in map_src_combo_to_tgts.values() if len(tgts) > 1)
    report["train"]["exact_duplicate_rows"] = dup_exact
    report["train"]["contradictions_same_src_same_controls"] = contradictions

    # 7) Control effect diagnostics (intradata)
    # Identify request-like *targets* (English), not Hindi sources
    req_rows = [(src, tgt, s, y) for src, tgt, s, y, _ in train if looks_like_request_en(tgt)]

    # counts
    inf_contr = sum(1 for _, tgt, s, _ in req_rows if s == "informal" and has_contraction(tgt))
    frm_no_contr = sum(1 for _, tgt, s, _ in req_rows if s == "formal" and not has_contraction(tgt))

    # denominators
    denom_inf = sum(1 for _, _, s, _ in req_rows if s == "informal")
    denom_frm = sum(1 for _, _, s, _ in req_rows if s == "formal")

    # ratios (original metrics)
    report["train"]["informal_req_with_contraction_ratio"] = round((inf_contr / denom_inf) if denom_inf else 0.0, 4)
    report["train"]["formal_req_without_contraction_ratio"] = round((frm_no_contr / denom_frm) if denom_frm else 0.0, 4)

    # additional informal signal: absence of "please"
    informal_no_please = sum(1 for _, tgt, s, _ in req_rows if s == "informal" and "please" not in tgt.lower())
    report["train"]["informal_req_without_please_ratio"] = round((informal_no_please / denom_inf) if denom_inf else 0.0, 4)

    # simplify effect: compare lengths for same src & style between yes vs no
    buckets = defaultdict(lambda: {"yes": [], "no": []})
    for src,tgt,s,y,_ in train:
        buckets[(src,s)][y].append(tgt)
    diffs = []
    for (src,s), d in buckets.items():
        if d["yes"] and d["no"]:
            len_yes = sum(len(t.split()) for t in d["yes"])/len(d["yes"])
            len_no  = sum(len(t.split()) for t in d["no"])/len(d["no"])
            if len_no > 0:
                diffs.append((len_yes, len_no))
    avg_ratio = (sum(l_yes / l_no for l_yes, l_no in diffs) / len(diffs)) if diffs else 1.0
    report["train"]["simplify_yes_vs_no_avg_len_ratio"] = round(avg_ratio, 4)

    # 8) Entity/list protection (count of entity-list lines marked simplify=yes)
    list_lines_marked_simplify = sum(1 for _,tgt,_,y,_ in train if is_entity_list(tgt) and y=="yes")
    report["train"]["simplify_marked_entity_lists_count"] = list_lines_marked_simplify

    # 9) Overlap checks with val/test (src overlap)
    def overlap(a, b):
        A = set(x[0] for x in a); B = set(x[0] for x in b)
        return round(len(A & B) / max(1, len(A)), 4)
    if path_val and os.path.exists(path_val):
        val = load(path_val); report["val"]["rows"] = len(val)
        report["train"]["src_overlap_with_val_ratio"] = overlap(train, val)
    if path_test and os.path.exists(path_test):
        test = load(path_test); report["test"]["rows"] = len(test)
        report["train"]["src_overlap_with_test_ratio"] = overlap(train, test)

    # 10) Write samples (UNIFIED to 4-tuples)
    random.seed(13)
    samples = []
    # 3 per combo where available
    for combo in [("formal","no"),("formal","yes"),("informal","no"),("informal","yes")]:
        cand = [(s,t,sty,smpl) for s,t,sty,smpl,_ in train if (sty,smpl)==combo]
        for s4 in random.sample(cand, min(3, len(cand))):
            samples.append(s4)
    # edge samples: entity list, request-like (EN), short numeric
    el = [x for x in train if is_entity_list(x[1])]
    rq = [x for x in train if looks_like_request_en(x[1])]
    sn = [x for x in train if is_short_or_numeric(x[1])]
    for bucket in (el, rq, sn):
        for x in random.sample(bucket, min(3, len(bucket))):
            samples.append((x[0], x[1], x[2], x[3]))  # drop pair; keep 4-tuple

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(OUTDIR/"train_samples.tsv", "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow(["src","tgt","style","simplify"])
        for src,tgt,sty,smpl in samples:
            w.writerow([src,tgt,sty,smpl])

    with open(OUTDIR/"train_report.json", "w", encoding="utf-8") as fo:
        json.dump(report, fo, indent=2, ensure_ascii=False)

    # Console summary with opinions
    print(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n=== Heuristic acceptance gates (what I look for) ===")
    gates = []
    total_rows = report["train"]["rows"] or 1
    gates.append(("pair_hi_en_ratio ≥ 0.99", report["train"]["pair_hi_en_ratio"] >= 0.99))
    gates.append(("src_mostly_devanagari ≥ 0.98", report["train"]["src_mostly_devanagari_ratio"] >= 0.98))
    gates.append(("tgt_mostly_latin ≥ 0.98", report["train"]["tgt_mostly_latin_ratio"] >= 0.98))
    gates.append(("style roughly balanced (each ≥ 35%)",
                  all(v/total_rows >= 0.35 for v in c_style.values()) if len(c_style)>=2 else True))
    gates.append(("simplify roughly balanced (each ≥ 35%)",
                  all(v/total_rows >= 0.35 for v in c_simpl.values()) if len(c_simpl)>=2 else True))
    gates.append(("starts_with_Please ≤ 1%", report["train"]["starts_with_Please_ratio"] <= 0.01))
    gates.append(("double_space ≤ 5%", report["train"]["double_space_ratio"] <= 0.05))
    gates.append(("html_like ≤ 0.5%", report["train"]["html_like_ratio"] <= 0.005))
    gates.append(("exact_duplicate_rows == 0", report["train"]["exact_duplicate_rows"] == 0))
    gates.append(("contradictions_same_src_same_controls == 0",
                  report["train"]["contradictions_same_src_same_controls"] == 0))
    gates.append(("simplify_yes_vs_no_avg_len_ratio ≤ 0.92", report["train"]["simplify_yes_vs_no_avg_len_ratio"] <= 0.92))
    # Our informal signal is mainly "no please" (direct tone), not contractions.
    gates.append(("informal_req_with_contraction ≥ 0.08",
                report["train"]["informal_req_with_contraction_ratio"] >= 0.08))
    gates.append(("informal_req_without_please ≥ 0.85",
                report["train"]["informal_req_without_please_ratio"] >= 0.85))
    gates.append(("formal_req_without_contraction ≥ 0.70",
                  report["train"]["formal_req_without_contraction_ratio"] >= 0.70))

    bad = [name for name,ok in gates if not ok]
    if not bad:
        print("✅ All gates passed. I’d proceed to training.")
    else:
        print("⚠️  Gates not met:", "; ".join(bad))
        print(f"Details + samples: {OUTDIR}/train_report.json , {OUTDIR}/train_samples.tsv")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/dataset_sanity_suite.py <train.tsv> [val.tsv] [test.tsv]")
        sys.exit(1)
    args = sys.argv[1:]
    run_checks(*args)
