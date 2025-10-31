#!/usr/bin/env python3
import csv, sys, re, os, argparse, random
from pathlib import Path

# ---------- detection helpers ----------
def detect_cols(fieldnames):
    # try common names; fallback to first two
    lfn = [c.lower() for c in fieldnames]
    def pick(names):
        for n in names:
            if n in lfn:
                return fieldnames[lfn.index(n)]
        return None
    src = pick(["src","source","text_en","en","english"])
    tgt = pick(["tgt","target","text_hi","hi","hindi"])
    if not src or not tgt:
        # fallback: assume first two columns
        src, tgt = fieldnames[0], fieldnames[1]
    return src, tgt

def is_entity_list(s: str) -> bool:
    caps = sum(1 for w in re.findall(r"\b[A-Z][a-z]+\b", s))
    return s.count(",") >= 4 or caps >= 8

def is_short_or_numeric(s: str) -> bool:
    txt = re.sub(r"[^A-Za-z]+", " ", s).strip()
    return len(txt.split()) < 4

# ---------- lexicons ----------
CONTRACTIONS = {
    "cannot":"can't","do not":"don't","does not":"doesn't","did not":"didn't",
    "is not":"isn't","was not":"wasn't","were not":"weren't",
    "will not":"won't","would not":"wouldn't","should not":"shouldn't",
    "could not":"couldn't","has not":"hasn't","have not":"haven't","had not":"hadn't",
    "it is":"it's","that is":"that's","there is":"there's","there are":"there're",
    "i am":"i'm","you are":"you're","we are":"we're","they are":"they're",
    "i will":"i'll","you will":"you'll","we will":"we'll","they will":"they'll",
    "i would":"i'd","you would":"you'd","we would":"we'd","they would":"they'd",
    "can not":"cannot",
}
EXPAND = {v:k for k,v in CONTRACTIONS.items()}

EASY_REPL = {
    "utilize":"use","approximately":"about","purchase":"buy","request":"ask",
    "assistance":"help","commence":"start","terminate":"end","endeavor":"try",
    "inform":"tell","obtain":"get","require":"need","inquire":"ask",
    "therefore":"so","however":"but","consequently":"so","nevertheless":"but",
    "assure":"promise","ensure":"make sure","indicate":"show","regarding":"about",
}

VERB_SET = {
    "close","open","check","tell","give","send","provide","share","explain",
    "call","confirm","submit","review","attach","reply","respond","turn",
    "start","stop","report","pay","bring","take","show","help","deliver"
}

# ---------- tiny transforms ----------
def replace_wordwise(text, mapping):
    def repl(m):
        w = m.group(0); lw = w.lower()
        if lw in mapping:
            rep = mapping[lw]
            if w and w[0].isupper(): rep = rep[0].upper() + rep[1:]
            return rep
        return w
    return re.sub(r"[A-Za-z']+", repl, text)

def expand_contractions(s):
    out = s
    for c, full in EXPAND.items():
        out = re.sub(rf"\b{re.escape(c)}\b", full, out, flags=re.I)
    return out

def make_contractions(s):
    out = s
    for full, c in CONTRACTIONS.items():
        out = re.sub(rf"\b{re.escape(full)}\b", c, out, flags=re.I)
    return out

# ---------- artifact clean ----------
def strip_spurious_please(en: str) -> str:
    s = en.lstrip()
    if not s.lower().startswith("please "): return en
    if s.endswith("?"): return en
    tokens = re.findall(r"[A-Za-z']+", s)
    if len(tokens) < 2: return en
    second = tokens[1].lower()
    early = s[:48].lower()
    if (second not in VERB_SET) and (" you " not in early):
        return re.sub(r"^[Pp]lease\s+", "", s, count=1)
    return en

# ---------- request detection ----------
def looks_like_request_or_question(en: str) -> bool:
    s = en.strip()
    if s.endswith("?"): return True
    if re.match(r"^(can|could|will|would)\s+you\b", s, re.I): return True
    m = re.match(r"^[Pp]lease\s+([A-Za-z']+)", s)
    if m and m.group(1).lower() in VERB_SET: return True
    m2 = re.match(r"^\s*([A-Za-z']+)\b", s)
    if m2 and m2.group(1).lower() in VERB_SET: return True
    return False

# ---------- style transforms ----------
def formalize(en: str) -> str:
    s = strip_spurious_please(en)
    s = expand_contractions(s).strip()
    if looks_like_request_or_question(s):
        s = re.sub(r"^(can|could|will|would)\s+you\s+", "Could you please ", s, flags=re.I)
        if not s.lower().startswith("could you please"):
            s = re.sub(r"^[Pp]lease\s+", "", s, count=1)
            s = "Could you please " + s
        s = re.sub(r"\s{2,}", " ", s).strip()
        if not s.endswith((".", "?", "!")): s += "."
    else:
        s = re.sub(r"\bI (can't|cannot|won't)\b", "I am afraid I cannot", s, flags=re.I)
    return s

def informalize(en: str) -> str:
    s = strip_spurious_please(en)
    s = make_contractions(s)
    s = replace_wordwise(s, EASY_REPL)
    if looks_like_request_or_question(s):
        s = re.sub(r"^[Pp]lease\s+", "", s, count=1)
        s = re.sub(r"\bCould you please\s+", "Can you ", s, flags=re.I)
    s = re.sub(r"\bI (am afraid I cannot|cannot)\b", "I can't", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

# ---------- simplify ----------
SIMPLIFY_PHRASES = {
    "in order to":"to","in the event that":"if","as a result of":"because",
}
def simplify_yes(en: str) -> str:
    if is_entity_list(en) or is_short_or_numeric(en):
        return en
    s = en
    s = re.sub(r"\s*\([^)]*\)", "", s)
    s = replace_wordwise(s, EASY_REPL)
    def repl_phrase(m): return SIMPLIFY_PHRASES[m.group(0).lower()]
    s = re.sub(r"\b(in order to|in the event that|as a result of)\b", repl_phrase, s, flags=re.I)
    if len(s.split()) >= 14:
        parts = re.split(r"[;:—–]|, and |, but ", s)
        if parts: s = parts[0]
    s = re.sub(r"\s{2,}", " ", s).strip()
    if s and s[-1] not in ".!?": s += "."
    return s

def apply_controls(base_en: str, style: str, simplify: str) -> str:
    out = base_en
    if style == "formal":
        out = formalize(out)
    elif style == "informal":
        out = informalize(out)
    if simplify == "yes":
        out = simplify_yes(out)
    return out

# ---------- build ----------
def main():
    ap = argparse.ArgumentParser("Build clean hi→en train with explicit controls from EN→HI all.tsv")
    ap.add_argument("--inp", required=True, help="Path to all.tsv (EN→HI)")
    ap.add_argument("--out", required=True, help="Path to write hi_en/train.v4.tsv")
    ap.add_argument("--max_src_len", type=int, default=256)
    ap.add_argument("--max_tgt_len", type=int, default=192)
    ap.add_argument("--combos", choices=["all4","fino"], default="all4",
                    help="all4 = 4 variants; fino = (formal,no) + (informal,yes)")
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.inp, encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        src_col, tgt_col = detect_cols(r.fieldnames)

        rows = []
        seen = set()
        for row in r:
            en = (row[src_col] or "").strip()   # EN (will become tgt)
            hi = (row[tgt_col] or "").strip()   # HI (will become src)
            if not en or not hi: continue

            # length clamps (rough)
            if len(hi.split()) > args.max_src_len: continue
            if len(en.split()) > 512: continue  # before edits

            # base English cleaned
            base_en = strip_spurious_please(en)

            # make variants
            pairs = [("formal","no"), ("formal","yes"), ("informal","no"), ("informal","yes")] \
                    if args.combos == "all4" else [("formal","no"), ("informal","yes")]

            for sty, simp in pairs:
                tgt = apply_controls(base_en, sty, simp)
                # final length check
                if len(tgt.split()) > args.max_tgt_len: 
                    tgt = " ".join(tgt.split()[:args.max_tgt_len])
                    if tgt and tgt[-1] not in ".!?": tgt += "."

                key = (hi, tgt, sty, simp)
                if key in seen: continue
                seen.add(key)
                rows.append({"src": hi, "tgt": tgt, "style": sty, "simplify": simp})

    if args.shuffle:
        random.shuffle(rows)

    with open(args.out, "w", encoding="utf-8", newline="") as fo:
        w = csv.DictWriter(fo, delimiter="\t", fieldnames=["src","tgt","style","simplify"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
