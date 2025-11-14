import os, warnings, argparse, torch
from pathlib import Path
from transformers.utils import logging as hf_logging
from peft import PeftModel
import re


# quiet
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.pop("TRANSFORMERS_CACHE", None)
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from IndicTransToolkit.processor import IndicProcessor

IT2_TAG = {"en":"eng_Latn","hi":"hin_Deva","ta":"tam_Taml","te":"tel_Telu","ml":"mal_Mlym"}

def parse_args():
    p = argparse.ArgumentParser("IndicTrans2 + LoRA (with IndicProcessor + controls)")
    p.add_argument("--base", default="models/indictrans2-indic-en-1B")
    p.add_argument("--adapter", required=True, help="Path to LoRA checkpoint dir (has adapter_model.safetensors).")
    p.add_argument("--src_lang", "--src", dest="src_lang", default="hi")
    p.add_argument("--tgt_lang", "--tgt", dest="tgt_lang", default="en")
    p.add_argument("--text", action="append", required=False,
                   help="Repeatable. If omitted, provide --file.")
    p.add_argument("--file", help="Read UTF-8 lines from this file.")
    p.add_argument("--style", choices=["formal","informal"], default="formal")
    p.add_argument("--simplify", choices=["yes","no"], default="no")
    p.add_argument("--token_format", choices=["legacy","special"], default="special",
        help="Control token format: 'legacy' uses text (formal/informal/yes/no), "
             "'special' uses tokens (<FORMAL>/<INFORMAL>/<SIMPL_Y>/<SIMPL_N>)")
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--use_cache", choices=["on","off"], default="off")
    p.add_argument("--quant", choices=["off","auto"], default="off")
    p.add_argument("--max_new_tokens", type=int, default=128)
    #enforce controls after generation
    p.add_argument("--enforce_controls", choices=["on","off"], default="on",
        help="Post-edit output to reflect style/simplify for demo. Set off for pure model output.")

    return p.parse_args()

def resolve_tag(code: str) -> str:
    return IT2_TAG.get(code.lower(), code)

def load_model(base: str, adapter: str, quant: str, device: str):
    if quant == "auto":
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes as _bnb  # noqa
            if device == "cuda":
                q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    base, trust_remote_code=True, quantization_config=q, device_map="auto"
                )
            else:
                base_model = AutoModelForSeq2SeqLM.from_pretrained(base, trust_remote_code=True)
        except Exception:
            dtype = torch.float16 if device == "cuda" else torch.float32
            base_model = AutoModelForSeq2SeqLM.from_pretrained(base, trust_remote_code=True, dtype=dtype).to(device)
    else:
        dtype = torch.float16 if device == "cuda" else torch.float32
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base, trust_remote_code=True, dtype=dtype).to(device)

    model = PeftModel.from_pretrained(base_model, adapter, is_trainable=False)
    bm = getattr(model, "base_model", model)
    if hasattr(bm, "tie_weights"):
        try: bm.tie_weights()
        except Exception: pass
    return model

_CONTRACTIONS = {
    "do not":"don't","does not":"doesn't","did not":"didn't","is not":"isn't","are not":"aren't",
    "was not":"wasn't","were not":"weren't","cannot":"can't","can not":"can't","will not":"won't",
    "would not":"wouldn't","should not":"shouldn't","could not":"couldn't","have not":"haven't",
    "has not":"hasn't","had not":"hadn't","it is":"it's","that is":"that's","there is":"there's",
    "there are":"there're","i am":"i'm","we will":"we'll","you will":"you'll","they will":"they'll",
}
def _make_contractions(s: str) -> str:
    t = s
    for full, c in _CONTRACTIONS.items():
        t = re.sub(rf"\b{re.escape(full)}\b", c, t, flags=re.IGNORECASE)
    return t

def _looks_like_request(en: str) -> bool:
    s = en.strip()
    if s.endswith("?"): return True
    if re.search(r"^(can|could|will|would)\s+you\b", s, re.I): return True
    m = re.match(r"^[Pp]lease\s+([A-Za-z']+)", s)
    if m: return True
    # tiny whitelist for bare imperatives
    first = re.match(r"^\s*([A-Za-z']+)\b", s)
    return bool(first and first.group(1).lower() in {
        "close","open","check","tell","give","send","provide","share","explain",
        "call","confirm","submit","review","attach","reply","respond","turn",
        "start","stop","report","pay","bring","take","show","help","deliver"
    })

def _formalize(en: str) -> str:
    s = en
    # expand contractions to sound formal
    for c, full in {v:k for k,v in _CONTRACTIONS.items()}.items():
        s = re.sub(rf"\b{re.escape(c)}\b", full, s, flags=re.IGNORECASE)
    # use formal vocabulary
    s = _formalize_vocab(s)
    if _looks_like_request(s):
        # Remove "please" from start and end first to avoid duplicates
        s = re.sub(r"^[Pp]lease\s+", "", s)
        s = re.sub(r"\s+[Pp]lease[.,!?]*$", "", s)  # remove trailing please with optional punctuation

        s = re.sub(r"^(can|could|will|would)\s+you\s+", "Could you please ", s, flags=re.IGNORECASE)
        if not s.lower().startswith("could you please "):
            s = "Could you please " + s.lstrip()

        # Clean up any remaining duplicate "please"
        s = re.sub(r"\bplease\s+please\b", "please", s, flags=re.IGNORECASE)

        # lower-case the first word after "please" (e.g., "Turn" -> "turn", "CLOSE" -> "close", "Close" -> "close")
        s = re.sub(r"(Could you please )([A-Za-z]+)", lambda m: m.group(1) + m.group(2).lower(), s, count=1)
        s = re.sub(r"\s{2,}", " ", s).strip()
        if s and s[-1] not in ".!?": s += "."
    # capitalize sentence start if it begins lower
    if s and s[0].islower():
        s = s[0].upper() + s[1:]

    # capitalize after sentence boundaries (., !, ?)
    def capitalize_after_period(m):
        return m.group(1) + m.group(2).upper()
    s = re.sub(r'([.!?]\s+)([a-z])', capitalize_after_period, s)

    return s


def _informalize(en: str) -> str:
    s = en
    # drop "please" anywhere (not only at start) for a more direct tone
    s = re.sub(r"\b[Pp]lease\b[, ]*", "", s)
    # soften formal scaffold
    s = re.sub(r"\bCould you please\s+", "Can you ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bWould you please\s+", "Can you ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bCould you\s+", "Can you ", s, flags=re.IGNORECASE)
    # use casual vocabulary
    s = _casualize_vocab(s)
    # add contractions
    s = _make_contractions(s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    # clean up spacing before punctuation (e.g., "word ." → "word.")
    s = re.sub(r'\s+([.,!?])', r'\1', s)
    # capitalize first letter
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    # capitalize after sentence boundaries
    def capitalize_after_period(m):
        return m.group(1) + m.group(2).upper()
    s = re.sub(r'([.!?]\s+)([a-z])', capitalize_after_period, s)
    return s


_FORMAL_TO_CASUAL = {
    r'\bhowever\b': 'but', r'\btherefore\b': 'so', r'\bconsequently\b': 'so',
    r'\bfurthermore\b': 'also', r'\bmoreover\b': 'also', r'\bnevertheless\b': 'but',
    r'\badditionally\b': 'also', r'\bthus\b': 'so', r'\bhence\b': 'so',
}

_CASUAL_TO_FORMAL = {
    r'\bbut\b': 'however', r'\bso\b': 'therefore', r'\balso\b': 'furthermore',
    r'\bokay\b': 'very well', r'\bok\b': 'very well', r'\bguys\b': 'everyone',
}

def _formalize_vocab(s: str) -> str:
    t = s
    for casual_pattern, formal_word in _CASUAL_TO_FORMAL.items():
        t = re.sub(casual_pattern, formal_word, t, flags=re.IGNORECASE)
    return t

def _casualize_vocab(s: str) -> str:
    t = s
    for formal_pattern, casual_word in _FORMAL_TO_CASUAL.items():
        t = re.sub(formal_pattern, casual_word, t, flags=re.IGNORECASE)
    return t

_EASY_REPL = {
    "approximately":"about","purchase":"buy","assistance":"help","commence":"start",
    "terminate":"end","endeavor":"try","inform":"tell","obtain":"get","require":"need",
    "inquire":"ask","therefore":"so","however":"but","consequently":"so","nevertheless":"but",
    "ensure":"make sure","indicate":"show","regarding":"about","utilize":"use",
    "demonstrate":"show","numerous":"many","substantial":"large","sufficient":"enough",
    "additional":"more","implement":"do","facilitate":"help","comprehensive":"full",
}

def _simplify_yes(en: str) -> str:
    """COMPREHENSIVE sentence simplification with restructuring."""
    s = en

    # Always apply these simplifications regardless of length:

    # 1. Remove parenthetical explanations
    s = re.sub(r'\s*\([^)]*\)', '', s)
    s = re.sub(r'\s*\[[^\]]*\]', '', s)

    # 2. Remove intensifiers and hedging (always helpful for simplicity)
    for modifier in [r'\bvery\s+', r'\breally\s+', r'\bquite\s+', r'\bextremely\s+',
                     r'\bparticularly\s+', r'\bespecially\s+', r'\bsignificantly\s+',
                     r'\bhighly\s+', r'\bexceptionally\s+', r'\bremarkably\s+',
                     r'\bsomewhat\s+', r'\brather\s+', r'\bfairly\s+']:
        s = re.sub(modifier, '', s, flags=re.IGNORECASE)

    # 3. Replace complex vocabulary with simple alternatives (always)
    def repl(m):
        w = m.group(0)
        lw = w.lower()
        rep = _EASY_REPL.get(lw)
        if not rep:
            return w
        return rep[0].upper() + rep[1:] if w and w[0].isupper() else rep
    s = re.sub(r'\b[A-Za-z]+\b', repl, s)

    # 4. Simplify common conjunctions and transitions (always)
    s = re.sub(r'\b(?:in order|so as) to\b', 'to', s, flags=re.IGNORECASE)
    s = re.sub(r'\bin the event that\b', 'if', s, flags=re.IGNORECASE)
    s = re.sub(r'\bdue to the fact that\b', 'because', s, flags=re.IGNORECASE)
    s = re.sub(r'\bin spite of the fact that\b', 'although', s, flags=re.IGNORECASE)
    s = re.sub(r'\bfor the purpose of\b', 'to', s, flags=re.IGNORECASE)
    s = re.sub(r'\bprior to\b', 'before', s, flags=re.IGNORECASE)
    s = re.sub(r'\bsubsequent to\b', 'after', s, flags=re.IGNORECASE)

    # Only apply complex restructuring for longer sentences:
    if len(s.split()) >= 8:
        # 5. Remove relative clauses (which/who/that introduce complexity)
        s = re.sub(r',\s*who\s+[^,]+,', ',', s)
        s = re.sub(r',\s*which\s+[^,]+,', ',', s)

        # 6. Convert passive to active voice patterns
        s = re.sub(r'\bwas\s+(\w+ed)\s+by\b', r'\1 by', s)
        s = re.sub(r'\bwere\s+(\w+ed)\s+by\b', r'\1 by', s)

        # 7. Break compound sentences - take main clause
        if len(s.split()) >= 14:
            parts = re.split(r'[;:]|,\s+(?:and|but|or|however|although|while|because)\s+', s)
            if len(parts) > 1:
                s = max(parts, key=len)  # Take longest clause (usually main)

    # 8. Remove redundant phrases
    s = re.sub(r'\bin my opinion,?\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\bit should be noted that\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\bit is important to note that\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\bas a matter of fact,?\s*', '', s, flags=re.IGNORECASE)

    # 9. Simplify time/date expressions
    s = re.sub(r'\bat the present time\b', 'now', s, flags=re.IGNORECASE)
    s = re.sub(r'\bat this point in time\b', 'now', s, flags=re.IGNORECASE)
    s = re.sub(r'\bin the near future\b', 'soon', s, flags=re.IGNORECASE)

    # 10. Clean up spacing and punctuation
    s = re.sub(r'\s{2,}', ' ', s).strip()
    s = re.sub(r',\s*,', ',', s)
    s = re.sub(r'\s+([.,!?])', r'\1', s)

    # 11. Ensure proper sentence ending
    if s and s[-1] not in '.!?':
        s += '.'

    # 12. Capitalize first letter
    if s and s[0].islower():
        s = s[0].upper() + s[1:]

    return s

def _apply_controls(en: str, style: str, simplify: str) -> str:
    out = en
    if style == "formal":
        out = _formalize(out)
    elif style == "informal":
        out = _informalize(out)
    if simplify == "yes":
        out = _simplify_yes(out)
    return out

def main():
    a = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    src, tgt = resolve_tag(a.src_lang), resolve_tag(a.tgt_lang)

    # gather inputs
    inputs = []
    if a.text: inputs.extend(a.text)
    if a.file:
        with open(a.file, "r", encoding="utf-8") as f:
            inputs.extend([ln.rstrip("\n") for ln in f if ln.strip()])
    if not inputs:
        raise SystemExit("Provide --text (repeatable) or --file with one input per line.")

    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = load_model(a.base, a.adapter, a.quant, device)
    use_cache = (a.use_cache == "on")
    if hasattr(model, "config"): model.config.use_cache = use_cache
    if hasattr(model, "generation_config"): model.generation_config.use_cache = use_cache
    model = model.eval()

    ip = IndicProcessor(inference=True)
    pres = ip.preprocess_batch(inputs, src_lang=src, tgt_lang=tgt)

    # insert controls BEFORE the separator on each item
    # Format depends on token_format: 'special' uses <FORMAL>, 'legacy' uses formal
    if a.token_format == "special":
        style_token = f"<{a.style.upper()}>"
        simplify_token = f"<SIMPL_{a.simplify.upper()}>"
    else:  # legacy
        style_token = a.style
        simplify_token = a.simplify

    with_ctrls = []
    for pre in pres:
        if " ||| " in pre:
            pre = pre.replace(" ||| ", f" {style_token} {simplify_token} ||| ", 1)
        with_ctrls.append(pre)

    enc = tok(with_ctrls, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=a.max_new_tokens,
                             num_beams=a.num_beams, do_sample=False, use_cache=use_cache)
    raw = tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    post = ip.postprocess_batch(raw, lang=tgt)
    # optional, demo-friendly control enforcement
    if a.enforce_controls == "on":
        post = [_apply_controls(line, a.style, a.simplify) for line in post]

    for line in post:
        print(line)

if __name__ == "__main__":
    main()
