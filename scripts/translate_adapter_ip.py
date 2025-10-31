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
    if _looks_like_request(s):
        s = re.sub(r"^(can|could|will|would)\s+you\s+", "Could you please ", s, flags=re.IGNORECASE)
        s = re.sub(r"^[Pp]lease\s+", "", s)  # avoid "please please"
        if not s.lower().startswith("could you please "):
            s = "Could you please " + s.lstrip()
        # lower-case the verb after "please" (e.g., "Turn" -> "turn")
        s = re.sub(r"(Could you please )([A-Z])", lambda m: m.group(1)+m.group(2).lower(), s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        if s and s[-1] not in ".!?": s += "."
    # capitalize sentence start if it begins lower
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s


def _informalize(en: str) -> str:
    s = en
    # drop "please" anywhere (not only at start) for a more direct tone
    s = re.sub(r"\b[Pp]lease\b[, ]*", "", s)
    # soften formal scaffold
    s = re.sub(r"\bCould you please\s+", "Can you ", s, flags=re.IGNORECASE)
    s = _make_contractions(s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


_EASY_REPL = {
    "approximately":"about","purchase":"buy","assistance":"help","commence":"start",
    "terminate":"end","endeavor":"try","inform":"tell","obtain":"get","require":"need",
    "inquire":"ask","therefore":"so","however":"but","consequently":"so","nevertheless":"but",
    "ensure":"make sure","indicate":"show","regarding":"about",
}
def _simplify_yes(en: str) -> str:
    # no aggressive edits for very short lines
    if len(en.split()) < 8: 
        return en
    s = re.sub(r"\s*\([^)]*\)", "", en)  # drop parentheticals
    # trim to first main clause if long
    if len(s.split()) >= 14:
        s = re.split(r"[;:—–]|, and |, but ", s)[0]
    # easy vocab
    def repl(m):
        w = m.group(0); lw = w.lower()
        rep = _EASY_REPL.get(lw)
        if not rep: return w
        return rep[0].upper()+rep[1:] if w and w[0].isupper() else rep
    s = re.sub(r"[A-Za-z']+", repl, s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    if s and s[-1] not in ".!?": s += "."
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
