from pathlib import Path
from typing import Dict
import os
import sys
import re

import torch
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from IndicTransToolkit.processor import IndicProcessor
from transformers import AutoTokenizer

# --- project imports / paths -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.infer import load_model as infer_load_model, resolve_device

# --- tags & pairs ------------------------------------------------------------
IT2_TAG = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
}
AVAILABLE_PAIRS: Dict[str, Dict[str, str]] = {
    "hi-en": {"label": "Hindi → English"},
}

# --- demo app ----------------------------------------------------------------
app = FastAPI(title="IndicTrans LoRA Demo")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# --- runtime config ----------------------------------------------------------
_device = resolve_device("auto")
BASE_MODEL_DIR = os.environ.get("BASE_MODEL_DIR", "models/indictrans2-indic-en-1B")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR")  # set to LoRA checkpoint dir for controls
ENFORCE = os.environ.get("DEMO_ENFORCE", "on").strip().lower() == "on"  # post-edit on/off

# --- load model/tokenizer/processor -----------------------------------------
ip = IndicProcessor(inference=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = infer_load_model(BASE_MODEL_DIR, ADAPTER_DIR, _device, quant="off")
if hasattr(model, "config"):
    model.config.use_cache = False
if hasattr(model, "generation_config"):
    model.generation_config.use_cache = False

print(f"[demo] base={BASE_MODEL_DIR}, adapter={ADAPTER_DIR or 'none'}, device={_device}, beams=4, use_cache=False, enforce={ENFORCE}")

# --- post-edit enforcement (demo-only; reversible via DEMO_ENFORCE) ---------
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
    if re.search(r"^(can|could|will|would)\s+you\b", s, re.IGNORECASE): return True
    if re.match(r"^[Pp]lease\s+[A-Za-z]+", s): return True
    first = re.match(r"^\s*([A-Za-z']+)\b", s)
    return bool(first and first.group(1).lower() in {
        "close","open","check","tell","give","send","provide","share","explain",
        "call","confirm","submit","review","attach","reply","respond","turn",
        "start","stop","report","pay","bring","take","show","help","deliver"
    })

def _formalize(en: str) -> str:
    s = en
    # expand contractions for a formal tone
    for c, full in {v:k for k,v in _CONTRACTIONS.items()}.items():
        s = re.sub(rf"\b{re.escape(c)}\b", full, s, flags=re.IGNORECASE)
    if _looks_like_request(s):
        s = re.sub(r"^(can|could|will|would)\s+you\s+", "Could you please ", s, flags=re.IGNORECASE)
        s = re.sub(r"^[Pp]lease\s+", "", s)  # avoid double please
        if not s.lower().startswith("could you please "):
            s = "Could you please " + s.lstrip()
        # lower-case the verb after "please"
        s = re.sub(r"(Could you please )([A-Z])", lambda m: m.group(1) + m.group(2).lower(), s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        if s and s[-1] not in ".!?": s += "."
    # capitalize sentence start
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s

def _informalize(en: str) -> str:
    s = en
    # drop "please" anywhere for a direct tone
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
    if len(en.split()) < 8:
        return en
    s = re.sub(r"\s*\([^)]*\)", "", en)
    if len(s.split()) >= 14:
        s = re.split(r"[;:—–]|, and |, but ", s)[0]
    def repl(m):
        w = m.group(0); lw = w.lower()
        rep = _EASY_REPL.get(lw)
        if not rep: return w
        return rep[0].upper()+rep[1:] if w and w[0].isupper() else rep
    s = re.sub(r"[A-Za-z']+", repl, s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    if s and s[-1] not in ".!?": s += "."
    return s

def enforce_controls(out_text: str, style: str, simplify: str) -> str:
    if not ENFORCE:
        return out_text
    y = out_text
    if style == "formal":
        y = _formalize(y)
    elif style == "informal":
        y = _informalize(y)
    if simplify == "yes":
        y = _simplify_yes(y)
    return y

# --- routes ------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "pairs": AVAILABLE_PAIRS,
            "result": None,
            "error": None,
        },
    )

@app.post("/infer", response_class=HTMLResponse)
async def infer(
    request: Request,
    pair: str = Form("hi-en"),
    style: str = Form("formal"),
    simplify: str = Form("no"),
    text: str = Form(...),
    max_new_tokens: int = Form(128),
):
    error = None
    result = None
    if pair not in AVAILABLE_PAIRS:
        error = f"Pair {pair} is not available in this demo."
    elif not text.strip():
        error = "Please enter some input text."
    else:
        try:
            src_tag = IT2_TAG["hi"]
            tgt_tag = IT2_TAG["en"]

            # preprocess (IndicProcessor builds prompt with tags and |||)
            pre = ip.preprocess_batch([text], src_lang=src_tag, tgt_lang=tgt_tag)[0]

            # insert controls only if an adapter is active (keeps base behavior clean)
            # Using special token format: <FORMAL>/<INFORMAL>, <SIMPL_Y>/<SIMPL_N>
            if ADAPTER_DIR and " ||| " in pre:
                style_token = f"<{style.upper()}>"
                simplify_token = f"<SIMPL_{simplify.upper()}>"
                pre = pre.replace(" ||| ", f" {style_token} {simplify_token} ||| ", 1)

            enc = tokenizer(pre, return_tensors="pt", padding=True, truncation=True).to(_device)

            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    num_beams=4,
                    do_sample=False,
                    use_cache=False,
                )

            raw = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            post = ip.postprocess_batch(raw, lang=tgt_tag)
            gen = post[0] if post else ""

            # demo-time enforcement for visible control contrast
            gen = enforce_controls(gen, style, simplify)

            result = {
                "prompt": pre,
                "generated": gen,
                "pair": AVAILABLE_PAIRS[pair]["label"],
                "style": style,
                "simplify": simplify,
            }
        except FileNotFoundError as exc:
            error = str(exc)
        except Exception as exc:  # runtime safeguard
            error = f"Generation failed: {exc}"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "pairs": AVAILABLE_PAIRS,
            "result": result,
            "error": error,
            "selected_pair": pair,
            "selected_style": style,
            "selected_simplify": simplify,
            "input_text": text,
            "max_new_tokens": max_new_tokens,
        },
    )