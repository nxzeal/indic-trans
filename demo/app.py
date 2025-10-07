from pathlib import Path
from typing import Dict
import os
import sys

import torch
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from IndicTransToolkit.processor import IndicProcessor
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.infer import load_model as infer_load_model, resolve_device

IT2_TAG = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
}

AVAILABLE_PAIRS: Dict[str, Dict[str, str]] = {
    "hi-en": {"label": "Hindi → English"},
}

app = FastAPI(title="IndicTrans LoRA Demo")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

_device = resolve_device("auto")
BASE_MODEL_DIR = os.environ.get("BASE_MODEL_DIR", "models/indictrans2-indic-en-1B")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR")
ip = IndicProcessor(inference=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = infer_load_model(BASE_MODEL_DIR, ADAPTER_DIR, _device, quant="off")
if hasattr(model, "config"):
    model.config.use_cache = False
if hasattr(model, "generation_config"):
    model.generation_config.use_cache = False

print(
    f"[demo] base={BASE_MODEL_DIR}, adapter={ADAPTER_DIR or 'none'}, device={_device}, beams=4, use_cache=False"
)


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
            preprocessed = ip.preprocess_batch([text], src_lang=src_tag, tgt_lang=tgt_tag)[0]
            if ADAPTER_DIR and " ||| " in preprocessed:
                preprocessed = preprocessed.replace(" ||| ", f" {style} {simplify} ||| ", 1)

            inputs = tokenizer(preprocessed, return_tensors="pt", padding=True, truncation=True).to(_device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=4,
                    do_sample=False,
                    use_cache=False,
                )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            postprocessed = ip.postprocess_batch(decoded, lang=tgt_tag)
            result = {
                "prompt": preprocessed,
                "generated": postprocessed[0],
                "pair": AVAILABLE_PAIRS[pair]["label"],
                "style": style,
                "simplify": simplify,
            }
        except FileNotFoundError as exc:
            error = str(exc)
        except Exception as exc:  # pragma: no cover - runtime safeguard
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
