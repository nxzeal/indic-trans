from pathlib import Path
from typing import Dict, Tuple
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import torch
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from scripts.infer import DEFAULT_MODEL_ARGS, load_pipeline, resolve_device

IT2_TAG = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ml": "mal_Mlym",
}

AVAILABLE_PAIRS: Dict[str, Dict[str, str]] = {
    "hi-en": {"model": "outputs/hi_en_r16", "src_lang": "hi", "tgt_lang": "en", "label": "Hindi -> English"},
    "ta-en": {"model": "outputs/ta_en_r16", "src_lang": "ta", "tgt_lang": "en", "label": "Tamil -> English"},
}

app = FastAPI(title="IndicTrans LoRA Demo")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

_device = resolve_device("auto")
_pipelines: Dict[str, Tuple[object, object]] = {}

# Ensure demo inference mirrors loader safeguards (explicit for clarity)
DEFAULT_MODEL_ARGS["allow_resize_token_embeddings"] = False


def get_pipeline(pair: str):
    if pair in _pipelines:
        return _pipelines[pair]
    cfg = AVAILABLE_PAIRS[pair]
    model_dir = Path(cfg["model"])
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Adapter for pair {pair} not found at {model_dir}. Train the model before using the demo."
        )
    pipeline = load_pipeline(model_dir, _device)
    _pipelines[pair] = pipeline
    return pipeline


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
    pair: str = Form(...),
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
            model, tokenizer = get_pipeline(pair)
            cfg = AVAILABLE_PAIRS[pair]
            src_tag = IT2_TAG.get(cfg["src_lang"].lower(), cfg["src_lang"])
            tgt_tag = IT2_TAG.get(cfg["tgt_lang"].lower(), cfg["tgt_lang"])
            prompt = f"{src_tag} {tgt_tag} {style} {simplify} ||| {text}"
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(_device)
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            result = {
                "prompt": prompt,
                "generated": generated,
                "pair": cfg["label"],
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
