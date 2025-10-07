import os, warnings, argparse, torch
from pathlib import Path
from transformers.utils import logging as hf_logging

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
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--use_cache", choices=["on","off"], default="off")
    p.add_argument("--quant", choices=["off","auto"], default="off")
    p.add_argument("--max_new_tokens", type=int, default=128)
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
    with_ctrls = []
    for pre in pres:
        if " ||| " in pre:
            pre = pre.replace(" ||| ", f" {a.style} {a.simplify} ||| ", 1)
        with_ctrls.append(pre)

    enc = tok(with_ctrls, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=a.max_new_tokens,
                             num_beams=a.num_beams, do_sample=False, use_cache=use_cache)
    raw = tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    post = ip.postprocess_batch(raw, lang=tgt)
    for line in post:
        print(line)

if __name__ == "__main__":
    main()
