import argparse
from pathlib import Path
from typing import Optional

import torch
from IndicTransToolkit.processor import IndicProcessor
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    BitsAndBytesConfig = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import bitsandbytes as _bnb  # noqa: F401

    BNB_AVAILABLE = True
except ImportError:  # pragma: no cover
    BNB_AVAILABLE = False

IT2_TAG = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ml": "mal_Mlym",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IndicTrans2 inference with optional LoRA adapter.")
    parser.add_argument(
        "--base",
        default="models/indictrans2-indic-en-1B",
        help="Base model path or Hugging Face repo id.",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional PEFT/LoRA adapter directory (e.g., outputs/hi_en_r16/checkpoint-6000).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Deprecated alias for --adapter (kept for backward compatibility).",
    )
    parser.add_argument("--src_lang", required=True, help="Source language code, e.g. hi.")
    parser.add_argument("--tgt_lang", required=True, help="Target language code, e.g. en.")
    parser.add_argument("--style", choices=["formal", "informal"], required=True)
    parser.add_argument("--simplify", choices=["yes", "no"], required=True)
    parser.add_argument("--text", required=True, help="Input text to translate or simplify.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument(
        "--use_cache",
        choices=["on", "off"],
        default="off",
        help="Enable decoder KV cache (default off to avoid IndicTrans2 bug).",
    )
    parser.add_argument(
        "--quant",
        choices=["auto", "off"],
        default="off",
        help="Quantization mode: 'auto' tries 4-bit NF4 if bitsandbytes is available.",
    )
    parser.add_argument("--device", default="auto", help="Device to run on: auto|cpu|cuda")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available on this machine.")
    return torch.device(device_arg)


def load_model(base: str, adapter: Optional[str], device: torch.device, quant: str) -> AutoModelForSeq2SeqLM:
    use_4bit = quant == "auto" and device.type == "cuda" and BitsAndBytesConfig is not None and BNB_AVAILABLE
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base,
            trust_remote_code=True,
            quantization_config=quant_config,
            device_map="auto",
        )
    else:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        model.to(device)

    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)

    base_model = getattr(model, "base_model", model)
    if hasattr(base_model, "tie_weights"):
        try:
            base_model.tie_weights()
        except Exception:
            pass

    model.eval()
    return model


def main() -> None:
    args = parse_args()
    base = args.base
    adapter = args.adapter or args.model
    use_cache = args.use_cache.lower() == "on"

    device = resolve_device(args.device)
    ip = IndicProcessor(inference=True)

    src_tag = IT2_TAG.get(args.src_lang.lower(), args.src_lang)
    tgt_tag = IT2_TAG.get(args.tgt_lang.lower(), args.tgt_lang)

    print(
        f"[loader] base={base}, adapter={adapter or 'none'}, quant={args.quant}, "
        f"beams={args.num_beams}, use_cache={use_cache}, src={src_tag}, tgt={tgt_tag}"
    )

    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(base, adapter, device, args.quant)

    if hasattr(model, "config"):
        model.config.use_cache = use_cache
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = use_cache

    preprocessed = ip.preprocess_batch([args.text], src_lang=src_tag, tgt_lang=tgt_tag)[0]
    if adapter is not None and " ||| " in preprocessed:
        preprocessed = preprocessed.replace(" ||| ", f" {args.style} {args.simplify} ||| ", 1)

    inputs = tokenizer(preprocessed, return_tensors="pt", padding=True, truncation=True).to(device)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "do_sample": False,
        "use_cache": use_cache,
    }
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)
    decoded = tokenizer.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    decoded = ip.postprocess_batch(decoded, lang=tgt_tag)

    print(decoded[0])


if __name__ == "__main__":
    main()
