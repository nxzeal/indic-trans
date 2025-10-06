import argparse
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_MODEL_ARGS = {
    "trust_remote_code": True,
    "use_fast_tokenizer": True,
    "allow_resize_token_embeddings": False,
}

IT2_TAG = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ml": "mal_Mlym",
}


def _load_tokenizer(source: str | Path, *, trust_remote_code: bool, use_fast: bool):
    try:
        return AutoTokenizer.from_pretrained(
            source,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
        )
    except Exception as exc:
        if use_fast:
            print(f"[loader] Tokenizer load failed with use_fast=True ({exc}); retrying with use_fast=False.")
            return AutoTokenizer.from_pretrained(
                source,
                trust_remote_code=trust_remote_code,
                use_fast=False,
            )
        raise


def _load_model_with_dtype(model_name: str, *, dtype_value: torch.dtype, trust_remote_code: bool):
    base_kwargs = {"trust_remote_code": trust_remote_code}
    try:
        return AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype_value, **base_kwargs)
    except TypeError as exc:
        print(f"[loader] dtype kwarg unsupported ({exc}); retrying with torch_dtype={dtype_value}.")
        return AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype_value, **base_kwargs)


def _maybe_resize_and_tie(model, tokenizer, *, allow_resize: bool) -> None:
    embedding_layer = None
    try:
        embedding_layer = model.get_input_embeddings()
    except (AttributeError, TypeError):
        embedding_layer = None

    tok_size = tokenizer.vocab_size
    model_vocab_size = None
    if embedding_layer is not None and hasattr(embedding_layer, "weight"):
        model_vocab_size = embedding_layer.weight.shape[0]

    model_class_name = model.__class__.__name__
    indictrans_model = "IndicTrans" in model_class_name

    if indictrans_model:
        print("[loader] Skipping resize_token_embeddings for IndicTrans* models.")
    elif model_vocab_size is not None and tok_size != model_vocab_size:
        if allow_resize:
            try:
                model.resize_token_embeddings(tok_size)
            except (NotImplementedError, AttributeError):
                print("[loader] resize_token_embeddings not supported; skipping.")
        else:
            print("[loader] Tokenizer/model vocab mismatch but resize disabled; continuing.")

    if hasattr(model, "tie_weights"):
        try:
            model.tie_weights()
            print("[loader] tie_weights() applied.")
        except Exception:
            print("[loader] tie_weights() not supported; continuing.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained IndicTrans LoRA adapter.")
    parser.add_argument("--model", required=True, help="Path to the LoRA adapter directory.")
    parser.add_argument("--src_lang", required=True, help="Source language code, e.g. hi.")
    parser.add_argument("--tgt_lang", required=True, help="Target language code, e.g. en.")
    parser.add_argument("--style", choices=["formal", "informal"], required=True)
    parser.add_argument("--simplify", choices=["yes", "no"], required=True)
    parser.add_argument("--text", required=True, help="Input text to translate or simplify.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", default="auto", help="Device to run on: auto|cpu|cuda")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available on this machine.")
    return torch.device(device_arg)


def load_pipeline(model_dir: Path, device: torch.device):
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
    peft_config = PeftConfig.from_pretrained(model_dir)
    base_model = peft_config.base_model_name_or_path

    dtype_value = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer_source = model_dir if (model_dir / "tokenizer_config.json").exists() else base_model
    tokenizer = _load_tokenizer(
        tokenizer_source,
        trust_remote_code=DEFAULT_MODEL_ARGS["trust_remote_code"],
        use_fast=DEFAULT_MODEL_ARGS["use_fast_tokenizer"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model_obj = _load_model_with_dtype(
        base_model,
        dtype_value=dtype_value,
        trust_remote_code=DEFAULT_MODEL_ARGS["trust_remote_code"],
    )
    _maybe_resize_and_tie(
        base_model_obj,
        tokenizer,
        allow_resize=DEFAULT_MODEL_ARGS.get("allow_resize_token_embeddings", False),
    )

    model = PeftModel.from_pretrained(base_model_obj, model_dir)
    model.to(device)
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, tokenizer = load_pipeline(Path(args.model), device)

    src_tag = IT2_TAG.get(args.src_lang.lower(), args.src_lang)
    tgt_tag = IT2_TAG.get(args.tgt_lang.lower(), args.tgt_lang)
    prompt = f"{src_tag} {tgt_tag} {args.style} {args.simplify} ||| {args.text}"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    output_text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    print("=== Prompt ===")
    print(prompt)
    print("=== Output ===")
    print(output_text)


if __name__ == "__main__":
    main()
