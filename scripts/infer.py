import argparse
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils_text import build_prompt


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

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_dir if (model_dir / "tokenizer_config.json").exists() else base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(base_model, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, model_dir)
    model.to(device)
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, tokenizer = load_pipeline(Path(args.model), device)

    prompt = build_prompt(args.src_lang, args.tgt_lang, args.style, args.simplify, args.text)
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
