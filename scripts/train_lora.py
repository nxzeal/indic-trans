import argparse
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - optional dependency
    BitsAndBytesConfig = None

try:
    import mlflow
except ImportError as exc:  # pragma: no cover - defensive import error
    raise SystemExit("mlflow is required for experiment tracking. Please install mlflow.") from exc

try:
    import yaml
except ImportError as exc:  # pragma: no cover - defensive import error
    raise SystemExit("PyYAML is required to parse configs. Please install pyyaml.") from exc

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from utils_io import ensure_dir, read_tsv, write_json


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a QLoRA adapter on IndicTrans data.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    return parser.parse_args(argv)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(name.lower(), torch.float16)


def prepare_datasets(cfg: Dict[str, Any]) -> DatasetDict:
    data_dir = Path(cfg["data_dir"])
    base_pair = cfg["pairs"][0]
    base_src, base_tgt = base_pair.split("-")
    template = cfg["task"]["prompt_template"]

    def expand_row(row: Dict[str, str]) -> List[Dict[str, str]]:
        examples: List[Dict[str, str]] = []
        for pair in cfg["pairs"]:
            src_lang, tgt_lang = pair.split("-")
            if src_lang == base_src and tgt_lang == base_tgt:
                source_text = row["src"]
                target_text = row["tgt"]
                style = row.get("style_tgt", "formal")
            elif src_lang == base_tgt and tgt_lang == base_src:
                source_text = row["tgt"]
                target_text = row["src"]
                style = row.get("style_src", "formal")
            else:
                continue
            simplify = row.get("simplify", "no")
            prompt = (
                template.replace("<SRC_LANG>", src_lang)
                .replace("<TGT_LANG>", tgt_lang)
                .replace("<STYLE>", style)
                .replace("<SIMPLIFY>", simplify)
                .replace("<TEXT>", source_text)
            )
            examples.append(
                {
                    "prompt": prompt,
                    "target": target_text,
                    "source": source_text,
                    "direction": pair,
                    "style": style,
                    "simplify": simplify,
                }
            )
        return examples

    def load_split(name: str) -> Dataset:
        path = data_dir / f"{name}.tsv"
        if not path.exists():
            raise FileNotFoundError(f"Expected split file {path} to exist.")
        rows = [row for row in read_tsv(path)]
        expanded: List[Dict[str, str]] = []
        for row in rows:
            expanded.extend(expand_row(row))
        if not expanded:
            raise ValueError(f"No usable rows after expansion for split {name}.")
        return Dataset.from_list(expanded)

    dataset = DatasetDict()
    dataset["train"] = load_split("train")
    dataset["validation"] = load_split("val")
    if (data_dir / "test.tsv").exists():
        dataset["test"] = load_split("test")
    return dataset


def tokenize_datasets(dataset: DatasetDict, tokenizer: AutoTokenizer, max_input: int = 1024, max_target: int = 512) -> DatasetDict:
    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        model_inputs = tokenizer(batch["prompt"], max_length=max_input, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(batch["target"], max_length=max_target, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    remove_cols = list(next(iter(dataset.values())).column_names)
    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=remove_cols)
    return tokenized


def configure_model(cfg: Dict[str, Any]) -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer, bool]:
    base_model = cfg["base_model"]
    quant_cfg = cfg.get("quantization", {})
    model_args = cfg.get("model_args", {})

    trust_remote_code = bool(model_args.get("trust_remote_code", True))
    use_fast_tokenizer = bool(model_args.get("use_fast_tokenizer", True))
    allow_resize = bool(model_args.get("allow_resize_token_embeddings", False))

    load_kwargs: Dict[str, Any] = {"device_map": "auto", "trust_remote_code": trust_remote_code}
    dtype_value = None
    use_4bit = quant_cfg.get("load_in_4bit", False)
    if use_4bit:
        if BitsAndBytesConfig is None:
            print("bitsandbytes not available in transformers; disabling 4-bit quantization.")
            use_4bit = False
        else:
            try:
                import bitsandbytes as _  # noqa: F401
            except ImportError:
                print("bitsandbytes package missing; disabling 4-bit quantization.")
                use_4bit = False
    if use_4bit:
        compute_dtype = dtype_from_name(quant_cfg.get("bnb_4bit_compute_dtype", "float16"))
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        load_kwargs["quantization_config"] = bnb_config
        dtype_value = compute_dtype
    else:
        default_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        dtype_value = default_dtype

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=use_fast_tokenizer,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        if use_fast_tokenizer:
            print(f"[loader] AutoTokenizer fast load failed ({exc}); retrying with use_fast=False.")
            tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                use_fast=False,
                trust_remote_code=trust_remote_code,
            )
        else:
            raise
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(load_kwargs)
    try:
        if dtype_value is not None:
            model = AutoModelForSeq2SeqLM.from_pretrained(base_model, dtype=dtype_value, **model_kwargs)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(base_model, **model_kwargs)
    except TypeError as exc:
        if dtype_value is not None:
            print(f"[loader] dtype kwarg unsupported ({exc}); retrying with torch_dtype={dtype_value}.")
            model = AutoModelForSeq2SeqLM.from_pretrained(base_model, torch_dtype=dtype_value, **model_kwargs)
        else:
            raise

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

    gradient_checkpointing = cfg["train"].get("gradient_checkpointing", False)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.0),
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)

    return model, tokenizer, use_4bit


def log_config(config_path: Path, cfg: Dict[str, Any]) -> None:
    mlflow.log_param("config_path", str(config_path))
    flat_params = {
        "lora_r": cfg["lora"]["r"],
        "lora_alpha": cfg["lora"]["alpha"],
        "lr": cfg["train"]["lr"],
        "batch_size": cfg["train"]["batch_size"],
        "grad_accum": cfg["train"]["grad_accum"],
        "max_steps": cfg["train"]["max_steps"],
    }
    mlflow.log_params(flat_params)
    mlflow.log_dict(cfg, "config/runtime_config.json")


def train(cfg: Dict[str, Any], config_path: Path) -> None:
    set_seed(cfg.get("seed", 42))
    output_dir = ensure_dir(cfg["output_dir"])

    model, tokenizer, used_4bit = configure_model(cfg)
    datasets = prepare_datasets(cfg)
    tokenized = tokenize_datasets(datasets, tokenizer)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    train_cfg = cfg["train"]
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        learning_rate=train_cfg["lr"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["grad_accum"],
        max_steps=train_cfg["max_steps"],
        warmup_ratio=train_cfg.get("warmup_ratio", 0.0),
        evaluation_strategy="steps",
        eval_steps=train_cfg["eval_every"],
        save_steps=train_cfg["save_every"],
        logging_steps=50,
        predict_with_generate=True,
        fp16=train_cfg.get("fp16", False) and torch.cuda.is_available(),
        bf16=train_cfg.get("bf16", False) and torch.cuda.is_available(),
        report_to=["mlflow"],
        load_best_model_at_end=False,
        save_total_limit=2,
    )

    mlflow.set_tracking_uri(f"file://{Path('mlruns').resolve()}")
    mlflow.set_experiment("indictrans-lora")
    run_name = Path(cfg["output_dir"]).name
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("used_4bit", used_4bit)
        log_config(config_path, cfg)
        mlflow.log_metric("train_examples", len(datasets["train"]))
        mlflow.log_metric("val_examples", len(datasets["validation"]))

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            tokenizer=tokenizer,
            data_collator=collator,
        )

        trainer.train()

        eval_output = trainer.predict(tokenized["validation"])
        predictions = tokenizer.batch_decode(eval_output.predictions, skip_special_tokens=True)
        preds_path = output_dir / "preds_val.txt"
        preds_path.write_text("\n".join(predictions), encoding="utf-8")
        metric_summary = {
            "eval_loss": float(eval_output.metrics.get("test_loss", eval_output.metrics.get("eval_loss", 0.0))),
            "samples": int(eval_output.metrics.get("test_num_samples", eval_output.metrics.get("eval_samples", 0))),
        }
        write_json(metric_summary, output_dir / "eval_summary.json")
        mlflow.log_artifact(str(config_path), artifact_path="config")
        mlflow.log_artifact(str(preds_path), artifact_path="predictions")
        mlflow.log_artifact(str(output_dir / "eval_summary.json"), artifact_path="metrics")

        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        state_dict_path = output_dir / "adapter_config.json"
        if state_dict_path.exists():
            mlflow.log_artifact(str(state_dict_path), artifact_path="adapter")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config)
    cfg = load_config(config_path)
    train(cfg, config_path)


if __name__ == "__main__":
    main()
