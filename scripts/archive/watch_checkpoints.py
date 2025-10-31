# scripts/watch_checkpoints.py
import time, subprocess, os
from pathlib import Path
from datetime import datetime

BASE = "models/indictrans2-indic-en-1B"
OUT_DIR = Path("outputs/hi_en_r8_v2")
ART = Path("artifacts/review2")
DEMO = ART / "demo_inputs.txt"
SEEN = ART / ".seen_ckpts.txt"
VAL_TSV = Path("data/clean/hi_en/val.tsv")
VAL_SRC = ART / "val50.src"
VAL_REF = ART / "val50.ref"

PY = os.environ.get("PYTHON", "python")

def ensure_demo_inputs():
    if DEMO.exists(): return
    ART.mkdir(parents=True, exist_ok=True)
    DEMO.write_text("\n".join([
        "कृपया दरवाज़ा बंद करें।",
        "क्या आप इसे कल तक भेज सकते हैं?",
        "मुझे लगता है कि समय सीमा बढ़ानी पड़ेगी।",
        "ये चीज़ थोड़ी झंझट वाली है।",
        "यार, यह तो बहुत टफ हो गया।",
        "कृपया इसे औपचारिक शैली में दोबारा लिखें।",
        "कृपया संक्षेप में समझाएँ कि मुख्य बिंदु क्या हैं, ताकि शुरुआती व्यक्ति भी समझ सके।",
        "कल हुई बैठक में बताई गई आवश्यकताएँ निम्नलिखित हैं, जिनमें समयसीमा और गुणवत्ता मानकों का पालन अनिवार्य है।",
    ]), encoding="utf-8")

def ensure_val50():
    if VAL_SRC.exists() and VAL_REF.exists(): return
    if not VAL_TSV.exists(): return
    lines = VAL_TSV.read_text(encoding="utf-8").splitlines()[:60][-50:]
    src = []
    ref = []
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) >= 2:
            src.append(parts[0])
            ref.append(parts[1])
    ART.mkdir(parents=True, exist_ok=True)
    VAL_SRC.write_text("\n".join(src), encoding="utf-8")
    VAL_REF.write_text("\n".join(ref), encoding="utf-8")

def run(cmd):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""        # force CPU for child processes
    env["TRANSFORMERS_VERBOSITY"] = "error"
    return subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)

def eval_checkpoint(ckpt: Path):
    name = ckpt.name
    print(f"[{datetime.now().strftime('%H:%M:%S')}] evaluating {name}")

    # adapter: formal/no
    of1 = ART / f"{name}_formal_no.txt"
    cmd1 = [PY, "scripts/translate_adapter_ip.py",
            "--base", BASE, "--adapter", str(ckpt),
            "--src", "hi", "--tgt", "en",
            "--style", "formal", "--simplify", "no",
            "--file", str(DEMO)]
    of1.write_text(run(cmd1).stdout, encoding="utf-8")

    # adapter: informal/yes
    of2 = ART / f"{name}_informal_yes.txt"
    cmd2 = [PY, "scripts/translate_adapter_ip.py",
            "--base", BASE, "--adapter", str(ckpt),
            "--src", "hi", "--tgt", "en",
            "--style", "informal", "--simplify", "yes",
            "--file", str(DEMO)]
    of2.write_text(run(cmd2).stdout, encoding="utf-8")

    # quick BLEU on 50 lines (neutral controls)
    if VAL_SRC.exists() and VAL_REF.exists():
        hyp = ART / f"{name}_val50.hyp"
        cmd3 = [PY, "scripts/translate_adapter_ip.py",
                "--base", BASE, "--adapter", str(ckpt),
                "--src", "hi", "--tgt", "en",
                "--style", "formal", "--simplify", "no",
                "--file", str(VAL_SRC)]
        hyp.write_text(run(cmd3).stdout, encoding="utf-8")

        # sacrebleu via Python API (no shell)
        try:
            import sacrebleu
            sys_ref = [VAL_REF.read_text(encoding="utf-8").splitlines()]
            sys_hyp = hyp.read_text(encoding="utf-8").splitlines()
            bleu = sacrebleu.corpus_bleu(sys_hyp, sys_ref)
            (ART / f"{name}_val50.bleu.txt").write_text(
                f"BLEU = {bleu.score:.2f}\nBP = {bleu.bp:.3f}  ratio = {bleu.sys_len/bleu.ref_len:.3f}\n",
                encoding="utf-8"
            )
        except Exception as e:
            (ART / f"{name}_val50.bleu.txt").write_text(f"BLEU failed: {e}\n", encoding="utf-8")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] done {name}")

def main():
    ensure_demo_inputs()
    ensure_val50()
    ART.mkdir(parents=True, exist_ok=True)
    SEEN.touch(exist_ok=True)
    seen = set(SEEN.read_text(encoding="utf-8").splitlines())

    print("[watch] monitoring:", OUT_DIR)
    while True:
        try:
            ckpts = sorted([p for p in OUT_DIR.glob("checkpoint-*") if p.is_dir()],
                           key=lambda p: p.stat().st_mtime)
            for ckpt in ckpts:
                if str(ckpt) not in seen:
                    eval_checkpoint(ckpt)
                    with SEEN.open("a", encoding="utf-8") as f:
                        f.write(str(ckpt) + "\n")
                    seen.add(str(ckpt))
            time.sleep(60)
        except KeyboardInterrupt:
            print("\n[watch] stopped.")
            break

if __name__ == "__main__":
    main()
