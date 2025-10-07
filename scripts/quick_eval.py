# scripts/quick_eval.py
import os, glob, subprocess
from pathlib import Path

BASE = "models/indictrans2-indic-en-1B"
OUT_DIR = Path("outputs/hi_en_r8_v2")
ART = Path("artifacts/review2")
DEMO = ART / "demo_inputs.txt"
VAL_SRC = ART / "val50.src"
VAL_REF = ART / "val50.ref"
PY = os.environ.get("PYTHON", "python")

def run(cmd):
    return subprocess.run(cmd, check=True, text=True, capture_output=True)

def ensure_demo():
    if not DEMO.exists():
        ART.mkdir(parents=True, exist_ok=True)
        DEMO.write_text("\n".join([
            "कृपया दरवाज़ा बंद करें।",
            "क्या आप इसे कल तक भेज सकते हैं?",
            "मुझे लगता है कि समय सीमा बढ़ानी पड़ेगी।",
        ]), encoding="utf-8")

def main():
    ensure_demo()
    # base demo
    base_out = ART / "base_demo.txt"
    cmd_base = [PY, "scripts/translate_base_ip.py",
                "--base", BASE, "--src", "hi", "--tgt", "en",
                "--file", str(DEMO)]
    base_out.write_text(run(cmd_base).stdout, encoding="utf-8")

    # pick latest checkpoint
    ckpts = sorted(OUT_DIR.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        print("No checkpoints yet.")
        return
    ckpt = ckpts[-1]
    print("Evaluating:", ckpt.name)

    # formal/no vs informal/yes on demo
    fn = ART / f"{ckpt.name}_formal_no.txt"
    iy = ART / f"{ckpt.name}_informal_yes.txt"
    cmd1 = [PY, "scripts/translate_adapter_ip.py",
            "--base", BASE, "--adapter", str(ckpt),
            "--src", "hi", "--tgt", "en", "--style", "formal", "--simplify", "no",
            "--file", str(DEMO)]
    cmd2 = [PY, "scripts/translate_adapter_ip.py",
            "--base", BASE, "--adapter", str(ckpt),
            "--src", "hi", "--tgt", "en", "--style", "informal", "--simplify", "yes",
            "--file", str(DEMO)]
    fn.write_text(run(cmd1).stdout, encoding="utf-8")
    iy.write_text(run(cmd2).stdout, encoding="utf-8")

    # BLEU on val50 if available
    if VAL_SRC.exists() and VAL_REF.exists():
        hyp = ART / f"{ckpt.name}_val50.hyp"
        cmd3 = [PY, "scripts/translate_adapter_ip.py",
                "--base", BASE, "--adapter", str(ckpt),
                "--src", "hi", "--tgt", "en", "--style", "formal", "--simplify", "no",
                "--file", str(VAL_SRC)]
        hyp.write_text(run(cmd3).stdout, encoding="utf-8")
        try:
            import sacrebleu
            refs = [VAL_REF.read_text(encoding="utf-8").splitlines()]
            hyps = hyp.read_text(encoding="utf-8").splitlines()
            bleu = sacrebleu.corpus_bleu(hyps, refs)
            (ART / f"{ckpt.name}_val50.bleu.txt").write_text(f"BLEU = {bleu.score:.2f}\n", encoding="utf-8")
        except Exception as e:
            print("BLEU failed:", e)

if __name__ == "__main__":
    main()
