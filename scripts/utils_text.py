import re
import unicodedata
from typing import Literal

Style = Literal["formal", "informal"]
YN = Literal["yes", "no"]


def normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def build_prompt(src_lang: str, tgt_lang: str, style: Style, simplify: YN, text: str) -> str:
    return f"{src_lang} {tgt_lang} {style} {simplify} ||| {text}"
