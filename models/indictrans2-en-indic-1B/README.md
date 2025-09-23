---
language:
- as
- bn
- brx
- doi
- en
- gom
- gu
- hi
- kn
- ks
- kas
- mai
- ml
- mr
- mni
- mnb
- ne
- or
- pa
- sa
- sat
- sd
- snd
- ta
- te
- ur
language_details: >-
  asm_Beng, ben_Beng, brx_Deva, doi_Deva, eng_Latn, gom_Deva, guj_Gujr,
  hin_Deva, kan_Knda, kas_Arab, kas_Deva, mai_Deva, mal_Mlym, mar_Deva,
  mni_Beng, mni_Mtei, npi_Deva, ory_Orya, pan_Guru, san_Deva, sat_Olck,
  snd_Arab, snd_Deva, tam_Taml, tel_Telu, urd_Arab
tags:
- indictrans2
- translation
- ai4bharat
- multilingual
license: mit
datasets:
- flores-200
- IN22-Gen
- IN22-Conv
metrics:
- bleu
- chrf
- chrf++
- comet
inference: false
---

# IndicTrans2

This is the model card of IndicTrans2 En-Indic 1.1B variant.

Here are the [metrics](https://drive.google.com/drive/folders/1lOOdaU0VdRSBgJEsNav5zC7wwLBis9NI?usp=sharing) for the particular checkpoint.

Please refer to `Appendix D: Model Card` of the [preprint](https://arxiv.org/abs/2305.16307) for further details on model training, intended use, data, metrics, limitations and recommendations.


### Usage Instructions

Please refer to the [github repository](https://github.com/AI4Bharat/IndicTrans2/tree/main/huggingface_interface) for a detail description on how to use HF compatible IndicTrans2 models for inference.

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
# recommended to run this on a gpu with flash_attn installed
# don't set attn_implemetation if you don't have flash_attn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

src_lang, tgt_lang = "eng_Latn", "hin_Deva"
model_name = "ai4bharat/indictrans2-en-indic-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    torch_dtype=torch.float16, # performance might slightly vary for bfloat16
    attn_implementation="flash_attention_2"
).to(DEVICE)

ip = IndicProcessor(inference=True)

input_sentences = [
    "When I was young, I used to go to the park every day.",
    "We watched a new movie last week, which was very inspiring.",
    "If you had met me at that time, we would have gone out to eat.",
    "My friend has invited me to his birthday party, and I will give him a gift.",
]

batch = ip.preprocess_batch(
    input_sentences,
    src_lang=src_lang,
    tgt_lang=tgt_lang,
)

# Tokenize the sentences and generate input encodings
inputs = tokenizer(
    batch,
    truncation=True,
    padding="longest",
    return_tensors="pt",
    return_attention_mask=True,
).to(DEVICE)

# Generate translations using the model
with torch.no_grad():
    generated_tokens = model.generate(
        **inputs,
        use_cache=True,
        min_length=0,
        max_length=256,
        num_beams=5,
        num_return_sequences=1,
    )

# Decode the generated tokens into text
generated_tokens = tokenizer.batch_decode(
    generated_tokens,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
)

# Postprocess the translations, including entity replacement
translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

for input_sentence, translation in zip(input_sentences, translations):
    print(f"{src_lang}: {input_sentence}")
    print(f"{tgt_lang}: {translation}")
```

### 📢 Long Context IT2 Models
- New RoPE based IndicTrans2 models which are capable of handling sequence lengths **upto 2048 tokens** are available [here](https://huggingface.co/collections/prajdabre/indictrans2-rope-6742ddac669a05db0804db35)
- These models can be used by just changing the `model_name` parameter. Please read the model card of the RoPE-IT2 models for more information about the generation.
- It is recommended to run these models with `flash_attention_2` for efficient generation. 


### Citation

If you consider using our work then please cite using:

```
@article{gala2023indictrans,
title={IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages},
author={Jay Gala and Pranjal A Chitale and A K Raghavan and Varun Gumma and Sumanth Doddapaneni and Aswanth Kumar M and Janki Atul Nawale and Anupama Sujatha and Ratish Puduppully and Vivek Raghavan and Pratyush Kumar and Mitesh M Khapra and Raj Dabre and Anoop Kunchukuttan},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=vfT4YuzAYA},
note={}
}
```
