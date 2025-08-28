# Open-sci-ref 0.01 release

Open-sci-ref 0.01 is a research dense transformer model family that includes all the intermediate checkpoints trained on 8 different reference open datasets (C4, Pile, SlimPajama, FineWeb-Edu-1.4T (v1.0.0), DCLM-baseline, Nemotron-CC-HQ, HPLT-2.0 (english subset), and CommonCorpus) on various model (0.13B - 0.4B - 1.3B - 1.7B) and token (50B, 300B, 1T) scales. It is suppposed to serve as baselines for comparison and for studies on training dynamics. All artifacts are released under permissive Apache 2.0 licence.

See the [Open-sci-ref 0.01 research release blog](https://laion.ai/blog/open-sci-ref-001/) for more details.

Here, we provide overview of all the released artifacts and further infos for reproducing training and evaluation. The page is continuously updated.

## Obtaining the models

We release models and intermediate checkpoints under Apache 2.0 license via [HugginFace open-sci-ref collection](https://huggingface.co/collections/open-sci/open-sci-ref-001-685905e598be658fbcebff4f). The  collection contains subcollections for each reference dataset, holding 0.13B, 0.4B, 1.3B, 1.7B models trained on 300B (all datasets) and 1T (DCLM, FineWeb-Edu, Nemontron-CC-HQ) tokens of the given reference dataset. For C4, also 50B token scale is available. 


## Usage

### Load open-sci models using HF transformers

_Note: the reference baseline models are research base models, and while capable of text generation, those are not meant for conversation-based interaction. For this, multi-stage post-training should be applied, e.g., SFT, RLHF etc._

```python
# transformers >= 4.49.0
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "open-sci/open-sci-ref-v0.01-1.7b-nemotron-hq-1T-4096-rope_theta-100k"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)


prompts = ["Tokyo is", "Cologne is", "Freiburg is", "Helsinki is", "Tuebingen is"]

tokenizer.pad_token_id = tokenizer.eos_token_id
inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
output = model.generate(
    **inputs,
    max_length=48,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,
)
generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)
print("\n\n".join(generated_text))
```

## Logs

See [this HuggingFace dataset](https://huggingface.co/datasets/open-sci/open-sci-ref-0.01-logs) to acccess the logs.

## Citation
If you like this work, please cite:

```

@misc{opensciref001repo,
  author = {Nezhurina, Marianna and Franke, Joerg and Nakamura, Taishi, and Carstensen, Timur, and Ajroldi, Niccolò and Komulainen, Ville, and Salinas, David and Jitsev, Jenia},
  title = {Open-sci-ref-0.01 repository},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LAION-AI/open-sci-ref-0.01}},
}

@misc{opensciref001blog,
  author = {Nezhurina, Marianna and Franke, Joerg and Nakamura, Taishi, and Carstensen, Timur, and Ajroldi, Niccolò and Komulainen, Ville, and Salinas, David and Jitsev, Jenia},
  title = {{Open-sci-ref-0.01: open and reproducible reference baselines for language model and dataset comparison}},
  howpublished = {https://laion.ai/blog/open-sci-ref-001},
  year = {2025}
}
```
