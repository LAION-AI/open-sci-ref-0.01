# Open-sci-ref 0.01 release

We release open-sci-ref 0.01 - a research dense transformer model family with all the intermediate checkpoints trained on 8 different reference open datasets (C4, Pile, SlimPajama, FineWeb-Edu-1.4T (v1.0.0), DCLM-baseline, Nemotron-CC-HQ, HPLT-2.0 (english subset), and CommonCorpus) on various model (0.13B - 0.4B - 1.3B - 1.7B) and token (50B, 300B, 1T) scales to serve as baselines for comparison and for studies on training dynamics. Our release includes all intermediate model weights, logs and training workflow code to enable easy learning procedure comparison on available reference scales and datasets and to support future research.

See the [Open-sci-ref 0.01 research release blog](https://laion.ai/blog/open-sci-ref-001/) for more details.

Here, we provide overview of all the released artifacts and further infos for reproducing training and evaluation. The page is continuously updated.

## Usage

### Load open-sci models using HF transformers

_Note: We do not advise you to use base language models for conversation-based interaction. For this, post-training should be applied, e.g., SFT, RLHF etc._

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
