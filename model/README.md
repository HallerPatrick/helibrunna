---
language: 
- en
license: mit
---

# An xLSTM Model

Trained with [Helibrunna](https://github.com/PatrickHaller/helibrunna) (fork)

To use this model the [xLSTM](https://github.com/NX-AI/xlstm) package is required. We recommend to install 
it locally with conda:

```bash
git clone https://github.com/NX-AI/xlstm
cd xlstm
conda env create -n xlstm -f environment_pt220cu121.yaml
conda activate xlstm
```


## Usage 

```python
from transformers import AutoModelForCasualLM, AutoTokenizer

model_name_or_path = "PatrickHaller/xlstm_dummy"

model = AutoModelForCasualLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")
output = model.generate(input_ids, max_length=100, temperature=0.7, do_sample=True)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)

```
