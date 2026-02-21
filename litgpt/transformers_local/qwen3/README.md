Paper: https://arxiv.org/abs/2602.06019

This model is trained with a Multi-Token Prediction (MTP) objective. It features a custom generation API that allows for accelerated decoding without modifying the core transformer model and without auxiliary draft models or other complicated harness code.

## Model Description

Unlike standard autoregressive models that predict one token at a time, this model can predict multiple future tokens (`k`) in a single forward pass. It utilizes a custom `generate()` implementation allowing for accelerated inference by predicting `k` tokens at once as well as an adaptive mode (ConfAdapt) that dynamically adjusts the number of predicted tokens based on model confidence.

## Quick Start

To use this model, you **must** enable `trust_remote_code=True` to load the custom generation logic.

### Recommended Usage

If you do not pass `do_mtp=True`, the model defaults to standard Hugging Face generation behavior (1 token at a time).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "this/repo"

tokenizer = AutoTokenizer.from_pretrained(model_id)
# trust_remote_code is required for the custom model class
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto")

prompt = "Q: There are 15 trees in the grove..."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Decode using the ConfAdapt strategy with threshold 90%, max k = 16
output = model.generate(
    input_ids=inputs.input_ids,
    max_returned_tokens=128,     # Limits total length (prompt + gen).
    do_mtp=True,                 # Enable custom MTP logic.
    k_toks=16,                   # Maximum tokens to attempt per step.
    mask_id=151669,              # NOTE Must match actual mask token id for your model.
    eos_id=[151645, 151643],     # NOTE Must match actual stop token id, can handle multiple.
    strategy=["conf_adapt", 0.9] # See examples below.
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## MTP Generation API

To enable Multi-Token Prediction, pass `do_mtp=True` to the `generate()` function.

### Required Special Tokens
You must identify the correct special tokens for this specific model version:
* **Mask Token ID (`mask_id`):** The token used to mask future positions (e.g., `151669` for Qwen3 models in this collection).
* **EOS Token ID (`eos_id`):** The token ID(s) that stop generation. Note that in the examples, sometimes multiple tokens, eg. `eos_id=[151645, 151643]` should be passed if the model is sometimes inconsistent about which stop token it emits.

Check the tokenizer files to confirm the mask token id and examine the various stop tokens the model may emit. The main reason there can be more than one stop token is if there is both a pretraining "eos" token and a post training "end msg" in the tokenizer. Most modern models are post-trained with chat templates and these often introduce new stop tokens.

### Example 1: Fallback
Falls back to the huggingface default generate loop implementation.

```python
output = model.generate(
    input_ids=inputs.input_ids,
    do_mtp=False,            # Disable custom MTP logic
    # other kwargs
)
```

### Example 2: K=1 but MTP logic
Predicts 1 token per step, useful to compare against default huggingface behavior.
Omitting the `strategy` arg, i.e. `= None`, means that the Static strategy is used and `k` value is fixed at every step.

```python
output = model.generate(
    input_ids=inputs.input_ids,
    max_returned_tokens=128, # Limits total length (prompt + gen)
    do_mtp=True,             # Enable custom MTP logic
    k_toks=1,                # Predict 1 token per step
    mask_id=151669,          # REPLACE with actual mask token id for your model
    eos_id=151643,           # REPLACE with actual eos token id
)
```

### Example 3: Fixed-K Generation
Predicts exactly `k` tokens per step, fixed acceleration, possibly lossy.
Omitting the `strategy` arg, i.e. `= None`, means that the Static strategy is used and `k` value is fixed at every step.

```python
output = model.generate(
    input_ids=inputs.input_ids,
    max_returned_tokens=128, # Limits total length (prompt + gen)
    do_mtp=True,             # Enable custom MTP logic
    k_toks=3,                # Predict 3 tokens per step
    mask_id=151669,          # REPLACE with actual mask token id for your model
    eos_id=151643,           # REPLACE with actual eos token id
)
```

### Example 4: Adaptive Strategy (`conf_adapt`)
Dynamically accepts between 1 and `k` tokens based on a confidence threshold, variable acceleration, nearly lossless. Check implementaton for other possible strategies, some of which are experimental and not discussed in paper.

```python
# Strategy spec: ["conf_adapt", threshold_float]
# Stops predicting k tokens if confidence drops below 0.9
strategy = ["conf_adapt", 0.9]

output = model.generate(
    input_ids=inputs.input_ids,
    max_returned_tokens=128,
    do_mtp=True,
    k_toks=16,                # Maximum tokens to attempt per step
    mask_id=151669,           # REPLACE with actual mask token id for your model
    eos_id=[151645, 151643],  # Can handle a list of stop tokens
    strategy=strategy
)
```

## API Reference

When `do_mtp=True`, standard sampling arguments (like `do_sample`) are ignored.

| Argument | Type | Description |
| :--- | :--- | :--- |
| `do_mtp` | `bool` | Set to `True` to enable the MTP generation path. |
| `k_toks` | `int` | The number of future tokens to predict per forward pass. |
| `mask_id` | `int` | The token ID used to mask future positions. **Required if `k_toks > 1`.** |
| `eos_id` | `int` or `list` | The End-of-Sequence token ID(s). |
| `strategy` | `tuple` | Decoding strategy. Supports `("conf_adapt", threshold)` or `("random", weights)`. |
| `include_prompt` | `bool` | Whether to return the full sequence or just generated tokens. Default: `True`. |

**Note:** MTP generation currently supports single-example generation only (no batching).
