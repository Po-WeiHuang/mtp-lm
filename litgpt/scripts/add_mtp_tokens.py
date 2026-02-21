import os
from jsonargparse import CLI, set_parsing_settings
from tokenizers import AddedToken
from transformers import AutoModelForCausalLM, AutoTokenizer

from litgpt.scripts.convert_hf_checkpoint import convert_hf_checkpoint
from pathlib import Path


def add_mtp_tokens(
    model_path: Path = None,
    output_dir: Path = None,
    num_mtp_toks: int = 32,
    token_pattern: str = "<|mtp_special_token_{i}|>",
    add_pad_token: bool = True,
    trust_remote_code: bool = True,
    mean_resizing: bool = True,  # default is True for hf resize_token_embeddings
    pad_to_multiple_of: int = 128,
    model_name: str = None, # must exist in litgpt/config.py
):
    """
    Adds a specified number of numbered special tokens to a model based on a pattern.

    Args:
        model_path: Hub ID or local path.
        output_dir: Path to save the modified artifacts.
        num_mtp_toks: How many tokens to add.
        token_pattern: The f-string pattern for tokens, e.g., "<|extra_{i}|>".
        trust_remote_code: Allow custom model code.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )

    # Construct the list using the provided pattern
    new_tokens = [
        AddedToken(
            token_pattern.format(i=i),
            lstrip=False,
            rstrip=False,
            single_word=False,
            normalized=False,
            special=True,
        )
        for i in range(num_mtp_toks)
    ]

    # Add to tokenizer
    num_added = tokenizer.add_special_tokens(
        special_tokens_dict={"additional_special_tokens": new_tokens},
        replace_additional_special_tokens=False,
    )
    if add_pad_token and tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        num_added += 1

    new_vocab_size = len(tokenizer)
    print(f"Added {num_added} tokens. New tokenizer length: {new_vocab_size}")
    
    os.makedirs(output_dir, exist_ok=True)

    tokenizer.save_pretrained(output_dir)
    print(f"Successfully saved tokenizer to {output_dir}")

    # verify loading
    reloaded_tokenizer = AutoTokenizer.from_pretrained(output_dir)

    print(f"Loading from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )

    # Logic: Only resize if the new vocab exceeds the physical matrix capacity
    embedding_layer = model.get_input_embeddings()
    current_matrix_size = embedding_layer.weight.shape[0]

    if new_vocab_size > current_matrix_size:
        print(f"Resizing: {new_vocab_size} > {current_matrix_size}")
        # This handles both input embeddings and LM head (if they exist/are tied)
        model.resize_token_embeddings(new_vocab_size, mean_resizing=mean_resizing, pad_to_multiple_of=pad_to_multiple_of)
    else:
        print(
            f"No resize required: {new_vocab_size} fits in current matrix of {current_matrix_size}"
        )

    model.save_pretrained(output_dir)
    print(f"Successfully saved model to {output_dir}")
    
    # verify loading one more time
    reloaded_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    # this will probably throw a message this second time but it's a bug
    # https://github.com/huggingface/transformers/pull/42605

    # now try doing the conversion to litgpt
    if model_name is not None:
        print("Converting checkpoint files to LitGPT format.")
        convert_hf_checkpoint(checkpoint_dir=output_dir, model_name=model_name)


if __name__ == "__main__":
    set_parsing_settings(
        config_read_mode_urls_enabled=True,
        docstring_parse_attribute_docstrings=True,
    )

    CLI(add_mtp_tokens)
