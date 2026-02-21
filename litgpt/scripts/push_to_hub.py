import importlib
from pprint import pprint
from pathlib import Path
import torch
import transformers
from huggingface_hub import create_repo, upload_file
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


@torch.inference_mode()
def push_to_hub(
    model_path: Path = None,
    model_name: str = None,
    model_class_path: str = None,  # an example would be "transformers.LlamaForCausalLM"
    readme_path: Path = None,
    readme_only: bool = False,
    precision: str = "bf16-true",
    org: str = "tomg-group-umd",
    private: bool = True,
    hub_token: str = None,
    tokenizer_path: str = None,
    dry_run: bool = True,
    update_existing: bool = False,
    verbose: bool = False,
):
    pprint(locals())

    if verbose:
        transformers.logging.set_verbosity_info()

    repo_name = f"{org}/{model_name}"
    if tokenizer_path is None:
        tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if model_class_path is not None:
        module_path, class_name = model_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)
    else:
        ModelClass = AutoModelForCausalLM
    print(f"Loading model from {model_path}")
    print(f"Using model class {ModelClass}")

    model = ModelClass.from_pretrained(
        model_path,
        dtype={
            "16-true": torch.float16,
            "float16": torch.float16,
            "bf16-true": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "32-true": torch.float32,
            "float32": torch.float32,
        }[precision],
    )

    print(model)
    print(tokenizer)

    if dry_run == True:
        print("Dry run, not pushing to hub")
        exit()

    if hub_token is None:
        hub_token = os.environ.get("HF_HUB_TOKEN", None)

    create_repo(repo_name, private=private, token=hub_token, exist_ok=update_existing)

    if readme_path is not None:
        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_name,
            token=hub_token,
        )
        print(f"Pushed README to {repo_name}")
        if readme_only:
            print("Readme only, not push/updating model.")
            exit()

    model.push_to_hub(repo_name, use_temp_dir=True, token=hub_token)
    tokenizer.push_to_hub(repo_name, use_temp_dir=True, token=hub_token)

    print(f"Model pushed to {repo_name}")
