# fmt: off
from jsonargparse import CLI, set_parsing_settings

from functools import partial
import os
import time
import json
from datasets import load_dataset

from multiprocessing import Pool

from transformers import AutoTokenizer

from args_data import DataSource, DataSources, P2PConfig

import torch.distributed as dist
from torch.distributed import timedelta


def extract_text_field(dataset, text_field="text", tokenizer=None, num_proc=None):
    fallback_fields = ["content", "body", "message", "document"]

    if text_field in dataset.column_names:
        if text_field != "text":
            dataset = dataset.rename_column(text_field, "text")
        return dataset.select_columns(["text"])

    for field in fallback_fields:
        if field in dataset.column_names:
            dataset = dataset.rename_column(field, "text")
            return dataset.select_columns(["text"])

    raise ValueError(f"No text field found. Available: {dataset.column_names}")


def extract_text_join_cols(dataset, text_fields, add_bos=False, tokenizer=None, num_proc=None):
    def join_text_fields(example):
        joined_text = "\n\n".join(
            [example[field] for field in text_fields if field in example]
        )
        if add_bos:
            assert tokenizer is not None, "Tokenizer must be provided to add bos token."
            bos_str = tokenizer.bos_token if tokenizer.bos_token is not None else ""
            joined_text = f"{bos_str}{joined_text}"
        return {"text": joined_text}

    ds = dataset.map(
        join_text_fields,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Joining text fields",
    )
    return ds.select_columns(["text"])


def join_query_resp(dataset, add_bos=False, tokenizer=None, num_proc=None):
    fields = ["query", "response"]
    return extract_text_join_cols(dataset, fields, add_bos=add_bos, tokenizer=tokenizer, num_proc=num_proc)

def query_resp_chat(dataset, tokenizer=None, num_proc=None):
    assert tokenizer is not None, "Tokenizer must be provided for this chat template."

    def format_chat(example, do_tokenize=False):
        messages = [
            {"role": "user", "content": example["query"]},
            {"role": "assistant", "content": example["response"]},
        ]
        return {
            "text": tokenizer.apply_chat_template(
                messages, tokenize=do_tokenize
            )
        }

    ds = dataset.map(
        format_chat,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Formatting query-response as chat",
    )
    return ds.select_columns(["text"])


def extract_chat_dbl_nl(dataset, tokenizer=None, num_proc=None):
    # chats are a top level list of dicts with 'role' and 'content'
    # we will just extract all the content joined by a dbl newline
    # no robustness in handling, designed to fail if structure is not as expected
    def format_chat(example):
        messages = example["messages"]
        chat_text = "\n\n".join([msg["content"] for msg in messages])
        return {"text": chat_text}

    ds = dataset.map(
        format_chat,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Formatting with chat template",
    )
    return ds.select_columns(["text"])


def convo_from_human_from_gpt(dataset, tokenizer=None, num_proc=None):
    assert tokenizer is not None, "Tokenizer must be provided for this chat template."

    def format_chat(example, do_tokenize=False):
        messages = example["conversations"]
        renamed_messages = []
        for msg in messages:
            if msg["from"] == "human":
                renamed_messages.append({"role": "user", "content": msg["value"]})
            elif msg["from"] == "gpt":
                renamed_messages.append({"role": "assistant", "content": msg["value"]})
            else:
                raise ValueError(f"Unknown message format: {msg}")
        return {
            "text": tokenizer.apply_chat_template(
                renamed_messages, tokenize=do_tokenize
            )
        }

    ds = dataset.map(
        format_chat,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Formatting convo from human and gpt",
    )
    return ds.select_columns(["text"])

def convo_from_bos_only(dataset, tokenizer=None, num_proc=None):
    assert tokenizer is not None, "Tokenizer must be provided for this chat template."

    def format_chat(example, do_tokenize=False):
        messages = example["conversations"]
        joined_msgs = '\n\n'.join([msg['value'] for msg in messages])
        bos_str = tokenizer.bos_token if tokenizer.bos_token is not None else ""
        text = f"{bos_str}{joined_msgs}"
        return {"text": text}

    ds = dataset.map(
        format_chat,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Formatting convo from bos only",
    )
    return ds.select_columns(["text"])


template_fn_registry = {
    "default_text": extract_text_field,
    "default_chat": extract_chat_dbl_nl,
    "join_query_resp": join_query_resp,
    "join_query_resp_bos": partial(join_query_resp, add_bos=True),
    "query_resp_chat": query_resp_chat,
    "convo_from_human_from_gpt": convo_from_human_from_gpt,
    "convo_from_bos_only": convo_from_bos_only,
}


def shard_and_save_to_parquet(dataset, output_path, num_shards, shard_index=0):
    """
    Shard the dataset and save it to Parquet format.
    """
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # Shard the dataset save to Parquet
    dataset.shard(num_shards=num_shards, index=shard_index).to_parquet(output_path)


def reload_parquet_dataset(dataset_dir, num_proc=None):
    """
    Reload a dataset from Parquet format.
    """

    return load_dataset(
        "parquet", data_dir=dataset_dir, split="train", num_proc=num_proc
    )


def pull_raw_datasets(
    script_configs: P2PConfig = P2PConfig(),
    srcs_config_file: str = None,
):

    download_configs = DataSources.from_yaml(srcs_config_file).sources
    
    if script_configs.raw_dir is None:
        script_configs.raw_dir = os.path.join(script_configs.output_dir, "raw")
    cache_dir = os.path.join(script_configs.output_dir, "cache")
    cache_locks = os.path.join(cache_dir, "cache_locks")
    
    # assert os.path.exists(script_configs.output_dir), f"Directory {script_configs.output_dir} does not exist."
    os.makedirs(script_configs.output_dir, exist_ok=True)
    os.makedirs(script_configs.raw_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(cache_locks, exist_ok=True)

    dist.init_process_group("nccl", device_id=int(os.environ["LOCAL_RANK"]), rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]), timeout=timedelta(seconds=script_configs.max_wait_time))

    array_task_id = int(os.getenv("SLURM_PROCID", "0"))
    array_task_count = int(os.getenv("SLURM_NTASKS", "1"))

    # Each array should write it's lockfile in the cache locks directory
    lockfile = os.path.join(cache_locks, f"pull_lock_{array_task_id:03d}.lock")
    if not os.path.exists(lockfile):
        with open(lockfile, "w") as f:
            f.write(
                "This is a lock file to prevent process continuation during data pull.\n"
            )
    else:
        if not script_configs.resume:
            # If the lock file already exists and we are not resuming, raise an error
            # This prevents multiple tasks from running at the same time
            raise ValueError(f"Lock file {lockfile} already exists.")
        else:
            # add a line to the lock file to indicate that we are resuming
            with open(lockfile, "a") as f:
                f.write(
                    f"Resuming task {array_task_id} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                )

    if array_task_id == 0:

        os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"] = cache_dir
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        if script_configs.hf_tokenizer is not None:
            tokenizer = AutoTokenizer.from_pretrained(script_configs.hf_tokenizer)
        elif hf_tokenizer is not None:
            tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer)
        else:
            tokenizer = None

        for dataset_name, dataset_config in download_configs.items():
            name_or_path = dataset_config.address
            config_name = dataset_config.subset
            split_name = dataset_config.split

            
            template_fn_name = dataset_config.template_fn if dataset_config.template_fn is not None else script_configs.template_fn

            if not name_or_path:
                print(f"Skipping {dataset_name}: no name_or_path specified.")
                continue

            print(f"Downloading {dataset_name}...")

            retries = 0
            while retries < script_configs.max_retries:
                try:
                    # Attempt to load the dataset
                    print(
                        f"Try number {retries + 1} for {dataset_name} with {script_configs.num_proc_pull} procs ..."
                    )
                    dataset = load_dataset(
                        name_or_path,
                        name=config_name,
                        split=split_name,
                        num_proc=script_configs.num_proc_pull,
                        cache_dir=cache_dir,
                    )
                    print(f"Original dataset loaded: {dataset_name}")
                    print(dataset)
                    break  # If successful, exit the retry loop
                except Exception as e:
                    print(f"Error loading {dataset_name}: {e}")
                    retries += 1
                    if retries >= script_configs.max_retries:
                        print(f"Failed to load {dataset_name} after {script_configs.max_retries} retries.")
                        break
                    print(f"Retrying in {script_configs.retry_patience} seconds...")
                    time.sleep(script_configs.retry_patience)

            
            template_fn = template_fn_registry[template_fn_name]
            print(
                f"Using template fn {template_fn_name} for {dataset_name}"
            )
            if tokenizer is not None:
                text_dataset = template_fn(
                    dataset, tokenizer=tokenizer, num_proc=script_configs.num_proc_pull
                )
            else:
                text_dataset = template_fn(dataset, num_proc=script_configs.num_proc_pull)

            # preview one row of the dataset
            print(
                f"Preview one row from text dataset extracted using {template_fn_name} for {dataset_name}:"
            )
            print(text_dataset[0]["text"])

            output_subdir = os.path.join(script_configs.raw_dir, f"{dataset_name}")
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            output_path = os.path.join(output_subdir, f"{dataset_name}.parquet")

            print(f"Saving {dataset_name} to {output_path}...")

            mp_pool = Pool(processes=script_configs.num_proc_pull)

            # use the pool to shard and save the dataset
            shard_tasks = [
                (
                    text_dataset,
                    output_path.replace(".parquet", f"_{i}.parquet"),
                    script_configs.num_raw_shards,
                    i,
                )
                for i in range(script_configs.num_raw_shards)
            ]
            mp_pool.starmap(shard_and_save_to_parquet, shard_tasks)
            mp_pool.close()
            mp_pool.join()
            # Ensure the output path is correct

            print(f"Dataset saved to {output_path} with {script_configs.num_raw_shards} shards.")

            # test that the dataset is reloadable
            try:
                print(f"Reloading {dataset_name} from parquet at {output_subdir}...")
                reloaded_dataset = reload_parquet_dataset(output_subdir)
                print(f"Reloaded dataset: {dataset_name}")
                print(reloaded_dataset)
            except Exception as e:
                print(f"Error reloading {dataset_name} from parquet: {e}")

    # now each task can remove its own lock file
    os.remove(lockfile)
    wait_time = 0
    last_logged_wait = time.time()

    # wait for all other tasks to finish
    while any(
        os.path.exists(os.path.join(cache_locks, f"pull_lock_{i:03d}.lock"))
        for i in range(array_task_count)
    ):
        if time.time() - last_logged_wait > (10 * script_configs.wait_interval):
            print("Waiting for other tasks to finish...")
            last_logged_wait = time.time()
        time.sleep(script_configs.wait_interval)
        wait_time += script_configs.wait_interval
        if wait_time > script_configs.max_wait_time:
            raise TimeoutError(
                f"Waiting for other tasks to finish took too long: {wait_time} seconds."
            )
    # then we can remove the cache if requested
    print("All tasks finished, removing pull cache if requested...")
    if script_configs.rm_cache:
        if array_task_id == 0:
            print("Removing pull cache on rank 0...")
            import shutil

            shutil.rmtree(cache_dir, ignore_errors=False)
            print("pull cache removed.")

    print(f"Task {array_task_id} finished data pull.")
    
    dist.barrier()
    dist.destroy_process_group()


# This script pulls raw pretraining datasets based on a configuration file.
# It downloads datasets from Hugging Face and saves them in Parquet format.
# Usage: python pull_raw_datasets.py

# Run the function to pull datasets
# pull_raw_datasets()

if __name__ == "__main__":

    set_parsing_settings(
        config_read_mode_urls_enabled=True,
        docstring_parse_attribute_docstrings=True,
    )

    CLI(pull_raw_datasets)
