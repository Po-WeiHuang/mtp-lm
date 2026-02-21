# fmt: off
import time
import os
import glob
import math
from pathlib import Path
import contextlib
import random

import itertools
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq

from typing import List, Optional
import yaml

import multiprocess
import datasets
from datasets import DatasetDict

from transformers import AutoTokenizer, BatchEncoding
from jsonargparse import CLI, set_parsing_settings

from args_data import DataSource, DataSources, P2PConfig

import torch.distributed as dist
from torch.distributed import timedelta


def is_valid_parquet_file(file_path):
    # Parquet files end with a 4-byte magic number: 'PAR1'
    with open(file_path, "rb") as f:
        f.seek(-4, os.SEEK_END)
        footer = f.read(4)
        if footer == b"PAR1":
            metadata = pq.read_metadata(file_path)
            schema = metadata.schema
            if "text" in schema.names:
                return True
            else:
                print(f"Unexpected schema in {file_path}: {schema.names}")
        else:
            print(f"Invalid footer in {file_path}: {footer}")
        return False


def weighted_file_selection(files, weight):
    return files * math.floor(weight) + random.sample(
        files, k=round((weight % 1) * len(files))
    )


def _generate_tables(self, files):
    for file_idx, file in enumerate(itertools.chain.from_iterable(files)):
        with open(file, "rb") as f:
            try:
                parquet_file = pq.ParquetFile(f)
                if parquet_file.metadata.num_row_groups > 0:
                    batch_size = (
                        self.config.batch_size
                        or parquet_file.metadata.row_group(0).num_rows
                    )
                    for batch_idx, record_batch in enumerate(
                        parquet_file.iter_batches(
                            batch_size=batch_size, columns=self.config.columns
                        )
                    ):
                        pa_table = pa.Table.from_batches([record_batch])
                        yield f"{file_idx}_{batch_idx}", self._cast_table(pa_table)
            except Exception as e:
                print(f"Failed to read file '{file}' with error {type(e)}: {e}")


datasets.packaged_modules.parquet.Parquet._generate_tables = _generate_tables


def process_single_shard(args):
    dataset, index, num_shards, output_template, rank, size, resume = args
    output_path = output_template.format(rank=rank, index=index)

    # As a failsafe, we create a modified temp template for the output path to use
    # during the initial write out. then when it completes, move it to the final path
    # replace .parquet with .parquet.tmp
    # in the scenario where it dies during the write, we can resume
    output_path_tmp = output_path.replace(".parquet", ".parquet.tmp")

    if Path(output_path_tmp).exists():
        try:
            print(
                f"Encountered a temp file during parquet write: {output_path_tmp}, removing it."
            )
            Path(output_path_tmp).unlink()
        except Exception as e:
            print(f"Failed to remove temporary file {output_path_tmp}: {e}")
            raise

    if not (Path(output_path).exists() and resume):
        shard = dataset.shard(index=index, num_shards=num_shards, contiguous=True)
        shard.to_parquet(output_path_tmp, batch_size=len(shard))
        # Now move the temporary file to the final output path
        Path(output_path_tmp).rename(output_path)
        time.sleep(5)
        # check if the file was moved successfully
        if not Path(output_path).exists():
            raise FileNotFoundError(
                f"Failed to move {output_path_tmp} to {output_path}."
            )
        if rank is not None:
            print(f"Rank {rank}/{size}: Saved shard {index + 1}/{num_shards}.")
        else:
            print(f"Saved shard {index + 1}/{num_shards}")


def process_shards(
    dataset, num_shards, output_template, resume, num_proc, rank=None, size=None
):
    if resume:
        unfinished_shards = [
            index
            for index in range(num_shards)
            if not Path(output_template.format(rank=rank, index=index)).exists()
        ]
    else:
        unfinished_shards = range(num_shards)

    with multiprocess.Pool(num_proc) as pool:
        args_list = [
            (dataset, index, num_shards, output_template, rank, size, resume)
            for index in unfinished_shards
        ]
        pool.map(process_single_shard, args_list)


def print_examples(
    array_task_id,
    processed_dir,
    hf_tokenizer,
    parquet_index=0,
    print_n=5,
    print_tok_limit=20,
    print_char_limit=500,
):
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer)
    example = os.path.join(
        processed_dir,
        "train",
        f"{array_task_id:03d}_{parquet_index:05d}.parquet",
    )
    parquet_file = pq.ParquetFile(example)
    print("Schema:")
    print(parquet_file.schema)

    first_batch = next(parquet_file.iter_batches(batch_size=print_n))
    print(f"\nFirst {print_tok_limit} entries from first {print_n} rows:")
    values = first_batch["input_ids"].to_pylist()
    for seq in values:
        print(seq[:print_tok_limit])

    for i in range(print_n):
        print(f"\nDecoded text from row {i}:")
        print(
            tokenizer.decode(
                first_batch["input_ids"][i].as_py(), skip_special_tokens=False
            )[:print_char_limit]
        )


def parquet_to_parquet_tokenization(
    script_configs: P2PConfig = P2PConfig(),
    srcs_config_file: str = None,
):
    data_sources = DataSources.from_yaml(srcs_config_file)

    random.seed(1597)
    if not script_configs.log_tokenization_progress:
        datasets.disable_progress_bars()

    if script_configs.raw_dir is None:
        script_configs.raw_dir = os.path.join(script_configs.output_dir, "raw")
    cache_dir = os.path.join(script_configs.output_dir, "cache")
    cache_locks = os.path.join(cache_dir, "cache_locks")
    processed_dir = os.path.join(script_configs.output_dir, "processed")
    
    # assert os.path.exists(script_configs.output_dir), f"Directory {script_configs.output_dir} does not exist."
    os.makedirs(script_configs.output_dir, exist_ok=True)
    assert os.path.exists(script_configs.raw_dir), f"Raw data directory {script_configs.raw_dir} does not exist."
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(cache_locks, exist_ok=True)

    os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"] = cache_dir
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    cache_locks = os.path.join(cache_dir, "cache_locks")
    os.makedirs(cache_locks, exist_ok=True)

    if script_configs.hf_tokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(script_configs.hf_tokenizer)
        hf_tokenizer_path = script_configs.hf_tokenizer
    else:
        raise ValueError("HF Tokenizer path must be provided in config.")

    #
    # - Phase I: Tokenizer parquet -> parquet via Huggingface datasets functionality (and patience) ####################
    #

    parquet_files = []
    os.makedirs(
        os.path.join(processed_dir, "train"),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(processed_dir, "val"),
        exist_ok=True,
    )
    source_reference = {}

    for name, source in data_sources.sources.items():
        try:
            files = glob.glob(f"{script_configs.raw_dir}/{name}/**/*.parquet", recursive=True)[
                :script_configs.limiter
            ]
        except Exception:
            files = []
        valid_files = [p for p in files if is_valid_parquet_file(p)]
        weighed_files = weighted_file_selection(valid_files, source.weight)
        print(
            f"Gathered {len(files):<4} parquet sources from {name:<30}, {len(valid_files):<4} were valid, "
            f"reweighing to {len(weighed_files):<4} with w={source.weight}."
        )
        parquet_files += weighed_files
        for source_file in weighed_files:
            source_reference[source_file] = name

    dist.init_process_group("nccl", device_id=int(os.environ["LOCAL_RANK"]), rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]), timeout=timedelta(seconds=script_configs.max_wait_time))

    # Instead of slurm arrays, we're using a multinode job with one task per node.
    # We can check that the number of nodes and the number of tasks match.
    # if so, the procid can be used as the array task id, and the number of tasks as the array task count.
    slurm_nnodes = int(os.getenv("SLURM_NNODES", "1"))
    slurm_ntasks = int(os.getenv("SLURM_NTASKS", "1"))
    slurm_ntasks_per_node = int(os.getenv("SLURM_NTASKS_PER_NODE", "1"))
    array_task_id = int(os.getenv("SLURM_PROCID", "0"))
    array_task_count = slurm_ntasks
    # if slurm_nnodes != slurm_ntasks:
    #     raise ValueError(
    #         f"Number of nodes ({slurm_nnodes}) does not match number of tasks ({slurm_ntasks})."
    #     )
    # we'll actually handle more than one task per node, but we need to drop the process count accordingly
    print(f"Slurm nodes: {slurm_nnodes}, tasks: {slurm_ntasks}, tasks per node: {slurm_ntasks_per_node}.")


    print(f"Array task ID: {array_task_id}, Array task count: {array_task_count}")

    if script_configs.num_proc is None:
        
        script_configs.num_proc = multiprocess.cpu_count()
        # first we round it to a mult of the slurm tasks
        script_configs.num_proc = max(1, int(script_configs.num_proc // slurm_ntasks_per_node) * slurm_ntasks_per_node)
        if script_configs.num_proc % slurm_ntasks_per_node != 0:
            raise ValueError(
                f"Number of processes ({script_configs.num_proc}) is still somehow not divisible by number of tasks ({slurm_ntasks_per_node})."
            )
        script_configs.num_proc = script_configs.num_proc // slurm_ntasks_per_node
        # now we round it down a bit after division performed to avoid clobbering issues
        script_configs.num_proc = max(1, int(script_configs.num_proc * 0.9 // 2) * 2)

    script_configs.num_proc_map_to_block = (
        script_configs.num_proc // 2 if script_configs.num_proc_map_to_block is None else script_configs.num_proc_map_to_block
    )
    print(
        f"Number of CPUs available: {multiprocess.cpu_count()}, each task using {script_configs.num_proc} processes for tokenization, {script_configs.num_proc_map_to_block} for mapping to blocks."
    )

    # Each array should write it's lockfile in the cache locks directory
    lockfile = os.path.join(cache_locks, f"p2p_tok_lock_{array_task_id:03d}.lock")
    if not os.path.exists(lockfile):
        with open(lockfile, "w") as f:
            f.write(
                "This is a lock file to prevent cache removal during tokenization.\n"
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

    # Shuffle the parquet files before assigning work if requested
    if script_configs.shuffle_filenames:
        random.seed(233)
        random.shuffle(parquet_files)
    parquet_files = parquet_files[array_task_id::array_task_count]

    print(
        f"Array task {array_task_id} out of {array_task_count} processing {len(parquet_files)} source files."
    )

    datasets.utils._filelock.FileLock = contextlib.nullcontext

    monolith = datasets.load_dataset(
        "parquet",
        data_files=parquet_files,
        columns=["text"],
        streaming=False,
        cache_dir=cache_dir,
        num_proc=script_configs.num_proc,
        verification_mode="no_checks",
    )["train"]

    # Get all sizes
    file_sizes = []
    for f in parquet_files:
        try:
            size = pq.read_metadata(f).num_rows
            file_sizes.append(size)
        except Exception as e:
            raise ValueError(f"Cannot read size of {f}: {e}")

    # Create flags
    keep_all_values = []
    for file, size in zip(parquet_files, file_sizes):
        value = bool(data_sources.sources[source_reference[file]].every_token_is_sacred)
        keep_all_values.extend([value] * size)

    # Strict verification
    if len(keep_all_values) != len(monolith):
        raise ValueError(
            f"Size mismatch: flags={len(keep_all_values)} vs dataset={len(monolith)}"
        )

    monolith = monolith.add_column("keep_all", keep_all_values)  # type: ignore
    # Add right after loading monolith:
    print(f"Dataset loaded with {len(monolith)} rows")
    print(f"File sizes: {list(zip(parquet_files, file_sizes))}")
    print(f"Total size from files: {sum(file_sizes)}")

    # After adding keep_all column:
    true_count = sum(1 for x in keep_all_values if x)
    print(
        f"Keep_all distribution: {true_count} True, {len(keep_all_values) - true_count} False"
    )
    # Add after everything:
    print(
        "-----------------------------------------------------Examples----------------------------------------------"
    )
    for i in range(min(5, len(monolith))):
        print(f"Row {i}: text={monolith[i]['text']}")
        print(f"Row {i}: keep_all={monolith[i]['keep_all']}")
    print(
        "-----------------------------------------------------------------------------------------------------------"
    )
    print(f"Monolith loaded with features {monolith.features}.")

    def tokenize_per_example(batch):
        input_batch = []
        for text, keep_all in zip(batch["text"], batch["keep_all"]):
            if keep_all:
                input_batch += [text]
            else:
                input_batch += [text[: (16 * script_configs.target_block_size)]]
        encoded_batch: BatchEncoding = tokenizer(
            input_batch, add_special_tokens=False, padding=False, truncation=False
        ).input_ids
        output_batch = []
        for encoding, keep_all in zip(encoded_batch, batch["keep_all"]):
            if keep_all:
                output_batch.append(encoding)
            else:
                output_batch.append(encoding[: script_configs.target_block_size + 64])
        return {"token_ids": output_batch}

    tokenized_monolith = monolith.map(
        tokenize_per_example,
        batch_size=1024,
        batched=True,
        num_proc=script_configs.num_proc,
        remove_columns=["text"],
        desc="Tokenizing monolith",
    )
    print(f"Monolith tokenized with features {tokenized_monolith.features}.")

    #
    # - Phase II: Full shuffle based on datasets shuffler ##############################################################
    #

    shuffled_tokenized_monolith = tokenized_monolith.shuffle(seed=777777777777)

    #
    # - Phase III: Map to blocks (of size S+1), measure incineration rate. #############################################
    #

    def map_to_block_operation(batch):
        """Desirable properties: Only block starts are used, blocks are filled from random rows from all datasets."""
        mined_blocks = []
        # incinerated_tokens = # should probably count this later, but we can estimate from loss in number of tokens
        # Concatenate batch contents:
        current_block = []
        last_block_had_keep_all = False
        for example, keep_all in zip(batch["token_ids"], batch["keep_all"]):
            if len(current_block) >= script_configs.target_block_size + 1:
                if last_block_had_keep_all:
                    while len(current_block) >= script_configs.target_block_size + 1:
                        mined_blocks.append(current_block[: (script_configs.target_block_size + 1)])
                        current_block = current_block[(script_configs.target_block_size + 1) :]
                else:
                    mined_blocks.append(current_block[: (script_configs.target_block_size + 1)])
                    current_block = []
            current_block += example
            last_block_had_keep_all = keep_all
        return {"input_ids": mined_blocks}

    def map_to_block_no_pack(batch):
        """Desirable properties: Only block starts are used, blocks are filled from random rows from all datasets."""
        mined_blocks = []
        # incinerated_tokens = # should probably count this later, but we can estimate from loss in number of tokens
        current_block = []
        for example in batch["token_ids"]:
            # instead of checking whether buffer is past the max, we just write if buffer has anything
            if len(current_block) > 0:
                mined_blocks.append(current_block[: (script_configs.target_block_size + 1)])
                current_block = []
            current_block += example
        return {"input_ids": mined_blocks}

    pack_setting_to_fn_map = {
        "default": map_to_block_operation,
        "no_pack": map_to_block_no_pack,
    }
    map_to_block_fn = pack_setting_to_fn_map[script_configs.pack_setting]

    blockset = shuffled_tokenized_monolith.map(
        map_to_block_fn,
        batch_size=4096,
        batched=True,
        num_proc=script_configs.num_proc_map_to_block,
        remove_columns=shuffled_tokenized_monolith.column_names,
        desc=f"Mapping to blocks using {map_to_block_fn.__name__}.",
    )
    print(f"Blockset written with features {blockset.features}.")
    num_blocks = len(blockset)
    print(f"Number of valid blocks: {num_blocks}")
    print(
        f"This is equivalent to {num_blocks * (script_configs.target_block_size + 1) / 1e9:7.4f}B tokens (assuming full blocks of {script_configs.target_block_size + 1} toks)."
    )
    print(
        f"Total token count is likely around {array_task_count * num_blocks * (script_configs.target_block_size + 1) / 1e12:7.4f}T tokens"
    )
    #
    # - Phase IV: Additional Shuffle ##############################################################################
    #

    shuffled_blockset = blockset.shuffle(seed=33333333333)

    #
    # - Phase V: Write back into parquet ##############################################################################
    #
    num_shards = math.ceil(script_configs.target_shard_num / array_task_count)
    print(f"Preparing for {num_shards} shards.")

    if script_configs.train_split_pct < 1.0:
        blockdict = shuffled_blockset.train_test_split(
            train_size=script_configs.train_split_pct, shuffle=False
        )
    else:
        blockdict = DatasetDict({"train": shuffled_blockset})

    output_template_train = os.path.join(
        processed_dir, "train", "{rank:03d}_{index:05d}.parquet"
    )
    output_template_val = os.path.join(
        processed_dir, "val", "{rank:03d}_{index:05d}.parquet"
    )

    process_shards(
        blockdict["train"],
        num_shards,
        output_template_train,
        script_configs.resume,
        script_configs.num_proc,
        rank=array_task_id,
        size=array_task_count,
    )
    if script_configs.train_split_pct < 1.0:
        process_shards(
            blockdict["test"],
            num_shards,
            output_template_val,
            script_configs.resume,
            script_configs.num_proc,
            rank=array_task_id,
            size=array_task_count,
        )

    print(f"Reshuffling into parquet finished on task {array_task_id}!")
    print(
        "------------------------------------------------------------------------------"
    )
    print(
        "-----------------------------Printing examples now ---------------------------"
    )
    print_examples(array_task_id, processed_dir, hf_tokenizer_path)

    # - Phase VI: Clear cache ########################################################################################

    # now each task can remove its own lock file
    os.remove(lockfile)
    wait_time = 0
    last_logged_wait = time.time()

    # wait for all other tasks to finish
    while any(
        os.path.exists(os.path.join(cache_locks, f"p2p_tok_lock_{i:03d}.lock"))
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
    print("All tasks finished, removing tok cache if requested...")
    if script_configs.rm_cache:
        if array_task_id == 0:
            print("Removing tok cache on rank 0...")
            import shutil

            shutil.rmtree(cache_dir, ignore_errors=False)
            print("tok cache removed.")

    print(f"Task {array_task_id} finished tokenization.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    set_parsing_settings(
        config_read_mode_urls_enabled=True,
        docstring_parse_attribute_docstrings=True,
    )
    CLI(parquet_to_parquet_tokenization)
