# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import glob
import json
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader
from tqdm import tqdm

from litgpt.data import DataModule
from litgpt.data.alpaca import download_if_missing
from litgpt.data.text_files import validate_tokenizer
from litgpt.tokenizer import Tokenizer

from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader
from datasets import Dataset, load_dataset

@dataclass
class Fineweb(DataModule):
    """See tinystories.py from which this is derived"""
    
    ds_name_or_path: str = "HuggingFaceFW/fineweb"
    subset: str = "sample-10BT"
    raw_split: str = "train"
    test_size: float = 0.02

    data_path: Path = Path("data/fineweb")
    """The path to the data directory, containing two folders 'train' and 'val'
    which are the output of the preprocessing step."""

    seed: int = 42
    """The seed to use for shuffling the dataset."""
    num_workers: int = 8
    """The number of workers to use for the dataloaders."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    train_max_seq_length: int = field(default=-1, init=False, repr=False)
    val_max_seq_length: int = field(default=-1, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.data_path_train = self.data_path / "train"
        self.data_path_val = self.data_path / "val"

    def connect(self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, train_max_seq_length: int = -1, val_max_seq_length: int = -1) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.train_max_seq_length = train_max_seq_length + 1  # Increase by one because we need the next token as well
        self.val_max_seq_length = val_max_seq_length + 1  # Increase by one because we need the next token as well


    # a hf load_dataset version based on the openwebtext.py example
    def prepare_data(self) -> None:
        from litdata import optimize, TokensLoader

        # num_workers = min(os.cpu_count() // 2, 32) # don't let it get out of hand
        # num_workers = 16 # startup cost is high, as each worker needs a view of the dataset
        num_workers = 32 # startup cost is high, as each worker needs a view of the dataset

        if Path(self.data_path_train).is_dir() and Path(self.data_path_val).is_dir():
            print(f"Found Fineweb train and val dir: {self.data_path}. Skipping preprocessing.")
            return
        
        data_dir = self.data_path / "fineweb_raw"
        data_dir.mkdir(exist_ok=True, parents=True)

        dataset = load_dataset(self.ds_name_or_path, self.subset, split=self.raw_split, cache_dir=data_dir, num_proc=num_workers)
        dataset = dataset.select_columns("text")
        dataset = dataset.train_test_split(test_size=self.test_size, seed=self.seed, train_indices_cache_file_name=f"{data_dir}/train", test_indices_cache_file_name=f"{data_dir}/test")

        # This dataset now contains train and "val" but name is wrong
        dataset["val"] = dataset.pop("test")
        
        # if we end up needing to make this a bit more robust, we can preshard the data and then make sure that
        # the workers each load one of their shards and then just get a shard len indices list
        optimize(
            fn=partial(tokenize, hf_ds=dataset["train"], tokenizer=self.tokenizer), # see below
            inputs=list(range(len(dataset["train"]))), # indices passed to partial
            output_dir=str(self.data_path_train),
            num_workers=num_workers,
            chunk_bytes="200MB",
            item_loader=TokensLoader(),
        )
        optimize(
            fn=partial(tokenize, hf_ds=dataset["val"], tokenizer=self.tokenizer), # see below
            inputs=list(range(len(dataset["val"]))), # indices passed to partial
            output_dir=str(self.data_path_val),
            num_workers=1, # its small
            chunk_bytes="200MB",
            item_loader=TokensLoader(),
        )

    def train_dataloader(self) -> DataLoader:

        train_dataset = StreamingDataset(
            input_dir=str(self.data_path_train),
            item_loader=TokensLoader(block_size=self.train_max_seq_length),
            shuffle=True,
        )
        train_dataloader = StreamingDataLoader(
            train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:

        val_dataset = StreamingDataset(
            input_dir=str(self.data_path_val),
            item_loader=TokensLoader(block_size=self.val_max_seq_length),
            shuffle=True,
        )
        val_dataloader = StreamingDataLoader(
            val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return val_dataloader


def tokenize(index: int = None, hf_ds: Dataset = None, tokenizer: Tokenizer = None):
    text = hf_ds[index]["text"].strip()  # get rid of leading/trailing whitespace
    tokens = tokenizer.encode(text, bos=True, eos=False)  # encode the text, use BOS
    yield tokens


# add FinewebEdu which only differs in some key arguments
@dataclass
class FinewebEdu(Fineweb):
    ds_name_or_path: str = "HuggingFaceFW/fineweb-edu"
    subset: str = "sample-100BT"
    raw_split: str = "train"
    test_size: float = 0.005
    data_path: Path = Path("data/fineweb_edu")
