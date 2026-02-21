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

from datasets import Dataset, load_dataset


@dataclass
class TinyStories(DataModule):
    """The TinyStories data module: https://huggingface.co/datasets/roneneldan/TinyStories

    Provides training and validation dataloaders that return batches of tokens. Every sample is set to a fixed length.
    """

    data_path: Path = Path("data/tinystories")
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
        num_workers = 8 # startup cost is high, as each worker needs a view of the dataset

        if Path(self.data_path_train).is_dir() and Path(self.data_path_val).is_dir():
            print(f"Found TinyStories train and val dir: {self.data_path}. Skipping preprocessing.")
            return
        
        data_dir = self.data_path / "TinyStories_raw"
        data_dir.mkdir(exist_ok=True, parents=True)

        dataset = load_dataset("roneneldan/TinyStories", cache_dir=data_dir, num_proc=num_workers, trust_remote_code=True)

        # This dataset already contains train and val but name is wrong
        dataset["val"] = dataset.pop("validation")

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
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

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
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

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

