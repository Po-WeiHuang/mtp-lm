# very loosely based on the original code from Lightning AI
# litgpt/packed_dataset.py
# and
# very loosely inspired by indexed_dataset in Fairseq, Megatron
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py

import random
import hashlib
from torch.utils.data import IterableDataset
from pathlib import Path
from tqdm import tqdm

import logging
import glob

import torch
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ParquetStream(IterableDataset):
    """Minimal implementation of a purely parquet based distributed friendly dataset."""

    def __init__(
        self,
        # runtime and fabric
        seed=12345,
        num_processes=1,
        process_rank=0,
        torch_device=None,
        block_size=None,
        pad_token_id=None,
        # cfg args
        dataset_folder_path="",
        prefix="",
        broadcast_glob=True,
        shuffle=True,
        shuffle_filenames=True,
        doc_wise=False,
        doc_wise_skip_tail=True,
        doc_wise_sep_tok=None,
        ignore_fingerprint_mismatch=False,
        verbose=False,
    ):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.torch_device = torch_device
        # block_size+1 for next token prediction mirroring litgpt's convention
        self.block_size = block_size + 1 if block_size is not None else None
        self.pad_token_id = pad_token_id
        if self.block_size is not None:
            assert self.pad_token_id is not None, "pad_token_id must be specified if block_size is set"

        self.doc_wise = doc_wise
        self.doc_wise_skip_tail = doc_wise_skip_tail
        self.doc_wise_sep_tok = doc_wise_sep_tok
        if self.doc_wise:
            assert self.doc_wise_sep_tok is not None, "Need a separator id for doc-wise pqdsp"

        # Get file list, with distributed broadcast if needed
        if broadcast_glob and torch.distributed.is_initialized():
            if process_rank == 0:
                filenames = sorted(str(p) for p in Path(dataset_folder_path).glob(f"{prefix}*.parquet"))
            else:
                filenames: list[str] = None  # type: ignore # believe
            obj = [filenames]
            torch.distributed.broadcast_object_list(obj, 0)
            parquet_files = obj[0]
        else:
            parquet_files = sorted(str(p) for p in Path(dataset_folder_path).glob(f"{prefix}*.parquet"))
        if shuffle_filenames:
            random.Random(seed).shuffle(parquet_files)

        # Shard files for distributed training
        self.parquet_files = parquet_files[process_rank::num_processes] if num_processes > 1 else parquet_files
        self._ds_fingerprint = hashlib.shake_128(str(self.parquet_files).encode()).hexdigest(4)
        self._ignore_fingerprint_mismatch = ignore_fingerprint_mismatch

        self.verbose = verbose
        if self.verbose:
            logger.info(
                f"Rank {process_rank}/{num_processes} has {len(self.parquet_files)} parquet files | identifier={self._ds_fingerprint}"
            )
            examples = pq.read_table(self.parquet_files[0], columns=["input_ids"]).slice(0, 3).to_pylist()  # Get 3 rows
            for i, example in enumerate(examples):
                logger.info(f"Example {i}: {example['input_ids'][:12]}")  # First 12 tokens of each row
        self.shuffle = shuffle
        self.seed = seed
        # Initialize default state
        self._state = {
            "rng": random.Random(seed),
            "rng_state": (-1, [-1], None),
            "buffer": [],
            "file_idx": 0,
            "epoch": 0, # TODO: deterministic shuffle on the file list for each epoch
            "row_group_idx": 0,
            "fingerprint": self._ds_fingerprint,
        }
    
    @property
    def epoch(self):
        return self._state["epoch"]

    def __iter__(self):
        while self._state["file_idx"] < len(self.parquet_files):
            if not self._state["buffer"]:
                # Refill buffer from current position
                pf = pq.ParquetFile(self.parquet_files[self._state["file_idx"]])
                if self._state["row_group_idx"] >= pf.num_row_groups:
                    self._state["file_idx"] += 1
                    # handle wrap around on epoch
                    if self._state["file_idx"] >= len(self.parquet_files):
                        self._state["file_idx"] = 0
                        self._state["epoch"] += 1
                        if self.verbose:
                            print(
                                f"({self.process_rank}/{self.num_processes}) cycled file_idx, starting epoch={self._state['epoch']}",
                                flush=True,
                            )
                        # raise StopIteration # this will prop up to the cycle iterator, eh not sure this works correctly
                    self._state["row_group_idx"] = 0
                    continue

                if self.verbose:
                    # proactively warn if loading the last row group of the last file
                    if (self._state["file_idx"] == len(self.parquet_files) - 1) and (
                        self._state["row_group_idx"] == pf.num_row_groups - 1
                    ):
                        print(f"Loading the last row group of the last file!", flush=True)

                    print(
                        f"Reading new row group | file_idx:{self._state['file_idx']}, row_group_idx:{self._state['row_group_idx']}",
                        flush=True,
                    )

                self._load_buffer(pf)
                self._state["row_group_idx"] += 1

            while self._state["buffer"]:
                row = self._state["buffer"].pop()
                while len(row) <= 1:
                    row = self._state["buffer"].pop()
                if self.block_size is not None:
                    row = row[: self.block_size]
                    # if ragged, we need to pad
                    if len(row) < self.block_size:
                        padding_length = self.block_size - len(row)
                        row = row + [self.pad_token_id] * padding_length
                yield torch.as_tensor(row, dtype=torch.long)

    def __len__(self):
        """Implemented without accounting for doc-wise loading as this cannot be inferred without scanning the data."""
        total_rows = 0
        # total_rows_by_row_group_ctr = 0
        
        for file_path in tqdm(self.parquet_files, total=len(self.parquet_files), desc="Counting rows in Parquet files using metadata."):
            pf = pq.ParquetFile(file_path)

            # num_row_groups = pf.num_row_groups
            # for rg_idx in range(num_row_groups):
            #     rows_in_group = pf.read_row_group(rg_idx).num_rows
            #     total_rows_by_row_group_ctr += rows_in_group
            
            total_rows += pf.metadata.num_rows
        
        # assert total_rows == total_rows_by_row_group_ctr, "Row count mismatch between metadata and row group counting."
        return total_rows
    
    def _estimate_total_tokens(self, k=1, block_size=None):
        """we count the tokens in the first few files and then multiply it out to estimate total tokens"""
        files_to_consider = len(self.parquet_files)
        if k is not None:
            files_to_consider = min(k, len(self.parquet_files))
        toks_counted = 0
        rows_counted = 0
        max_tok_per_row = float('-inf')
        min_tok_per_row = float('inf')
        for file_idx in tqdm(range(files_to_consider), total=files_to_consider, desc=f"Estimating total tokens in Parquet files using {files_to_consider} files."):
            pf = pq.ParquetFile(self.parquet_files[file_idx])
            num_row_groups = pf.num_row_groups
            for rg_idx in tqdm(range(num_row_groups), total=num_row_groups, desc=f"Counting tokens in file {file_idx}/{files_to_consider}"):
                batch = pf.read_row_group(rg_idx)
                input_ids_column = batch.column("input_ids")
                for row in input_ids_column.to_pylist():
                    if block_size is not None:
                        row = row[:block_size]
                        # while logic below mirrors the actual loading, we don't want to count
                        # padding during estimation. the goal is only to factor in truncation.
                        # # if ragged, we need to pad
                        # if len(row) < block_size:
                        #     padding_length = block_size - len(row)
                        #     row = row + [self.pad_token_id] * padding_length
                    toks_in_row = len(row)
                    toks_counted += toks_in_row
                    rows_counted += 1
                    if toks_in_row > max_tok_per_row:
                        max_tok_per_row = toks_in_row
                    if toks_in_row < min_tok_per_row:
                        min_tok_per_row = toks_in_row
        avg_toks_per_file = toks_counted / files_to_consider
        estimated_total_toks = int(avg_toks_per_file * len(self.parquet_files))
        avg_toks_per_row = toks_counted / rows_counted if rows_counted > 0 else 0
        return estimated_total_toks, (avg_toks_per_row, min_tok_per_row, max_tok_per_row)
        

    def _load_buffer(self, parquet_file):
        batch = parquet_file.read_row_group(self._state["row_group_idx"])
        self._state["buffer"] = batch.column("input_ids").to_pylist()

        if self.doc_wise:
            doc_wise_buffer = []
            for row in self._state["buffer"]:
                curr_sep_idx = 0
                while curr_sep_idx < len(row):
                    try:
                        next_sep_idx = row.index(self.doc_wise_sep_tok, curr_sep_idx + 1)
                    except ValueError:
                        # if we can't find another sep, were either the tail,
                        # or if curr is 0, we're the head==only doc, and so we dont skip that
                        if self.doc_wise_skip_tail and curr_sep_idx > 0:
                            break
                        next_sep_idx = len(row)
                    doc_wise_buffer.append(row[curr_sep_idx:next_sep_idx])
                    curr_sep_idx = next_sep_idx

            self._state["buffer"] = doc_wise_buffer

        if self.shuffle:
            self._state["rng_state"] = self._state["rng"].getstate()  # the last used state for a shuffle op
            self._state["rng"].shuffle(self._state["buffer"])

    # def state_dict(self):

    #     if self.verbose:
    #         print(f"({self.process_rank}/{self.num_processes}) BEGIN pqds state_dict function.", flush=True)

    #     if self.shuffle:
    #         # rng state has three parts , one int, one list, and another int or None
    #         rng_0, rng_1, rng_2 = self._state["rng_state"]
    #         rank_rng_0 = torch.tensor([rng_0], device=self.torch_device)
    #         rank_rng_1 = torch.tensor(rng_1, device=self.torch_device)
    #         rank_rng_2 = torch.tensor([rng_2] if rng_2 is not None else [-1], device=self.torch_device)
    #         # make world size containers for each part
    #         all_rank_rng_0 = [torch.zeros_like(rank_rng_0) for _ in range(self.num_processes)]
    #         all_rank_rng_1 = [torch.zeros_like(rank_rng_1) for _ in range(self.num_processes)]
    #         all_rank_rng_2 = [torch.zeros_like(rank_rng_2) for _ in range(self.num_processes)]
    #         # gather the parts
    #         torch.distributed.all_gather(all_rank_rng_0, rank_rng_0)
    #         if self.verbose:
    #             print(
    #                 f"({self.process_rank}/{self.num_processes}) state_dict function: passed rng0 gather",
    #                 flush=True,
    #             )
    #         torch.distributed.all_gather(all_rank_rng_1, rank_rng_1)
    #         if self.verbose:
    #             print(
    #                 f"({self.process_rank}/{self.num_processes}) state_dict function: passed rng1 gather",
    #                 flush=True,
    #             )
    #         torch.distributed.all_gather(all_rank_rng_2, rank_rng_2)
    #         if self.verbose:
    #             print(
    #                 f"({self.process_rank}/{self.num_processes}) state_dict function: passed rng2 gather",
    #                 flush=True,
    #             )
    #         # pack them up
    #         all_rank_rng_states = (all_rank_rng_0, all_rank_rng_1, all_rank_rng_2)
    #     else:
    #         all_rank_rng_states = None

    #     if self.verbose:
    #         print(f"({self.process_rank}/{self.num_processes}) state_dict function: passed rng gathers", flush=True)

    #     # we also need to save independent row_indices for each worker
    #     row_idx = torch.tensor([len(self._state["buffer"])], device=self.torch_device)
    #     all_row_indices = [torch.zeros_like(row_idx) for _ in range(self.num_processes)]
    #     torch.distributed.all_gather(all_row_indices, row_idx)
    #     all_row_indices = [int(ari.item()) for ari in all_row_indices]

    #     if self.verbose:
    #         print(f"({self.process_rank}/{self.num_processes}) state_dict function: passed idx gathers", flush=True)

    #     # we also need to save the file_idx
    #     file_idx = torch.tensor([self._state["file_idx"]], device=self.torch_device)

    #     # and the epoch
    #     epoch = torch.tensor([self._state["epoch"]], device=self.torch_device)

    #     # for row_group_idx we need to
    #     # sub 1 to reload the currently buffer'd row group on resume (rather than the next)
    #     # but only do it on a copy of the value
    #     row_group_idx = torch.tensor([self._state["row_group_idx"]], device=self.torch_device)
    #     if len(self._state["buffer"]) > 0:
    #         row_group_idx -= 1

    #     all_file_indices = [torch.zeros_like(file_idx) for _ in range(self.num_processes)]
    #     all_epochs = [torch.zeros_like(epoch) for _ in range(self.num_processes)]
    #     all_row_group_indices = [torch.zeros_like(row_group_idx) for _ in range(self.num_processes)]
    #     torch.distributed.all_gather(all_file_indices, file_idx)
    #     torch.distributed.all_gather(all_epochs, epoch)
    #     torch.distributed.all_gather(all_row_group_indices, row_group_idx)
    #     all_file_indices = [int(afi.item()) for afi in all_file_indices]
    #     all_epochs = [int(ae.item()) for ae in all_epochs]
    #     all_row_group_indices = [int(argi.item()) for argi in all_row_group_indices]
    #     # sub 1 to reload the currently buffer'd row group on resume (rather than the next)
    #     # all_row_group_indices = [argi - 1 for argi in all_row_group_indices]
    #     if self.verbose:
    #         print(
    #             f"({self.process_rank}/{self.num_processes}) state_dict function: passed file, epoch, and row group gathers",
    #             flush=True,
    #         )

    #     # and finally the fingerprint
    #     fingerprint_tensor = torch.tensor([int(self._ds_fingerprint, 16)], device=self.torch_device)
    #     all_fingerprints = [torch.zeros_like(fingerprint_tensor) for _ in range(self.num_processes)]
    #     torch.distributed.all_gather(all_fingerprints, fingerprint_tensor)
    #     all_fingerprints = [hex(int(af.item()))[2:] for af in all_fingerprints]
    #     if self.verbose:
    #         print(
    #             f"({self.process_rank}/{self.num_processes}) state_dict function: passed fingerprint gathers",
    #             flush=True,
    #         )

    #     if self.verbose:
    #         print(f"({self.process_rank}/{self.num_processes}) END pqds state_dict function.", flush=True)
    #     return {
    #         "file_idx": all_file_indices,
    #         "epoch": all_epochs,
    #         "row_group_idx": all_row_group_indices,
    #         "row_idx": all_row_indices,
    #         "rng_state": all_rank_rng_states,
    #         "fingerprint": all_fingerprints,
    #     }

    # lets drop in the gemini rewrite to allreduce based  function

    def state_dict(self):
        if self.verbose:
            print(f"({self.process_rank}/{self.num_processes}) BEGIN pqds state_dict function (All-Reduce Mode).", flush=True)

        def _all_gather_via_reduce(local_tensor):
            """
            Helper to simulate all_gather using all_reduce.
            Assumes local_tensor is a 1D tensor or scalar.
            """
            # Create a buffer to hold data from all ranks
            # Ensure it's on the correct device and dtype
            global_buffer = torch.zeros(
                (self.num_processes, *local_tensor.shape), 
                device=self.torch_device, 
                dtype=local_tensor.dtype
            )
            # Place local data in the assigned slot
            global_buffer[self.process_rank] = local_tensor
            
            # Sum across all ranks
            torch.distributed.all_reduce(global_buffer, op=torch.distributed.ReduceOp.SUM)
            return global_buffer

        # --- RNG State Gathering ---
        if self.shuffle:
            rng_0, rng_1, rng_2 = self._state["rng_state"]
            
            # Convert to tensors
            rank_rng_0 = torch.tensor([rng_0], device=self.torch_device, dtype=torch.long)
            rank_rng_1 = torch.tensor(rng_1, device=self.torch_device, dtype=torch.long)
            rank_rng_2 = torch.tensor([rng_2] if rng_2 is not None else [-1], device=self.torch_device, dtype=torch.long)

            # Gather using all_reduce
            if self.verbose:
                print(
                    f"({self.process_rank}/{self.num_processes}) state_dict function: before rng0 reducegather",
                    flush=True,
                )
            all_rank_rng_0 = _all_gather_via_reduce(rank_rng_0)
            if self.verbose:
                print(
                    f"({self.process_rank}/{self.num_processes}) state_dict function: passed rng0 reducegather",
                    flush=True,
                )
            all_rank_rng_1 = _all_gather_via_reduce(rank_rng_1)
            all_rank_rng_2 = _all_gather_via_reduce(rank_rng_2)
            
            # Structure as a tuple of lists to match original functionality
            # Original code returned (List[Tensor], List[Tensor], List[Tensor])
            all_rank_rng_states = (
                [all_rank_rng_0[i] for i in range(self.num_processes)],
                [all_rank_rng_1[i] for i in range(self.num_processes)],
                [all_rank_rng_2[i] for i in range(self.num_processes)]
            )
        else:
            all_rank_rng_states = None

        # --- Row Indices, File Indices, Epochs, Row Group Indices ---
        # We can batch these together into a single all_reduce for better performance,
        # but for "exact same functionality" we'll do them as requested.

        row_idx = torch.tensor([len(self._state["buffer"])], device=self.torch_device, dtype=torch.long)
        all_row_indices_tensor = _all_gather_via_reduce(row_idx)
        all_row_indices = [int(x.item()) for x in all_row_indices_tensor]

        file_idx = torch.tensor([self._state["file_idx"]], device=self.torch_device, dtype=torch.long)
        epoch = torch.tensor([self._state["epoch"]], device=self.torch_device, dtype=torch.long)
        
        # Calculate row_group_idx with the buffer logic
        rg_val = self._state["row_group_idx"]
        if len(self._state["buffer"]) > 0:
            rg_val -= 1
        row_group_idx = torch.tensor([rg_val], device=self.torch_device, dtype=torch.long)

        all_file_indices = [int(x.item()) for x in _all_gather_via_reduce(file_idx)]
        all_epochs = [int(x.item()) for x in _all_gather_via_reduce(epoch)]
        all_row_group_indices = [int(x.item()) for x in _all_gather_via_reduce(row_group_idx)]

        # --- Fingerprint ---
        fingerprint_tensor = torch.tensor([int(self._ds_fingerprint, 16)], device=self.torch_device, dtype=torch.long)
        all_fingerprints_tensor = _all_gather_via_reduce(fingerprint_tensor)
        all_fingerprints = [hex(int(x.item()))[2:] for x in all_fingerprints_tensor]

        if self.verbose:
            print(f"({self.process_rank}/{self.num_processes}) END pqds state_dict function.", flush=True)

        return {
            "file_idx": all_file_indices,
            "epoch": all_epochs,
            "row_group_idx": all_row_group_indices,
            "row_idx": all_row_indices,
            "rng_state": all_rank_rng_states,
            "fingerprint": all_fingerprints,
        }


    def load_state_dict(self, state_dict):
        if self.verbose:
            print(f"BEGIN pqds load_state_dict function.", flush=True)

        # Unpack fingerprint
        fingerprint = state_dict.get("fingerprint")[torch.distributed.get_rank()]
        if self._ignore_fingerprint_mismatch:
            if self.verbose:
                print(f"Ignoring dataset fingerprint mismatch, better know what you're doing!", flush=True)
        else:
            if int(fingerprint, 16) != int(self._ds_fingerprint, 16):
                raise ValueError(
                    f"Dataset fingerprint mismatch. Expected {self._ds_fingerprint}, "
                    f"got {state_dict.get('fingerprint')}. This may indicate attempting to "
                    "load a state from a different dataset."
                )
            if self.verbose:
                print(f"Dataset fingerprint match: {fingerprint}=={self._ds_fingerprint}", flush=True)
        # Unpack the file_idx, epoch, and row_group_idx states
        self._state["file_idx"] = state_dict["file_idx"][torch.distributed.get_rank()]
        self._state["epoch"] = state_dict["epoch"][torch.distributed.get_rank()]
        self._state["row_group_idx"] = state_dict["row_group_idx"][torch.distributed.get_rank()]
        if self._state["row_group_idx"] == -1:  # this is a special case for the first row group
            self._state["row_group_idx"] = 0

        if self.verbose:
            print(f"file_idx: {self._state['file_idx']}, epoch: {self._state['epoch']}, row_group_idx: {self._state['row_group_idx']}", flush=True)

        # Validate file_idx bounds
        if not (0 <= self._state["file_idx"] < len(self.parquet_files)):
            raise ValueError(
                f"Invalid file_idx {self._state['file_idx']}. " f"Must be between 0 and {len(self.parquet_files)-1}"
            )

        if self.verbose:
            print(f"file_idx bounds check passed", flush=True)

        # Load the current file
        pf = pq.ParquetFile(self.parquet_files[self._state["file_idx"]])

        if self.verbose:
            print(f"loaded parquet file: {self.parquet_files[self._state['file_idx']]}", flush=True)

        # Reload RNG before trying to fill the buffer as we may shuffle within the read fn
        if state_dict["rng_state"] is not None:
            assert self.shuffle, "RNG state provided but shuffle is disabled, potential resume mismatch."
            # now we unpack the world of rng states
            all_rank_rng_states = state_dict["rng_state"]
            all_rank_rng_0, all_rank_rng_1, all_rank_rng_2 = all_rank_rng_states
            rank_rng_0 = all_rank_rng_0[torch.distributed.get_rank()]
            rank_rng_1 = all_rank_rng_1[torch.distributed.get_rank()]
            rank_rng_2 = all_rank_rng_2[torch.distributed.get_rank()]
            self._state["rng_state"] = (
                rank_rng_0.item(),
                tuple(rank_rng_1.tolist()),
                rank_rng_2.item() if rank_rng_2.item() != -1 else None,
            )

            self._state["rng"] = random.Random()
            self._state["rng"].setstate(self._state["rng_state"])

        if self.verbose:
            print(f"loaded rng state", flush=True)

        # Then actually reload the token buffer
        self._load_buffer(pf)

        if self.verbose:
            print(f"reloaded buffer: len={len(self._state['buffer'])}", flush=True)

        # Then get the row_idx
        all_row_indices = state_dict["row_idx"]
        row_idx = all_row_indices[torch.distributed.get_rank()]

        if self.verbose:
            print(f"reloaded row_idx: {row_idx}", flush=True)

        # to trim the buffer down to point we previously consumed through
        self._state["buffer"] = self._state["buffer"][:row_idx]

        if self.verbose:
            print(f"trimmed buffer: len={len(self._state['buffer'])}", flush=True)
        if self.verbose:
            print(f"END pqds load_state_dict function.", flush=True)
