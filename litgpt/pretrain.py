# fmt: off
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from jsonargparse import CLI, set_parsing_settings

import os
import math
import pprint
import time
from dataclasses import asdict
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import shutil
import random
import gc

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from lightning.fabric.strategies import FSDPStrategy, DDPStrategy
from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from torchmetrics.aggregation import RunningMean
from typing_extensions import Literal
from wandb import Table

# Catch warnings from the torch data package
import warnings

warnings.filterwarnings('ignore', message='.*Length of IterableDataset.*was reported to be.*', category=UserWarning)

import hashlib
from attn_gym import visualize_attention_scores

from litgpt import Tokenizer
from litgpt.args import EvalArgs, LogArgs, TrainArgs, SingleShotArgs, WandbArgs, PQDSArgs, parse_prelude_token_ids, parse_temp_sep_token_id_range, parse_k_toks, parse_topk_values, parse_extra_val_trunc_lengths, parse_extra_val_k_toks_values
from litgpt.config import name_to_config
from litgpt.data import DataModule, TinyLlama
from litgpt.model import GPT, Block, CausalSelfAttention, Config, LLaMAMLP
from litgpt.generate.base import generate as generate_fn
from litgpt.parquet_dataset import ParquetStream
from litgpt.utils import (
    _TORCH_EQUAL_2_7,
    _TORCH_EQUAL_2_8,
    CycleIterator,
    capture_hparams,
    check_nvlink_connectivity,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    extend_checkpoint_dir,
    find_resume_path,
    get_default_supported_precision,
    init_out_dir,
    instantiate_torch_optimizer,
    num_parameters,
    parse_devices,
    reset_parameters,
    save_config,
    save_hyperparameters,
    dict2attr,
)

from litgpt.repetition_diversity_tokens import measure_repetition_and_diversity


def pt_ce_plus_ent_loss(logits_student=None, logits_teacher=None, labels_teacher=None, beta=None):
    # BxLxV, BxLxV
    # kl isolating formula implemented using the available reference cross entropy function
    # logic is structured to limit any extraneous computation

    stud_logits = logits_student.flatten(end_dim=-2) # (BxL)xV

    if labels_teacher is not None:
        ent_teach = torch.tensor(0.0) # ent_teach will be treated as 0 in this case, bc teach_labels is 1-hot.
        teach_labels = labels_teacher.flatten() # (BxL)x1erges all dimensions up to the second-to-last one into one dimension.
        ce_teach_stud = F.cross_entropy(stud_logits, teach_labels, reduction="mean") # 1x
        kl_teach_stud = ce_teach_stud # ... - ent_teach but 0 in this case.
    else:
        teach_logits = logits_teacher.flatten(end_dim=-2) # (BxL)xV
        teach_probs = F.softmax(teach_logits, -1)
        ent_teach = F.cross_entropy(teach_logits, teach_probs, reduction="mean") # 1x
        ce_teach_stud = F.cross_entropy(stud_logits, teach_probs, reduction="mean") # 1x
        kl_teach_stud = ce_teach_stud - ent_teach
    
    # we only need to do the student entropy calc if beta > 0
    if beta > 0.0:
        stud_probs = F.softmax(stud_logits, -1)
        ent_stud = F.cross_entropy(stud_logits, stud_probs, reduction="mean") # 1x
    else:
        ent_stud = torch.tensor(0.0) # ent_stud will be treated as 0 in this case.
    
    loss = kl_teach_stud + beta * ent_stud
    
    return loss, ce_teach_stud, kl_teach_stud, ent_teach, ent_stud


def batch_find_subarray(batch_tensor, sub_tensor):
    # batch_tensor: [Batch, SeqLength]
    # sub_tensor: [SubLength]
    batch_size = batch_tensor.size(0)
    sub_len = len(sub_tensor)
    
    # Unfold on the sequence dimension (dim=1)
    # Resulting shape: [Batch, NumWindows, SubLength]
    windows = batch_tensor.unfold(1, sub_len, 1)
    
    # Compare each window to the sub_tensor (broadcasting)
    # Result: [Batch, NumWindows] boolean mask
    matches = (windows == sub_tensor).all(dim=2)
    
    # Efficiently find the first True in each row
    # torch.argmax returns the first index of the maximum value (True > False)
    # Note: If no match exists, it returns 0. Use a mask to handle misses.
    first_indices = matches.int().argmax(dim=1)
    
    # Check if a match actually exists in each batch
    has_match = matches.any(dim=1)
    
    # Fill -1 where no match was found
    first_indices[~has_match] = -1
    
    return first_indices

# # Example
# batch_data = torch.tensor([
#     [1, 2, 3, 4, 5],
#     [0, 1, 2, 3, 0],
#     [9, 9, 9, 9, 9]
# ])
# target = torch.tensor([2, 3])
# print(batch_find_subarray(batch_data, target)) # Output: tensor([1, 2, -1])


# Begin Gemini :] version for multi region striding
# and then a rewrite, and a rewrite, and a rewrite... :<
def truncate_and_mask(input_ids=None, target_ids=None, k_toks=None, mask_id=None, truncation_length=None, mask_region_ct=1, offset=0, verbose=False, pad_token_id=None, prelude_token_ids=None, min_mask_id=None, max_mask_id=None, skip_max_mask_id_check=False):
   
    B, og_slen = input_ids.shape
    S = truncation_length

    # assert S == og_slen, "For this debug rewrite, truncation_length must equal input sequence length."
    # relax slightly
    # assert S <= og_slen, "truncation_length must be less than or equal to input sequence length."
    # relax even more, covering this below with the consumption calculation

    K = k_toks - 1
    P = truncation_length // mask_region_ct - K
    region_width = P + K

    assert abs(offset) < P, "Offset magnitude must be less than prefix length P."

    source_block_starts = torch.arange(0, S, P) 
    if verbose: print("Source Block Starts:", source_block_starts)

    base_range = torch.arange(region_width, device=input_ids.device)
    if verbose: print("Base Range:", base_range)

    num_blocks = source_block_starts.size(0)
    assert num_blocks == (S + P - 1) // P, "Not sure if eq always."
    if verbose: print("Num Blocks:", num_blocks)
    
    block_starts = torch.arange(0, num_blocks * P, P, device=input_ids.device).unsqueeze(1)
    if verbose: print("Block Starts:", block_starts)

    full_indices = block_starts + base_range
    if verbose: print("Full Indices:", full_indices)

    # at this moment, we know where to put masks, the last K tokens of each region
    # for the mask_id_mask and the last K+1 tokens for the pred_pos_mask
    # so we can create the corresponding mask_id_mask and pred_pos_mask here
    mask_id_mask = torch.zeros_like(full_indices, dtype=torch.bool)
    pred_pos_mask = torch.zeros_like(full_indices, dtype=torch.bool)
    mask_id_mask[:, P:P+K] = True
    pred_pos_mask[:, P-1:P+K] = True

    full_indices = full_indices.flatten()
    if verbose: print("Full Indices Flattened:", full_indices)
    
    mask_id_mask = mask_id_mask.flatten()
    pred_pos_mask = pred_pos_mask.flatten()
    if verbose: print("Mask ID Mask:", mask_id_mask)
    if verbose: print("Pred Pos Mask:", pred_pos_mask)

    final_indices = full_indices[:S]
    assert torch.equal(final_indices,torch.clamp(final_indices, max=S - 1)), "Final indices exceed sequence length."
    if verbose: print("Final Indices after Truncation:", final_indices)

    mask_id_mask = mask_id_mask[:S]
    pred_pos_mask = pred_pos_mask[:S]
    if verbose: print("Mask ID Mask after Truncation:", mask_id_mask)
    if verbose: print("Pred Pos Mask after Truncation:", pred_pos_mask)

    # result of reworking train dataloading to limit token waste
    if verbose: print(f"Max src idx we will access: {final_indices.max().item()} vs. og_slen: {og_slen}")
    assert final_indices.max().item() <= (og_slen-1), "Final indices exceed original sequence length."
    # this next check is bc we use this logic to set the train_block_size for the dataloader
    num_toks_consumed = (mask_region_ct * P) + K
    assert final_indices.max().item() == (num_toks_consumed - 1), "Final indices max does not match expected consumed toks."

    # special roll slide thing for offset
    if verbose: print(f"Applying Offset={offset} in context of P={P}, K={K}")
    if offset < 0:
        if K == 0:
            # we need to simulate one extra index because indexing with -0 will be the first element
            # which is wrong. we want the last + 1 element in this special case
            old_final_mask_start_pos = final_indices[-1] + 1
        else:
            old_final_mask_start_pos = final_indices[-K]
        final_indices = final_indices.roll(offset)
        final_indices[offset:] = torch.arange(old_final_mask_start_pos, old_final_mask_start_pos + (-offset), device=input_ids.device)
        final_indices += offset  # shift all indices by offset amount
        # mask arrays are simpler, as we just roll and then overwrite with Falses
        mask_id_mask = mask_id_mask.roll(offset)
        mask_id_mask[offset:] = False
        pred_pos_mask = pred_pos_mask.roll(offset)
        pred_pos_mask[offset:] = False
    elif offset > 0:
        final_indices = final_indices.roll(offset)
        final_indices[0:offset] = torch.arange(-offset, 0, device=input_ids.device)
        final_indices += offset  # shift all indices by offset amount
        # mask arrays are simpler, as we just roll and then overwrite with Falses
        mask_id_mask = mask_id_mask.roll(offset)
        mask_id_mask[0:offset] = False
        pred_pos_mask = pred_pos_mask.roll(offset)
        pred_pos_mask[0:offset] = False
    
    if verbose: print("Final Indices after Offset Adjustment:", final_indices)
    if verbose: print("Mask ID Mask after Offset Adjustment:", mask_id_mask)
    if verbose: print("Pred Pos Mask after Offset Adjustment:", pred_pos_mask) 
    
    prepared_input_ids = torch.index_select(input_ids, dim=1, index=final_indices)
    prepared_target_ids = torch.index_select(target_ids, dim=1, index=final_indices)
    # The use of torch.index_select is equivalent to advanced indexing:
    # prepared_input_ids = input_ids[:, final_indices]
    # prepared_target_ids = target_ids[:, final_indices]
    # but apparently index_select is more efficient in some cases.

    # repeat the masks in the batch dim to match the new prepared inputs and targets
    mask_id_mask = mask_id_mask.unsqueeze(0).repeat(B, 1)
    pred_pos_mask = pred_pos_mask.unsqueeze(0).repeat(B, 1)

    # overwrite the mask positions in the input using mask_id_mask
    prepared_input_ids[mask_id_mask] = ((torch.arange(S, device=input_ids.device) - offset) % region_width - P + (min_mask_id if min_mask_id is not None else mask_id)).unsqueeze(0).expand(B, -1)[mask_id_mask]
    # check that we didn't insert something larger than max_mask_id
    if (min_mask_id is not None) and (not skip_max_mask_id_check):
        assert prepared_input_ids[mask_id_mask].max().item() <= max_mask_id, "Inserted mask ID exceeds specified max_mask_id."

    # normal case is where the prefix is the complement of the pred_pos_mask
    # so we establish that here, before the potential edits below
    prefix_pos_mask = ~pred_pos_mask

    # final step is that we want to edit the pred pos mask to throw away positions after the first 
    # padding/EOS token is observed.
    if pad_token_id is not None:
        eos_mask = (prepared_target_ids == pad_token_id)  # BxS
        if torch.any(eos_mask):
            # this oneliner gives us the position of the first PAD/EOS token per sequence
            # or a noop-y S if none found
            eos_positions = torch.where(eos_mask.any(dim=1), eos_mask.float().argmax(dim=1), torch.tensor(S, device=prepared_target_ids.device))  # B
            if verbose: print("PAD/EOS Positions:", eos_positions)
            # we identify regions, being mindful of the offset
            if offset >= 0:
                tail_region_indices = torch.clamp((eos_positions - offset) // region_width, min=0)  # B
            else:
                tail_region_indices = torch.clamp((eos_positions) // region_width, min=0)  # B 
            if verbose: print("First PAD/EOS Occurence Region Indices:", tail_region_indices)
            # now create a mask for positions to zero out in pred_pos_mask
            batch_indices = torch.arange(B, device=prepared_target_ids.device).unsqueeze(1)  # Bx1
            position_indices = torch.arange(S, device=prepared_target_ids.device).unsqueeze(0)  # 1xS
            tail_region_index = (tail_region_indices.unsqueeze(1) + 1)  # Bx1 # +1 is bc we want the first pos _after_ the eos region
            tail_region_boundaries = tail_region_index * region_width  # Bx1 # * region_width to get the actual position index
            if verbose: print("Tail Region Index:", tail_region_index)
            if verbose: print("Tail Region Boundaries:", tail_region_boundaries)
            positions_to_zero = position_indices >= tail_region_boundaries  # BxS
            if verbose: print("Positions to Zero Out in Pred Pos Mask:", positions_to_zero)
            if verbose:
                num_to_zero = positions_to_zero.sum().item()
                print(f"Number of elms in Pred Pos Mask to zero out due to tail PAD/EOS: {num_to_zero}")
            pred_pos_mask[positions_to_zero] = False
            if verbose: print("Final Pred Pos Mask after PAD/EOS Adjustment:", pred_pos_mask)
            # do the same for mask_id_mask to avoid computing loss on those positions
            mask_id_mask[positions_to_zero] = False
            if verbose: print("Final Mask ID Mask after PAD/EOS Adjustment:", mask_id_mask)
            # and finally, do the same for prefix_pos_mask as well
            prefix_pos_mask[positions_to_zero] = False
            if verbose: print("Final Prefix Pos Mask after PAD/EOS Adjustment:", prefix_pos_mask)
    
    # Actual final step is that we also want to edit pred_mask to also throw away positions before and including 
    # the first occurence of a special prelude subsequence of tokens, if provided.
    # Note that this means that the exact subsequence needs to be present, not just occurences of the individual tokens.
    # and we can use helper function to efficiently find the first occurence making sure to mask the sequence match region itself too.
    if prelude_token_ids is not None and len(prelude_token_ids) > 0:
        prelude_tensor = torch.tensor(prelude_token_ids, device=prepared_target_ids.device)
        
        first_prelude_pos_batch = batch_find_subarray(prepared_target_ids, prelude_tensor)  # B
        if verbose: print("First Prelude Positions per Sequence:", first_prelude_pos_batch)
        # now we use this to zero out positions in pred_pos_mask without a loop
        # mirror the region boundaries so that we still end up with a square number of regions
        if offset >= 0:
            prelude_region_indices = torch.clamp((first_prelude_pos_batch - offset) // region_width, min=0)  # B
        else:
            prelude_region_indices = torch.clamp((first_prelude_pos_batch) // region_width, min=0)  # B
        if verbose: print("First Prelude Occurence Region Indices:", prelude_region_indices)
        batch_indices = torch.arange(B, device=prepared_target_ids.device).unsqueeze(1)  # Bx1
        position_indices = torch.arange(S, device=prepared_target_ids.device).unsqueeze(0)  # 1xS
        # now we find the region in which the prelude ends, and go back one, zeroing everything in and before that region
        prelude_end_region_index = (prelude_region_indices.unsqueeze(1) + 1)  # Bx1 # +1 is bc we want the first pos _after_ the prelude region
        prelude_end_region_boundaries = prelude_end_region_index * region_width  # Bx1 # * region_width to get the actual position index
        if verbose: print("Prelude End Region Index:", prelude_end_region_index)
        if verbose: print("Prelude End Region Boundaries:", prelude_end_region_boundaries)
        positions_to_zero = position_indices < prelude_end_region_boundaries  # BxS
        if verbose: print("Positions to Zero Out in Pred Pos Mask due to Prelude:", positions_to_zero)
        if verbose:
            num_to_zero = positions_to_zero.sum().item()
            print(f"Number of elms in Pred Pos Mask to zero out due to Prelude: {num_to_zero}")
            pred_pos_ct_before = pred_pos_mask.sum().item()
        pred_pos_mask[positions_to_zero] = False
        if verbose:
            delta_in_pred_pos_ct = pred_pos_ct_before - pred_pos_mask.sum().item()
            print(f"Delta in Pred Pos Mask count after Prelude adjustment: {delta_in_pred_pos_ct}")
            print("Final Pred Pos Mask after Prelude Adjustment:", pred_pos_mask)
        # do the same for mask_id_mask to avoid computing loss on those positions
        mask_id_mask[positions_to_zero] = False
        if verbose: print("Final Mask ID Mask after Prelude Adjustment:", mask_id_mask)
        # and finally, do the same for prefix_pos_mask as well
        prefix_pos_mask[positions_to_zero] = False
        if verbose: print("Final Prefix Pos Mask after Prelude Adjustment:", prefix_pos_mask)

    return prepared_input_ids, prepared_target_ids, mask_id_mask, pred_pos_mask, prefix_pos_mask


def extend_w_mask(input_ids=None, k_toks=None, mask_id=None, min_mask_id=None, max_mask_id=None):
    bsz, _ = input_ids.shape
    
    if k_toks-1 > 0:
        if min_mask_id is not None:
            mask_tensor = torch.arange(min_mask_id, min_mask_id + k_toks - 1, dtype=torch.int64, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
            # check that we didn't insert something larger than max_mask_id
            assert mask_tensor.max().item() <= max_mask_id, "Inserted mask ID exceeds specified max_mask_id."
        else:
            mask_tensor = torch.ones((bsz,k_toks-1),dtype=torch.int64, device=input_ids.device) * mask_id
        return torch.cat([input_ids, mask_tensor], dim=-1)

    return input_ids


def topk_tok_accuracy(pred_toks: torch.Tensor = None, gt_logits: torch.Tensor = None, k_values: int = [1]):

    k_values = sorted(k_values)
    max_k = k_values[-1]
    results = {k: None for k in k_values}
    
    _, top_k_indices = torch.topk(gt_logits, k=max_k, dim=-1)
    pred_toks = pred_toks.unsqueeze(1)
    
    for k in k_values:
        correct_k = torch.eq(pred_toks, top_k_indices[:,:k])
        correct_bool_per_position = torch.any(correct_k, dim=1)
        
        topk_acc = correct_bool_per_position.float().mean().item()
        cu_sums = torch.cumsum(correct_bool_per_position.float(), dim=0)
        cu_lens = torch.arange(1, correct_bool_per_position.shape[0]+1, device=correct_bool_per_position.device).float()
        cu_accs = cu_sums / cu_lens

        results[k] = [topk_acc, cu_accs, correct_bool_per_position]

    return results


def topk_tok_cu_confidence(logits: torch.Tensor = None, k_values: int = [1]):

    k_values = sorted(k_values)
    max_k = k_values[-1]
    results = {k: None for k in k_values}
    
    probs = F.softmax(logits, dim=-1) # LxV
    topk_probs, top_k_indices = torch.topk(probs, k=max_k, dim=-1)
    
    for k in k_values:
        
        topk_cu_probs_per_pos = topk_probs[:,:k].sum(dim=-1) # Lx1

        topk_cu_prob_avg = topk_cu_probs_per_pos.mean().item()
        cu_sums = torch.cumsum(topk_cu_probs_per_pos, dim=0)
        cu_lens = torch.arange(1, topk_cu_probs_per_pos.shape[0]+1, device=topk_cu_probs_per_pos.device).float()
        topk_cu_prob_cu = cu_sums / cu_lens

        results[k] = [topk_cu_prob_avg, topk_cu_prob_cu, topk_cu_probs_per_pos]

    return results


def ent_and_top1_confidence(logits: torch.Tensor = None):
    probs = F.softmax(logits, dim=-1) # LxV
    ents = F.cross_entropy(logits, probs, reduction='none') # Lx1
    top1_idx = torch.argmax(probs, dim=-1) # Lx1
    top1_confs = probs[torch.arange(probs.shape[0], device=probs.device), top1_idx] # Lx1
    return ents, top1_confs


def nll_metric(logits: torch.Tensor = None, tok_ids: torch.Tensor = None):
    losses = F.cross_entropy(logits.flatten(end_dim=-2), tok_ids.flatten(), reduction='none') # (B x L) x 1
    per_pos_losses = losses.view(tok_ids.shape[0], tok_ids.shape[1]) # B x L
    cu_sums = torch.cumsum(per_pos_losses, dim=1) # B x L
    cu_lens = torch.arange(1, tok_ids.shape[1]+1, device=tok_ids.device).float().unsqueeze(0) # 1 x L
    cu_losses = cu_sums / cu_lens # B x L
    per_seq_losses = cu_losses[:,-1] # B x 1

    return per_seq_losses, cu_losses, per_pos_losses


def compile_utilities(compile_mode=None):
    if compile_mode is None:
        globals().update(dict(
            pt_ce_plus_ent_loss=torch.compile(pt_ce_plus_ent_loss),
            # truncate_and_mask=torch.compile(truncate_and_mask),
            # extend_w_mask=torch.compile(extend_w_mask),
            topk_tok_accuracy=torch.compile(topk_tok_accuracy),
            topk_tok_cu_confidence=torch.compile(topk_tok_cu_confidence),
            ent_and_top1_confidence=torch.compile(ent_and_top1_confidence),
            nll_metric=torch.compile(nll_metric),
        ))
    else:
        globals().update(dict(
            pt_ce_plus_ent_loss=torch.compile(pt_ce_plus_ent_loss, mode=compile_mode),
            # truncate_and_mask=torch.compile(truncate_and_mask, mode=compile_mode),
            # extend_w_mask=torch.compile(extend_w_mask, mode=compile_mode),
            topk_tok_accuracy=torch.compile(topk_tok_accuracy, mode=compile_mode),
            topk_tok_cu_confidence=torch.compile(topk_tok_cu_confidence, mode=compile_mode),
            ent_and_top1_confidence=torch.compile(ent_and_top1_confidence, mode=compile_mode),
            nll_metric=torch.compile(nll_metric, mode=compile_mode),
        ))
    # print("Skipping utility compilation for now.")


def setup(
    pdb: Optional[bool] = False,
    model_name: Optional[str] = None,
    model_config: Optional[Config] = None,
    out_dir: Path = Path("out/pretrain"),
    precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
    initial_checkpoint_dir: Optional[Path] = None,
    resume: Union[bool, Literal["auto"], Path] = False,
    data: Optional[Union[DataModule, str]] = None,
    pqds: PQDSArgs = PQDSArgs(), # see args.py, only active if data == "pqds"
    train: TrainArgs = TrainArgs(), # see args.py
    eval: EvalArgs = EvalArgs(), # see args.py
    log: LogArgs = LogArgs(), # see args.py
    singleshot: SingleShotArgs = SingleShotArgs(), # see args.py
    wandb: WandbArgs = WandbArgs(), # see args.py
    optimizer: Union[str, Dict] = "AdamW",
    devices: Union[int, str] = "auto",
    num_nodes: int = 1,
    tokenizer_dir: Optional[Path] = None,
    logger_name: Literal["wandb", "tensorboard", "csv", "mlflow"] = "wandb",
    seed: int = 42,
):
    """Pretrain a model.

    Arguments:
        model_name: The name of the model to pretrain. Choose from names in ``litgpt.config``. Use "list" to list the supported models.
        model_config: A ``litgpt.Config`` object to define the model architecture. Mutually exclusive with
            ``model_config``. Overrides the `model_name` if specified.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Determines a compatible precision setting by default.
        initial_checkpoint_dir: Optional path to a checkpoint directory to initialize the model from.
            Useful for continued pretraining. Mutually exclusive with ``resume``.
        resume: Path to a checkpoint directory to resume from in case training was interrupted, or ``True`` to resume
            from the latest checkpoint in ``out_dir``. An error will be raised if no checkpoint is found. Passing
            ``'auto'`` will resume from the latest checkpoint but not error if no checkpoint exists.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.TinyLlama``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        optimizer: An optimizer name (such as "AdamW") or config.

        devices: How many devices/GPUs to use. Uses all GPUs by default.
        num_nodes: How many nodes the code is being run on.
        tokenizer_dir: Optional path to the tokenizer dir that was used for preprocessing the dataset. Only some data
            module require this.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
    """
    assert model_name is not None, "Most provide either `model_name` or `model_config`."

    if model_name == "list":
        available_models = "\n".join(sorted(name_to_config))
        print(f"Available values:\n{available_models}")
        quit()

    if initial_checkpoint_dir is not None:
        initial_checkpoint_dir = extend_checkpoint_dir(initial_checkpoint_dir)

    if tokenizer_dir is not None:
        tokenizer_dir = extend_checkpoint_dir(tokenizer_dir)

    if model_config is None:
        # Support both model_name options: meta-llama/Meta-Llama-3-8B & Meta-Llama-3-8B
        try:
            model_config = Config.from_name(model_name)
        except ValueError:
            print(f"Model name {model_name} is not supported.\n")
            available_models = "\n".join(sorted(name_to_config))
            print(f"Available values:\n{available_models}")
            quit()

    # handle optimizer cfgs before the cli args are captured
    optimizer["init_args"]["lr"] = train.peak_lr
    
    hparams = capture_hparams()
    # for ease of use and cleaner accessing, has a to_dict method if necessary
    hparams = dict2attr(hparams)
    
    # FIXME there should be a more elegant way to do this, but I couldnt figure out the
    # runtime call order for the dataclass inits versus the overrides via yaml and cmdline.
    # handle the k toks spec using methods attached to the dataclass
    hparams.singleshot.k_toks = parse_k_toks(hparams.singleshot.k_toks)
    if hparams.singleshot.lockstep_rand_k_toks or hparams.singleshot.rand_rank_k_toks:
        hparams.singleshot.k_toks_min = parse_k_toks(hparams.singleshot.k_toks_min)
        hparams.singleshot.k_toks_max = parse_k_toks(hparams.singleshot.k_toks_max)

    # handle the topk values spec
    hparams.singleshot.topk_values = parse_topk_values(hparams.singleshot.topk_values)
    # handle the extra val trunc lengths spec
    hparams.singleshot.extra_val_trunc_lengths = parse_extra_val_trunc_lengths(hparams.singleshot.extra_val_trunc_lengths)
    # handle the extra val k toks values spec
    hparams.singleshot.extra_val_k_toks_values = parse_extra_val_k_toks_values(hparams.singleshot.extra_val_k_toks_values)

    # create the k_tok curriculum scheduler object
    if isinstance(hparams.singleshot.k_toks, list):
        hparams.singleshot.k_toks = PiecewiseKTokCurriculum(hparams.singleshot.k_toks)
    elif isinstance(hparams.singleshot.k_toks, int):
        hparams.singleshot.k_toks = ConstantKTokCurriculum(hparams.singleshot.k_toks)
    else:
        raise ValueError(f"Unexpected value passed for k_toks: {k_toks}")
    
    if isinstance(hparams.singleshot.k_toks_min, list):
        hparams.singleshot.k_toks_min = PiecewiseKTokCurriculum(hparams.singleshot.k_toks_min)
    elif isinstance(hparams.singleshot.k_toks_min, int):
        hparams.singleshot.k_toks_min = ConstantKTokCurriculum(hparams.singleshot.k_toks_min)
    else:
        assert not hparams.singleshot.lockstep_rand_k_toks, "lockstep_rand_k_toks is True but k_toks_min is not int or list."
    
    if isinstance(hparams.singleshot.k_toks_max, list):
        hparams.singleshot.k_toks_max = PiecewiseKTokCurriculum(hparams.singleshot.k_toks_max)
    elif isinstance(hparams.singleshot.k_toks_max, int):
        hparams.singleshot.k_toks_max = ConstantKTokCurriculum(hparams.singleshot.k_toks_max)
    else:
        assert not hparams.singleshot.lockstep_rand_k_toks, "lockstep_rand_k_toks is True but k_toks_max is not int or list."

    assert data is not None, "Please provide a data module or a data module name."

    config = Config.from_name(model_name) if model_config is None else model_config
    precision = precision or get_default_supported_precision(training=True)
    devices = parse_devices(devices)
    out_dir = init_out_dir(out_dir)
    # in case the dataset requires the Tokenizer
    tokenizer = Tokenizer(tokenizer_dir) if tokenizer_dir is not None else None

    # finally, if we aren't resuming we may need to adjust the tokenizer and embedding for the init ckpt
    hparams.singleshot.temp_sep_token_id_range = parse_temp_sep_token_id_range(hparams.singleshot.temp_sep_token_id_range)
    # do the same for the prelude_token_ids
    if hparams.pqds.prelude_token_ids is not None:
        hparams.pqds.prelude_token_ids = parse_prelude_token_ids(hparams.pqds.prelude_token_ids)

    # handle the alternate method of mask and temp token id specification
    if hparams.singleshot.mtp_special_token_pattern is not None:
        assert tokenizer_dir is not None, "If using mtp_special_token_pattern, please provide a tokenizer_dir to resolve the token ids."
        assert hparams.singleshot.mask_id is None and hparams.singleshot.temp_sep_token_id_range is None, "If using mtp_special_token_pattern, do not set mask_id or temp_sep_token_id_range."
        special_token_strs = [hparams.singleshot.mtp_special_token_pattern.format(i=i) for i in range(hparams.singleshot.num_mtp_special_tokens)]
        # We split the available mtp special tokens range into the ones we use, and then extras for the temp sep tokens we need in the val loop.
        if hparams.singleshot.pos_wise_unique_mtp_tok_ids:
            hparams.singleshot.min_mask_id = tokenizer.token_to_id(special_token_strs[0])
            max_k = hparams.singleshot.k_toks_max.max_k_toks_value if hparams.singleshot.k_toks_max is not None else hparams.singleshot.k_toks
            hparams.singleshot.max_mask_id = tokenizer.token_to_id(special_token_strs[max_k-1])
            hparams.singleshot.temp_sep_token_id_range = [tokenizer.token_to_id(special_token_strs[i]) for i in range(max_k, len(special_token_strs))]
            print(f"Resolved MTP special tokens: {special_token_strs} to ids: MASK_RANGE = {hparams.singleshot.min_mask_id} to {hparams.singleshot.max_mask_id}, TEMP_SEP_RANGE = {hparams.singleshot.temp_sep_token_id_range}")
        else:
            hparams.singleshot.mask_id = tokenizer.token_to_id(special_token_strs[0])
            hparams.singleshot.temp_sep_token_id_range = [tokenizer.token_to_id(special_token_strs[i]) for i in range(1, len(special_token_strs))]
            print(f"Resolved MTP special tokens: {special_token_strs} to ids: MASK_RANGE = {hparams.singleshot.mask_id}, TEMP_SEP_RANGE = {hparams.singleshot.temp_sep_token_id_range}")

    # logger = choose_logger( ... )
    assert logger_name == "wandb", "Only supporting wandb logging."
    logger = WandbLogger(
        entity=hparams.wandb.entity,
        project=hparams.wandb.project,
        name=hparams.wandb.run_name,
        group=hparams.wandb.group,
        save_dir=hparams.out_dir,
        tags=hparams.wandb.tags,
        offline=(hparams.wandb.offline or hparams.pdb),
    )
    
    # grab multi-node args from the env
    num_nodes = int(os.environ.get("SLURM_NNODES", num_nodes))
    devices = int(os.environ.get("SLURM_NTASKS")) // num_nodes if os.environ.get("SLURM_NTASKS") is not None else int(devices)
    hparams.num_nodes = num_nodes
    hparams.devices = devices

    mesh = None
    strategy = None
    torch_dist_timeout = timedelta(minutes=hparams.train.pt_dist_timeout_mins) if hparams.train.pt_dist_timeout_mins is not None else None
    if train.fabric_strategy == "ddp":
        # strategy = "ddp"
        strategy = DDPStrategy(process_group_backend="nccl", timeout=torch_dist_timeout)
    elif train.fabric_strategy=="fsdp" or (devices * num_nodes > 1):
        if train.fsdp_device_mesh is not None:
            mesh = tuple([int(elm) for elm in train.fsdp_device_mesh.split("x")])
        else:
            mesh = (num_nodes, devices)
        # strategy = FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full", sharding_strategy="HYBRID_SHARD", device_mesh=mesh)
        strategy = FSDPStrategy(
            auto_wrap_policy={Block}, 
            activation_checkpointing_policy=({Block} if train.fsdp_activation_checkpointing else None),
            state_dict_type=train.fsdp_state_dict_type, 
            sharding_strategy="HYBRID_SHARD", 
            device_mesh=mesh,
            process_group_backend="nccl",
            timeout=torch_dist_timeout, # Note that this really doesn't seem to have the correct effect on Tuo, but does on Daint.
            limit_all_gathers=train.fsdp_limit_all_gathers,
        )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, num_nodes=num_nodes, strategy=strategy, precision=precision, loggers=[logger])
    fabric.print(f"num_nodes: {num_nodes}, devices per node: {devices}, strategy: {strategy}, fsdp mesh: {mesh}")

    if torch.cuda.is_available() and devices > 1:
        check_nvlink_connectivity(fabric)

    fabric.launch()

    fabric.print(pprint.pformat(hparams.to_dict()))
    
    # if logger_name in ("tensorboard", "wandb", "mlflow"):
    fabric.logger.log_hyperparams(hparams.to_dict())


    main(
        hparams=hparams,
        fabric=fabric,
        devices=devices,
        num_nodes=num_nodes,
        seed=seed,
        initial_checkpoint_dir=initial_checkpoint_dir,
        resume=resume,
        config=config,
        data=data,
        pqds=pqds,
        out_dir=out_dir,
        tokenizer_dir=tokenizer_dir,
        tokenizer=tokenizer,
        train=train,
        eval=eval,
        optimizer=optimizer,
    )


def main(
    hparams: dict2attr,
    fabric: L.Fabric,
    devices: int,
    seed: int,
    initial_checkpoint_dir: Optional[Path],
    resume: Union[bool, Literal["auto"], Path],
    config: Config,
    data: Union[DataModule, str],
    pqds: PQDSArgs,
    out_dir: Path,
    tokenizer_dir: Optional[Path],
    tokenizer: Optional[Tokenizer],
    train: TrainArgs,
    eval: EvalArgs,
    optimizer: Union[str, Dict],
    num_nodes: int = 1,
) -> None:
    validate_args(hparams, train, eval, initial_checkpoint_dir, resume)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config, hparams, use_block_mask=hparams.singleshot.train_with_block_mask, is_teacher=False)
        model_teacher = GPT(config, hparams, use_block_mask=hparams.singleshot.train_with_block_mask, is_teacher=True)

    initialize_weights(fabric, model, n_layer=config.n_layer, n_embd=config.n_embd)
    initialize_weights(fabric, model_teacher, n_layer=config.n_layer, n_embd=config.n_embd)

    if train.tie_embeddings:
        model.transformer.wte.weight = model.lm_head.weight
        model_teacher.transformer.wte.weight = model_teacher.lm_head.weight
    if train.max_seq_length:
        model.max_seq_length = train.max_seq_length
        model_teacher.max_seq_length = train.max_seq_length
    if train.train_block_size is None:
        train.train_block_size = model.max_seq_length
    if eval.val_block_size is None:
        eval.val_block_size = model.max_seq_length

    # if training with block mask, we know we don't need a full truncation len's worth of raw input
    # this is currently just an automagic setting under the hood
    if hparams.singleshot.train_with_block_mask:
        if hparams.singleshot.lockstep_rand_k_toks or hparams.singleshot.rand_rank_k_toks:
            # we need to handle the range of raw input token values needed for all values of k
            # note that the largest input tok count is determined by the _min_ k value
            # logic is touchy, be careful
            assert hparams.train.max_seq_length == hparams.singleshot.truncation_length, "When using randomized k_toks with block masking, train.max_seq_length must equal singleshot.truncation_length."
            assert model.S == hparams.singleshot.truncation_length, "When using randomized k_toks with block masking, model.S must equal singleshot.truncation_length."
            min_K = hparams.singleshot.k_toks_min.min_k_toks_value - 1
            corresp_P = (model.S // model.mask_region_ct) - min_K
            train.train_block_size = (model.mask_region_ct * corresp_P) + min_K
        else:
            train.train_block_size = (model.mask_region_ct * model.P) + model.K

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters: {num_parameters(model):,}")
    fabric.print(f"Total parameters teacher: {num_parameters(model_teacher):,}")

    if hparams.train.do_compile:
        if hparams.train.dynamo_cache_size_limit is not None:
            # torch._dynamo.config.cache_size_limit = 32
            torch._dynamo.config.cache_size_limit = hparams.train.dynamo_cache_size_limit
            fabric.print(f"Setting torch._dynamo.config.cache_size_limit to {hparams.train.dynamo_cache_size_limit}")

        fabric.print(f"Using compile mode: '{hparams.train.compile_mode}'")
        if hparams.train.compile_mode is None:
            fabric.print(f"Compiling utilities.")
            compile_utilities()
            fabric.print(f"Compiling models.")
            model = torch.compile(model)
            model_teacher = torch.compile(model_teacher)
        else:
            fabric.print(f"Compiling utilities.")
            compile_utilities(compile_mode=hparams.train.compile_mode)
            fabric.print(f"Compiling models.")
            model = torch.compile(model, mode=hparams.train.compile_mode)
            model_teacher = torch.compile(model_teacher, mode=hparams.train.compile_mode)
    
    model = fabric.setup(model)
    model_teacher = fabric.setup(model_teacher)

    # Now that the model is not on meta tensor anymore...
    # Visualization, unnecessary
    if model.use_block_mask:
        def make_tensor():
            return torch.ones(hparams.train.micro_batch_size, model.config.n_query_groups, model.max_seq_length, model.H, device="cpu")
        step_idx = 0
        viz_name = f"mtp_causal_mask_step-{step_idx:08d}_P{model.P}_K{model.K}_S{model.S}_O{model.Ofs}_B{model.B}_H{model.H}_bda{model.bidirect_ss_attn}"
        base_save_path = f"{hparams.out_dir}/mtp_masks"
        os.makedirs(base_save_path, exist_ok=True)
        visualize_attention_scores(
            make_tensor(), # query
            make_tensor(), # key
            mask_mod=model.mask_mod, # interleaved_mtp_mask_mod,
            device="cpu",
            name=viz_name,
            path=Path(f"{base_save_path}/{viz_name}"),
        )
    # End visualization

    # disable all teacher gradients
    for param in model_teacher.parameters():
        param.requires_grad = False

    extra_kwargs = {"fused": fabric.device.type == "cuda"}

    optimizer = instantiate_torch_optimizer(optimizer, model.parameters(), **extra_kwargs)
    optimizer = fabric.setup_optimizers(optimizer)

    if data == "pqds":
        if hparams.pqds.pad_token_id is None:
            if tokenizer.pad_id is not None:
                pqds.pad_token_id = tokenizer.pad_id
                hparams.pqds.pad_token_id = tokenizer.pad_id
            elif tokenizer.eos_id is not None:
                pqds.pad_token_id = tokenizer.eos_id
                hparams.pqds.pad_token_id = tokenizer.eos_id
        assert hparams.pqds.pad_token_id is not None, "Please provide a pad_token_id for the Parquet dataset."
        print(f"Using prelude token ids of {hparams.pqds.prelude_token_ids} and pad token id of {hparams.pqds.pad_token_id} for pqds dataset mask processing.")

    train_dataloader, val_dataloader = get_dataloaders(hparams, fabric, data, pqds, tokenizer, train, train_block_size=train.train_block_size, val_block_size=eval.val_block_size)
    
    # extend seq len just for val data so there will be enough tokens of gt to cover the rollout multiplier
    if hparams.singleshot.rollout_multiplier > 1:
        adjustment = hparams.singleshot.k_toks.max_k_toks_value*(hparams.singleshot.rollout_multiplier-1)
        del val_dataloader
        _, val_dataloader = get_dataloaders(hparams, fabric, data, pqds, tokenizer, train, train_block_size=train.train_block_size, val_block_size=eval.val_block_size+adjustment)

    # the data sub obj has been manipulated since capture_hparams
    hparams.data = data

    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    if initial_checkpoint_dir:

        if train.ignore_extra_keys_in_init_ckpt:
            fabric.print(f"Loading initial checkpoint from {initial_checkpoint_dir} with strict=False to ignore extra keys.")
            fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", {"model": model}, strict=False)
            fabric.print(f"Finished loading initial checkpoint for student model with strict=False.")
            fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", {"model": model_teacher}, strict=False)
            fabric.print(f"Finished loading initial checkpoint for teacher model with strict=False.")
        else:
            fabric.print(f"Loading initial checkpoint from {initial_checkpoint_dir}")
            fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", model)
            fabric.print(f"Finished loading initial checkpoint for student model.")
            fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", model_teacher)
            fabric.print(f"Finished loading initial checkpoint for teacher model.")

    state = {
        "model": model,
        "model_teacher": model_teacher,
        "optimizer": optimizer,
        "train_dataloader": train_dataloader,
        "iter_num": 0,
        "step_count": 0,
    }


    resume = find_resume_path(fabric, resume, out_dir)
    if resume:
        fabric.print(f"Resuming training from {resume}")
        
        # prep the state dict with any necessary removals
        teacher_model_ref = state.pop("model_teacher")
        if train.ignore_dataloader_state_on_resume:
            train_dataloader = state.pop("train_dataloader")
        
        fabric.load(resume, state)
        
        # reinsert any removed items necessary
        state["model_teacher"] = teacher_model_ref
        if train.ignore_dataloader_state_on_resume:
            state["train_dataloader"] = train_dataloader

    train_time = time.perf_counter()

    # work around PyTorch issue https://github.com/pytorch/pytorch/issues/152162
    # which does not like the lazy initialization to be called in dynamo.
    # TODO: Happens with PyTorch 2.7+
    if (
        (_TORCH_EQUAL_2_7 or _TORCH_EQUAL_2_8)
        and (model._forward_module.__class__.__name__ == "OptimizedModule")
        and (model._forward_module._orig_mod.__class__.__name__ == "FullyShardedDataParallel")
    ):
        from torch.distributed.fsdp._runtime_utils import _root_pre_forward

        _root_pre_forward(model._forward_module._orig_mod, model._forward_module._orig_mod, [], {})
        _root_pre_forward(model_teacher._forward_module._orig_mod, model_teacher._forward_module._orig_mod, [], {})

    fit(
        hparams=hparams,
        fabric=fabric,
        devices=devices,
        num_nodes=num_nodes,
        state=state,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        out_dir=out_dir,
        tokenizer=tokenizer,
        tokenizer_dir=tokenizer_dir,
        train=train,
        eval=eval,
    )

    total_tokens = state["iter_num"] * train.micro_batch_size * model.max_seq_length * fabric.world_size

    # Print formatted output
    separator = "-" * 40
    fabric.print(separator)
    fabric.print("| Performance")
    fabric.print(f"| - Total tokens  : {total_tokens:,}")
    fabric.print(f"| - Training Time : {(time.perf_counter() - train_time):.2f} s")
    fabric.print(f"| - Tok/sec       : {total_tokens / train_time:.2f} tok/s")
    fabric.print("| " + "-" * 40)

    if fabric.device.type == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        fabric.print("| Memory Usage")
        fabric.print(f"| - Memory Used   : {memory_used:.2f} GB")
    fabric.print(separator)


def fit(
    hparams: dict2attr,
    fabric: L.Fabric,
    devices: int,
    state: dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    out_dir: Path,
    tokenizer: Tokenizer,
    tokenizer_dir: Optional[Path],
    train: TrainArgs,
    eval: EvalArgs,
    num_nodes: int = 1,
) -> None:
    model = state["model"]
    model_teacher = state["model_teacher"]
    optimizer = state["optimizer"]

    # Initial Save
    if train.initial_save and state["step_count"] == 0:
        fabric.print("Performing initial save ...")
        teacher_model_ref = state.pop("model_teacher")
        save_checkpoint(hparams, fabric, state, tokenizer_dir, out_dir / f"step-{state['step_count']:08d}" / "lit_model.pth")
        state["model_teacher"] = teacher_model_ref
        fabric.barrier()

    if eval.initial_validation:

        metrics = {"step": state["step_count"]}

        gen_val_t0 = time.perf_counter()
        val_lengths = [None]
        if hparams.singleshot.extra_val_trunc_lengths is not None:
            val_lengths += hparams.singleshot.extra_val_trunc_lengths
        val_k_toks_values = [None]
        if hparams.singleshot.extra_val_k_toks_values is not None:
            val_k_toks_values += hparams.singleshot.extra_val_k_toks_values
        for val_k_toks in val_k_toks_values:
            for val_len in val_lengths:
                # basename = "generation" if val_len is None else f"generation_slen{val_len}"
                basename = "generation"
                if val_k_toks is not None:
                    basename += f"_ktoks{val_k_toks}"
                if val_len is not None:
                    basename += f"_slen{val_len}"
                generations, gen_stats, gen_stats_lists, gen_losses, gen_losses_lists = generative_validate(hparams, state, fabric, model, model_teacher, tokenizer, val_dataloader, max_iters=eval.max_iters, truncation_length=val_len, k_toks=val_k_toks)
                
                flat_gen_stats = {}
                for col_name in gen_stats.keys():
                    for metric_name in gen_stats[col_name].keys():
                        flat_gen_stats[f"{basename}_stats/{col_name.replace(' ','_')}/{metric_name}"] = gen_stats[col_name][metric_name]
                
                flat_gen_stats_lists = {}
                for col_name in gen_stats_lists.keys():
                    for metric_name in gen_stats_lists[col_name].keys():
                        flat_gen_stats_lists[f"{col_name.replace(' ','_')}/{metric_name}"] = gen_stats_lists[col_name][metric_name]

                flat_gen_losses = {}
                for col_name in gen_losses.keys():
                    for metric_name in gen_losses[col_name].keys():
                        flat_gen_losses[f"{basename}_losses/{col_name.replace(' ','_')}/{metric_name}"] = gen_losses[col_name][metric_name]
                
                flat_gen_losses_lists = {}
                for col_name in gen_losses_lists.keys():
                        flat_gen_losses_lists[f"{col_name.replace(' ','_')}"] = gen_losses_lists[col_name]

                metrics.update(flat_gen_stats)
                metrics.update(flat_gen_losses)

                run_name = fabric.logger.experiment.name
                run_id = fabric.logger.experiment.id
                step = metrics["step"]
                num_records = len(list(flat_gen_stats_lists.values())[0])
                metadata_cols = {
                    "run_name": [run_name]*num_records,
                    "run_id": [run_id]*num_records,
                    "step": [step]*num_records,
                }
                flat_gen_stats_lists.update(metadata_cols)
                flat_gen_losses_lists.update(metadata_cols)

                gen_stats_table_columns = list(flat_gen_stats_lists.keys())
                gen_stats_table = Table(columns=gen_stats_table_columns)
                for stat_tup in zip(*list(flat_gen_stats_lists.values())):
                    gen_stats_table.add_data(*stat_tup)
                metrics[f"{basename}_gen_stats_table"] = gen_stats_table

                gen_losses_table_columns = list(flat_gen_losses_lists.keys())
                gen_losses_table = Table(columns=gen_losses_table_columns)
                for loss_tup in zip(*list(flat_gen_losses_lists.values())):
                    gen_losses_table.add_data(*loss_tup)
                metrics[f"{basename}_gen_losses_table"] = gen_losses_table
                
                gen_table_columns = ["Prompt", "GT Compl", "AR Teach Gen", "AR Stud Gen", "SS Stud Gen"]
                if hparams.singleshot.num_samples is not None:
                    gen_table_columns += [f"SS Stud Sample{i}"for i in range(1, hparams.singleshot.num_samples+1)]
                gen_table = Table(columns=gen_table_columns)
                for gen_tup in generations:
                    gen_table.add_data(*gen_tup)
                for k,v in metadata_cols.items():
                    gen_table.add_column(name=k, data=v)
                metrics[basename] = gen_table
        
        gen_val_t1 = time.perf_counter()
        fabric.print(f"Generative validation time: {gen_val_t1 - gen_val_t0:.2f} seconds.")
        metrics["time_gen_val"] = gen_val_t1 - gen_val_t0

        val_loss = validate(fabric, model, val_dataloader, max_iters=eval.max_iters).item()

        fabric.print("Init generations (prompt omitted) | ", [dict(zip(gen_table_columns[1:],tup[1:])) for tup in generations])
        fabric.print(f"Init validation | val loss: {val_loss:.3f} | val ppl: {math.exp(val_loss):.3f}")
        metrics.update({"val_loss": val_loss, "val_ppl": math.exp(val_loss)})
        fabric.log_dict(metrics, step=state["iter_num"] - 1)
        fabric.barrier()
    else:
        # fabric.print("Verifying settings ...")
        # validate(fabric, model, val_dataloader, max_iters=2, verbose=False)  # sanity check
        # generative_validate(hparams, state, fabric, model, model_teacher, tokenizer, val_dataloader, max_iters=2) # sanity check would be nice, but it's slow
        generations = "n/a"
        val_loss = "n/a"

    throughput = ThroughputMonitor(fabric, window_size=5)

    with torch.device("meta"):
        meta_model = GPT(model.config, hparams, use_block_mask=False)
        x = torch.randint(0, 1, (train.micro_batch_size, meta_model.max_seq_length))
        model_fwd = lambda: meta_model(x)  # noqa: F821
        model_loss = lambda y: chunked_cross_entropy(y, x, chunk_size=0)  # noqa: F821
        measured_flops = measure_flops(meta_model, model_fwd, model_loss)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    max_tokens_per_device = train.max_tokens // fabric.world_size
    tokens_per_iter = train.micro_batch_size * train.train_block_size
    max_iters = max_tokens_per_device // tokens_per_iter

    fabric.print(
        f"Max tokens for run: {train.max_tokens:,}\n"
        f"Max tokens per device: {max_tokens_per_device:,}\n"
        f"Tokens per iteration (per device): {tokens_per_iter:,}\n"
        f"Resulting max iterations: {max_iters:,}"
    )

    # estimate the resulting epochs based on dataloader and underlying dataset
    fabric.print(
        f"Len of the train dataset: {len(train_dataloader.dataset):,}\n"
        f"Len of the val dataset: {len(val_dataloader.dataset):,}\n"
        f"Len of the train dataloader: {len(train_dataloader):,}\n"
        f"Len of the val dataloader: {len(val_dataloader):,}"
    )
    if hparams.data == "pqds" and hparams.pqds.estimate_token_counts:
        if fabric.global_rank == 0:
            fabric.print("Estimating total token counts in PQDS dataset ...")
            pqds_est_tot_toks_raw, tp_row_stats_raw = train_dataloader.dataset._estimate_total_tokens(k=hparams.pqds.estimate_file_count)
            pqds_est_tot_toks_blocked, tp_row_stats_raw_blocked = train_dataloader.dataset._estimate_total_tokens(k=hparams.pqds.estimate_file_count, block_size=train_dataloader.dataset.block_size)
            pqds_est_tot_toks_raw_world = pqds_est_tot_toks_raw * fabric.world_size
            pqds_est_tot_toks_blocked_world = pqds_est_tot_toks_blocked * fabric.world_size
            fabric.print(
                f"PQDS: Estimated tokens in all files loading raw rows (world): {pqds_est_tot_toks_raw_world:,}\n"
                f"PQDS: Estimated tokens in all files loading raw rows (device): {pqds_est_tot_toks_raw:,}\n"
                f"PQDS: Estimated tokens per row loading raw rows, avg (min/max): {tp_row_stats_raw[0]:.02f} ({tp_row_stats_raw[1]:.02f}/{tp_row_stats_raw[2]:.02f})\n"
                f"PQDS: Estimated tokens in all files accounting for block_size truncation (world): {pqds_est_tot_toks_blocked_world:,}\n"
                f"PQDS: Estimated tokens in all files accounting for block_size truncation (device): {pqds_est_tot_toks_blocked:,}\n"
                f"PQDS: Estimated tokens per row accounting for block_size truncation, avg (min/max): {tp_row_stats_raw_blocked[0]:.02f} ({tp_row_stats_raw_blocked[1]:.02f}/{tp_row_stats_raw_blocked[2]:.02f})\n"
            )
    # Generically estimate tokens per epoch, and therefore num epochs with a few sanity checks for good measure
    assert train_dataloader.batch_size == hparams.train.micro_batch_size
    assert model.max_seq_length == hparams.train.max_seq_length
    # note that the data is set up to load one extra token wrt the target seq len so that the target
    # right shift by one still allows for seq len _trained_ toks per iter rather than slen - 1
    actual_dl_toks_per_iter = next(iter(train_dataloader)).numel()
    assert actual_dl_toks_per_iter - (1 * train.micro_batch_size) == tokens_per_iter, f"Tokens per iter {tokens_per_iter} != dataloader {actual_dl_toks_per_iter} even under rightshift correction."
    toks_in_train_dataloader_raw = len(train_dataloader) * actual_dl_toks_per_iter # raw
    toks_in_train_dataloader_sup = len(train_dataloader) * tokens_per_iter # trained on
    toks_in_full_train_dataset_raw = toks_in_train_dataloader_raw * fabric.world_size
    expected_epochs_iters = max_iters / len(train_dataloader) # per device
    expected_epochs_tokens = max_tokens_per_device / toks_in_train_dataloader_sup # per device
    fabric.print(
        f"Estimated tokens available in full train dataset: {toks_in_full_train_dataset_raw:,}\n"
        f"Estimated tokens in train dataloader to be loaded (device): {toks_in_train_dataloader_raw:,}\n"
        f"Estimated tokens in train dataloader to be trained on (device): {toks_in_train_dataloader_sup:,}\n"
        f"Expected epochs based on iteration target: {expected_epochs_iters:.03f}\n"
        f"Expected epochs based on token target (using trained on est): {expected_epochs_tokens:.03f}\n"
    )
    fabric.barrier()


    log_iter_interval = train.log_interval * train.gradient_accumulation_iters(devices, num_nodes)
    initial_iter = state["iter_num"]
    train_iterator = CycleIterator(train_dataloader)

    running_loss = RunningMean(window=train.gradient_accumulation_iters(devices, num_nodes), sync_on_compute=hparams.train.sync_running_metrics, nan_strategy="error").to(fabric.device)
    running_grad_norm = RunningMean(window=train.gradient_accumulation_iters(devices, num_nodes), sync_on_compute=hparams.train.sync_running_metrics, nan_strategy="error").to(fabric.device)

    running_ce_teach_stud = RunningMean(window=train.gradient_accumulation_iters(devices, num_nodes), sync_on_compute=hparams.train.sync_running_metrics).to(fabric.device)
    running_kl_teach_stud = RunningMean(window=train.gradient_accumulation_iters(devices, num_nodes), sync_on_compute=hparams.train.sync_running_metrics).to(fabric.device)
    running_ent_teach = RunningMean(window=train.gradient_accumulation_iters(devices, num_nodes), sync_on_compute=hparams.train.sync_running_metrics).to(fabric.device)
    running_ent_stud = RunningMean(window=train.gradient_accumulation_iters(devices, num_nodes), sync_on_compute=hparams.train.sync_running_metrics).to(fabric.device)
    
    running_gt_forced_prefix_loss = RunningMean(window=train.gradient_accumulation_iters(devices, num_nodes), sync_on_compute=hparams.train.sync_running_metrics).to(fabric.device)

    # extra metrics block
    if hparams.singleshot.extra_train_metrics:
        running_stud_forced_teach_loss = RunningMean(window=train.gradient_accumulation_iters(devices, num_nodes), sync_on_compute=hparams.train.sync_running_metrics).to(fabric.device)
        running_cu_pos_stud_forced_teach_losses = {} 
        running_pos_stud_forced_teach_losses = {}
        running_accuracies = {k:None for k in hparams.singleshot.topk_values}
        running_cu_pos_accuracies = {k:{} for k in hparams.singleshot.topk_values}
        running_pos_accuracies = {k:{} for k in hparams.singleshot.topk_values}
        running_confidences = {k:None for k in hparams.singleshot.topk_values}
        running_cu_pos_confidences = {k:{} for k in hparams.singleshot.topk_values}
        running_pos_confidences = {k:{} for k in hparams.singleshot.topk_values}
        for pos in range(hparams.singleshot.k_toks.max_k_toks_value):
            running_cu_pos_stud_forced_teach_losses[pos] = RunningMean(window=train.gradient_accumulation_iters(devices, num_nodes), sync_on_compute=hparams.train.sync_running_metrics).to(fabric.device)
            running_pos_stud_forced_teach_losses[pos] = RunningMean(window=train.gradient_accumulation_iters(devices, num_nodes), sync_on_compute=hparams.train.sync_running_metrics).to(fabric.device)
            for k in hparams.singleshot.topk_values:
                running_cu_pos_accuracies[k][pos] = RunningMean(window=train.gradient_accumulation_iters(devices, num_nodes), sync_on_compute=hparams.train.sync_running_metrics).to(fabric.device)
                running_pos_accuracies[k][pos] = RunningMean(window=train.gradient_accumulation_iters(devices, num_nodes), sync_on_compute=hparams.train.sync_running_metrics).to(fabric.device)
                running_cu_pos_confidences[k][pos] = RunningMean(window=train.gradient_accumulation_iters(devices, num_nodes), sync_on_compute=hparams.train.sync_running_metrics).to(fabric.device)
                running_pos_confidences[k][pos] = RunningMean(window=train.gradient_accumulation_iters(devices, num_nodes), sync_on_compute=hparams.train.sync_running_metrics).to(fabric.device)
        for k in hparams.singleshot.topk_values:
            running_accuracies[k] = RunningMean(window=train.gradient_accumulation_iters(devices, num_nodes), sync_on_compute=hparams.train.sync_running_metrics).to(fabric.device)
            running_confidences[k] = RunningMean(window=train.gradient_accumulation_iters(devices, num_nodes), sync_on_compute=hparams.train.sync_running_metrics).to(fabric.device)
            
    fabric.barrier()
    total_t0 = time.perf_counter()

    # Clear memory
    if fabric.global_rank == 0:
        fabric.print("Manual GC and CUDA cache clear...")
    gc.collect()
    torch.cuda.empty_cache()

    warmup_iters = train.warmup_iters(devices, num_nodes, max_iters, train_dataloader)

    for train_data in train_iterator:
        if state["iter_num"] >= max_iters:
            break

        iter_t0 = time.perf_counter()

        # determine and set the learning rate for this iteration
        lr = get_lr(hparams.train.lr_schedule, optimizer.defaults["lr"], state["iter_num"], warmup_iters, max_iters, train.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        state["iter_num"] += 1

        data_mask_prep_t0 = time.perf_counter()
        train_data_len = train_data.shape[1] - 1
        input_ids = train_data[:, 0 : train_data_len].contiguous().long()
        target_ids = train_data[:, 1 : (train_data_len + 1)].contiguous().long()

        # Read up front from hparams (not from `model.S`/`model.mask_region_ct`):
        # validate() reconstructs the block mask on this same shared model with its
        # own S/mask_region_ct, which otherwise leaks into the P/offset math below
        # on the very next training step.
        truncation_length = hparams.singleshot.truncation_length
        mask_region_ct = hparams.singleshot.mask_region_ct

        if hparams.singleshot.rand_rank_k_toks:
            # rank based random k_toks sampling for multi-node multi-gpu runs
            rank_step_seed = state["step_count"] + fabric.global_rank * 12345678
            k_toks_rng = random.Random(rank_step_seed)
            k_toks_min = hparams.singleshot.k_toks_min.get_value(state["step_count"])
            k_toks_max = hparams.singleshot.k_toks_max.get_value(state["step_count"])
            k_toks = k_toks_rng.randint(k_toks_min, k_toks_max)
            # now, we need to reset the value of P because it will be used in offset computation below
            # but is now stale because we have a different k_toks than before
            model.P = (truncation_length // mask_region_ct) - (k_toks - 1)
            model.block_mask_config["P"] = model.P
            model_teacher.P = model.P
            model_teacher.block_mask_config["P"] = model.P
        elif hparams.singleshot.lockstep_rand_k_toks:
            # sample a random k_toks between min and max for this step
            step_seed = state["step_count"] * 12345
            k_toks_rng = random.Random(step_seed)
            k_toks_min = hparams.singleshot.k_toks_min.get_value(state["step_count"])
            k_toks_max = hparams.singleshot.k_toks_max.get_value(state["step_count"])
            k_toks = k_toks_rng.randint(k_toks_min, k_toks_max)
            # now, we need to reset the value of P because it will be used in offset computation below
            # but is now stale because we have a different k_toks than before
            model.P = (truncation_length // mask_region_ct) - (k_toks - 1)
            model.block_mask_config["P"] = model.P
            model_teacher.P = model.P
            model_teacher.block_mask_config["P"] = model.P
        else:
            k_toks = hparams.singleshot.k_toks.get_value(state["step_count"])

        # Single source of truth for P this step, independent of whatever
        # validate() may have mutated on model.block_mask_config since the last
        # training step ran.
        current_P = (truncation_length // mask_region_ct) - (k_toks - 1)

        data_bsz = input_ids.shape[0]

        tot_mask_regions = data_bsz * mask_region_ct # is also the total prefix regions, gt regions etc

        # dynamically reconstruct the block masks for this k_toks, trunc length, bsz etc.
        rolling_offset = 0
        if hparams.singleshot.train_with_block_mask:

            if hparams.singleshot.roll_offsets:
                if hparams.singleshot.rand_rank_roll_offsets:
                    # rank based offset rolling for multi-node multi-gpu runs
                    rank_step_seed = state["step_count"] + fabric.global_rank * 12345678
                    offset_rng = random.Random(rank_step_seed)
                    rank_factor = offset_rng.randint(0, current_P - 1)
                    rolling_offset = rank_factor * -1
                elif hparams.singleshot.lockstep_rand_roll_offsets:
                    # lockstep random offset rolling across all ranks
                    step_seed = state["step_count"] * 12345
                    offset_rng = random.Random(step_seed)
                    factor = offset_rng.randint(0, current_P - 1)
                    rolling_offset = factor * -1
                else:
                    rolling_offset = state["step_count"] % current_P * -1

            if hparams.pdb: print(f"Rank: {fabric.global_rank} | Step: {state['step_count']} | Sampled k_toks: {k_toks} | Rolling Offset: {rolling_offset}")

            model.reconstruct_block_mask(
                K = k_toks - 1,
                S = truncation_length,
                B = data_bsz,
                mask_region_ct = mask_region_ct,
                offset = rolling_offset,
                bidirect_ss_attn=hparams.singleshot.bidirect_ss_attn,
                # device = fabric.device, # not absolutely sure which instantiation style is faster
            )
            model_teacher.reconstruct_block_mask(
                K = k_toks - 1,
                S = truncation_length,
                B = data_bsz,
                mask_region_ct = mask_region_ct,
                offset = rolling_offset,
                bidirect_ss_attn=False, # we don't want the teacher to have this on
                # device = fabric.device, # not absolutely sure which instantiation style is faster
            )
        
        pad_token_id = None
        tot_nonpad_tokens = None
        if hparams.data == "pqds" and hparams.pqds.omit_tail_padding_in_loss:
            pad_token_id = hparams.pqds.pad_token_id

            # before we do anything, count the number of non-pad tokens in the input for logging later
            tot_nonpad_tokens = (input_ids != pad_token_id).sum().item()
            tot_pad_tokens = (input_ids == pad_token_id).sum().item()
            if hparams.pdb: print(f"Rank: {fabric.global_rank} | Step: {state['step_count']} | Fraction of total that are non-pad: {tot_nonpad_tokens}/{input_ids.numel()} = {tot_nonpad_tokens/input_ids.numel():.4f}, pads: {tot_pad_tokens}/{input_ids.numel()} = {tot_pad_tokens/input_ids.numel():.4f}")
        prelude_token_ids = hparams.pqds.prelude_token_ids

        orig_input_ids = input_ids.clone().detach()
        input_ids, target_ids, mask_id_mask, pred_pos_mask, prefix_pos_mask = truncate_and_mask(
            input_ids=input_ids, 
            target_ids=target_ids, 
            k_toks=k_toks, 
            mask_id=hparams.singleshot.mask_id,
            min_mask_id=hparams.singleshot.min_mask_id,
            max_mask_id=hparams.singleshot.max_mask_id,
            truncation_length=truncation_length, 
            mask_region_ct=mask_region_ct, 
            offset=rolling_offset, 
            pad_token_id=pad_token_id, 
            prelude_token_ids=prelude_token_ids, 
            verbose=hparams.pdb
        )

        # for now we assert that we have the expected number of predicted elems to extract
        # as a weak check for all complete, contig masked regions, there are still ways for this to
        # be wrong with true mask rollover, but this will catch some errors
        tot_pred_elems = pred_pos_mask.sum().item()
        expected_tot_pred_elems = tot_mask_regions * k_toks
        if hparams.pdb: print(f"Rank: {fabric.global_rank} | Step: {state['step_count']} | Fraction of expected predicted elements: {tot_pred_elems}/{expected_tot_pred_elems} = {tot_pred_elems/expected_tot_pred_elems:.4f}")

        if hparams.data == "pqds" and hparams.pqds.omit_tail_padding_in_loss:
            assert tot_pred_elems % k_toks == 0, f"Total predicted elements {tot_pred_elems} not multiple of k_toks {k_toks}."
            old_tot_mask_regions = tot_mask_regions
            tot_mask_regions = tot_pred_elems // k_toks
            # if old_tot_mask_regions != tot_mask_regions: fabric.print(f"Adjusted tot_mask_regions from {old_tot_mask_regions} to {tot_mask_regions} due to tail padding.")
        else:
            assert tot_pred_elems == expected_tot_pred_elems, (
                f"Total predicted elements {tot_pred_elems} != expected {expected_tot_pred_elems}"
                f"This might be because of offset rollover that cuts off some mask regions."
            )

        new_settings_hash = hashlib.shake_128(str(model.block_mask_config).encode()).hexdigest(8)
        
        if hparams.singleshot.log_masks_and_inputs and (new_settings_hash not in model.old_settings_hashset):
            fabric.print(f"Reconstructed new block mask for k_toks: {k_toks}, trunc_length: {truncation_length}, offset: {rolling_offset}, bsz: {data_bsz}, mask_region_ct: {mask_region_ct}")

            # Visualization, unnecessary
            def make_tensor():
                return torch.ones(model.B, model.config.n_query_groups, model.S, model.H, device="cpu")
            step_idx = state["step_count"]
            viz_name = f"mtp_causal_mask_step-{step_idx:08d}_P{model.P}_K{model.K}_S{model.S}_O{model.Ofs}_B{model.B}_H{model.H}_bda{model.bidirect_ss_attn}"
            base_save_path = f"{hparams.out_dir}/mtp_masks"
            os.makedirs(base_save_path, exist_ok=True)
            visualize_attention_scores(
                make_tensor(), # query
                make_tensor(), # key
                mask_mod=model.mask_mod, # interleaved_mtp_mask_mod,
                device="cpu",
                name=viz_name,
                path=Path(f"{base_save_path}/{viz_name}"),
            )
            # End visualization
            # log one set of all prepped data for first iter only or when mask changes
            # if state["iter_num"] == 1 or (new_settings_hash not in model.old_settings_hashset):
            if fabric.global_rank == 0:
                fabric.print("Sample preprocessed training data (only shown when block masking changes):")
                for i in range(min(1, input_ids.shape[0])):
                    orig_inp_ids = orig_input_ids[i].cpu().tolist()
                    inp_ids = input_ids[i].cpu().tolist()
                    tgt_ids = target_ids[i].cpu().tolist()
                    mask_id_mask_ids = mask_id_mask[i].cpu().tolist()
                    pred_pos_mask_ids = pred_pos_mask[i].cpu().tolist()
                    fabric.print(f"Original Input IDs {i}: {orig_inp_ids}")
                    fabric.print(f"Input IDs  {i}: {inp_ids}")
                    fabric.print(f"Target IDs {i}: {tgt_ids}")
                    fabric.print(f"Mask ID Mask {i}: {mask_id_mask_ids}")
                    fabric.print(f"Pred Pos Mask{i}: {pred_pos_mask_ids}")
        
        # if rank rolling, not all will save a new mask at the same time, so this must come after the logging block
        fabric.barrier() 
        model.old_settings_hashset.add(new_settings_hash)

        data_mask_prep_t1 = time.perf_counter()

        is_accumulating = state["iter_num"] % train.gradient_accumulation_iters(devices, num_nodes) != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            
            fwd_t0 = time.perf_counter()
            # 1. student prediction pass
            logits = model(input_ids)
            soft_stud_preds = logits[pred_pos_mask].view(tot_mask_regions, k_toks, -1)

            # 1.1 prep for aux prefix supervision
            stud_prefix_logits = logits[prefix_pos_mask].view(tot_mask_regions, -1, logits.shape[-1])
            gt_prefix_ids = target_ids[prefix_pos_mask].view(tot_mask_regions, -1)
            gt_suffix_ids = target_ids[pred_pos_mask].view(tot_mask_regions, -1)

            # 2. student forced teacher feedback pass
            # student output discretization
            if hparams.singleshot.sample_during_train:
                bsz,num_pos,vocab = soft_stud_preds.shape
                hard_stud_preds = torch.multinomial(F.softmax(soft_stud_preds, dim=-1).reshape(-1,vocab), num_samples=1).reshape(bsz,k_toks)
            else:
                hard_stud_preds = torch.argmax(soft_stud_preds, dim=-1)
            
            # prep the student forcing sequence for the teacher
            # starting with the original prefix for the sequence before mask positions
            stud_forcing_input_ids = input_ids.clone()
            if k_toks-1 > 0:
                # then 'cat' the student predictions to follow the prefix if there was more than 1 
                stud_forcing_input_ids[mask_id_mask] = hard_stud_preds[:,:(k_toks-1)].reshape(-1)
            
            # pass the prompt prefix and the student preds through the teacher
            # note: considered omitting ctx mgr as teach req's grads == False and used ~1gb more mem
            # but reinstated while debugging speed issues with multi-region

            if not hparams.singleshot.gt_teacher_supervision:
                with torch.no_grad():
                    logits_teacher = model_teacher(stud_forcing_input_ids)
                    soft_teach_preds = logits_teacher[pred_pos_mask].view(tot_mask_regions, k_toks, -1)
                    hard_teach_preds = torch.argmax(soft_teach_preds, dim=-1)

            if hparams.singleshot.last_region_loss_only:
                assert not (hparams.data == "pqds" and hparams.pqds.omit_tail_padding_in_loss), "Not supported to omit tail padding when using last_region_loss_only."
                # we consider only the last region per sequence for loss and metrics
                # need to reshape the student soft and hard and teacher soft and hard preds
                # into the original data_bsz x num_regions x whatever, then slice to last region only, then reshape back
                if mask_region_ct > 1:
                    soft_stud_preds = soft_stud_preds.view(data_bsz, mask_region_ct, k_toks, -1)[:,-1:,:,:].reshape(data_bsz, k_toks, -1)
                    hard_stud_preds = hard_stud_preds.view(data_bsz, mask_region_ct, k_toks)[:,-1:,:].reshape(data_bsz, k_toks)
                    soft_teach_preds = soft_teach_preds.view(data_bsz, mask_region_ct, k_toks, -1)[:,-1:,:,:].reshape(data_bsz, k_toks, -1)
                    hard_teach_preds = hard_teach_preds.view(data_bsz, mask_region_ct, k_toks)[:,-1:,:].reshape(data_bsz, k_toks)
                    # we now have fewer than tot_mask_regions
                    # but we no longer technically need that var, as the metric loops below size themselves
                    # however, if we want to use the mask_id_mask or pred_pos_mask later they would be incorrect
                    # but we can't overwrite them bc that would be an in-place op error and break autograd
            
            fwd_t1 = time.perf_counter()

            metrics_t0 = time.perf_counter()
            # extra metrics block
            if hparams.singleshot.extra_train_metrics:
                # 2.1 Compute a bunch of metrics comparing the stud and teach predictions
                stud_forced_teach_loss = nll_metric(logits=soft_teach_preds, tok_ids=hard_stud_preds)
                avg_stud_forced_teach_loss = torch.mean(stud_forced_teach_loss[0],dim=0)
                avg_cu_pos_stud_forced_teach_loss = torch.mean(stud_forced_teach_loss[1],dim=0)
                avg_pos_stud_forced_teach_loss = torch.mean(stud_forced_teach_loss[2],dim=0)

                # set the resulting loop dim based on whatever we have from logic above
                metrics_bsz = hard_stud_preds.shape[0]
                if not hparams.singleshot.last_region_loss_only:
                    assert metrics_bsz == tot_mask_regions, "Should be equal as both are num masked regions."

                # compute topk cumulative confidences
                k_values = hparams.singleshot.topk_values
                topk_cu_confidences = {k: [] for k in k_values}

                for i in range(metrics_bsz):
                    topk_cu_confs = topk_tok_cu_confidence(logits=soft_teach_preds[i], k_values=k_values)
                    for k in k_values:
                        topk_cu_confidences[k].append(topk_cu_confs[k])
                
                avg_conf = {k: sum([tup[0] for tup in topk_cu_confidences[k]])/metrics_bsz for k in k_values}
                avg_cu_pos_confs = {k: torch.sum(torch.stack([tup[1] for tup in topk_cu_confidences[k]],dim=0),dim=0)/metrics_bsz for k in k_values}
                avg_pos_confs = {k: torch.sum(torch.stack([tup[2].float() for tup in topk_cu_confidences[k]],dim=0),dim=0)/metrics_bsz for k in k_values}
                
                # compute topk accuracies
                k_values = hparams.singleshot.topk_values
                accuracies = {k: [] for k in k_values}

                for i in range(metrics_bsz):
                    stud_teach_topk_accs = topk_tok_accuracy(pred_toks=hard_stud_preds[i], gt_logits=soft_teach_preds[i], k_values = k_values)
                    for k in k_values: 
                        accuracies[k].append(stud_teach_topk_accs[k])

                avg_acc = {k: sum([tup[0] for tup in accuracies[k]])/metrics_bsz for k in k_values}
                avg_cu_pos_accs = {k: torch.sum(torch.stack([tup[1] for tup in accuracies[k]],dim=0),dim=0)/metrics_bsz for k in k_values}
                avg_pos_accs = {k: torch.sum(torch.stack([tup[2].float() for tup in accuracies[k]],dim=0),dim=0)/metrics_bsz for k in k_values}

            metrics_t1 = time.perf_counter()

            main_loss_t0 = time.perf_counter()
            # 3. full loss calculation based on stud preds and stud-forced teacher preds
            # assert not (hparams.singleshot.hard_teacher_supervision and hparams.singleshot.gt_teacher_supervision), "One or the other."
            assert sum([hparams.singleshot.hard_teacher_supervision, hparams.singleshot.gt_teacher_supervision, hparams.singleshot.hard_self_teacher_supervision]) <= 1, "Supervision styles are mutually exclusive, but more than one was selected."
            if hparams.singleshot.hard_teacher_supervision:
                loss_terms = pt_ce_plus_ent_loss(logits_student=soft_stud_preds, logits_teacher=soft_teach_preds, labels_teacher=hard_teach_preds, beta=hparams.singleshot.beta)
            elif hparams.singleshot.gt_teacher_supervision:
                loss_terms = pt_ce_plus_ent_loss(logits_student=soft_stud_preds, logits_teacher=None, labels_teacher=gt_suffix_ids, beta=hparams.singleshot.beta)
            elif hparams.singleshot.hard_self_teacher_supervision:
                # at the last second here, we isolate only the positions where the student's hard pred equals the teacher's hard pred, and only use those for supervision.
                match_mask = hard_stud_preds == hard_teach_preds
                masked_soft_stud_preds = soft_stud_preds[match_mask]
                masked_hard_stud_preds = hard_stud_preds[match_mask]
                if masked_soft_stud_preds.shape[0] == 0:
                    # if there are no matches, we can't compute a loss, so we skip the loss computation and backward step for this iteration
                    if fabric.global_rank == 0:
                        fabric.print(f"Step {state['step_count']}: No matches between student and teacher hard predictions, skipping loss computation and backward step for this iteration.")
                    optimizer.zero_grad() # probably not required due to no backward.
                    continue
                loss_terms = pt_ce_plus_ent_loss(logits_student=masked_soft_stud_preds, logits_teacher=None, labels_teacher=masked_hard_stud_preds, beta=hparams.singleshot.beta)
            else:
                loss_terms = pt_ce_plus_ent_loss(logits_student=soft_stud_preds, logits_teacher=soft_teach_preds, beta=hparams.singleshot.beta)
            loss, ce_teach_stud, kl_teach_stud, ent_teach, ent_stud = loss_terms
            main_loss_t1 = time.perf_counter()
            
            prefix_loss_t0 = time.perf_counter()
            # 3.1 add aux prefix loss if we're using this
            # Note that new logic can cause there to be no prefix tokens at all
            if gt_prefix_ids.shape[1] > 0:
                gt_forced_prefix_loss = nll_metric(logits=stud_prefix_logits, tok_ids=gt_prefix_ids) 
                avg_gt_forced_prefix_loss = torch.mean(gt_forced_prefix_loss[0],dim=0) # we only need the per seq elm 0
            else:
                avg_gt_forced_prefix_loss = torch.tensor(0.0, device=fabric.device)
            
            prefix_loss_t1 = time.perf_counter()
                
            if hparams.singleshot.supervise_prefix:
                loss = loss + avg_gt_forced_prefix_loss # implicit equal weighting despite uneven num toks
            elif hparams.singleshot.supervise_prefix_only:
                assert gt_prefix_ids.shape[1] > 0, "There must be some prefix tokens to supervise if using this setting."
                loss = avg_gt_forced_prefix_loss

            bwd_t0 = time.perf_counter()       
            fabric.backward(loss / train.gradient_accumulation_iters(devices, num_nodes))
            bwd_t1 = time.perf_counter()       

        post_bwd_t0 = time.perf_counter()
        loss_rank0 = loss.detach()
        running_loss.update(loss_rank0)

        running_ce_teach_stud.update(ce_teach_stud.detach())
        running_kl_teach_stud.update(kl_teach_stud.detach())
        running_ent_teach.update(ent_teach.detach())
        running_ent_stud.update(ent_stud.detach())

        running_gt_forced_prefix_loss.update(avg_gt_forced_prefix_loss.detach())

        # extra metrics block
        if hparams.singleshot.extra_train_metrics:
            running_stud_forced_teach_loss.update(avg_stud_forced_teach_loss.detach())
            for k in hparams.singleshot.topk_values:
                running_accuracies[k].update(avg_acc[k])
                running_confidences[k].update(avg_conf[k])
            
            for pos in range(k_toks):
                running_cu_pos_stud_forced_teach_losses[pos].update(avg_cu_pos_stud_forced_teach_loss[pos].detach())
                running_pos_stud_forced_teach_losses[pos].update(avg_pos_stud_forced_teach_loss[pos].detach())
                for k in hparams.singleshot.topk_values:
                    running_cu_pos_accuracies[k][pos].update(avg_cu_pos_accs[k][pos].detach())
                    running_pos_accuracies[k][pos].update(avg_pos_accs[k][pos].detach())
                    running_cu_pos_confidences[k][pos].update(avg_cu_pos_confs[k][pos].detach())
                    running_pos_confidences[k][pos].update(avg_pos_confs[k][pos].detach())
        post_bwd_t1 = time.perf_counter()

        grad_clip_t0, grad_clip_t1 = 0.0, 0.0
        optim_step_t0, optim_step_t1 = 0.0, 0.0
        if not is_accumulating:
            grad_clip_t0 = time.perf_counter()
            grad_norm = fabric.clip_gradients(model, optimizer, max_norm=train.max_norm)
            running_grad_norm.update(grad_norm)
            grad_clip_t1 = time.perf_counter()
            
            optim_step_t0 = time.perf_counter()
            optimizer.step()
            optimizer.zero_grad()
            optim_step_t1 = time.perf_counter()
            state["step_count"] += 1

        if state["iter_num"] % log_iter_interval == 0:

            log_calcs_t0 = time.perf_counter()
            loss = running_loss.compute().item()  # sync_on_compute=False makes this cheap, but local only
            
            grad_norm = running_grad_norm.compute().item()
            ce_teach_stud = running_ce_teach_stud.compute().item()
            kl_teach_stud = running_kl_teach_stud.compute().item()
            ent_teach = running_ent_teach.compute().item()
            ent_stud = running_ent_stud.compute().item()

            gt_forced_prefix_loss = {
                f"gt_forced_prefix_loss/avg":running_gt_forced_prefix_loss.compute().item()
            }
            # extra metrics block
            if hparams.singleshot.extra_train_metrics:
                stud_forced_teach_loss = {
                    f"stud_forced_teach_loss/avg":running_stud_forced_teach_loss.compute().item()
                }
                accuracies = {}
                confidences = {}
                for k in hparams.singleshot.topk_values:
                    accuracies[f"top{k}_acc/avg"] = running_accuracies[k].compute().item()
                    confidences[f"top{k}_conf/avg"] = running_confidences[k].compute().item()
                
                cu_pos_stud_forced_teach_losses = {}
                pos_stud_forced_teach_losses = {}
                cu_pos_accuracies = {}
                pos_accuracies = {}
                cu_pos_confidences = {}
                pos_confidences = {}
                for pos in range(k_toks):
                    cu_pos_stud_forced_teach_losses[f"stud_forced_teach_loss/cu_pos{pos}"] = running_cu_pos_stud_forced_teach_losses[pos].compute().item()
                    pos_stud_forced_teach_losses[f"stud_forced_teach_loss/pos{pos}"] = running_pos_stud_forced_teach_losses[pos].compute().item()
                    for k in hparams.singleshot.topk_values:
                        cu_pos_accuracies[f"top{k}_acc/cu_pos{pos}"] = running_cu_pos_accuracies[k][pos].compute().item()
                        pos_accuracies[f"top{k}_acc/pos{pos}"] = running_pos_accuracies[k][pos].compute().item()
                        cu_pos_confidences[f"top{k}_conf/cu_pos{pos}"] = running_cu_pos_confidences[k][pos].compute().item()
                        pos_confidences[f"top{k}_conf/pos{pos}"] = running_pos_confidences[k][pos].compute().item()

            t1 = time.perf_counter()
            throughput.update(
                time=(t1 - total_t0),
                flops=(measured_flops * log_iter_interval),
                batches=state["iter_num"],
                samples=(state["iter_num"] * train.micro_batch_size),
                lengths=(state["iter_num"] * train.micro_batch_size * model.max_seq_length),
            )

            max_memory_allocated_per_gpu = torch.cuda.max_memory_allocated(fabric.device) / 1024**3
            max_memory_reserved_per_gpu = torch.cuda.max_memory_reserved(fabric.device) / 1024**3
            torch.cuda.reset_peak_memory_stats(fabric.device)

            # this is the traditional tokens := hstates per second metric
            toks_per_iter_per_device = train.micro_batch_size * model.max_seq_length
            toks_per_sec_per_device = toks_per_iter_per_device / (t1 - iter_t0)
            toks_per_iter_world = toks_per_iter_per_device * fabric.world_size
            toks_per_sec_world = toks_per_iter_world / (t1 - iter_t0)
            
            # this is our singleshot supervised tokens per second, which might be the important one
            tot_pred_elems = pred_pos_mask.sum().item()
            if hparams.singleshot.last_region_loss_only:
                tot_pred_elems /= mask_region_ct # as we sliced to last region only
            assert hard_stud_preds.numel() == tot_pred_elems, "Should be equal as both are num final predicted elements incl in loss."
            sup_toks_per_iter_per_device = tot_pred_elems
            sup_toks_per_sec_per_device = sup_toks_per_iter_per_device / (t1 - iter_t0)
            sup_toks_per_iter_world = sup_toks_per_iter_per_device * fabric.world_size
            sup_toks_per_sec_world = sup_toks_per_iter_world / (t1 - iter_t0)

            # and then this is the raw ingestion rate, which depending on the regions can be much lower
            # since the dataloader is configured to give only the min raw unique inputs required
            if tot_nonpad_tokens is not None:
                # it's iffy how to count this, but it bounded depending on the scenario
                consumed_toks_per_iter_per_device = min(tot_nonpad_tokens, train.micro_batch_size * train.train_block_size)
            else:
                consumed_toks_per_iter_per_device = train.micro_batch_size * train.train_block_size
            consumed_toks_per_sec_per_device = consumed_toks_per_iter_per_device / (t1 - iter_t0)
            consumed_toks_per_iter_world = consumed_toks_per_iter_per_device * fabric.world_size
            consumed_toks_per_sec_world = consumed_toks_per_iter_world / (t1 - iter_t0)
            log_calcs_t1 = time.perf_counter()
            

            metrics = {
                "loss": loss,
                "loss_rank0": loss_rank0, # this gives an understanding of the spread
                "grad_norm": grad_norm,
                "k_toks": k_toks, # this is the current k_toks used, which may be changing over time
                "ce_teach_stud": ce_teach_stud,
                "kl_teach_stud": kl_teach_stud,
                "ent_teach": ent_teach,
                "ent_stud": ent_stud,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": (train_iterator.iterable.dataset.epoch if hparams.data=="pqds" else train_iterator.epoch),
                "iter_time": t1 - iter_t0,
                "time_data_mask_prep": data_mask_prep_t1 - data_mask_prep_t0,
                "time_fwd": fwd_t1 - fwd_t0,
                "time_metrics": metrics_t1 - metrics_t0,
                "time_main_loss": main_loss_t1 - main_loss_t0,
                "time_prefix_loss": prefix_loss_t1 - prefix_loss_t0,
                "time_bwd": bwd_t1 - bwd_t0,
                "time_post_bwd": post_bwd_t1 - post_bwd_t0,
                "time_grad_clip": grad_clip_t1 - grad_clip_t0,
                "time_optim_step": optim_step_t1 - optim_step_t0,
                "time_log_calcs": log_calcs_t1 - log_calcs_t0,
                "time_iter_total": t1 - iter_t0,
                "remaining_time": (
                    (t1 - total_t0) / (state["iter_num"] - initial_iter) * (max_iters - state["iter_num"])
                ),
                "tokens": state["iter_num"] * toks_per_iter_per_device,
                "total_tokens": state["iter_num"] * toks_per_iter_world,
                "sup_tokens": state["iter_num"] * sup_toks_per_iter_per_device,
                "total_sup_tokens": state["iter_num"] * sup_toks_per_iter_world,
                "consumed_tokens": state["iter_num"] * consumed_toks_per_iter_per_device,
                "total_consumed_tokens": state["iter_num"] * consumed_toks_per_iter_world,
                "learning_rate": lr,
                "max_mem_allocated_per_gpu": max_memory_allocated_per_gpu,
                "max_mem_reserved_per_gpu": max_memory_reserved_per_gpu,
                "toks_per_sec_world": toks_per_sec_world,
                "toks_per_sec_per_device": toks_per_sec_per_device,
                "sup_toks_per_sec_world": sup_toks_per_sec_world,
                "sup_toks_per_sec_per_device": sup_toks_per_sec_per_device,
                "consumed_toks_per_sec_world": consumed_toks_per_sec_world,
                "consumed_toks_per_sec_per_device": consumed_toks_per_sec_per_device,
            }
            metrics.update(gt_forced_prefix_loss)
            # extra metrics block
            if hparams.singleshot.extra_train_metrics:
                metrics.update(stud_forced_teach_loss)
                metrics.update(cu_pos_stud_forced_teach_losses)
                metrics.update(pos_stud_forced_teach_losses)
                metrics.update(accuracies)
                metrics.update(cu_pos_accuracies)
                metrics.update(pos_accuracies)
                metrics.update(confidences)
                metrics.update(cu_pos_confidences)
                metrics.update(pos_confidences)

            if isinstance(val_loss, float):
                val_loss = f"{val_loss:.3f}"
            if fabric.global_rank == 0:
                fabric.print(
                    f"Epoch {metrics['epoch'] + 1} | iter {metrics['iter']} step {metrics['step']} |"
                    f" loss train: {metrics['loss']:.3f},"
                    f" val: {val_loss} |"
                    f" grad_norm: {grad_norm:.3f} |"
                    f" lr: {metrics['learning_rate']:.3e} |"
                    # f" ce_teach_stud:{ce_teach_stud:.5f}, kl_teach_stud:{kl_teach_stud:.5f}, ent_teach:{ent_teach:.3f}, ent_stud:{ent_stud:.3f} |"
                    # f" stud_forced_teach_loss:{stud_forced_teach_loss['stud_forced_teach_loss/avg']:.5f},"
                    # f" {',  '.join([f'{k}:{v:.2f}' for k,v in accuracies.items()])} |"
                    f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                    f"{' (step)' if not is_accumulating else ''}"
                    f" remaining time: {timedelta(seconds=int(metrics['remaining_time']))!s}"
                    f" | max mem (alloc/reserv): {metrics['max_mem_allocated_per_gpu']:.2f}/{metrics['max_mem_reserved_per_gpu']:.2f} GB |"
                    f" toks/sec: {metrics['toks_per_sec_world'] :.2e}"
                    f" toks/sec/dev: {metrics['toks_per_sec_per_device'] :.2e}"
                    f" sup_toks/sec/dev: {metrics['sup_toks_per_sec_per_device'] :.2e}"
                    f" consumed_toks/sec/dev: {metrics['consumed_toks_per_sec_per_device'] :.2e}"
                )

            throughput_metrics = throughput.compute()
            metrics.update(throughput_metrics)
            fabric.log_dict(metrics, step=state["iter_num"] - 1)

        if val_dataloader is not None and not is_accumulating and state["step_count"] % eval.interval == 0:
            
            metrics = {"step": state["step_count"]}

            gen_val_t0 = time.perf_counter()
            val_lengths = [None]
            if hparams.singleshot.extra_val_trunc_lengths is not None:
                val_lengths += hparams.singleshot.extra_val_trunc_lengths
            val_k_toks_values = [None]
            if hparams.singleshot.extra_val_k_toks_values is not None:
                val_k_toks_values += hparams.singleshot.extra_val_k_toks_values
            for val_k_toks in val_k_toks_values:
                for val_len in val_lengths:
                    # basename = "generation" if val_len is None else f"generation_slen{val_len}"
                    basename = "generation"
                    if val_k_toks is not None:
                        basename += f"_ktoks{val_k_toks}"
                    if val_len is not None:
                        basename += f"_slen{val_len}"
                    generations, gen_stats, gen_stats_lists, gen_losses, gen_losses_lists = generative_validate(hparams, state, fabric, model, model_teacher, tokenizer, val_dataloader, max_iters=eval.max_iters, truncation_length=val_len, k_toks=val_k_toks)
                    
                    flat_gen_stats = {}
                    for col_name in gen_stats.keys():
                        for metric_name in gen_stats[col_name].keys():
                            flat_gen_stats[f"{basename}_stats/{col_name.replace(' ','_')}/{metric_name}"] = gen_stats[col_name][metric_name]
                    
                    flat_gen_stats_lists = {}
                    for col_name in gen_stats_lists.keys():
                        for metric_name in gen_stats_lists[col_name].keys():
                            flat_gen_stats_lists[f"{col_name.replace(' ','_')}/{metric_name}"] = gen_stats_lists[col_name][metric_name]

                    flat_gen_losses = {}
                    for col_name in gen_losses.keys():
                        for metric_name in gen_losses[col_name].keys():
                            flat_gen_losses[f"{basename}_losses/{col_name.replace(' ','_')}/{metric_name}"] = gen_losses[col_name][metric_name]
                    
                    flat_gen_losses_lists = {}
                    for col_name in gen_losses_lists.keys():
                            flat_gen_losses_lists[f"{col_name.replace(' ','_')}"] = gen_losses_lists[col_name]

                    metrics.update(flat_gen_stats)
                    metrics.update(flat_gen_losses)

                    run_name = fabric.logger.experiment.name
                    run_id = fabric.logger.experiment.id
                    step = metrics["step"]
                    num_records = len(list(flat_gen_stats_lists.values())[0])
                    metadata_cols = {
                        "run_name": [run_name]*num_records,
                        "run_id": [run_id]*num_records,
                        "step": [step]*num_records,
                    }
                    flat_gen_stats_lists.update(metadata_cols)
                    flat_gen_losses_lists.update(metadata_cols)

                    gen_stats_table_columns = list(flat_gen_stats_lists.keys())
                    gen_stats_table = Table(columns=gen_stats_table_columns)
                    for stat_tup in zip(*list(flat_gen_stats_lists.values())):
                        gen_stats_table.add_data(*stat_tup)
                    metrics[f"{basename}_gen_stats_table"] = gen_stats_table

                    gen_losses_table_columns = list(flat_gen_losses_lists.keys())
                    gen_losses_table = Table(columns=gen_losses_table_columns)
                    for loss_tup in zip(*list(flat_gen_losses_lists.values())):
                        gen_losses_table.add_data(*loss_tup)
                    metrics[f"{basename}_gen_losses_table"] = gen_losses_table
                    
                    gen_table_columns = ["Prompt", "GT Compl", "AR Teach Gen", "AR Stud Gen", "SS Stud Gen"]
                    if hparams.singleshot.num_samples is not None:
                        gen_table_columns += [f"SS Stud Sample{i}"for i in range(1, hparams.singleshot.num_samples+1)]
                    gen_table = Table(columns=gen_table_columns)
                    for gen_tup in generations:
                        gen_table.add_data(*gen_tup)
                    for k,v in metadata_cols.items():
                        gen_table.add_column(name=k, data=v)
                    metrics[basename] = gen_table
            
            gen_val_t1 = time.perf_counter()
            fabric.print(f"Generative validation time: {gen_val_t1 - gen_val_t0:.2f} seconds.")
            metrics["time_gen_val"] = gen_val_t1 - gen_val_t0
            
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader, max_iters=eval.max_iters).item()
            td = time.perf_counter() - t0

            fabric.print("Generations (prompt omitted) | ", [dict(zip(gen_table_columns[1:],tup[1:])) for tup in generations])
            fabric.print(f"Validation | iter {state['iter_num']} | val loss: {val_loss:.3f}, val ppl: {math.exp(val_loss):.3f} | val time: {td * 1000:.2f} ms")
            metrics.update({"val_loss": val_loss, "val_ppl": math.exp(val_loss)})
            fabric.log_dict(metrics, step=state["iter_num"] - 1)
            fabric.barrier()
        
        save_step = train.save_interval is not None and state["step_count"] % train.save_interval == 0
        save_latest_step = train.save_latest_interval is not None and state["step_count"] % train.save_latest_interval == 0
        if not is_accumulating and (save_step or save_latest_step):
            is_latest_only_save = save_latest_step and not save_step
            teacher_model_ref = state.pop("model_teacher")
            save_checkpoint(hparams, fabric, state, tokenizer_dir, out_dir / f"step-{state['step_count']:08d}" / "lit_model.pth", is_latest_only_save=is_latest_only_save)
            state["model_teacher"] = teacher_model_ref
            fabric.barrier()

    # Sync on train loop exiting
    fabric.barrier()
    
    # Final checkpoint
    if train.final_save:
        teacher_model_ref = state.pop("model_teacher")
        save_checkpoint(hparams, fabric, state, tokenizer_dir, out_dir / f"step-{state['step_count']:08d}" / "lit_model.pth")
        state["model_teacher"] = teacher_model_ref
        fabric.barrier()

    # Final validation
    if eval.final_validation:

        metrics = {"step": state["step_count"]}

        gen_val_t0 = time.perf_counter()
        val_lengths = [None]
        if hparams.singleshot.extra_val_trunc_lengths is not None:
            val_lengths += hparams.singleshot.extra_val_trunc_lengths
        val_k_toks_values = [None]
        if hparams.singleshot.extra_val_k_toks_values is not None:
            val_k_toks_values += hparams.singleshot.extra_val_k_toks_values
        for val_k_toks in val_k_toks_values:
            for val_len in val_lengths:
                # basename = "generation" if val_len is None else f"generation_slen{val_len}"
                basename = "generation"
                if val_k_toks is not None:
                    basename += f"_ktoks{val_k_toks}"
                if val_len is not None:
                    basename += f"_slen{val_len}"
                generations, gen_stats, gen_stats_lists, gen_losses, gen_losses_lists = generative_validate(hparams, state, fabric, model, model_teacher, tokenizer, val_dataloader, max_iters=eval.max_iters, truncation_length=val_len, k_toks=val_k_toks)
                
                flat_gen_stats = {}
                for col_name in gen_stats.keys():
                    for metric_name in gen_stats[col_name].keys():
                        flat_gen_stats[f"{basename}_stats/{col_name.replace(' ','_')}/{metric_name}"] = gen_stats[col_name][metric_name]
                
                flat_gen_stats_lists = {}
                for col_name in gen_stats_lists.keys():
                    for metric_name in gen_stats_lists[col_name].keys():
                        flat_gen_stats_lists[f"{col_name.replace(' ','_')}/{metric_name}"] = gen_stats_lists[col_name][metric_name]

                flat_gen_losses = {}
                for col_name in gen_losses.keys():
                    for metric_name in gen_losses[col_name].keys():
                        flat_gen_losses[f"{basename}_losses/{col_name.replace(' ','_')}/{metric_name}"] = gen_losses[col_name][metric_name]
                
                flat_gen_losses_lists = {}
                for col_name in gen_losses_lists.keys():
                        flat_gen_losses_lists[f"{col_name.replace(' ','_')}"] = gen_losses_lists[col_name]

                metrics.update(flat_gen_stats)
                metrics.update(flat_gen_losses)

                run_name = fabric.logger.experiment.name
                run_id = fabric.logger.experiment.id
                step = metrics["step"]
                num_records = len(list(flat_gen_stats_lists.values())[0])
                metadata_cols = {
                    "run_name": [run_name]*num_records,
                    "run_id": [run_id]*num_records,
                    "step": [step]*num_records,
                }
                flat_gen_stats_lists.update(metadata_cols)
                flat_gen_losses_lists.update(metadata_cols)

                gen_stats_table_columns = list(flat_gen_stats_lists.keys())
                gen_stats_table = Table(columns=gen_stats_table_columns)
                for stat_tup in zip(*list(flat_gen_stats_lists.values())):
                    gen_stats_table.add_data(*stat_tup)
                metrics[f"{basename}_gen_stats_table"] = gen_stats_table

                gen_losses_table_columns = list(flat_gen_losses_lists.keys())
                gen_losses_table = Table(columns=gen_losses_table_columns)
                for loss_tup in zip(*list(flat_gen_losses_lists.values())):
                    gen_losses_table.add_data(*loss_tup)
                metrics[f"{basename}_gen_losses_table"] = gen_losses_table
                
                gen_table_columns = ["Prompt", "GT Compl", "AR Teach Gen", "AR Stud Gen", "SS Stud Gen"]
                if hparams.singleshot.num_samples is not None:
                    gen_table_columns += [f"SS Stud Sample{i}"for i in range(1, hparams.singleshot.num_samples+1)]
                gen_table = Table(columns=gen_table_columns)
                for gen_tup in generations:
                    gen_table.add_data(*gen_tup)
                for k,v in metadata_cols.items():
                    gen_table.add_column(name=k, data=v)
                metrics[basename] = gen_table

        gen_val_t1 = time.perf_counter()
        fabric.print(f"Generative validation time: {gen_val_t1 - gen_val_t0:.2f} seconds.")
        metrics["time_gen_val"] = gen_val_t1 - gen_val_t0

        val_loss = validate(fabric, model, val_dataloader, max_iters=eval.max_iters).item()
        
        fabric.print("Final generations: ",  [dict(zip(gen_table_columns[1:],tup[1:])) for tup in generations])
        fabric.print(f"Final validation | val loss: {val_loss:.3f} | val ppl: {math.exp(val_loss):.3f}")
        metrics.update({"val_loss": val_loss, "val_ppl": math.exp(val_loss)})
        fabric.log_dict(metrics, step=state["iter_num"])
        fabric.barrier()


@torch.no_grad()
def validate(
    fabric: L.Fabric, model: nn.Module, val_dataloader: DataLoader, max_iters: int, verbose: bool = True
) -> torch.Tensor:
    fabric.barrier()
    if verbose:
        fabric.print("Validating ...")
    model.eval()

    # turn off the block masking in student for forced loss
    if model.block_mask is not None:
        model.use_block_mask = False

    losses = []
    for k, batch in enumerate(val_dataloader):
        if k >= max_iters:
            break
        # note this one loop, we just use whatever length the model is set for since we aren't 
        # correcting for varying lengths here
        input_ids = batch[:, 0 : model.max_seq_length].contiguous().long()
        target_ids = batch[:, 1 : (model.max_seq_length + 1)].contiguous().long()
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, target_ids)
        losses.append(loss)

    val_loss = torch.stack(losses).mean()
    model.train()
    
    # turn back on the block masking in student
    if model.block_mask is not None:
        model.use_block_mask = True
    
    fabric.barrier()
    return val_loss


def reset_model_kv_cache(model: nn.Module=None, batch_size: int=None, max_returned_tokens: int=None, device: torch.device=None):
    # following some api.py commands to set up kv cache and things
    model.clear_kv_cache()
    model.set_kv_cache(batch_size=batch_size, max_seq_length=max_returned_tokens, device=device)

    for block in model.transformer.h:
        block.attn.kv_cache.reset_parameters()

@torch.no_grad()
def generative_validate(hparams: dict2attr, state: dict, fabric: L.Fabric, model: nn.Module, model_teacher: nn.Module, tokenizer: Tokenizer, val_dataloader: DataLoader, max_iters: int, truncation_length: int = None, k_toks: int = None, verbose: bool = True):

    fabric.barrier()
    if verbose:
        fabric.print("Generating ...")

    # Clear memory
    if fabric.global_rank == 0:
        fabric.print("Manual GC and CUDA cache clear...")
    gc.collect()
    torch.cuda.empty_cache()

    # extend seq len temporarily for the models to handle the rollout multiplier
    if hparams.singleshot.rollout_multiplier > 1:
        adjustment = hparams.singleshot.k_toks.max_k_toks_value*(hparams.singleshot.rollout_multiplier-1)
        model.max_seq_length += adjustment
        model_teacher.max_seq_length += adjustment
    
    model_teacher.eval()
    model.eval()

    # with randomized k_toks possibility, we need to make sure this is wrt the actual max, not just the last step
    if k_toks is not None:
        val_k_toks = k_toks
    else:
        val_k_toks = hparams.singleshot.k_toks.max_k_toks_value
    # then we update the model blockmask configs accordingly
    if hparams.singleshot.train_with_block_mask:
        model.K = val_k_toks - 1
        model.block_mask_config["K"] = model.K
        model_teacher.K = model.K
        model_teacher.block_mask_config["K"] = model.K
        model.P = (model.S // model.mask_region_ct) - (val_k_toks - 1)
        model.block_mask_config["P"] = model.P
        model_teacher.P = model.P
        model_teacher.block_mask_config["P"] = model.P

    if truncation_length is not None:
        val_truncation_length = truncation_length
        fabric.print(f"val length override: {val_truncation_length}")
    elif hparams.singleshot.train_with_block_mask and hparams.singleshot.multi_region_val_correction:
        max_prefix_seen = hparams.singleshot.mask_region_ct * model.block_mask_config["P"]
        val_truncation_length = max_prefix_seen + model.block_mask_config["K"] # eg. +(k_toks-1)
        fabric.print(f"multi-region block adjusted val length: max_prefix_seen+k_toks-1={max_prefix_seen}+{model.block_mask_config['K']}={val_truncation_length}")
    else:
        val_truncation_length = hparams.singleshot.truncation_length

    # strs
    generations = []
    # toks
    all_ss_outputs = []
    all_ss_sampled_outputs = [] # raveled
    all_prompts = []
    all_gts = []
    all_ar_outputs_teacher = []
    all_ar_outputs = []
    # ents and confs
    all_ss_outputs_ents = []
    all_ss_outputs_top1_confs = []
    # just starting with the minimum ones for now
    # all_ss_sampled_outputs_ents = [] # raveled
    # all_ss_sampled_outputs_top1_confs = [] # raveled
    # all_ar_outputs_teacher_ents = []
    # all_ar_outputs_teacher_top1_confs = []
    # all_ar_outputs_ents = []
    # all_ar_outputs_top1_confs = []
    # forced losses
    all_output_forced_teach_losses = {
        "GT Compl":[],
        "AR Teach Gen":[],
        "AR Stud Gen":[],
        "SS Stud Gen":[],
    }
    for k in list(all_output_forced_teach_losses.keys()):
        for ri in range(hparams.singleshot.rollout_multiplier):
            all_output_forced_teach_losses[f"{k} kx{ri}"] = []

    for k, batch in enumerate(val_dataloader):

        if k >= max_iters:
            break
        val_data_len = batch.shape[1] - 1
        input_ids = batch[:, 0 : val_data_len].contiguous().long()
        target_ids = batch[:, 1 : (val_data_len + 1)].contiguous().long()

        # outputs = batched_generate_fn( ... )
        # for now we do this sequentially, but batching would be nice
        for i, (row_in, row_tgt) in enumerate(zip(input_ids,target_ids)):

            # k_toks = hparams.singleshot.k_toks.get_value(state["step_count"])
            
            # idk _exactly_ why, but clone is req'd else causes rept'd samples in gen fn
            # if the truncation routine is compiled
            processed_row = truncate_and_mask(
                input_ids=row_in.clone().unsqueeze(0),
                target_ids=row_tgt.clone().unsqueeze(0),
                k_toks=val_k_toks,
                mask_id=hparams.singleshot.mask_id,
                min_mask_id=hparams.singleshot.min_mask_id,
                max_mask_id=hparams.singleshot.max_mask_id,
                truncation_length=val_truncation_length
            )
            trunc_masked_row_in,trunc_masked_row_tgt = processed_row[0].squeeze(0), processed_row[1].squeeze(0)
            
            # do SS stud rollouts by initializing with the masked input
            # and then cat-ing as we go
            ss_outputs = trunc_masked_row_in.clone()
            ss_outputs_ents = torch.empty(0, device=fabric.device)
            ss_outputs_top1_confs = torch.empty(0, device=fabric.device)
            for m in range(hparams.singleshot.rollout_multiplier):

                # the main SS pred
                assert ss_outputs.ndim == 1
                model.reconstruct_block_mask(
                    K = val_k_toks - 1,
                    S = ss_outputs.shape[0],
                    B = 1,
                    mask_region_ct = 1,
                    offset = 0,
                    bidirect_ss_attn=hparams.singleshot.bidirect_ss_attn,
                    device = fabric.device,
                )
                fabric.barrier()
                new_settings_hash = hashlib.shake_128(str(model.block_mask_config).encode()).hexdigest(8)
                
                if hparams.singleshot.log_masks_and_inputs and (new_settings_hash not in model.old_settings_hashset):
                    fabric.print(f"Reconstructed new block mask for val_k_toks: {val_k_toks}, trunc_length: {ss_outputs.shape[0]}, offset: {0}, bsz: {1}, mask_region_ct: {1}")

                    # Visualization, unnecessary
                    def make_tensor():
                        return torch.ones(model.B, model.config.n_query_groups, model.S, model.H, device="cpu")
                    step_idx = state["step_count"]
                    viz_name = f"validation_mtp_causal_mask_step-{step_idx:08d}_P{model.P}_K{model.K}_S{model.S}_O{model.Ofs}_B{model.B}_H{model.H}_bda{model.bidirect_ss_attn}"
                    base_save_path = f"{hparams.out_dir}/mtp_masks"
                    os.makedirs(base_save_path, exist_ok=True)
                    visualize_attention_scores(
                        make_tensor(), # query
                        make_tensor(), # key
                        mask_mod=model.mask_mod, # interleaved_mtp_mask_mod,
                        device="cpu",
                        name=viz_name,
                        path=Path(f"{base_save_path}/{viz_name}"),
                    )
                    # End visualization
                # if rank rolling, not all will save a new mask at the same time, so this must come after the logging block
                fabric.barrier() 
                model.old_settings_hashset.add(new_settings_hash)
                
                logits = model(ss_outputs.unsqueeze(0))
                soft_preds = logits[:,-val_k_toks:]
                
                ents, top1_confs = ent_and_top1_confidence(soft_preds.squeeze(0))
                ss_outputs_ents = torch.cat([ss_outputs_ents, ents], dim=-1)
                ss_outputs_top1_confs = torch.cat([ss_outputs_top1_confs, top1_confs], dim=-1)
                
                new_ss_outputs = torch.argmax(soft_preds, dim=-1) # .squeeze(0)

                if m < hparams.singleshot.rollout_multiplier-1:
                    new_ss_outputs = extend_w_mask(
                        input_ids=new_ss_outputs.clone(),
                        k_toks=val_k_toks,
                        mask_id=hparams.singleshot.mask_id,
                        min_mask_id=hparams.singleshot.min_mask_id,
                        max_mask_id=hparams.singleshot.max_mask_id
                    )
                
                new_ss_outputs = new_ss_outputs.squeeze(0)

                if val_k_toks - 1 > 0:
                    unmasked_prefix = ss_outputs[:-(val_k_toks-1)]
                else:
                    unmasked_prefix = ss_outputs

                ss_outputs = torch.cat([unmasked_prefix,new_ss_outputs],dim=-1)

            # now do SS sampled rollouts the same way
            if hparams.singleshot.num_samples is not None:
                ss_sampled_output_set = torch.stack([trunc_masked_row_in.clone()]*hparams.singleshot.num_samples)
                for m in range(hparams.singleshot.rollout_multiplier):
                    new_ss_sampled_output_set = []
                    for s in range(hparams.singleshot.num_samples):
                        assert ss_sampled_output_set[s].ndim == 1
                        model.reconstruct_block_mask(
                            K = val_k_toks - 1,
                            S = ss_sampled_output_set[s].shape[0],
                            B = 1,
                            mask_region_ct = 1,
                            offset = 0,
                            bidirect_ss_attn=hparams.singleshot.bidirect_ss_attn,
                            device = fabric.device,
                        )
                        logits = model(ss_sampled_output_set[s].unsqueeze(0))
                        soft_preds = logits[:,-val_k_toks:]
                        ss_sampled_outputs = torch.multinomial(F.softmax(soft_preds, dim=-1).squeeze(0),num_samples=1).reshape(1,val_k_toks)

                        if m < hparams.singleshot.rollout_multiplier-1:
                            ss_sampled_outputs = extend_w_mask(
                                input_ids=ss_sampled_outputs.clone(),
                                k_toks=val_k_toks,
                                mask_id=hparams.singleshot.mask_id,
                                min_mask_id=hparams.singleshot.min_mask_id,
                                max_mask_id=hparams.singleshot.max_mask_id
                            )
                        ss_sampled_outputs = ss_sampled_outputs.squeeze(0)
                        new_ss_sampled_output_set.append(ss_sampled_outputs)
                    new_ss_sampled_output_set = torch.stack(new_ss_sampled_output_set)

                    if val_k_toks - 1 > 0:
                        unmasked_prefix = ss_sampled_output_set[:,:-(val_k_toks-1)]
                    else:
                        unmasked_prefix = ss_sampled_output_set
                    ss_sampled_output_set = torch.cat([unmasked_prefix,new_ss_sampled_output_set], dim=-1)
            else:
                ss_sampled_output_set = [torch.empty(0)] # dummy placeholder for later code

            # then proceed with the AR rollout

            # turn off the block masking in student and teacher for the AR and forced loss parts
            if model.block_mask is not None:
                model.use_block_mask = False
            if model_teacher.block_mask is not None:
                model_teacher.use_block_mask = False
            fabric.barrier()

            ar_truncation_length = val_truncation_length + val_k_toks * (hparams.singleshot.rollout_multiplier-1)
            eff_k_toks = val_k_toks*hparams.singleshot.rollout_multiplier
            # clone req'd see above prior use of the truncation fn
            processed_row = truncate_and_mask(
                input_ids=row_in.clone().unsqueeze(0),
                target_ids=row_tgt.clone().unsqueeze(0),
                k_toks=eff_k_toks,
                mask_id=hparams.singleshot.mask_id,
                min_mask_id=hparams.singleshot.min_mask_id,
                max_mask_id=hparams.singleshot.max_mask_id,
                skip_max_mask_id_check=True, # we are repurposing the fn to just do the truncation, and eff_k_toks will be out of the MTP range
                truncation_length=ar_truncation_length
            )
            trunc_masked_row_in,trunc_masked_row_tgt = processed_row[0].squeeze(0), processed_row[1].squeeze(0)

            if eff_k_toks - 1 > 0:
                prompt = trunc_masked_row_in[:-(eff_k_toks-1)]
            else:
                prompt = trunc_masked_row_in
            gt_ids = trunc_masked_row_tgt[-eff_k_toks:]

            orig_len = len(trunc_masked_row_in) + 1

            max_returned_tokens = len(prompt) + eff_k_toks
            assert max_returned_tokens == orig_len
            assert max_returned_tokens == len(prompt) + len(gt_ids)
            
            reset_model_kv_cache(model=model_teacher, batch_size=1, max_returned_tokens=max_returned_tokens, device=fabric.device)
            ar_outputs_teacher = generate_fn(
                model=model_teacher,
                prompt=prompt,
                max_returned_tokens=max_returned_tokens,
                temperature=0.0, # eg. greedy
                top_k=None, # inactive value
                top_p=1.0, # inactive value
                # eos_id=tokenizer.eos_id,
                eos_id=None, # NOTE will hang bc all ranks must generate in sync for max ret tok
                include_prompt=False,
            )


            reset_model_kv_cache(model=model, batch_size=1, max_returned_tokens=max_returned_tokens, device=fabric.device)
            ar_outputs = generate_fn(
                model=model,
                prompt=prompt,
                max_returned_tokens=max_returned_tokens,
                temperature=0.0, # eg. greedy
                top_k=None, # inactive value
                top_p=1.0, # inactive value
                # eos_id=tokenizer.eos_id,
                eos_id=None, # NOTE will hang bc all ranks must generate in sync for max ret tok
                include_prompt=False,
            )

            # ss outputs now have the prompt prepended to them, remove this
            ss_outputs = ss_outputs[prompt.shape[0]:]
            if hparams.singleshot.num_samples is not None:
                ss_sampled_output_set = ss_sampled_output_set[:,prompt.shape[0]:]
            
            # store the raw toks before we modify anything
            all_ss_outputs.append(ss_outputs)
            if hparams.singleshot.num_samples is not None:
                for samp in ss_sampled_output_set:
                    all_ss_sampled_outputs.append(samp)
            all_prompts.append(prompt)
            all_gts.append(gt_ids)
            all_ar_outputs_teacher.append(ar_outputs_teacher)
            all_ar_outputs.append(ar_outputs)

            # store the ents and confs
            all_ss_outputs_ents.append(ss_outputs_ents)
            all_ss_outputs_top1_confs.append(ss_outputs_top1_confs)

            # Grab forced loss measures before we do anything else.
            # names match the wandb output tables for generations
            outputs_to_test = {
                "GT Compl":gt_ids,
                "AR Teach Gen":ar_outputs_teacher,
                "AR Stud Gen":ar_outputs,
                "SS Stud Gen":ss_outputs,
            }
            model_teacher.clear_kv_cache() # just in case

            batch_prompt = torch.stack([prompt]*len(outputs_to_test))
            batch_output = torch.stack(list(outputs_to_test.values()))

            combined_input = torch.cat([batch_prompt, batch_output],dim=-1)
            combined_shift_len = combined_input.shape[1] - 1
            input_ids = combined_input[:, 0 : combined_shift_len].contiguous().long()
            target_ids = combined_input[:, 1 : (combined_shift_len + 1)].contiguous().long()

            teach_logits = model_teacher(input_ids)
            output_logits = teach_logits[:,-eff_k_toks:]
            output_targets = target_ids[:,-eff_k_toks:]

            losses_tup = nll_metric(output_logits,batch_output)
            teach_loss, cu_teach_loss, per_pos_teach_loss = losses_tup
            # create aggs over each k_toks in hparams.singleshot.rollout_multiplier passes
            per_k_teach_loss = {}
            for ri in range(hparams.singleshot.rollout_multiplier):
                start, end = ri*val_k_toks, (ri+1)*val_k_toks
                per_k_teach_loss[ri] = torch.sum(per_pos_teach_loss[:,start:end], dim=-1) / val_k_toks
            
            for outp_name,outp_loss in zip(outputs_to_test.keys(), teach_loss):
                all_output_forced_teach_losses[outp_name].append(outp_loss.item())
            
            for ri in range(hparams.singleshot.rollout_multiplier):
                for outp_name,outp_loss in zip(outputs_to_test.keys(), per_k_teach_loss[ri]):
                    all_output_forced_teach_losses[f"{outp_name} kx{ri}"].append(outp_loss.item())

            # Need to interleave a visual separator in order to show the rollout segments clearly
            # we'll use the mask token + 1 just to avoid the edge case where it is generated though this
            # is still pretty brittle
            # tmp_sep_tok = mask_id+1
            # still brittle, but at least configurable/adaptive now
            tmp_sep_tok = hparams.singleshot.temp_sep_token_id_range[0]
            # hopefully it doesnt happen often, but if it does, find a new id for this instance
            sets_to_check = [ss_outputs]
            if hparams.singleshot.num_samples is not None:
                sets_to_check.append(ss_sampled_output_set)
            if any([torch.any(s == tmp_sep_tok) for s in sets_to_check]):
                start, end = hparams.singleshot.temp_sep_token_id_range
                for cand_id in range(start, end+1):
                    if not any([torch.any(s == cand_id) for s in sets_to_check]):
                        tmp_sep_tok = cand_id
                        break
            assert not any([torch.any(s == tmp_sep_tok) for s in sets_to_check]), "heck why, thought I 'fixed' this"
            tmp_sep_str = tokenizer.decode(torch.tensor([tmp_sep_tok]),skip_special_tokens=False)

            def replace_sep_toks(full_str, tok_id=tmp_sep_tok, tok_str=tmp_sep_str):
                visual_separator = lambda ridx: f"<ss{ridx}>"
                new_str = visual_separator(0)
                rem_str = str(full_str).replace(tok_str,"", 1)
                for i in range(1, hparams.singleshot.rollout_multiplier+1):
                    idx = rem_str.find(tok_str)
                    if idx == -1: idx = None
                    new_str += rem_str[:idx]
                    if i < hparams.singleshot.rollout_multiplier:
                        new_str += visual_separator(i) 
                    rem_str = rem_str[idx:].replace(tok_str,"", 1)
                return new_str

            for ri in range(hparams.singleshot.rollout_multiplier,0,-1):
                offset = ri * val_k_toks
                sep_tensor = torch.tensor([tmp_sep_tok],device=ss_outputs.device)
                ss_outputs = torch.cat([ss_outputs[:-offset], sep_tensor, ss_outputs[-offset:]], dim=-1)
                if hparams.singleshot.num_samples is not None:
                    sep_tensor = torch.tensor([[tmp_sep_tok]]*ss_sampled_output_set.shape[0],device=ss_sampled_output_set.device)
                    ss_sampled_output_set = torch.cat([ss_sampled_output_set[:,:-offset], sep_tensor, ss_sampled_output_set[:,-offset:]], dim=-1)
            
            # decode non ss elements
            decoded_prompts = tokenizer.decode(prompt,skip_special_tokens=False)
            decoded_gts = tokenizer.decode(gt_ids,skip_special_tokens=False)
            decoded_ar_outputs_teacher = tokenizer.decode(ar_outputs_teacher,skip_special_tokens=False)
            decoded_ar_outputs = tokenizer.decode(ar_outputs,skip_special_tokens=False)

            # decode ss elements
            decoded_ss_outputs = tokenizer.decode(ss_outputs,skip_special_tokens=False)
            if hparams.singleshot.num_samples is not None:
                decoded_ss_sampled_outputs = [tokenizer.decode(elm,skip_special_tokens=False) for elm in ss_sampled_output_set]
            
            # convert seps in ss elements
            decoded_ss_outputs = replace_sep_toks(decoded_ss_outputs)
            if hparams.singleshot.num_samples is not None:
                decoded_ss_sampled_outputs = [replace_sep_toks(s) for s in decoded_ss_sampled_outputs]

            # collate
            new_generations = [
                decoded_prompts,
                decoded_gts,
                decoded_ar_outputs_teacher,
                decoded_ar_outputs,
                decoded_ss_outputs,
                ]
            if hparams.singleshot.num_samples is not None:
                new_generations += decoded_ss_sampled_outputs
            generations.append(new_generations)

            # turn back on the block masking in student and teacher
            if model.block_mask is not None:
                model.use_block_mask = True
            if model_teacher.block_mask is not None:
                model_teacher.use_block_mask = True

    model.clear_kv_cache()
    model.train()
    model_teacher.clear_kv_cache()
    model_teacher.train()

    # undo the temp extension if we did one
    if hparams.singleshot.rollout_multiplier > 1:
        adjustment = hparams.singleshot.k_toks.max_k_toks_value*(hparams.singleshot.rollout_multiplier-1)
        model.max_seq_length -= adjustment
        model_teacher.max_seq_length -= adjustment

    # turn back on the block masking in student and teacher
    if model.block_mask is not None:
        model.use_block_mask = True
    if model_teacher.block_mask is not None:
        model_teacher.use_block_mask = True
    # and reset to the base training cfg
    model.reconstruct_block_mask(
        K = hparams.singleshot.k_toks.get_value(state["step_count"]) - 1,
        S = hparams.singleshot.truncation_length,
        B = hparams.train.micro_batch_size,
        mask_region_ct = hparams.singleshot.mask_region_ct,
        offset = 0,
        bidirect_ss_attn=hparams.singleshot.bidirect_ss_attn,
        device = fabric.device,
    )
    model_teacher.reconstruct_block_mask(
        K = hparams.singleshot.k_toks.get_value(state["step_count"]) - 1,
        S = hparams.singleshot.truncation_length,
        B = hparams.train.micro_batch_size,
        mask_region_ct = hparams.singleshot.mask_region_ct,
        offset = 0,
        bidirect_ss_attn=False, # we don't want the teacher to have this on
        device = fabric.device,
    )

    fabric.barrier()

    gen_table_columns = ["Prompt", "GT Compl", "AR Teach Gen", "AR Stud Gen", "SS Stud Gen"]
    data_cols = [all_prompts, all_gts, all_ar_outputs_teacher, all_ar_outputs, all_ss_outputs]
    if hparams.singleshot.num_samples is not None:
        gen_table_columns += [f"SS Stud Sample"] # no indiv cols for sample trials in stats
        data_cols += [all_ss_sampled_outputs]
    generation_stats_agg = {}
    generation_stats_lists = {}
    for col_name,col_data in zip(gen_table_columns, data_cols):
        rep_stats, rep_stats_list = compute_repetition_metrics(col_data)
        generation_stats_agg[col_name] = rep_stats
        generation_stats_lists[col_name] = rep_stats_list
        if col_name != "Prompt":
            col_data = torch.stack(col_data)
            for ri in range(hparams.singleshot.rollout_multiplier):
                val_k_toks = hparams.singleshot.k_toks.get_value(state["step_count"])
                start, end = ri*val_k_toks, (ri+1)*val_k_toks
                rep_stats, rep_stats_list = compute_repetition_metrics(col_data[:,start:end])
                generation_stats_agg[f"{col_name} kx{ri}"] = rep_stats
                generation_stats_lists[f"{col_name} kx{ri}"] = rep_stats_list

    # handle ents and confs similar to the start and slicing of the rep stats
    # but were just storing the avgs ents and top1 confs per ri and for the whole row for now
    gen_col_names = ["SS Stud Gen"]
    ents_confs_names = ["ents", "top1_confs"] # these must not conflict with other metric names above
    data_cols_ents_confs = [all_ss_outputs_ents, all_ss_outputs_top1_confs]
    ents_confs_agg = {}
    ents_confs_lists = {}
    for col_name in gen_col_names:
        ents_confs_agg[col_name] = {}
        ents_confs_lists[col_name] = {}
        for ri in range(hparams.singleshot.rollout_multiplier):
            ents_confs_agg[f"{col_name} kx{ri}"] = {}
            ents_confs_lists[f"{col_name} kx{ri}"] = {}
        for metric_name,col_data in zip(ents_confs_names, data_cols_ents_confs):
            avg_per_row = [torch.mean(elm).item() for elm in col_data]
            avg_per_row = torch.tensor(avg_per_row)
            ents_confs_lists[col_name][metric_name] = avg_per_row.tolist()
            # Only log mean/median/std to wandb to keep the per-run chart
            # count manageable; full per-example values are still available
            # via ents_confs_lists for anyone who needs the full distribution.
            ents_confs_agg[col_name].update({
                f"{metric_name}": avg_per_row.mean().item(),
                f"{metric_name}_median": avg_per_row.median().item(),
                f"{metric_name}_std": avg_per_row.std().item(),
            })
            for ri in range(hparams.singleshot.rollout_multiplier):
                val_k_toks = hparams.singleshot.k_toks.get_value(state["step_count"])
                start, end = ri*val_k_toks, (ri+1)*val_k_toks
                avg_per_row_ri = [torch.mean(elm[start:end]).item() for elm in col_data]
                avg_per_row_ri = torch.tensor(avg_per_row_ri)
                ents_confs_lists[f"{col_name} kx{ri}"][metric_name] = avg_per_row_ri.tolist()
                ents_confs_agg[f"{col_name} kx{ri}"].update({
                    f"{metric_name}": avg_per_row_ri.mean().item(),
                    f"{metric_name}_median": avg_per_row_ri.median().item(),
                    f"{metric_name}_std": avg_per_row_ri.std().item(),
                })
    for col_name in ents_confs_agg.keys():
        generation_stats_agg[col_name].update(ents_confs_agg[col_name])
        generation_stats_lists[col_name].update(ents_confs_lists[col_name])

    # aggregate the forced teach losses
    all_output_forced_teach_losses_agg = {}
    all_output_forced_teach_losses_lists = {}
    for k,lst in all_output_forced_teach_losses.items():
        all_output_forced_teach_losses_lists[k] = lst
        loss_tensor = torch.tensor(lst)
        # Only log mean/median/std to wandb to keep the per-run chart count
        # manageable; full per-example values are still available via
        # all_output_forced_teach_losses_lists for anyone who needs the full
        # distribution.
        all_output_forced_teach_losses_agg[k] = {
            "teach_loss": loss_tensor.mean().item(),
            "teach_loss_median": loss_tensor.median().item(),
            "teach_loss_std": loss_tensor.std().item(),
        }


    # Clear memory
    if fabric.global_rank == 0:
        fabric.print("Manual GC and CUDA cache clear...")
    gc.collect()
    torch.cuda.empty_cache()

    fabric.barrier()
    return generations, generation_stats_agg, generation_stats_lists, all_output_forced_teach_losses_agg, all_output_forced_teach_losses_lists


def compute_repetition_metrics(inputs):

    # Track unique_1-4 (repetition_n is dropped since repetition_n == 1 -
    # unique_n, pure redundancy) plus the two summary metrics. Only
    # mean/median/std get logged to wandb to keep the per-run chart count
    # manageable; full per-example values are still available via
    # stats_table (the "_lists" variant) for anyone who needs the full
    # distribution.
    tracked_keys = ["unique_1", "unique_2", "unique_3", "unique_4", "diversity", "log_diversity"]
    stats_table = {k: [] for k in tracked_keys}

    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            inp = inp.tolist()
        stats = measure_repetition_and_diversity(inp)
        for k in stats_table.keys():
            stats_table[k].append(stats[k])
    stats_table_agg = {}
    for k in stats_table.keys():
        stat_tensor = torch.tensor(stats_table[k])
        stats_table_agg.update({
            f"{k}": stat_tensor.mean().item(),
            f"{k}_median": stat_tensor.median().item(),
            f"{k}_std": stat_tensor.std().item(),
        })

    return stats_table_agg, stats_table


def get_dataloaders(
    hparams: dict2attr, fabric: L.Fabric, data: DataModule, pqds: PQDSArgs, tokenizer: Tokenizer, train: TrainArgs, train_block_size: int, val_block_size: int
) -> Tuple[DataLoader, DataLoader]:

    if data == "pqds":
        train_dataset = ParquetStream(
                seed=hparams.seed, # this is the same for all ranks
                num_processes=fabric.world_size,
                process_rank=fabric.global_rank,
                torch_device=fabric.device,
                block_size=train_block_size,
                pad_token_id=pqds.pad_token_id,
                dataset_folder_path=pqds.train_dataset_folder_path,
                prefix=pqds.train_prefix,
                broadcast_glob=pqds.broadcast_glob,
                shuffle=pqds.shuffle,
                shuffle_filenames=pqds.shuffle_filenames,
                doc_wise=pqds.doc_wise,
                doc_wise_skip_tail=pqds.doc_wise_skip_tail,
                doc_wise_sep_tok=pqds.doc_wise_sep_tok,
                ignore_fingerprint_mismatch=pqds.ignore_fingerprint_mismatch,
                verbose=pqds.verbose,
            )
        val_dataset = ParquetStream(
                seed=hparams.seed,
                num_processes=fabric.world_size,
                process_rank=fabric.global_rank,
                torch_device=fabric.device,
                block_size=val_block_size,
                pad_token_id=pqds.pad_token_id,
                dataset_folder_path=pqds.val_dataset_folder_path,
                prefix=pqds.val_prefix,
                broadcast_glob=pqds.broadcast_glob,
                shuffle=pqds.shuffle,
                shuffle_filenames=pqds.shuffle_filenames,
                doc_wise=pqds.doc_wise,
                doc_wise_skip_tail=pqds.doc_wise_skip_tail,
                doc_wise_sep_tok=pqds.doc_wise_sep_tok,
                ignore_fingerprint_mismatch=pqds.ignore_fingerprint_mismatch,
                verbose=pqds.verbose,
            )
        train_dataloader = StatefulDataLoader(
            train_dataset,
            batch_size=train.micro_batch_size,
            shuffle=False, # we handle this in the dataset
            pin_memory=True,
            num_workers=pqds.num_workers,
            prefetch_factor=pqds.prefetch_factor if pqds.num_workers > 0 else None,
        )
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=train.micro_batch_size,
            shuffle=False, # we handle this in the dataset
            pin_memory=True,
            num_workers=pqds.num_workers,
            prefetch_factor=pqds.prefetch_factor if pqds.num_workers > 0 else None,
        )
    else:
        data.connect(tokenizer=tokenizer, batch_size=train.micro_batch_size, train_max_seq_length=train_block_size, val_max_seq_length=val_block_size)
        with fabric.rank_zero_first():
            data.prepare_data()
        data.setup()
        train_dataloader = data.train_dataloader()
        val_dataloader = data.val_dataloader()
    
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with linear warmup)
def get_lr(lr_schedule: str, learning_rate: float, it: int, warmup_iters: int, max_iters: int, min_lr: float) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    if lr_schedule == "constant":
        return learning_rate
    elif lr_schedule == "cosine":
        assert min_lr is not None, "min_lr must be specified for cosine lr schedule"
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")


class ConstantKTokCurriculum(object):
    def __init__(self, static_k_toks_value):
        """Expects a single int."""
        self.static_k_toks_value = static_k_toks_value
        self.max_k_toks_value = static_k_toks_value
        self.min_k_toks_value = static_k_toks_value

    def get_value(self, current_step):
        return self.static_k_toks_value

    def __repr__(self):
        return f"ConstantKTokCurriculum: {self.static_k_toks_value}"


class PiecewiseKTokCurriculum(object):
    def __init__(self, curriculum_intervals):
        """Expects a list of tuples like (start_step, k_toks_value)."""
        self.curriculum_intervals = curriculum_intervals
        self.max_k_toks_value = max([v for k,v in self.curriculum_intervals])
        self.min_k_toks_value = min([v for k,v in self.curriculum_intervals])

    def get_value(self, current_step):
        for i, (start, value) in enumerate(self.curriculum_intervals):
            if i == len(self.curriculum_intervals) - 1:
                end = float("inf") 
            else:
                end = self.curriculum_intervals[i+1][0]

            if start <= current_step < end:
                return value
        
        raise RuntimeError("Shoudn't be here, something's incorrect about the k_tok curric logic.")

    def __repr__(self):
        repr_pieces = []
        for i, (start, value) in enumerate(self.curriculum_intervals):
            if i == len(self.curriculum_intervals) - 1:
                end = float("inf") 
            else:
                end = self.curriculum_intervals[i+1][0]
            repr_pieces.append(f"[{start}->{end})={value}")
        return f"PiecewiseKTokCurriculum: [{', '.join(repr_pieces)}]"


def initialize_weights(fabric: L.Fabric, model: GPT, n_layer: int, n_embd: int) -> None:
    """GPT-NeoX weight initialization (https://arxiv.org/abs/2204.06745)."""
    # Adapted from https://github.com/jzhang38/TinyLlama

    def init_weights(module, std):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)

    for mod in model.modules():
        if isinstance(mod, (nn.Embedding, nn.Linear)):
            mod.reset_parameters = partial(init_weights, mod, std=math.sqrt(2.0 / 5 / n_embd))

    # need a separate loop because `mod.proj` below is a `nn.Linear` too
    for mod in model.modules():
        if isinstance(mod, (LLaMAMLP, CausalSelfAttention)):
            mod.proj.reset_parameters = partial(init_weights, mod.proj, std=(1 / math.sqrt(n_embd) / n_layer))

    if not isinstance(fabric.strategy, FSDPStrategy):
        reset_parameters(model)

def save_checkpoint(hparams, fabric, state, tokenizer_dir, checkpoint_file, is_latest_only_save=False):

    save_t0 = time.perf_counter()
    
    # Clear memory
    if fabric.global_rank == 0:
        fabric.print("Manual GC and CUDA cache clear...")
    gc.collect()
    torch.cuda.empty_cache()

    model = state["model"]
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    fabric.print(f"Saving checkpoint to {str(checkpoint_file)!r}")

    fabric.save(checkpoint_file, state)
    if fabric.global_rank == 0:
        save_hyperparameters(setup, checkpoint_file.parent)
        if tokenizer_dir is not None:
            copy_config_files(tokenizer_dir, checkpoint_file.parent)
        save_config(model.config, checkpoint_file.parent)

    # latest and pruning logic
    base_ckpt_dir = checkpoint_file.parent.parent
    fabric.barrier() 
    # update what the "latest" is by moving existing one to _prev and then making a copy of the new one
    if hparams.train.save_latest_ckpt:
        latest_ckpt_path = base_ckpt_dir / "latest"
        temp_prev_ckpt_path = base_ckpt_dir / "latest_prev"
        if fabric.global_rank == 0:
            if temp_prev_ckpt_path.exists():
                shutil.rmtree(temp_prev_ckpt_path)
            if latest_ckpt_path.exists():
                shutil.move(str(latest_ckpt_path), str(temp_prev_ckpt_path))
            # main operation
            shutil.copytree(checkpoint_file.parent, latest_ckpt_path)
            # rm the temp
            if temp_prev_ckpt_path.exists():
                shutil.rmtree(temp_prev_ckpt_path)
            # potentially rm the ckpt we saved this step if this is a latest only save
            if is_latest_only_save:
                fabric.print(f"Removing checkpoint at {str(checkpoint_file.parent)!r} since this is a latest-only save.")
                shutil.rmtree(checkpoint_file.parent)
    
    fabric.barrier() 
    # we don't want funny business ansynchony here
    prune_ckpt_dirs(hparams, fabric, base_ckpt_dir)
    fabric.barrier()

    # Clear memory
    if fabric.global_rank == 0:
        fabric.print("Manual GC and CUDA cache clear...")
    gc.collect()
    torch.cuda.empty_cache()

    save_t1 = time.perf_counter()
    fabric.log_dict({"time_save_ckpt": save_t1 - save_t0, "step": state["step_count"]},  step=state["iter_num"] - 1)


def prune_ckpt_dirs(hparams, fabric, ckpt_dir):
    fabric.print(f"Pruning checkpoints at {str(ckpt_dir)!r}")
    if hparams.train.max_ckpts_to_keep is None:
        fabric.print(f"Not pruning any checkpoints, as max_ckpts_to_keep is set to None")
        return # keep all
    if fabric.global_rank == 0:
        # assume justified integer names for step dirs, else fix sort criteria
        all_ckpts = sorted(ckpt_dir.glob("step-*"))
        fabric.print(f"Found {len(all_ckpts)} pruning to max of {hparams.train.max_ckpts_to_keep} most recent...")
        while len(all_ckpts) > hparams.train.max_ckpts_to_keep:
            oldest_ckpt = all_ckpts.pop(0)
            shutil.rmtree(oldest_ckpt)
        ckpts_str = '\n'.join([' * '+str(p.stem) for p in all_ckpts])
        fabric.print(f"\nRemaining {len(all_ckpts)} checkpoints after pruning:\n{ckpts_str}\n", flush=True)

def validate_args(hparams: dict2attr, train: TrainArgs, eval: EvalArgs, initial_checkpoint_dir, resume) -> None:
    issues = []
    unsupported = [(train, ["max_steps", "epochs"]), (eval, ["max_new_tokens"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(f"{__file__} doesn't support the {name!r} argument. This is set in {args}")
    required = [(train, ["max_tokens", "max_norm"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(f"{__file__} requires the {name!r} argument. This is set in {args}")
    # this should be safe to ignore given order of loading operations, and specifics of our setup
    # if initial_checkpoint_dir and resume:
    #     issues.append("Can't provide both `--resume` and `--initial_checkpoint_dir`. Choose one.")

    # saving
    if train.save_latest_ckpt:
        assert train.save_latest_interval is not None, "If save_latest_ckpt is True, save_latest_interval must be set to an integer value."
    
    # Check singleshot stuff
    if (hparams.singleshot.train_with_block_mask == False) and (hparams.singleshot.mask_region_ct != 1):
        issues.append("If not training with block mask, mask_region_ct must be 1.")
    if hparams.singleshot.train_with_block_mask == False:
        if hparams.singleshot.roll_offsets:
            issues.append("roll_offsets can only be True if train_with_block_mask is also True.")
        if hparams.singleshot.log_masks_and_inputs:
            issues.append("log_masks_and_inputs can only be True if train_with_block_mask is also True.")
    if hparams.singleshot.rand_rank_roll_offsets:
        if not hparams.singleshot.roll_offsets:
            issues.append("rand_rank_roll_offsets can only be True if roll_offsets is also True.")
    if (hparams.singleshot.rand_rank_roll_offsets and hparams.singleshot.lockstep_rand_roll_offsets):
        issues.append("rand_rank_roll_offsets and lockstep_rand_roll_offsets cannot both be True.")

    if (hparams.singleshot.lockstep_rand_k_toks and hparams.singleshot.rand_rank_k_toks):
        issues.append("lockstep_rand_k_toks and rand_rank_k_toks cannot both be True.")

    if issues:
        raise ValueError("\n".join(issues))


if __name__ == "__main__":

    set_parsing_settings(
        config_read_mode_urls_enabled=True,
        docstring_parse_attribute_docstrings=True,
    )

    torch.set_float32_matmul_precision("high")
    
    CLI(setup)
