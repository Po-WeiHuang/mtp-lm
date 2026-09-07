# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class TrainArgs:
    """Training-related arguments"""

    save_interval: Optional[int] = 1000
    """Number of optimizer steps between saving checkpoints"""
    save_latest_interval: Optional[int] = 1000
    """Number of optimizer steps between saving 'latest' checkpoints"""
    max_ckpts_to_keep: Optional[int] = 1
    """Number of saved checkpoints to keep, older ones are del'd, unless None."""
    save_latest_ckpt: bool = True
    """Whether to always save a 'latest' checkpoint that gets overwritten."""
    initial_save: bool = False
    """Whether to save a checkpoint at the beginning of training"""
    final_save: bool = True
    """Whether to save a checkpoint at the end of training"""
    log_interval: int = 1
    """Number of iterations between logging calls"""
    global_batch_size: int = 512
    """Number of samples between optimizer steps across data-parallel ranks"""
    micro_batch_size: int = 4
    """Number of samples per data-parallel rank"""
    lr_warmup_steps: Optional[int] = 5000
    """Number of iterations with learning rate warmup active"""
    lr_warmup_fraction: Optional[float] = None
    """The fraction of an epoch to use for learning rate warmup"""
    epochs: Optional[int] = None
    """Number of epochs to train on"""
    # TODO: `pretrain` is the only script using `max_tokens` explicitly. replace it with epoch_size*epochs?
    max_tokens: Optional[int] = int(3e12)  # 3 trillion
    """Total number of tokens to train on"""
    max_steps: Optional[int] = None
    """Limits the number of optimizer steps to run"""
    max_time: Optional[float] = None
    """Limits the number of seconds to train for"""
    max_seq_length: Optional[int] = None
    """Limits the length of samples that will be passed to the model"""
    train_block_size: Optional[int] = None
    """Sets the block size for the train dataloader, separate from max_seq_length"""
    tie_embeddings: Optional[bool] = False
    """Whether to tie the embedding weights with the language modeling head weights"""
    do_compile: bool = True
    """Whether to use torch.compile to optimize the model and other functions."""
    compile_mode: Optional[str] = None
    """The mode to use for torch.compile, e.g., 'default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'"""
    dynamo_cache_size_limit: Optional[int] = None
    """If set, overrides torch._dynamo.config.cache_size_limit with this value."""
    fabric_strategy: Optional[str] = None
    """Optionally override the Fabric strategy choice otherwise a default configured."""
    fsdp_device_mesh: Optional[str] = None
    """If set, enables FSDP with the specified device mesh, e.g., 4x4 for 4N to node boundary or 2x8 for 4N across 2N"""
    fsdp_activation_checkpointing: bool = False
    """Whether to use activation checkpointing to save memory during training."""
    fsdp_limit_all_gathers: bool = False
    """Whether to limit allgathers during training when using FSDP."""
    fsdp_state_dict_type: str = "full"
    """If set, overrides the FSDP state dict type, e.g., 'full' or 'sharded'."""
    ignore_dataloader_state_on_resume: bool = False
    """Whether to ignore the dataloader state when resuming from a checkpoint. This is a hack when continuing a run with modified dataset/dataloader settings."""
    ignore_extra_keys_in_init_ckpt: bool = False
    """Whether to ignore extra keys in the checkpoint when initializing the model, eg. we already trained it some and it has optim states."""
    sync_running_metrics: bool = True
    """Whether to sync the running TorchMetrics across all processes during training."""
    pt_dist_timeout_mins: int = None
    """Timeout in minutes for PyTorch distributed process group operations."""

    # Optimization args
    lr_schedule: str = "cosine"
    """Learning rate schedule to use during training"""
    max_norm: Optional[float] = 1.0
    """Maximum norm for gradient clipping."""
    peak_lr: float = 6e-6
    """Peak learning rate for the learning rate scheduler."""
    min_lr: Optional[float] = 6e-7
    """Minimum learning rate for the learning rate scheduler."""

    def __post_init__(self) -> None:
        if self.lr_warmup_fraction and self.lr_warmup_steps:
            raise ValueError(
                "Can't provide both `--train.lr_warmup_fraction` and `--train.lr_warmup_steps`. Choose one."
            )
        if self.lr_warmup_fraction and not (0 <= self.lr_warmup_fraction <= 1):
            raise ValueError("`--train.lr_warmup_fraction` must be between 0 and 1.")

        if (
            self.lr_warmup_steps
            and self.max_steps
            and (self.lr_warmup_steps >= self.max_steps)
        ):
            warnings.warn(
                "`--train.lr_warmup_steps` should be less than `--train.max_steps`."
                f" Got {self.lr_warmup_steps} lr_warmup_steps and {self.max_steps} max_steps.",
                UserWarning,
            )

    def gradient_accumulation_iters(self, devices: int, num_nodes: int = 1) -> int:
        """Number of iterations between gradient synchronizations"""
        gradient_accumulation_iters = (
            self.batch_size(devices, num_nodes) // self.micro_batch_size
        )
        assert gradient_accumulation_iters > 0
        return gradient_accumulation_iters

    def batch_size(self, devices: int, num_nodes: int = 1) -> int:
        """Number of samples between optimizer steps per data-parallel rank"""
        batch_size = self.global_batch_size // (devices * num_nodes)
        assert batch_size > 0
        return batch_size

    def warmup_iters(
        self, devices: int, num_nodes: int, max_iters: int, train_dataloader
    ) -> int:
        """Number of iterations to warm up the learning rate."""
        if self.lr_warmup_fraction:
            return min(
                max_iters, math.ceil(self.lr_warmup_fraction * len(train_dataloader))
            )
        if self.lr_warmup_steps:
            return min(
                max_iters,
                self.lr_warmup_steps
                * self.gradient_accumulation_iters(devices, num_nodes),
            )
        return 0


@dataclass
class EvalArgs:
    """Evaluation-related arguments"""

    interval: int = 1000
    """Number of optimizer steps between evaluation calls"""
    max_new_tokens: Optional[int] = None
    """Number of tokens to generate"""
    max_iters: int = 100
    """Number of iterations"""
    initial_validation: bool = True
    """Whether to evaluate on the validation set at the beginning of the training"""
    final_validation: bool = True
    """Whether to evaluate on the validation set at the end of the training"""
    evaluate_example: Union[str, int] = "first"
    """How to pick an example instruction to evaluate periodically during training.
       Can be "first", "random", or an integer index to pick a specific example."""
    val_block_size: Optional[int] = None
    """Sets the block size for the val dataloader, separate from max_seq_length"""


@dataclass
class LogArgs:
    """Logging-related arguments"""

    project: Optional[str] = None
    """Project name"""
    run: Optional[str] = None
    """Run name"""
    group: Optional[str] = None
    """Group name"""


@dataclass
class WandbArgs:
    """Wandb-related arguments"""

    entity: Optional[str] = "tomg-group-umd"
    """Wandb entity"""
    project: Optional[str] = "singleshot"
    """Wandb project name"""
    run_name: Optional[str] = "debug"
    """Wandb run name"""
    group: Optional[str] = None
    """Wandb group name"""
    tags: Optional[list[str]] = None
    """Wandb tags"""
    offline: bool = False
    """Whether to run wandb in offline mode"""


@dataclass
class PQDSArgs:
    """Parquet Dataset related args"""

    dataset_script_config_file: Optional[str] = None
    """Path to dataset script configs YAML file. Just passed for reporting purposes."""
    dataset_sources_config_file: Optional[str] = None
    """Path to dataset sources YAML file. Just passed for reporting purposes."""

    num_workers: int = 0
    """Number of workers to use for loading the Parquet dataset"""
    prefetch_factor: int = 4
    """Number of batches to prefetch per worker. Only used if num_workers > 0"""
    # dataset_folder_path: Optional[str] = ""
    train_dataset_folder_path: Optional[str] = (
        "/p/vast1/kirchenb/retrieval_output/p2p_dataset_patch/wir_unpacked/train"
    )
    val_dataset_folder_path: Optional[str] = (
        "/p/vast1/kirchenb/retrieval_output/p2p_dataset_patch/wir_unpacked/val"
    )
    """Path to the Parquet Dataset"""
    train_prefix: Optional[str] = ""
    val_prefix: Optional[str] = ""
    """Prefix to use when globbing for Parquet files in the dataset folder"""
    broadcast_glob: bool = True
    """Whether to broadcast the globbed list of Parquet filenames to all ranks rather than having each rank glob independently"""
    shuffle: bool = True
    """Whether to shuffle the dataset"""
    shuffle_filenames: bool = True
    """Whether to shuffle the list of Parquet filenames before loading"""
    doc_wise: bool = False
    """Whether to use document-wise Parquet dataset processing"""
    doc_wise_skip_tail: bool = True
    """Whether to skip the tail of documents that don't fit exactly into blocks during document-wise Parquet dataset processing"""
    doc_wise_sep_tok: Optional[int] = None
    """Separator token ID to use between documents during document-wise Parquet dataset processing"""
    ignore_fingerprint_mismatch: bool = False
    """Whether to ignore fingerprint mismatches when loading Parquet datasets."""
    pad_token_id: Optional[int] = None
    """Pad token ID to use for padding sequences"""
    prelude_token_ids: Optional[str] = None
    """An subarray of token IDs that indicate prelude tokens in the dataset, used to adjust masking logic, eg 128009-128006-882-128007"""
    omit_tail_padding_in_loss: bool = True
    """Whether to omit the loss contributions from eos/padding tokens at the end of sequences"""
    verbose: bool = True
    """Whether to print verbose logs during Parquet dataset loading"""
    estimate_token_counts: bool = True
    """Whether to estimate token counts in the Parquet dataset for reporting purposes"""
    estimate_file_count: int = 1
    """Number of Parquet files to use for token count estimation"""


@dataclass
class SingleShotArgs:
    """Single Shot LM recipe related args"""

    truncation_length: int = 32
    """Number of tokens to truncate each sample to, separate from train.max_seq_length"""
    mask_id: Optional[int] = None
    """Token ID used as a mask token during single-shot prediction, eg. 128002 when sneaking it into the L3 tokenizer"""
    temp_sep_token_id_range: Optional[str] = None
    """Token ID range to use for special markers during some output postproc ops, eg. '128011-128255' when sneaking it into the L3 tokenizer"""
    mtp_special_token_pattern: Optional[str] = "<|mtp_special_token_{i}|>"
    """Pattern for MTP special tokens to add, e.g., "<|mtp_special_token_{i}|>", exclusive with """
    num_mtp_special_tokens: Optional[int] = 32
    """Number of MTP special tokens to add to the tokenizer and model, Used to materialize the mtp_special_token_pattern."""
    pos_wise_unique_mtp_tok_ids: bool = False
    """Whether to use unique MTP special tokens for each masked position in the input sequence."""
    min_mask_id: Optional[int] = None
    """Minimum token ID to use for mask tokens when pos_wise_unique_mtp_tok_ids, set automatically in the unique case."""
    max_mask_id: Optional[int] = None
    """Maximum token ID to use for mask tokens when pos_wise_unique_mtp_tok_ids, set automatically in the unique case."""
    k_toks: Union[int | str | list] = 1
    """Number of tokens to predict in a single shot. If special string, it's a curriculum def."""
    lockstep_rand_k_toks: bool = False
    """Whether to randomize the k_toks value per step same across ranks during training."""
    rand_rank_k_toks: bool = False
    """Whether to randomize the k_toks value per rank and per step during training."""
    k_toks_min: Optional[Union[int | str | list]] = None
    """Minimum k_toks value when randomizing k_toks. If special string, it's a curriculum def."""
    k_toks_max: Optional[Union[int | str | list]] = None
    """Maximum k_toks value when randomizing k_toks. If special string, it's a curriculum def."""
    beta: float = 0.0
    """Weighting factor for the entropy term in the loss function"""

    topk_values: Union[str | list] = "1-3-10-50-1000"
    """List of values to monitor top-k token accuracy for."""

    extra_train_metrics: bool = False
    """Whether to compute extra metrics (beyond loss components) during training step."""

    num_samples: Optional[int] = None
    """Number of multinomial sampled generations to produce during SS evaluation."""

    sample_during_train: bool = False
    """Whether to use multinomial sampling instead of argmax during the SS forward in train step."""

    hard_teacher_supervision: bool = True
    """Whether to use sampled/argmax tokens from teacher for CE/KL loss rather than soft probs."""

    gt_teacher_supervision: bool = False
    """Whether to train against the ground truth suffixes rather than stud forced teacher outputs."""

    supervise_prefix: bool = False
    """Whether to also supervise standard offline next token prediction on the prefixes."""

    supervise_prefix_only: bool = False
    """Whether to only supervise standard offline next token prediction on the prefixes, mainly for debugging."""

    hard_self_teacher_supervision: bool = True
    """Whether to use sampled/argmax tokens from student model itself as targets."""

    rollout_multiplier: int = 1
    """How many multiples of the current k-value to roll out for during the generation loop (both SS and AR)."""

    extra_val_trunc_lengths: Union[None, str | list] = None
    """Additional truncation lengths to evaluate at during validation, same format as topk_values"""

    extra_val_k_toks_values: Union[None, str | list] = None
    """Additional k_toks values to evaluate at during validation, same format as topk_values"""

    train_with_block_mask: bool = True
    """Whether to use the BlockMask + flex attention setup during training, else standard causal SDPA."""

    bidirect_ss_attn: bool = False
    """Whether to use bidirectional attention within the masked region during single-shot prediction with BlockMask."""

    mask_region_ct: int = 1
    """Number of masked regions to use during training with BlockMask + flex attention. Note must be 1 if not using block mask."""

    roll_offsets: bool = True
    """Whether to use variable offsets for the block masks during training with BlockMask + flex attention."""

    rand_rank_roll_offsets: bool = True
    """Whether to use rank-specific random variable offsets for the block masks during training with BlockMask + flex attention."""

    lockstep_rand_roll_offsets: bool = False
    """Whether to use the same random offset across all ranks for the block masks during training with BlockMask + flex attention."""

    log_masks_and_inputs: bool = False
    """Whether to log the generated masks and masked inputs during training for visualization."""

    multi_region_val_correction: bool = True
    """Whether to apply correction to the single-shot loss during eval when using multiple masked regions."""

    last_region_loss_only: bool = False
    """Debug flag for only computing single-shot loss on the last masked region during training when using multiple masked regions."""

    drift_diagnostics_enabled: bool = False
    """DriftMTP Phase 2: whether to compute diagnostic-only ECE/MMD/Sinkhorn metrics comparing
    student and teacher predictive states during training (see src/driftmtp/validation.py and
    claudedriftingplan.md > Validation Metrics). Purely diagnostic: never affects the training
    objective, gradients, or model parameters. Computed only on the same iterations already
    gated by train.log_interval, reusing that step's existing student/teacher forward passes
    (no extra forward pass, no extra dataloader). Default False recovers exactly the prior
    (pre-DriftMTP) training behavior."""

    drift_diagnostics_mmd_warmup_steps: int = 2000
    """DriftMTP Phase 2: number of training iterations (state["iter_num"], NOT optimizer steps
    or diagnostic-call count) over which the shared MMD kernel bandwidth is recomputed from
    horizon-1 features on each diagnostic step; frozen at its last value for every iteration
    after this. Deliberately independent of train.lr_warmup_steps (a different concept -- LR
    schedule vs. kernel-scale stabilization), not derived from it. Must be several multiples of
    log_iter_interval (train.log_interval * gradient_accumulation_iters) to actually let the
    bandwidth recompute more than once before freezing, since diagnostics -- and therefore
    bandwidth updates -- only run on log_iter_interval-aligned iterations. See
    src/driftmtp/metrics/mmd.py (MMDBandwidthSchedule) for the policy this implements."""

    drift_enabled: bool = False
    """DriftMTP Phase 4: whether to add lambda_drift * L_drift^token (see
    src/driftmtp/training.py and claudedriftingplan.md > PHASE 4 -- Training Integration) to the
    training objective, so L_total = L_MTP + lambda_drift * L_drift^token. Requires the
    student-forced teacher pass, so it is inapplicable under gt_teacher_supervision (asserted at
    runtime). Default False recovers exactly the prior (pre-DriftMTP) training behavior. Setting
    drift_enabled=True with drift_weight=0.0 MUST reproduce native MTP training numerically
    (the drift loss is still computed every iteration, but its exact-zero contribution to the
    backward pass leaves gradients unchanged) -- this is the Phase 4 backward-compatibility test."""

    drift_weight: float = 0.0
    """DriftMTP Phase 4: lambda_drift, the token drifting loss weight. Only has an effect when
    drift_enabled=True. Default 0.0 is deliberately inert even when drift_enabled=True, so
    turning drift_enabled on by itself never changes the training objective.

    This is a FIXED constant for the whole run. Calibrate it so the weighted drift term is a
    chosen fraction of L_MTP at the START of training (~10% is the working default), then let
    it decay on its own: rms_field_normalize divides by (lambda_tau + eps), so a temperature
    whose raw field falls below eps stops being renormalized back to unit scale and its
    contribution decays toward zero as the student population converges to the teacher. An
    adaptive weight that pinned lambda_drift * L_drift / L_MTP to a constant was tried and
    removed, because it holds the drift term up forever and defeats exactly that fade."""

    drift_exclude_self_positive: bool = True
    """DriftMTP: mask the query's OWN paired teacher token out of the positive population,
    mirroring the self-exclusion that has always been applied to the negative population.

    y_pos[r] is the teacher state for the SAME token as the student query x[r]. As the student
    converges toward the teacher that pair's distance goes to zero and, at low tau, its softmax
    weight goes to 1 -- the positive mass collapses onto a single self-pair, while the negative
    population still has its own self-pair removed. That asymmetry leaves a systematic residual
    field that does not vanish at equilibrium, so "student == teacher" is NOT a fixed point of
    the original construction. Masking both sides restores the symmetry the paper's unpaired
    positives have for free.

    True (default) is the symmetric construction. False reproduces the original asymmetric one;
    use it only to reproduce runs from before this change."""

    drift_temperatures: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.2])
    """DriftMTP Phase 4: the base Laplace-kernel temperatures tau (claudedriftingplan.md >
    Drifting Normalization > Kernel Temperature), SUMMED after per-temperature RMS field
    normalization (Multi-Temperature Field Normalization). Default matches the paper's
    {0.02, 0.05, 0.2}."""

    drift_step_size: float = 1.0
    """DriftMTP Phase 4: eta, the step size applied to the multi-temperature drift field before
    forming the stopped-gradient drift target in L_drift^token (claudedriftingplan.md > Token
    Drifting Loss)."""

    drift_use_column_norm: bool = True
    """DriftMTP Phase 4: forwards to src/driftmtp/drifting.py's `use_column_norm` toggle
    (claudedriftingplan.md > Column-Normalization Toggle). True (default) uses Algorithm 2's
    row+column geometric-mean affinity; False uses Eq. 8's separately-normalized affinity."""


def parse_temp_sep_token_id_range(temp_sep_token_id_range: str):
    """'128011-128255' -> (128011, 128255)"""
    if temp_sep_token_id_range is None:
        return None

    start_id, end_id = temp_sep_token_id_range.split("-")
    return int(start_id), int(end_id)


def parse_prelude_token_ids(prelude_token_ids: str):
    """'128009-128006-882-128007' -> [128009,128006,882,128007]"""
    if prelude_token_ids is None:
        return None

    return [int(elm) for elm in prelude_token_ids.split("-")]


def parse_k_toks(k_toks: str):
    """'0-1_10-2_20-3' -> [ [0,1], [10,2], [20,3] ]"""
    if k_toks is None:
        return None

    if isinstance(k_toks, int):
        return k_toks
    else:
        intervals = k_toks.split("_")
        intervals = [list(map(int, interval.split("-"))) for interval in intervals]
        return intervals


def parse_topk_values(topk_values: str):
    """'1-2-3-5-10' -> [1,2,3,5,10]"""
    if isinstance(topk_values, int):
        return [topk_values]
    else:
        return [int(elm) for elm in topk_values.split("-")]


def parse_extra_val_trunc_lengths(extra_val_trunc_lengths: Optional[str]):
    """'32-64-128' -> [32,64,128]"""
    if extra_val_trunc_lengths is None:
        return None
    elif isinstance(extra_val_trunc_lengths, int):
        return [extra_val_trunc_lengths]
    else:
        return [int(elm) for elm in extra_val_trunc_lengths.split("-")]


def parse_extra_val_k_toks_values(extra_val_k_toks_values: Optional[str]):
    """'1-2-3-5' -> [1,2,3,5]"""
    if extra_val_k_toks_values is None:
        return None
    elif isinstance(extra_val_k_toks_values, int):
        return [extra_val_k_toks_values]
    else:
        return [int(elm) for elm in extra_val_k_toks_values.split("-")]
