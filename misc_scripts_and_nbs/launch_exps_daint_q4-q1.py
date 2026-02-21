# fmt: off
import os
from itertools import product, chain
import hashlib

LIST_CFGS = True
# LIST_CFGS = False

# WRITE_ONLY = True
WRITE_ONLY = False

RELAUNCH_ONLY = True
# RELAUNCH_ONLY = False

# RELAUNCH_STATE_CHECK = "running" # can be used to confirm we're adding to a real running job
RELAUNCH_STATE_CHECK = "pending"

# TARGET_JOB_CT = 1
TARGET_JOB_CT = 3
# TARGET_JOB_CT = 10
# TARGET_JOB_CT = 100

# LAUNCHER_FILEPATH = f"{os.environ['PROJ_USER']}/llnl-tools/launch_daint.py"
LAUNCHER_FILEPATH = f"{os.environ['WRKSPC']}/llnl-tools/launch_daint.py"

ENV_ACT_STYLE = "conda_activate"

# COMPILE_CACHE_DIR = None
COMPILE_CACHE_DIR = "/dev/shm/$USER/torchinductor"

# RM_CORE_DUMPS = False
RM_CORE_DUMPS = True

# for manual uenv based workflow
ENVIRONMENT = None
UENV = None 
CONTAINER = None
MODULES = None
# env activation is a bash macro sslm_env

# NCCL_CFG = "daint-defaults"
# NCCL_CFG = "daint-aggressive"
# NCCL_CFG = "daint-docs-asof-121725"
NCCL_CFG = "daint-asof-010626"

WANDB_OFFLINE = False

# debug interactive
# PARTITION = "debug"
# TIME_LIMIT = 29
# REPETITIONS = 1
# DEPENDENCY = None
# WANDB_OFFLINE = True

# # prod smoke test
# PARTITION = "normal"
# TIME_LIMIT = 59
# REPETITIONS = 1
# DEPENDENCY = None
# WANDB_OFFLINE = True

# prod
PARTITION = "normal"
TIME_LIMIT = 1440
# TIME_LIMIT = 720
# TIME_LIMIT = 360
# REPETITIONS = 1
REPETITIONS = 3
# REPETITIONS = 10
# REPETITIONS = 100
DEPENDENCY = "singleton"
WANDB_OFFLINE = False


# stage control
# DO_DATA_PREP = True
DO_DATA_PREP = False

DO_TRAIN = True
# DO_TRAIN = False

# EVAL_ITERS = 1
# EVAL_ITERS = 4
# EVAL_ITERS = 16

EVAL_INTERVAL = 200000 # past max
# EVAL_INTERVAL = 5000
# EVAL_INTERVAL = 1000
# EVAL_INTERVAL = 500
# EVAL_INTERVAL = 100

SAVE_INTERVAL = 10000
# SAVE_INTERVAL = 5000
# SAVE_INTERVAL = 1000
# SAVE_INTERVAL = 50
# MAX_CKPTS_TO_KEEP = 1
# MAX_CKPTS_TO_KEEP = 3
MAX_CKPTS_TO_KEEP = "null"

SAVE_LATEST_INTERVAL = 1000
# SAVE_LATEST_INTERVAL = 500
# SAVE_LATEST_INTERVAL = 100


# Fast startup settings

INIT_SAVE = True
# INIT_SAVE = False

# INIT_EVAL = True
INIT_EVAL = False

DO_COMPILE = True # currently seeing if both 32N and 128N can be compiled... nah
# DO_COMPILE = False

# DYNAMO_CACHE_SIZE_LIMIT = None
# DYNAMO_CACHE_SIZE_LIMIT = 32
DYNAMO_CACHE_SIZE_LIMIT = 256

COMPILE_MODE = "null"
# COMPILE_MODE = "max-autotune"
# COMPILE_MODE = "max-autotune-no-cudagraphs"

TORCH_DIST_TIMEOUT_MINS = "null"
# TORCH_DIST_TIMEOUT_MINS = 59


# BASE_OUT_DIR = f"{os.environ['PROJ_USER']}/singleshot-root/singleshot/outputs"
BASE_OUT_DIR = f"{os.environ['WRKSPC']}/singleshot-root/singleshot/outputs"

# BASE_RUN_NAME = f"debug_interactive"

# prod
# BASE_RUN_NAME = f"daint_prod_q4"
# BASE_RUN_NAME = f"daint_debug_q1"
# BASE_RUN_NAME = f"daint_debug_q1_long_timeout"
# BASE_RUN_NAME = f"daint_prod_q1_k2-16_abl"
# BASE_RUN_NAME = f"daint_debug_ift_ckpts"
# BASE_RUN_NAME = f"daint_debug_e2e_data_run"
# BASE_RUN_NAME = f"daint_prod_ift_sweep"
# BASE_RUN_NAME = f"daint_prod_ift_mask_fix"
# BASE_RUN_NAME = f"daint_prod_ift_magpie"
# BASE_RUN_NAME = f"daint_prod_ift_q3-4b"
# BASE_RUN_NAME = f"daint_prod_ift_q3-14b"
# BASE_RUN_NAME = f"daint_prod_ift_q3-32b"
# BASE_RUN_NAME = f"daint_prod_suprv_abl"
BASE_RUN_NAME = f"daint_prod_pre_arxiv_extra"

# Ad hoc indy run settings

# IGNORE_DATALOADER_STATE_ON_RESUME = True
IGNORE_DATALOADER_STATE_ON_RESUME = False

# IGNORE_FINGERPRINT_MISMATCH = True
IGNORE_FINGERPRINT_MISMATCH = False

GPN = 4

# Cfgs
exp_list = [
    ["litgpt/pretrain.py", "config_hub/pretrain/ss.yaml"],
    # ["litgpt/pretrain.py", "config_hub/pretrain/ss_gt_baseline.yaml"],
    # ["litgpt/pretrain.py", "config_hub/pretrain/ss_soft_teacher_baseline.yaml"],
    # ["litgpt/pretrain.py", "config_hub/pretrain/ss_prefix_suprv_baseline.yaml"],
    # ["litgpt/pretrain.py", "config_hub/pretrain/ss_beta1_baseline.yaml"],
    # ["litgpt/pretrain.py", "config_hub/pretrain/ss_beta2_baseline.yaml"],
]

# models
sweep_hparam = [
    # ["Llama-3.2-1B", "checkpoints/meta-llama/Llama-3.2-1B", "checkpoints/meta-llama/Llama-3.2-1B"],
    # ["Llama-3.1-8B", "checkpoints/meta-llama/Meta-Llama-3.1-8B", "checkpoints/meta-llama/Meta-Llama-3.1-8B"],
    # IFT model sweep
    # ["Llama-3.1-8B", None, "checkpoints/meta-llama/Meta-Llama-3.1-8B", 128002, "128011-128255"],
    # ["Llama-3.1-8B-Instruct", None, "checkpoints/meta-llama/Meta-Llama-3.1-8B-Instruct", 128002, "128011-128255"],
    # ["Llama-3.1-Tulu-3-8B-SFT", None, "checkpoints/allenai/Llama-3.1-Tulu-3-8B-SFT", 128002, "128011-128255"],
    # ["Llama-3.1-8B-Magpie-Align-SFT-v0.1", None, "checkpoints/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1", 128002, "128010-128255"],
    # ["Llama-3.1-8B-Instruct", None, "checkpoints/nvidia/OpenMath2-Llama3.1-8B, "128002, "128010-128255"],
    # Using the alternate mtp token spec variation
    # ["Llama-3.1-8B", None, "checkpoints/meta-llama/Meta-Llama-3.1-8B", 128002, "128011-128255", "null", "null", "null"], # control
    # ["Llama-3.1-8B-MTPV128384", None, "checkpoints/extended/meta-llama/Meta-Llama-3.1-8B-MTPV128384", "null", "null", '"<|mtp_special_token_{i}|>"', 32, "null"],
    # ["Llama-3.1-8B-Instruct-MTPV128384", None, "checkpoints/extended/meta-llama/Meta-Llama-3.1-8B-Instruct-MTPV128384", "null", "null", '"<|mtp_special_token_{i}|>"', 32, "128009-128006-882-128007"],
    # ["Llama-3.1-Tulu-3-8B-SFT-MTPV128384", None, "checkpoints/extended/allenai/Llama-3.1-Tulu-3-8B-SFT-MTPV128384", "null", "null", '"<|mtp_special_token_{i}|>"', 32, "null"],
    ["Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384", None, "checkpoints/extended/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384", "null", "null", '"<|mtp_special_token_{i}|>"', 32, "null"],
    # ["OpenMath2-Llama3.1-8B-MTPV128384", None, "checkpoints/extended/nvidia/OpenMath2-Llama3.1-8B-MTPV128384", "null", "null", '"<|mtp_special_token_{i}|>"', 32, "128009-128006-882-128007"],
    # ["Qwen3-8B-Base", None, "checkpoints/extended/Qwen/Qwen3-8B-Base-MTP", "null", "null", '"<|mtp_special_token_{i}|>"', 32, "null"],
    # ["Qwen3-8B", None, "checkpoints/extended/Qwen/Qwen3-8B-MTP", "null", "null", '"<|mtp_special_token_{i}|>"', 32, "null"],
    # ["Qwen3-4B-Instruct-2507", None, "checkpoints/extended/Qwen/Qwen3-4B-Instruct-2507-MTP", "null", "null", '"<|mtp_special_token_{i}|>"', 32, "null"],
    # ["Qwen3-30B-A3B-Instruct-2507", None, "checkpoints/extended/Qwen/Qwen3-30B-A3B-Instruct-2507-MTP", "null", "null", '"<|mtp_special_token_{i}|>"', 32, "null"],
    # ["Qwen3-14B-Base", None, "checkpoints/extended/Qwen/Qwen3-14B-Base-MTP", "null", "null", '"<|mtp_special_token_{i}|>"', 32, "null"],
    # ["Qwen3-32B-Instruct", None, "checkpoints/extended/qywu/Qwen3-32B-Instruct-MTP", "null", "null", '"<|mtp_special_token_{i}|>"', 32, "null"], # has no chat tempalate provided
]
exp_list = list(chain(*[[exp + hp for hp in sweep_hparam] for exp in exp_list]))

# dynamic data tok and source cfgs and whether to do indy train/val prep
sweep_hparam = [
    # ["config_hub/data/p2p_generic_metamath.yaml", "config_hub/data/metamath_strat_split_bos.yaml", True],
    ["config_hub/data/p2p_generic_metamath.yaml", "config_hub/data/metamath_strat_split_chat.yaml", True],
    # ["config_hub/data/p2p_generic_magpie.yaml", "config_hub/data/sources_magpie_bos.yaml", False],
    # ["config_hub/data/p2p_generic_magpie.yaml", "config_hub/data/sources_magpie_chat.yaml", False],
]
exp_list = list(chain(*[[exp + hp for hp in sweep_hparam] for exp in exp_list]))

# static data module and budget
sweep_hparam = [
    # ["TinyStories", int(450e6 * 10)],
    # [["pqds",
    #   f"{BASE_OUT_DIR}/p2p_tok_yolomix_test_256shard/multinode_tok/processed/pretrain/train",
    #   f"{BASE_OUT_DIR}/p2p_tok_yolomix_test_256shard/multinode_tok/processed/pretrain/val",
    #   ],
    #   int(17.0e9 * 1)],
    # [["pqds",
    #   f"{BASE_OUT_DIR}/p2p_tok_fineweb-edu_slice/multinode_tok/processed/pretrain/train",
    #   f"{BASE_OUT_DIR}/p2p_tok_fineweb-edu_slice/multinode_tok/processed/pretrain/val",
    #   ],
    #   int(5.95e9 * 1)],
    # [["pqds",
    #   f"{BASE_OUT_DIR}/p2p_tok_fineweb-edu_slice/multinode_tok/processed_block8192/pretrain/train",
    #   f"{BASE_OUT_DIR}/p2p_tok_fineweb-edu_slice/multinode_tok/processed_block8192/pretrain/val",
    #   ],
    #   int(5.95e9 * 1)],
    # [["pqds",
    #   f"{BASE_OUT_DIR}/p2p_tok_metamath_strat_split_train/multinode_tok/processed/pretrain/train",
    #   f"{BASE_OUT_DIR}/p2p_tok_metamath_strat_split_val/multinode_tok/processed/pretrain/train", # artifact of how p2p writes it
    #   ],
    # #   int(11.6e6 * 200)], # extended metamath target for k=16
    #   int(11.6e6 * 1000)], # large N extended
    # prod data
    # [["pqds",
    #   f"{BASE_OUT_DIR}/p2p_tok_metamath_strat_split_train/multinode_tok/processed/pretrain/train",
    #   f"{BASE_OUT_DIR}/p2p_tok_metamath_strat_split_val/multinode_tok/processed/pretrain/train", # artifact of how p2p writes it
    #   ],
    #   int(79.7e6 * 1000)],
    # [["pqds",
    #   f"{BASE_OUT_DIR}/p2p_tok_nemo_sci_spec/multinode_tok/processed/pretrain/train",
    #   f"{BASE_OUT_DIR}/p2p_tok_nemo_sci_spec/multinode_tok/processed/pretrain/val",
    #   ],
    #   int(100e9 * 3)],
    [[None,
      None,
      None,
      ],
      int(2e9)],
]
exp_list = list(chain(*[[exp + hp for hp in sweep_hparam] for exp in exp_list]))

# fabric strategy
sweep_hparam = [
    # ["null", "null"],
    # ["fsdp","null"]
    ["fsdp",4]
    # ["fsdp",8]
    # ["fsdp",16]
    # ["ddp","null"],
]
exp_list = list(chain(*[[exp + hp for hp in sweep_hparam] for exp in exp_list]))

# fsdp state dict
sweep_hparam = [
    "full",
    # "sharded",
]
exp_list = list(chain(*[[exp + [hp] for hp in sweep_hparam] for exp in exp_list]))

# topo and batch sizes, raw data max len and truncation len are the same for now
# plus ktoks, 
# use rank random k_toks or lockstep random k_toks
# and num regions etc
# make topo, bsz, and slen a set of paired hparams bc of memory use
sweep_hparam = [
    # moved k2-16 setup from from tuo
    # [1, GPN, 32, 1*GPN*32, 160, "0-16", "0-2", "0-16", "null", 5, "null", 1], # metamath ift
    # [1, GPN, 4, 1*GPN*4, 1024, "0-16", "0-2", "0-16", "null", 32, "null", 8], # magpie ift
    # [1, GPN, 32, 1*GPN*32, 160, "0-8", "0-2", "0-8", "null", 10, "null", 1], # lower k for metamath
    
    [1, GPN, 32, 1*GPN*32, 160, "0-9", "null", "null", "null", 5, "null", 1], # special metamath ift abl for ashwinee
    
    # prod 14b and 32b cfgs, equiv to existing 8B and 4B setup
    # [2, GPN, 16, 2*GPN*16, 160, "0-16", "0-2", "0-16", "null", 5, "null", 1], # 14b, 4 way fsdp
    # [8, GPN, 4, 8*GPN*4, 160, "0-16", "0-2", "0-16", "null", 5, "null", 1], # # 32b, 8 way fsdp
    # faster/strong scaled 14b and 32b cfgs
    # [8, GPN, 4, 8*GPN*4, 160, "0-16", "0-2", "0-16", "null", 5, "null", 1], # 14b, 4 way fsdp
    # [16, GPN, 2, 16*GPN*2, 160, "0-16", "0-2", "0-16", "null", 5, "null", 1], # 32b, 8 way fsdp
]
exp_list = list(chain(*[[exp + hp for hp in sweep_hparam] for exp in exp_list]))

# for bigger node counts this must be incr
TARGET_SHARD_NUM = 32
# TARGET_SHARD_NUM = 128

# rollout multiplier
sweep_hparam = [
    4,
]
exp_list = list(chain(*[[exp + [hp] for hp in sweep_hparam] for exp in exp_list]))

# use rank random k_toks or lockstep random k_toks
sweep_hparam = [
    [False, False], # baseline
    # [True, False],
    # [False, True],
]
exp_list = list(chain(*[[exp + hp for hp in sweep_hparam] for exp in exp_list]))

# whether to roll and 
# rank roll offsets and 
# lockstep rand roll offsets
# log masks and inputs 
# are tied
sweep_hparam = [
    # [False, False, False, True], # no roll, baseline
    # [True, False, False, True], # roll
    # [True, True, False, True], # rank roll
    [True, True, False, False], # rank roll quiet
    # [True, False, True, True], # lockstep roll
    # [True, False, True, False], # lockstep roll quiet
]
exp_list = list(chain(*[[exp + hp for hp in sweep_hparam] for exp in exp_list]))

# use bidirect attn or not
sweep_hparam = [
    # True,
    False,
]
exp_list = list(chain(*[[exp + [hp] for hp in sweep_hparam] for exp in exp_list]))

# warmup steps
sweep_hparam = [2000]
exp_list = list(chain(*[[exp + [hp] for hp in sweep_hparam] for exp in exp_list]))

# peak and min lr
sweep_hparam = [
    # ["constant",1e-4,"null"],
    # ["constant",5e-5,"null"],
    ["constant",1e-5,"null"], # tuo standard
    # ["constant",5e-6,"null"],
    # try cosine
    # ["cosine",1e-4,1e-5], # hot peak but maybe okay at scale and with decay
    # ["cosine",5e-5,5e-6], # more conservative
    # ["cosine",1e-5,1e-6], # low/standard
]
exp_list = list(chain(*[[exp + hp for hp in sweep_hparam] for exp in exp_list]))

final_exp_list = exp_list
for exp in final_exp_list:
    print(exp)

total_launches = 0
total_skips = 0
total_relaunches = 0
tot_incl_repetitions = 0

unique_run_names = set()

# queue all jobs
for exp in final_exp_list:

    (
        script,
        cfg,
        model_name,
        tok_dir,
        init_ckpt_dir,
        mask_id,
        sep_tok_range,
        mtp_pattern,
        num_mtp_toks,
        prelude_toks,
        ds_script_cfg,
        ds_src_cfg,
        indy_train_val,
        data,
        budget,
        fabric_strat,
        fsdp_model_dim,
        fsdp_state_type,
        nodes,
        gpn,
        mbsz,
        wbsz,
        slen,
        ktok,
        ktok_min,
        ktok_max,
        ex_val_ktoks,
        mask_regs,
        ex_val_lens,
        eval_iters,
        roll_mult,
        randk_ktok,
        lock_ktok,
        roll_offs,
        randk_roll,
        lock_roll,
        log_masks,
        bda,
        wu_steps,
        lr_sched,
        peak_lr,
        min_lr,
    ) = exp

    gpus = nodes * gpn

    cli_args = ""

    cfg_str = cfg.split("/")[-1].replace(".yaml","")
    cli_args += f" --config={cfg}"

    if model_name == "Llama-3.1-8B":
        model_str = "L3-8B"
    elif model_name == "Llama-3.2-1B":
        model_str = "L3-1B"
    else:
        model_str = model_name

    if "-Instruct" in model_name or "-SFT" in model_name:
        model_str += "-Inst" + init_ckpt_dir.replace("/", "-").replace(".", "-")
    
    if tok_dir is None:
        tok_dir = init_ckpt_dir
    
    cli_args += f" --model_name={model_name} --tokenizer_dir={tok_dir} --initial_checkpoint_dir={init_ckpt_dir}"

    # train dataloader state
    if IGNORE_DATALOADER_STATE_ON_RESUME:
        cli_args += f" --train.ignore_dataloader_state_on_resume={IGNORE_DATALOADER_STATE_ON_RESUME}"

    # fabric
    cli_args += f" --train.fabric_strategy={fabric_strat}"
    if fsdp_model_dim != "null":
        fsdp_device_mesh = f"{(nodes*gpn)//fsdp_model_dim}x{fsdp_model_dim}"
        cli_args += f" --train.fsdp_device_mesh={fsdp_device_mesh}"
    else:
        fsdp_device_mesh = "null"

    # fsdp state dict
    cli_args += f" --train.fsdp_state_dict_type={fsdp_state_type}"

    bsz_str = f"mb{mbsz}-wb{wbsz}-sl{slen}"
    cli_args += f" --train.micro_batch_size={mbsz} --train.global_batch_size={wbsz} --train.max_seq_length={slen}"

    if (not randk_ktok and not lock_ktok):
        print("Warning: both randk_ktok and lock_ktok are False, this is equivalent to no k_tok randomization.")
        ktok_min = "null"
        ktok_max = "null"

    ss_str = f"trnc{slen}-ktok{ktok}-ktokmin{ktok_min}-ktokmax{ktok_max}-rndktok{randk_ktok}-lockktok{lock_ktok}-mreg{mask_regs}-roll{roll_offs}-rndkrl{randk_roll}-lockrl{lock_roll}-logmasks{log_masks}-rmult{roll_mult}-exvlen{ex_val_lens}-exvk{ex_val_ktoks}-bda{bda}"
    ss_hash = hashlib.shake_128(str(ss_str).encode()).hexdigest(4)
    ss_str = f"ss-cfg{ss_hash}"

    cli_args += (
        f" --singleshot.mask_id={mask_id}"
        f" --singleshot.temp_sep_token_id_range={sep_tok_range}"
        f" --singleshot.mtp_special_token_pattern={mtp_pattern}"
        f" --singleshot.num_mtp_special_tokens={num_mtp_toks}"
        f" --singleshot.truncation_length={slen}"
        f" --singleshot.k_toks={ktok}"
        f" --singleshot.k_toks_min={ktok_min}"
        f" --singleshot.k_toks_max={ktok_max}"
        f" --singleshot.rand_rank_k_toks={randk_ktok}"
        f" --singleshot.lockstep_rand_k_toks={lock_ktok}"
        f" --singleshot.mask_region_ct={mask_regs}"
        f" --singleshot.roll_offsets={roll_offs}"
        f" --singleshot.rand_rank_roll_offsets={randk_roll}"
        f" --singleshot.lockstep_rand_roll_offsets={lock_roll}"
        f" --singleshot.log_masks_and_inputs={log_masks}"
        f" --singleshot.rollout_multiplier={roll_mult}"
        f" --singleshot.extra_val_trunc_lengths={ex_val_lens}"
        f" --singleshot.extra_val_k_toks_values={ex_val_ktoks}"
        f" --singleshot.bidirect_ss_attn={bda}"
    )
    
    # save eval stuff
    cli_args += f" --train.initial_save={INIT_SAVE} --train.save_latest_interval={SAVE_LATEST_INTERVAL} --train.save_interval={SAVE_INTERVAL} --train.max_ckpts_to_keep={MAX_CKPTS_TO_KEEP} --eval.interval={EVAL_INTERVAL} --eval.max_iters={eval_iters} --eval.initial_validation={INIT_EVAL}"

    # learning rate stuff
    if lr_sched == "cosine":
        sched_shortname = "cos"
    elif lr_sched == "constant":
        sched_shortname = "const"
    else:
        raise ValueError("Unknown lr_sched")
    lr_str = f"lr-{wu_steps}steps-{sched_shortname}{peak_lr}to{min_lr}"
    cli_args += f"  --train.lr_warmup_steps={wu_steps} --train.lr_schedule={lr_sched} --train.peak_lr={peak_lr} --train.min_lr={min_lr}"

    # Compile stuff
    compile_str = f"compile{DO_COMPILE}-mode{COMPILE_MODE}-dcs{DYNAMO_CACHE_SIZE_LIMIT}"
    compile_hash = hashlib.shake_128(str(compile_str).encode()).hexdigest(4)
    compile_str = f"comp-cfg{compile_hash}"
    cli_args += f" --train.do_compile={DO_COMPILE} --train.compile_mode={COMPILE_MODE} --train.dynamo_cache_size_limit={DYNAMO_CACHE_SIZE_LIMIT}"

    # timeout control
    cli_args += f" --train.pt_dist_timeout_mins={TORCH_DIST_TIMEOUT_MINS}"

    # data preliminary stuff
    data_str = ""
    if isinstance(data,list) and data[0] is not None:
        data_str += "-".join(data)
    if ds_script_cfg is not None and ds_src_cfg is not None:
        data_str += f"-dscr{ds_script_cfg.split('/')[-1].replace('.yaml','')}-dsrc{ds_src_cfg.split('/')[-1].replace('.yaml','')}"
        cli_args += f" --pqds.dataset_script_config_file={ds_script_cfg} --pqds.dataset_sources_config_file={ds_src_cfg}"
    
    data_str += f"preltoks{prelude_toks}"
    cli_args += f" --pqds.prelude_token_ids={prelude_toks}"

    cli_args += f" --pqds.ignore_fingerprint_mismatch={IGNORE_FINGERPRINT_MISMATCH}"
    
    pqds_hash = hashlib.shake_128(str(data_str).encode()).hexdigest(4)
    data_str = f"pqds-cfg{pqds_hash}"

    if budget > 1e9:
        bgt = f"{round(budget / 1e9,1)}B"
    elif budget > 1e6:
        bgt = f"{round(budget / 1e6)}M"
    elif bgt > 1e3:
        budget = f"{round(budget / 1e3)}k"
    bgt_str = f"toks{bgt}"
    cli_args += f" --train.max_tokens={budget}"
    
    # mod more things
    # ...

    # join to a unique run name for the experiment
    full_hash_str = f"{cfg_str}_{model_str}_{data_str}_{bgt_str}_{ss_str}_{bsz_str}_{lr_str}_{compile_str}"
    full_hash_str = hashlib.shake_128(full_hash_str.encode()).hexdigest(4)
    run_name = (
        f"{BASE_RUN_NAME}_{nodes}N{gpus}n_{full_hash_str}"
    )
    unique_run_names.add(run_name)

    launch_out_dir = f"{BASE_OUT_DIR}/{BASE_RUN_NAME}"
    full_out_dir = f"{launch_out_dir}/{run_name}"

    # handle data prep
    data_prep_command = ""
    if DO_DATA_PREP and ds_script_cfg is not None and ds_src_cfg is not None:
        if indy_train_val:
            suffixes = ["_train", "_val"]
        else:
            suffixes = [""]
        for suffix in suffixes:
            data_prep_command += f"""
python litgpt/pull_raw_datasets.py \
--srcs_config_file={ds_src_cfg.replace('.yaml','')}{suffix}.yaml \
--script_configs={ds_script_cfg} \
--script_configs.num_raw_shards={TARGET_SHARD_NUM} \
--script_configs.target_shard_num={TARGET_SHARD_NUM} \
--script_configs.hf_tokenizer={tok_dir} \
--script_configs.raw_dir={full_out_dir}{suffix.replace('_','/')}/raw \
--script_configs.output_dir={full_out_dir}{suffix.replace('_','/')} || exit 1

python litgpt/p2p_tokenizer.py \
--srcs_config_file={ds_src_cfg.replace('.yaml','')}{suffix}.yaml \
--script_configs={ds_script_cfg} \
--script_configs.num_raw_shards={TARGET_SHARD_NUM} \
--script_configs.target_shard_num={TARGET_SHARD_NUM} \
--script_configs.hf_tokenizer={tok_dir} \
--script_configs.raw_dir={full_out_dir}{suffix.replace('_','/')}/raw \
--script_configs.output_dir={full_out_dir}{suffix.replace('_','/')} \
{f'--script_configs.train_split_pct=1.0' if indy_train_val else ''} || exit 1

""" 
        if indy_train_val:
            data_prep_command += f"""\
if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "Process 0: Moving files..."
    
    # Perform your move operation
    mkdir -p {full_out_dir}/processed
    mv {full_out_dir}/train/processed/train {full_out_dir}/processed/train || exit 1
    mv {full_out_dir}/val/processed/train {full_out_dir}/processed/val || exit 1

    echo "Process 0: Move complete."
fi

"""
    train_path = f"{full_out_dir}/processed/train"
    val_path = f"{full_out_dir}/processed/val"
    if all([elm is None for elm in data]):
        data = ["pqds", train_path, val_path]
    
    if isinstance(data,list):
        cli_args += f"  --data={data[0]} --pqds.train_dataset_folder_path={data[1]} --pqds.val_dataset_folder_path={data[2]}"
    else:
        data_str = data
        cli_args += f" --data={data}"

    # put together the actual "train.py" command
    train_command = ""
    if DO_TRAIN:
        train_command += f"python -u {script} {cli_args} --out_dir={full_out_dir} --wandb.run_name={run_name} --wandb.offline={WANDB_OFFLINE}"
    
    custom_invocation = f"{data_prep_command}{train_command}"

    REMAINING_REPS = REPETITIONS
    if RELAUNCH_ONLY:
        # check squeue for a run with the same name
        sq_out = os.popen(f"squeue -u $USER -t {RELAUNCH_STATE_CHECK} -n {run_name}").read()
        # count the lines
        res = sq_out.strip("\n").split("\n")
        nlines = len(res)
        njobs = nlines - 1  # subtract header
        # assert nlines in [1,2], "Only cases I expect"
        if njobs >= TARGET_JOB_CT:
            jobhit = res[1]
            # print(f"Skipping {run_name}, {njobs} instances already in queue: \n{'\n'.join(res)}")
            print(f"Skipping {run_name}, {njobs} instances already in queue.")
            total_skips += 1
            continue
        else:
            # set repetitions to remaining needed
            REMAINING_REPS = TARGET_JOB_CT - njobs
            REMAINING_REPS = min(REMAINING_REPS, REPETITIONS)
            print(f"Relaunching {run_name} w/ {REMAINING_REPS} repetitions, as {'no' if njobs==0 else f'only {njobs}'} runs found in queue with same name and state={RELAUNCH_STATE_CHECK}.")
            total_relaunches += 1

    # make the complete launcher command
        # --wandb_offline={WANDB_OFFLINE} \
    command = f"""\
    python {LAUNCHER_FILEPATH} \
        --output_dir={launch_out_dir} \
        --nccl_cfg={NCCL_CFG} \
        --env_act_style={ENV_ACT_STYLE} \
        {'--environment='+ENVIRONMENT+' ' if ENVIRONMENT is not None else ''}\
        {'--uenv='+UENV+' ' if UENV is not None else ''}\
        {'--modules='+MODULES+' ' if MODULES is not None else ''}\
        {'--container='+CONTAINER+' ' if CONTAINER is not None else ''}\
        {'--cache_dir='+COMPILE_CACHE_DIR+' ' if COMPILE_CACHE_DIR is not None else ''}\
        {'--rm_core_dumps='+str(RM_CORE_DUMPS)+' ' if RM_CORE_DUMPS is not None else ''}\
        --partition={PARTITION} \
        --repetitions={REMAINING_REPS}{f' --dependency={DEPENDENCY}' if DEPENDENCY is not None else ''} \
        --minutes={TIME_LIMIT} \
        --nodes={nodes} \
        --gpus_per_node={gpn} \
        --run_name={run_name} \
        --custom_invocation='{custom_invocation}' \
        --pass_run_name=False \
        {'--dryrun' if WRITE_ONLY else ''}
    """

    total_launches += 1
    tot_incl_repetitions += REMAINING_REPS    
    if not LIST_CFGS:
        os.system(command)
    else:
        print(run_name)
        # print(command)
if RELAUNCH_ONLY:
    print(f"Total skips/relaunches (incl repetitions): {total_skips}/{total_relaunches} ({tot_incl_repetitions})")
    assert (total_skips+total_relaunches) == len(unique_run_names), f"{total_launches} != {len(unique_run_names)} Jobs might be overwriting eachother, plz check that all hparams factor into naming."
else:
    print(f"Total launches (incl repetitions): {total_launches} ({tot_incl_repetitions})")
    assert total_launches == len(unique_run_names), f"{total_launches} != {len(unique_run_names)} Jobs might be overwriting eachother, plz check that all hparams factor into naming."
