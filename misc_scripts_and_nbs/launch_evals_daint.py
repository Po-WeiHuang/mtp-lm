# fmt: off
import os
from itertools import product, chain
import hashlib

LIST_CFGS = True
# LIST_CFGS = False

# WRITE_ONLY = True
WRITE_ONLY = False

# LAUNCHER_FILEPATH = f"{os.environ['PROJ_USER']}/llnl-tools/launch_daint.py"
LAUNCHER_FILEPATH = f"{os.environ['WRKSPC']}/llnl-tools/launch_daint.py"

ENV_ACT_STYLE = "conda_activate"

# COMPILE_CACHE_DIR = None
COMPILE_CACHE_DIR = "/dev/shm/$USER/torchinductor"

# RM_CORE_DUMPS = False
RM_CORE_DUMPS = True

# NCCL_CFG = "daint-defaults"
# NCCL_CFG = "daint-aggressive"
# NCCL_CFG = "daint-docs-asof-121725"
NCCL_CFG = "daint-asof-010626"

# for manual uenv based workflow
ENVIRONMENT = None
UENV = None 
CONTAINER = None
MODULES = None

# # debug
# PARTITION = "debug"
# TIME_LIMIT = 29
# REPETITIONS = 1
# DEPENDENCY = None

# # smoke test
# PARTITION = "normal"
# TIME_LIMIT = 20
# REPETITIONS = 1
# DEPENDENCY = "singleton"

# prod 
PARTITION = "normal"
# TIME_LIMIT = 180
# TIME_LIMIT = 480
TIME_LIMIT = 1440
REPETITIONS = 1
DEPENDENCY = "singleton"


# BASE_OUT_DIR = f"{os.environ['PROJ_USER']}/singleshot-root/singleshot/outputs"
BASE_OUT_DIR = f"{os.environ['WRKSPC']}/singleshot-root/singleshot/outputs"

BASE_TOKS_DIR = f"{os.environ['WRKSPC']}/singleshot-root/singleshot/checkpoints"

# BASE_RUN_NAME = f"debug_lm_eval_metrics"

# BASE_RUN_NAME = f"lm_eval_ift_mask_fix"
# BASE_RUN_NAME = f"lm_eval_ift_magpie"
# BASE_RUN_NAME = f"lm_eval_ift_mask_fix_imprv_metrics"
# BASE_RUN_NAME = f"lm_eval_ift_q3_len_incr"
# BASE_RUN_NAME = f"lm_eval_ift_magpie_convert"
# BASE_RUN_NAME = f"lm_eval_ift_q3-4b_convert"
# BASE_RUN_NAME = f"lm_eval_ift_dp"
# BASE_RUN_NAME = f"lm_eval_ift_dp_ml2048"
# BASE_RUN_NAME = f"lm_eval_push_for_rl"
# BASE_RUN_NAME = f"lm_eval_retest_dp"
# BASE_RUN_NAME = f"lm_eval_new_tasks"
# BASE_RUN_NAME = f"lm_eval_final_abl"
# BASE_RUN_NAME = f"lm_eval_big_models"
# BASE_RUN_NAME = f"lm_eval_pre_arxiv"
# BASE_RUN_NAME = f"lm_eval_pre_arxiv_extra"
# BASE_RUN_NAME = f"push_for_public_release"
# BASE_RUN_NAME = f"lm_eval_ent_crusher"
# BASE_RUN_NAME = f"lm_eval_ent_crusher_tm"
BASE_RUN_NAME = f"lm_eval_unique_masks"

# WANDB_TAGS = None
# WANDB_TAGS = ["debug","stepwise"]
# WANDB_TAGS = ["daint","stepwise"]
# WANDB_TAGS = ["daint","stepwise","new_tasks"]
# WANDB_TAGS = ["daint","stepwise","ablation"]
# WANDB_TAGS = ["daint","stepwise","big_models"]
# WANDB_TAGS = ["daint","stepwise","pre_arxiv"]
# WANDB_TAGS = ["daint","stepwise","pre_arxiv_extra"]
WANDB_TAGS = ["daint","stepwise","ent_crusher"]

#### LIT to HF CONVERT controls ####
DO_CONVERT=True
# DO_CONVERT=False
SKIP_CONVERT_IF_EXISTS=True
# SKIP_CONVERT_IF_EXISTS=False

# fallback method to prepare dirs for local loading
# LOCAL_HF_PREP=False
LOCAL_HF_PREP=True

# this is a temp patch for old runs that did not propagate all tokenizer files, esp chat templates
PROP_TOK_FILES=True
# PROP_TOK_FILES=False

#### HUB PUSH controls ####
# DO_PUSH=True
DO_PUSH=False
UPDATE_EXISTING_HUB_REPOS = True
# UPDATE_EXISTING_HUB_REPOS = False

#### LM EVAL controls ####
DO_EVAL=True
# DO_EVAL=False

# USE_LOCAL_MODEL=False
USE_LOCAL_MODEL=True

# DO_LMEVAL_INTEG_WANDB = True
DO_LMEVAL_INTEG_WANDB = False

#### Final WandB push controls ####

# DO_MANUAL_WANDB_PUSH=False
DO_MANUAL_WANDB_PUSH=True

MANUAL_WANDB_PUSH_DRY_RUN=False
# MANUAL_WANDB_PUSH_DRY_RUN=True

#### Other controls ####

PUSH_DTYPE="bfloat16"
# PUSH_ORG="tomg-group-umd"
PUSH_ORG="jwkirchenbauer"
# PUSH_PRIVATE=True
PUSH_PRIVATE=False

# WANDB_PROJECT="singleshot"
WANDB_PROJECT="singleshot-evals"


NODES = 1
TPN = 1
GPN = 4

# BASE_EVAL_COMMAND="lm-eval run"
BASE_EVAL_COMMAND="accelerate launch --config_file config_hub/lm_eval/accelerate_config_1N.yaml -m lm_eval run"

# Cfgs

# raw model/run dirs
exp_list = [
    # new style of masking and templating
    # # New L3 Magpie trained on chat fmt metamath
    # [
    #     # None, # push basename, mostly unused
    #     "L3-1-8B-Magpie-MTP", # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_ift_mask_fix/daint_prod_ift_mask_fix_1N4n_9d30cad5",
    #     f"{BASE_TOKS_DIR}/extended/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384",
    #     128259,
    #     "128009+128001",
    #     "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
    #     "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
    #     True, # add chat args
    #     # [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 100160]],
    #     [f"step-{str(s).zfill(8)}" for s in [100160]],
    # ],
    # # New Qwen3-8B-Base trained on bos only metamath
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_ift_mask_fix/daint_prod_ift_mask_fix_1N4n_3c08f241",
    #     f"{BASE_TOKS_DIR}/extended/Qwen/Qwen3-8B-Base-MTP",
    #     151669,
    #     "151645+151643",
    #     "litgpt.transformers_local.qwen3.configuration_qwen3.Qwen3Config",
    #     "litgpt.transformers_local.qwen3.modeling_qwen3.Qwen3ForCausalLM",
    #     True, # add chat args
    #     [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 100160]],
    # ],
    # # New Qwen3-8B trained on chat fmt metamath
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_ift_mask_fix/daint_prod_ift_mask_fix_1N4n_e5e98ecf",
    #     f"{BASE_TOKS_DIR}/extended/Qwen/Qwen3-8B-MTP",
    #     151669,
    #     "151645+151643",
    #     "litgpt.transformers_local.qwen3.configuration_qwen3.Qwen3Config",
    #     "litgpt.transformers_local.qwen3.modeling_qwen3.Qwen3ForCausalLM",
    #     # True, # add chat args
    #     False, # add chat args # NOTE try /nothink "no think" or prefill the think region something else add max leng 2x
    #     [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 100160]],
    # ],
    # # New Qwen3-8B trained on bos only metamath
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_ift_mask_fix/daint_prod_ift_mask_fix_1N4n_b3436b89",
    #     f"{BASE_TOKS_DIR}/extended/Qwen/Qwen3-8B-MTP",
    #     151669,
    #     "151645+151643",
    #     "litgpt.transformers_local.qwen3.configuration_qwen3.Qwen3Config",
    #     "litgpt.transformers_local.qwen3.modeling_qwen3.Qwen3ForCausalLM",
    #     # True, # add chat args
    #     False, # add chat args
    #     [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 100160]],
    # ],
    # # New L3 Openmath trained on chat fmt metamath
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_ift_mask_fix/daint_prod_ift_mask_fix_1N4n_141a737e",
    #     f"{BASE_TOKS_DIR}/extended/nvidia/OpenMath2-Llama3.1-8B-MTPV128384",
    #     128256,
    #     "128009+128001",
    #     "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
    #     "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
    #     True, # add chat args
    #     [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 100160]],
    # ],
    # # New L3 Tulu trained on chat fmt metamath
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_ift_mask_fix/daint_prod_ift_mask_fix_1N4n_f7e24b2c",
    #     f"{BASE_TOKS_DIR}/extended/allenai/Llama-3.1-Tulu-3-8B-SFT-MTPV128384",
    #     128257,
    #     128001,
    #     "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
    #     "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
    #     True, # add chat args
    #     [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 100160]],
    # ],
    # # Reproduction of old L3 base trained on bos only metamath with old mask style
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_ift_mask_fix/daint_prod_ift_mask_fix_1N4n_450c1bb8",
    #     f"{BASE_TOKS_DIR}/meta-llama/Meta-Llama-3.1-8B",
    #     128002,
    #     128001,
    #     "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
    #     "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
    #     # [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 100160]],
    #     [f"step-{str(s).zfill(8)}" for s in [100160]],
    # ],
    # Regression check for L3 base trained on bos only metamath with new mask style
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_ift_mask_fix/daint_prod_ift_mask_fix_1N4n_a8229c95",
    #     f"{BASE_TOKS_DIR}/extended/meta-llama/Meta-Llama-3.1-8B-MTPV128384",
    #     128256,
    #     128001,
    #     "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
    #     "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
    #     False, # add chat args
    #     [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 100160]],
    # ],
    # # New L3 Magpie trained on chat fmt magpie
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_ift_magpie/daint_prod_ift_magpie_1N4n_44004b35",
    #     f"{BASE_TOKS_DIR}/extended/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384",
    #     128259,
    #     "128009+128001",
    #     "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
    #     "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
    #     True, # add chat args
    #     [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 125881]],
    # ],
    # # New L3 Magpie trained on bos only magpie
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_ift_magpie/daint_prod_ift_magpie_1N4n_e379bd46",
    #     f"{BASE_TOKS_DIR}/extended/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384",
    #     128259,
    #     "128009+128001",
    #     "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
    #     "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
    #     False, # add chat args
    #     [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 125881]],
    # ],
    # # New Qwen3-4B-Instruct-2507 trained on bos only metamath, max k 16
    # [
    #     # None, # push basename, mostly unused
    #     "Qwen3-4B-Inst-2507-MTP", # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_ift_q3-4b/daint_prod_ift_q3-4b_1N4n_16cdce0f",
    #     f"{BASE_TOKS_DIR}/extended/Qwen/Qwen3-4B-Instruct-2507-MTP",
    #     151669,
    #     "151645+151643",
    #     "litgpt.transformers_local.qwen3.configuration_qwen3.Qwen3Config",
    #     "litgpt.transformers_local.qwen3.modeling_qwen3.Qwen3ForCausalLM",
    #     # True, # add chat args
    #     False, # add chat args
    #     # [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 100160]],
    #     [f"step-{str(s).zfill(8)}" for s in [100160]],
    #     # [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000]],
    # ],
    # # New Qwen3-4B-Instruct-2507 trained on chat fmt metamath, max k 16
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_ift_q3-4b/daint_prod_ift_q3-4b_1N4n_d2aa1bac",
    #     f"{BASE_TOKS_DIR}/extended/Qwen/Qwen3-4B-Instruct-2507-MTP",
    #     151669,
    #     "151645+151643",
    #     "litgpt.transformers_local.qwen3.configuration_qwen3.Qwen3Config",
    #     "litgpt.transformers_local.qwen3.modeling_qwen3.Qwen3ForCausalLM",
    #     True, # add chat args
    #     # False, # add chat args
    #     [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 100160]],
    #     # [f"step-{str(s).zfill(8)}" for s in [100160]],
    #     # [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000]],
    # ],
    # # New Qwen3-4B-Instruct-2507 trained on bos only metamath, max k 8
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_ift_q3-4b/daint_prod_ift_q3-4b_1N4n_ccfee559",
    #     f"{BASE_TOKS_DIR}/extended/Qwen/Qwen3-4B-Instruct-2507-MTP",
    #     151669,
    #     "151645+151643",
    #     "litgpt.transformers_local.qwen3.configuration_qwen3.Qwen3Config",
    #     "litgpt.transformers_local.qwen3.modeling_qwen3.Qwen3ForCausalLM",
    #     # True, # add chat args
    #     False, # add chat args
    #     [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 103476]],
    # ],
    # # New Qwen3-4B-Instruct-2507 trained on chat fmt metamath, max k 8
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_ift_q3-4b/daint_prod_ift_q3-4b_1N4n_426eff81",
    #     f"{BASE_TOKS_DIR}/extended/Qwen/Qwen3-4B-Instruct-2507-MTP",
    #     151669,
    #     "151645+151643",
    #     "litgpt.transformers_local.qwen3.configuration_qwen3.Qwen3Config",
    #     "litgpt.transformers_local.qwen3.modeling_qwen3.Qwen3ForCausalLM",
    #     True, # add chat args
    #     # False, # add chat args
    #     [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 103476]],
    # ],
    # Final ablation series
    # All are L3 Magpie trained on chat fmt metamath
    # # duplicate of flagship, for sanity
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_suprv_abl/daint_prod_suprv_abl_1N4n_9d30cad5",
    #     f"{BASE_TOKS_DIR}/extended/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384",
    #     128259,
    #     "128009+128001",
    #     "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
    #     "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
    #     True, # add chat args
    #     [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 100160]],
    # ],
    # # beta 2
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_suprv_abl/daint_prod_suprv_abl_1N4n_32112117",
    #     f"{BASE_TOKS_DIR}/extended/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384",
    #     128259,
    #     "128009+128001",
    #     "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
    #     "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
    #     True, # add chat args
    #     [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 100160]],
    # ],
    # # gt supervision
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_suprv_abl/daint_prod_suprv_abl_1N4n_82938517",
    #     f"{BASE_TOKS_DIR}/extended/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384",
    #     128259,
    #     "128009+128001",
    #     "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
    #     "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
    #     True, # add chat args
    #     [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 100160]],
    # ],
    # # bda
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_suprv_abl/daint_prod_suprv_abl_1N4n_b258ae13",
    #     f"{BASE_TOKS_DIR}/extended/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384",
    #     128259,
    #     "128009+128001",
    #     "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
    #     "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
    #     True, # add chat args
    #     [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 100160]],
    # ],
    # # beta1
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_suprv_abl/daint_prod_suprv_abl_1N4n_d2e5a6cb",
    #     f"{BASE_TOKS_DIR}/extended/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384",
    #     128259,
    #     "128009+128001",
    #     "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
    #     "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
    #     True, # add chat args
    #     [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 100160]],
    # ],
    # # soft teacher
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_suprv_abl/daint_prod_suprv_abl_1N4n_d30c404e",
    #     f"{BASE_TOKS_DIR}/extended/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384",
    #     128259,
    #     "128009+128001",
    #     "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
    #     "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
    #     True, # add chat args
    #     [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 100160]],
    # ],
    # # fixed k
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_suprv_abl/daint_prod_suprv_abl_1N4n_e12fd460",
    #     f"{BASE_TOKS_DIR}/extended/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384",
    #     128259,
    #     "128009+128001",
    #     "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
    #     "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
    #     True, # add chat args
    #     [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 110000]],
    # ],
    # # prefix supervision
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_suprv_abl/daint_prod_suprv_abl_1N4n_fcdeefba",
    #     f"{BASE_TOKS_DIR}/extended/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384",
    #     128259,
    #     "128009+128001",
    #     "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
    #     "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
    #     True, # add chat args
    #     # [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 100160]],
    #     # [f"step-{str(s).zfill(8)}" for s in [100000]],
    #     [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 100000]],
    # ],
    # # fixed k but at 9
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_pre_arxiv_extra/daint_prod_pre_arxiv_extra_1N4n_8aa66673",
    #     f"{BASE_TOKS_DIR}/extended/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384",
    #     128259,
    #     "128009+128001",
    #     "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
    #     "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
    #     True, # add chat args
    #     [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 100000, 122070]],
    # ],
    # # New Qwen3-14B-Base trained on bos only metamath
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_ift_q3-14b/daint_prod_ift_q3-14b_8N32n_90634667",
    #     f"{BASE_TOKS_DIR}/extended/Qwen/Qwen3-14B-Base-MTP",
    #     151669,
    #     "151645+151643",
    #     "litgpt.transformers_local.qwen3.configuration_qwen3.Qwen3Config",
    #     "litgpt.transformers_local.qwen3.modeling_qwen3.Qwen3ForCausalLM",
    #     # True, # add chat args
    #     False, # add chat args
    #     # [f"step-{str(s).zfill(8)}" for s in [0]],
    #     # [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000]],
    #     # [f"step-{str(s).zfill(8)}" for s in [90000]],
    #     [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000, 100160]],
    # ],
    # # ent crusher cont'd
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_ent_crusher/daint_prod_ent_crusher_1N4n_0eb278f3",
    #     f"{BASE_TOKS_DIR}/extended/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384",
    #     128259,
    #     "128009+128001",
    #     "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
    #     "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
    #     True, # add chat args
    #     ["latest"],
    # ],
    # # ent crusher scratch
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_ent_crusher/daint_prod_ent_crusher_1N4n_44ec004b",
    #     f"{BASE_TOKS_DIR}/extended/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384",
    #     128259,
    #     "128009+128001",
    #     "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
    #     "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
    #     True, # add chat args
    #     ["latest"],
    # ],
    # # ent crusher 1B at k max = 16
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_ent_crusher/daint_prod_ent_crusher_1N4n_fb6c7d50",
    #     f"{BASE_TOKS_DIR}/extended/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384",
    #     128259,
    #     "128009+128001",
    #     "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
    #     "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
    #     True, # add chat args
    #     # ["latest"],
    #     [f"step-{str(s).zfill(8)}" for s in [0, 5000, 10000, 20000, 30000, 50080]],
    # ],
    # # ent crusher 1B at k max = 5
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_ent_crusher/daint_prod_ent_crusher_1N4n_892eb612",
    #     f"{BASE_TOKS_DIR}/extended/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384",
    #     128259,
    #     "128009+128001",
    #     "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
    #     "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
    #     True, # add chat args
    #     # ["latest"],
    #     [f"step-{str(s).zfill(8)}" for s in [0, 5000, 10000, 20000, 30000, 53879]],
    # ],
    # # TM fixed ent crusher 1B at k max = 16
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_ent_crusher_tm/daint_prod_ent_crusher_tm_1N4n_b90624c4",
    #     f"{BASE_TOKS_DIR}/extended/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384",
    #     128259,
    #     None,
    #     None,
    #     "128009+128001",
    #     "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
    #     "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
    #     True, # add chat args
    #     # ["latest"],
    #     [f"step-{str(s).zfill(8)}" for s in [0, 5000, 10000, 20000, 30000, 50080]],
    # ],
    # # TM fixed ent crusher 1B at k max = 5
    # [
    #     None, # push basename, mostly unused
    #     f"{BASE_OUT_DIR}/daint_prod_ent_crusher_tm/daint_prod_ent_crusher_tm_1N4n_c9c858e5",
    #     f"{BASE_TOKS_DIR}/extended/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384",
    #     128259,
    #     None,
    #     None,
    #     "128009+128001",
    #     "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
    #     "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
    #     True, # add chat args
    #     # ["latest"],
    #     [f"step-{str(s).zfill(8)}" for s in [0, 5000, 10000, 20000, 30000, 53879]],
    # ],
    # unique mask token ids test
    [
        None, # push basename, mostly unused
        f"{BASE_OUT_DIR}/daint_prod_unique_masks/daint_prod_unique_masks_1N4n_25dbd712",
        f"{BASE_TOKS_DIR}/extended/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384",
        None,
        128259,
        128259+15,
        "128009+128001",
        "litgpt.transformers_local.llama.configuration_llama.LlamaConfig",
        "litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM",
        True, # add chat args
        # [f"step-{str(s).zfill(8)}" for s in [0, 10000, 20000, 30000, 50000]],
        [f"step-{str(s).zfill(8)}" for s in [100160]],
    ],
]
# product the inner list with its container list
exp_list = list(chain(*[[exp[:10] + [hp] for hp in exp[10]] for exp in exp_list]))

# evaluation configs
sweep_hparam = [
    # [
    #     "config_hub/lm_eval/gsm8k_baseline.yaml", 
    #     "/capstor/store/cscs/userlab/lp98/jkirchen/singleshot-root/singleshot/checkpoints/meta-llama/Meta-Llama-3.1-8B"
    # ],
    [
        "config_hub/lm_eval/default_mtp.yaml",
        None
    ],
]
exp_list = list(chain(*[[exp + hp for hp in sweep_hparam] for exp in exp_list]))

STATIC_GEN_KWARGS = [
    'do_sample=False',
    'do_mtp=True',
    'include_prompt=True',
    'return_mtp_result_dict=True', # new arg for piping out mtp specific metrics
    # 'max_length=32767',
    # 'max_gen_toks=32767',
]

# tasks and gen_kwargs overrides
# not running indiv helps keep output dirs separate
sweep_hparam = [
    ["gsm8k_cot_retrorec"]+[STATIC_GEN_KWARGS+['\"until=Q:+</s>+<|end_of_text|>+<|eot_id|>+<|endoftext|>+<|im_end|>\"']],
    # ["aime25"]+[STATIC_GEN_KWARGS+['\"until=</s>+<|end_of_text|>+<|eot_id|>+<|endoftext|>+<|im_end|>\"']],
    # ["ifeval"]+[STATIC_GEN_KWARGS+['\"until=</s>+<|end_of_text|>+<|eot_id|>+<|endoftext|>+<|im_end|>\"']],
    # ["cnn_dailymail_abisee"]+[STATIC_GEN_KWARGS+['\"until=</s>+<|end_of_text|>+<|eot_id|>+<|endoftext|>+<|im_end|>\"']],
    # ["bbh_cot_fewshot"]+[STATIC_GEN_KWARGS+['\"until=\\n\\n+Q:+</s>+<|end_of_text|>+<|eot_id|>+<|endoftext|>+<|im_end|>\"']],
    # ["gpqa_main_cot_n_shot"]+[STATIC_GEN_KWARGS+['\"until=</s>+<|end_of_text|>+<|eot_id|>+<|endoftext|>+<|im_end|>\"']],
]
exp_list = list(chain(*[[exp + hp for hp in sweep_hparam] for exp in exp_list]))

# ktoks and strategies values
sweep_hparam = [
    [1, None],
    [2, None],
    [3, None],
    [4, None],
    [5, None],
    # [8, None],
    # [16, None],
    # [16, "conf_adapt+0.9999995"],
    # [16, "conf_adapt+0.999999"],
    # [16, "conf_adapt+0.999995"],
    # [16, "conf_adapt+0.99999"],
    # [16, "conf_adapt+0.99995"],
    # [16, "conf_adapt+0.9999"],
    # [16, "conf_adapt+0.9995"],
    # [16, "conf_adapt+0.999"],
    [16, "conf_adapt+0.995"],
    [16, "conf_adapt+0.99"],
    [16, "conf_adapt+0.98"],
    [16, "conf_adapt+0.96"],
    [16, "conf_adapt+0.97"],
    [16, "conf_adapt+0.95"],
    [16, "conf_adapt+0.9"],
    [16, "conf_adapt+0.87"],
    [16, "conf_adapt+0.85"],
    [16, "conf_adapt+0.80"],
    [16, "conf_adapt+0.75"],
    [16, "conf_adapt+0.70"],
    [16, "conf_adapt+0.65"],
    [16, "conf_adapt+0.6"],

]
exp_list = list(chain(*[[exp + hp for hp in sweep_hparam] for exp in exp_list]))

# dtype
sweep_hparam = [
    "float32", # default for accuracy amongst implementations
]
exp_list = list(chain(*[[exp + [hp] for hp in sweep_hparam] for exp in exp_list]))


# limit rows of lm eval
sweep_hparam = [
    # 10,
    None,
]
exp_list = list(chain(*[[exp + [hp] for hp in sweep_hparam] for exp in exp_list]))

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
        push_basename,
        raw_run_dir,
        orig_tok_dir,
        mask_id,
        min_mask_id,
        max_mask_id,
        eos_id,
        config_class,
        model_class,
        add_chat_args,
        step,
        cfg,
        model_name_path,
        tasks,
        gen_kwargs,
        k_toks,
        strategy,
        dtype,
        limit,
    ) = exp

    custom_invocation = ""

    if push_basename is None:
        if raw_run_dir is not None and step is not None:
            push_name = f"{raw_run_dir.split('/')[-1]}_{step}"
    else:
        push_name = f"{push_basename}_{step}"

    if model_name_path is None:
        model_name_path = f"{PUSH_ORG}/{push_name}"

    assert model_name_path is not None, "model_name_path cannot be None"    

    if DO_CONVERT:
        if raw_run_dir is not None:
            if PROP_TOK_FILES and orig_tok_dir is not None:
                # The quotes around 'EOF' are the secret sauce here
                custom_invocation += f"""\
python << EOF
from litgpt.utils import copy_config_files
from pathlib import Path
copy_config_files(Path(\"{orig_tok_dir}\"), Path(\"{raw_run_dir}/{step}\"))
EOF

"""
            if LOCAL_HF_PREP:
                assert config_class is not None, "config_class cannot be None"
                assert model_class is not None, "model_class cannot be None"
            convert_cmd = f"litgpt convert_from_litgpt --checkpoint_dir={raw_run_dir}/{step} --output_dir={raw_run_dir}/{step} --output_name=pytorch_model.bin --skip_if_exists={SKIP_CONVERT_IF_EXISTS} {f'--config_class_path={config_class} --model_class_path={model_class}' if LOCAL_HF_PREP else ''}"
            custom_invocation += f"{convert_cmd}\n\n"         

    if DO_PUSH:
        if model_class is not None:
            assert push_name is not None, "push_name cannot be None"
            eval_cmd = f"litgpt push_to_hub --model_path={raw_run_dir}/{step} --model_class_path={model_class} --org={PUSH_ORG} --private={PUSH_PRIVATE} --model_name={push_name} --precision={PUSH_DTYPE} --dry_run=False --update_existing={UPDATE_EXISTING_HUB_REPOS}"
            custom_invocation += f"{eval_cmd}\n\n"

    # for eval, we always set some stuff even if not running
    cli_args = ""

    cfg_str = cfg.split("/")[-1].replace(".yaml","")
    cli_args += f" --config {cfg}"

    model_str = model_name_path.split("/")[-1].replace(".","-")
    # unless we are evaluating locally
    if USE_LOCAL_MODEL:
        model_name_path = raw_run_dir + "/" + step

    cli_args += f" --model_args pretrained={model_name_path},dtype={dtype}"
    task_str = tasks.replace(",","-")
    cli_args += f" --tasks {tasks}"

    if limit is not None:
        cli_args += f" --limit {limit}"
        cfg_str += f"_lim{limit}"
    
    if add_chat_args:
        cli_args += f" --apply_chat_template --fewshot_as_multiturn"
        cfg_str += f"_chatfmt"
    
    assert gen_kwargs is not None, "gen_kwargs cannot be None"
    assert k_toks is not None, "k_toks cannot be None"
    assert (mask_id is not None) or (min_mask_id is not None), "mask_id cannot be None"
    assert eos_id is not None, "eos_id cannot be None"
    
    cli_args += f" --gen_kwargs {','.join(gen_kwargs)},mask_id={mask_id},eos_id={eos_id},k_toks={k_toks}"

    if min_mask_id is not None and max_mask_id is not None:
        cli_args += f",min_mask_id={min_mask_id},max_mask_id={max_mask_id}"

    if strategy is not None:
        cli_args += f",strategy={strategy}"
    
    cfg_str += f"_ktoks{k_toks}_strat{strategy.replace('+','@') if strategy is not None else 'none'}"

    # mod more things
    # ...

    # join to a unique run name for the experiment
    run_name = (
        f"{BASE_RUN_NAME}_{task_str}_{cfg_str}_{model_str}"
    )
    unique_run_names.add(run_name)

    # build the final call, tacking on a few run name based things
    # we need to strip the step from within the run_name so that we can plot over it on wandb
    wandb_run_name = f"{model_str.replace(f'_{step}','')}"

    # put together the actual "train.py" command
    launch_out_dir = f"{BASE_OUT_DIR}/{BASE_RUN_NAME}"
    full_out_dir = f"{launch_out_dir}/{run_name}"

    cli_args += f" --output_path {full_out_dir}"
    if DO_LMEVAL_INTEG_WANDB:
        cli_args += f" --wandb_args name={wandb_run_name},project={WANDB_PROJECT}"
        if step != "latest":
            # extract step int
            step_int = int(step.split("-")[-1])
            cli_args += f",step={step_int}"
        if WANDB_TAGS is not None:
            cli_args += f",tags={'+'.join(WANDB_TAGS)}"
    
    if DO_EVAL:
        custom_invocation += f"{BASE_EVAL_COMMAND}{cli_args}\n"
    
    if DO_MANUAL_WANDB_PUSH:
        custom_invocation += f"""
python -u litgpt/scripts/push_lmeval_metrics_to_wandb.py \
--run_dir {full_out_dir} \
--hf_tokenizer_path={raw_run_dir}/{step} \
--wandb_args name={wandb_run_name},project={WANDB_PROJECT}\
{f",step={int(step.split('-')[-1])}" if step != "latest" else ""}\
{f",tags={'+'.join(WANDB_TAGS)}+manual_pusher" if WANDB_TAGS is not None else ",tags=manual_pusher"} \
--dry_run={MANUAL_WANDB_PUSH_DRY_RUN} \
"""
    
    # make the complete launcher command
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
        --repetitions={REPETITIONS}{f' --dependency={DEPENDENCY}' if DEPENDENCY is not None else ''} \
        --minutes={TIME_LIMIT} \
        --nodes={NODES} \
        --ntasks_per_node={TPN} \
        --gpus_per_node={GPN} \
        --run_name={run_name} \
        --custom_invocation='''{custom_invocation}''' \
        --pass_run_name=False \
        {'--dryrun' if WRITE_ONLY else ''}
    """

    total_launches += 1
    tot_incl_repetitions += REPETITIONS    
    if not LIST_CFGS:
        os.system(command)
    else:
        print(run_name)
        # print(command)

print(f"Total launches (incl repetitions): {total_launches} ({tot_incl_repetitions})")
assert total_launches == len(unique_run_names), f"{total_launches} != {len(unique_run_names)} Jobs might be overwriting eachother, plz check that all hparams factor into naming."
