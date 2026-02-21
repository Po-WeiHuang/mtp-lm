initial testing

```
srun --pty --job-name=test_litgpt --partition=tron --qos=high -N1 --ntasks-per-node=4 --mem=50G --cpus-per-task=2 --gres=gpu:rtxa4000:4 --time=4:00:00 bash

ml cuda/12.9.1
conda_activate $WRKSPC/nexus_28_stable_litgpt


litgpt download EleutherAI/pythia-14m
litgpt download QwenQwen3-0.6B
litgpt download Qwen/Qwen3-0.6B-Base
litgpt download Qwen/Qwen2.5-0.5B
litgpt download Qwen/Qwen2.5-1.5B
litgpt download Qwen/Qwen2.5-7B
litgpt download meta-llama/Meta-Llama-3-8B
litgpt download meta-llama/Meta-Llama-3.1-8B
litgpt download meta-llama/Llama-3.2-1B

litgpt download Qwen/Qwen3-8B-Base
litgpt download Qwen/Qwen3-8B

litgpt download Qwen/Qwen3-4B-Base && \
litgpt download Qwen/Qwen3-4B && \
litgpt download Qwen/Qwen3-4B-Instruct-2507 && \
litgpt download Qwen/Qwen3-4B-Thinking-2507
litgpt download Qwen/Qwen3-30B-A3B-Instruct-2507 && \
litgpt download qywu/Qwen3-32B-Instruct && \

litgpt download Qwen/Qwen3-14B-Base

litgpt download Qwen/Qwen3-1.7B-Base && \
litgpt download Qwen/Qwen3-1.7B && \
litgpt download Qwen/Qwen3-0.6B-Base && \
litgpt download Qwen/Qwen3-0.6B && \

# 1 gpu (a4000)
srun -N1 -n1 --ntasks-per-node=1 --gpus-per-task=1 --mem=50G --cpus-per-task=2 --unbuffered \
    litgpt pretrain \
    --config=config_hub/pretrain/debug.yaml \
    --model_name=Qwen2.5-0.5B \
    --tokenizer_dir=checkpoints/Qwen/Qwen2.5-0.5B \
    --train.micro_batch_size=1 \
    --train.global_batch_size=1 \
# 4 gpus (a4000)
srun -N1 -n4 --ntasks-per-node=4 --mem=50G --cpus-per-task=2 --unbuffered \
    python -u litgpt/pretrain.py \
    --config=config_hub/pretrain/debug.yaml \
    --model_name=Qwen2.5-0.5B \
    --tokenizer_dir=checkpoints/Qwen/Qwen2.5-0.5B \
    --train.micro_batch_size=1 \
    --train.global_batch_size=4 \
# 2  (h100)
srun -N1 -n2 --ntasks-per-node=2 --mem=128G --cpus-per-task=2 --unbuffered \
    python -u litgpt/pretrain.py \
    --config=config_hub/pretrain/debug.yaml \
    --model_name=Qwen2.5-1.5B \
    --tokenizer_dir=checkpoints/Qwen/Qwen2.5-1.5B \
    --train.micro_batch_size=4 \
    --train.global_batch_size=8 \
srun -N1 -n2 --ntasks-per-node=2 --mem=128G --cpus-per-task=2 --unbuffered \
    python -u litgpt/pretrain.py \
    --config=config_hub/pretrain/debug.yaml \
    --model_name=Qwen2.5-7B \
    --tokenizer_dir=checkpoints/Qwen/Qwen2.5-7B \
    --train.micro_batch_size=1 \
    --train.global_batch_size=2 \
    --train.max_seq_length=2048 \

# 1  (h100)
srun -N1 -n1 --ntasks-per-node=1 --gpus-per-task=1 --mem=128G --cpus-per-task=2 --unbuffered \
    python -u litgpt/pretrain.py \
    --config=config_hub/pretrain/debug.yaml \
    --model_name=Qwen2.5-7B \
    --tokenizer_dir=checkpoints/Qwen/Qwen2.5-7B \
    --train.micro_batch_size=1 \
    --train.global_batch_size=1 \
    --train.max_seq_length=2048 \


# 4  (h100)
srun -N1 -n4 --ntasks-per-node=4 --mem=128G --cpus-per-task=2 --unbuffered \
    python -u litgpt/pretrain.py \
    --config=config_hub/pretrain/debug.yaml \
    --model_name=Qwen2.5-7B \
    --tokenizer_dir=checkpoints/Qwen/Qwen2.5-7B \
    --train.micro_batch_size=2 \
    --train.global_batch_size=8 \
    --train.max_seq_length=4096 \

8192 toks / 0.770 s/step = 10.6k tps/gpu



# 4  (h100)
srun -N1 -n4 --ntasks-per-node=4 --mem=128G --cpus-per-task=2 --unbuffered \
    python -u litgpt/pretrain.py \
    --config=config_hub/pretrain/debug.yaml \
    --model_name=Llama-3-8B \
    --tokenizer_dir=checkpoints/meta-llama/Meta-Llama-3-8B \
    --train.micro_batch_size=2 \
    --train.global_batch_size=8 \
    --train.max_seq_length=4096 \

8192 toks / 0.830 s/step = 9.8k tps/gpu
at alloc/reserv 62gb/75gb of 80 so 77%/93%

note that early it does it 10.1k tps/gpu 
```

note that some cache related env vars are expected to be managed as you run out of space fast
```
# torch compile
export TRITON_CACHE_DIR="/cmlscratch/jkirchen/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="/cmlscratch/jkirchen/.cache/inductor"

# general temp
export TMPDIR="/cmlscratch/jkirchen/.cache/tmp"
export TEMP=$TMPDIR
export TMP=$TMPDIR
```

issues:
- make sure the slurm procs and gpus launching is correct, its diff than usual
- https://github.com/Lightning-AI/litData/issues/482 suggest issue with the pretrain tutorial litdata usage
- Qwen3-0.6B-Base threw a cuda error on various bsz. Assumed it was vocab/tokenization, but did pass the correct tokenizer dir
- trying Qwen2.5-0.5B ... 
- okay nevermind cli args were wack nd wrong model was being loaded
- a4000s are so damn slow I forgot lol
- on 2xH100 we get 30k tps/gpu for the Qwen 1.5B, and  BAD?   for the 7B
- okay bf16 gives more reasonable thorughput on 7B like 8k. this could explain the original gap between lingua and litgptdev tbh, but will take more testing
against lingua in parallel



# setup on Tuo, main driver is the normal style

launch_exps_tuo.py


# set up generation utils

python -u litgpt/generate/base.py checkpoints/meta-llama/Meta-Llama-3.1-8B \
  --prompt "Hello, my name is"

inline util is in pretrain file now and calls the base generate routine in a distributed manner


# datasets

litgpt/tinystories.py module updated in process of dialing in token count and epoch logging, but def need
to move on from this soon

added the litgpt/fineweb.py module, but again reminded this has got to go, it's so slow



# exporting our SSLM checkpoints

we export in place to give access to all the other files we need

Daint
export RUN_DIR=/capstor/store/cscs/userlab/lp98/jkirchen/singleshot-root/singleshot/outputs/daint_prod_q4/daint_prod_q4_32N128n_12cacdc9 && \
Tuo
export RUN_DIR=/p/vast1/kirchenb/singleshot-root/singleshot/outputs/debug_metamath_full_rand_k2-8_ex_valk/debug_metamath_full_rand_k2-8_ex_valk_ss_L3-8B_pqds-cfgfc55c3da_toks11.6B_ss-cfg2059c710_mb32-wb128-sl160_lr-2000steps-const1e-05tonull_1N4n && \
export RUN_DIR=/p/vast1/kirchenb/daint/daint_prod_q4/daint_prod_q4_32N128n_e3c83f91 && \
export RUN_DIR=/p/vast1/kirchenb/daint/daint_prod_q4/daint_prod_q4_32N128n_12cacdc9 && \
export RUN_DIR=/p/vast1/kirchenb/daint/daint_prod_q4/daint_prod_q4_128N512n_fd7261ea && \
export STEP_DIR=latest && \
litgpt convert_from_litgpt \
--checkpoint_dir=$RUN_DIR/$STEP_DIR \
--output_dir=$RUN_DIR/$STEP_DIR \
--output_name=pytorch_model.bin && \

# pushing sslm models to hub

export MODEL_PATH=/p/vast1/kirchenb/singleshot-root/singleshot/outputs/debug_metamath_full_rand_k2-8_ex_valk/debug_metamath_full_rand_k2-8_ex_valk_ss_L3-8B_pqds-cfgfc55c3da_toks11.6B_ss-cfg2059c710_mb32-wb128-sl160_lr-2000steps-const1e-05tonull_1N4n/latest && \
export PUSH_NAME=debug_metamath_full_rand_k2-8_ex_valk_baseline_latest && \

export MODEL_PATH=/p/vast1/kirchenb/daint/daint_prod_q4/daint_prod_q4_32N128n_12cacdc9/latest && \
export PUSH_NAME=daint_prod_q4_32N128n_12cacdc9_latest && \
litgpt push_to_hub \
    --model_path=$MODEL_PATH \
    --model_class_path=litgpt.transformers.llama.modeling_llama.LlamaForCausalLM \
    --model_name=$PUSH_NAME \
    --dry_run=False && \
export MODEL_PATH=/p/vast1/kirchenb/daint/daint_prod_q4/daint_prod_q4_32N128n_e3c83f91/latest && \
export PUSH_NAME=daint_prod_q4_32N128n_e3c83f91_latest && \
litgpt push_to_hub \
    --model_path=$MODEL_PATH \
    --model_class_path=litgpt.transformers.llama.modeling_llama.LlamaForCausalLM \
    --model_name=$PUSH_NAME \
    --dry_run=False && \
export MODEL_PATH=/p/vast1/kirchenb/daint/daint_prod_q4/daint_prod_q4_128N512n_fd7261ea/latest && \
export PUSH_NAME=daint_prod_q4_128N512n_fd7261ea && \
litgpt push_to_hub \
    --model_path=$MODEL_PATH \
    --model_class_path=litgpt.transformers.llama.modeling_llama.LlamaForCausalLM \
    --model_name=$PUSH_NAME \
    --dry_run=False && \

# prototyping a mtp generate method within litgpt

litgpt generate \

export MODEL_PATH=/capstor/store/cscs/userlab/lp98/jkirchen/singleshot-root/singleshot/checkpoints/meta-llama/Meta-Llama-3.1-8B && \
export MODEL_PATH=/capstor/store/cscs/userlab/lp98/jkirchen/singleshot-root/singleshot/outputs/daint_prod_q4/daint_prod_q4_32N128n_12cacdc9/step-00010000 && \
export MODEL_PATH=/capstor/store/cscs/userlab/lp98/jkirchen/singleshot-root/singleshot/outputs/daint_prod_q4/daint_prod_q4_32N128n_12cacdc9/step-00156603 && \
export MODEL_PATH=/capstor/store/cscs/userlab/lp98/jkirchen/singleshot-root/singleshot/outputs/daint_prod_q4/daint_prod_q4_32N128n_12cacdc9/latest && \
export MODEL_PATH=checkpoints/meta-llama/Meta-Llama-3.1-8B && \
python -u litgpt/generate/base_mtp.py \
    --checkpoint_dir=$MODEL_PATH \
    --max_new_tokens=32 \
    --num_samples=3 \
    --precision=bf16-true \
    --compile=False \
    --k_toks=16 \
    --k_toks=1 \

make this useable in a notebook without reloading the model ...

# attempting to run the sslm conversion inside lm eval harness

lm-eval run --config eval_config.yaml --tasks gsm8k_cot --limit 10


export RUN_NAME="lmeval-gsm8k-cot-baseline" && \
export MODEL_ARGS="--model_args pretrained=/p/vast1/kirchenb/singleshot-root/singleshot/checkpoints/meta-llama/Meta-Llama-3.1-8B,dtype=float32" && \
export CONFIG="--config config_hub/lm_eval/debug_eval_baseline.yaml" && \
export TASKS="--tasks=gsm8k_cot" && \
python /p/vast1/$USER/llnl-tools/launch_tuo.py \
    --output_dir=/p/vast1/kirchenb/singleshot-root/singleshot/outputs/lm_eval \
    --wandb_offline=True \
    --rocm_version=6.4.2 \
    --rccl_installdir=/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.4.1/install/lib \
    --rccl_cfg=rdzv-lbann \
    --qos=pdebug \
    --bank=effml \
    --minutes=10 \
    --nodes=1 \
    --tasks_per_node=1 \
    --gpus_per_node=4 \
    --run_name=$RUN_NAME \
    --custom_invocation="lm-eval run $CONFIG $MODEL_ARGS $TASKS" \
    --pass_run_name=False && \
export RUN_NAME="lmeval-gsm8k-cot-mtp1" && \
export MODEL_ARGS="--model_args pretrained=tomg-group-umd/debug_metamath_full_rand_k2-8_ex_valk_baseline_latest,dtype=float32" && \
export CONFIG="--config config_hub/lm_eval/debug_eval_mtp1.yaml" && \
export TASKS="--tasks=gsm8k_cot" && \
python /p/vast1/$USER/llnl-tools/launch_tuo.py \
    --output_dir=/p/vast1/kirchenb/singleshot-root/singleshot/outputs/lm_eval \
    --wandb_offline=True \
    --rocm_version=6.4.2 \
    --rccl_installdir=/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.4.1/install/lib \
    --rccl_cfg=rdzv-lbann \
    --qos=pdebug \
    --bank=effml \
    --minutes=10 \
    --nodes=1 \
    --tasks_per_node=1 \
    --gpus_per_node=4 \
    --run_name=$RUN_NAME \
    --custom_invocation="lm-eval run $CONFIG $MODEL_ARGS $TASKS" \
    --pass_run_name=False && \
export RUN_NAME="lmeval-gsm8k-cot-mtp3" && \
export MODEL_ARGS="--model_args pretrained=tomg-group-umd/debug_metamath_full_rand_k2-8_ex_valk_baseline_latest,dtype=float32" && \
export CONFIG="--config config_hub/lm_eval/debug_eval_mtp3.yaml" && \
export TASKS="--tasks=gsm8k_cot" && \
python /p/vast1/$USER/llnl-tools/launch_tuo.py \
    --output_dir=/p/vast1/kirchenb/singleshot-root/singleshot/outputs/lm_eval \
    --wandb_offline=True \
    --rocm_version=6.4.2 \
    --rccl_installdir=/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.4.1/install/lib \
    --rccl_cfg=rdzv-lbann \
    --qos=pdebug \
    --bank=effml \
    --minutes=10 \
    --nodes=1 \
    --tasks_per_node=1 \
    --gpus_per_node=4 \
    --run_name=$RUN_NAME \
    --custom_invocation="lm-eval run $CONFIG $MODEL_ARGS $TASKS" \
    --pass_run_name=False && \
export RUN_NAME="lmeval-gsm8k-cot-mtp8" && \
export MODEL_ARGS="--model_args pretrained=tomg-group-umd/debug_metamath_full_rand_k2-8_ex_valk_baseline_latest,dtype=float32" && \
export CONFIG="--config config_hub/lm_eval/debug_eval_mtp8.yaml" && \
export TASKS="--tasks=gsm8k_cot" && \
python /p/vast1/$USER/llnl-tools/launch_tuo.py \
    --output_dir=/p/vast1/kirchenb/singleshot-root/singleshot/outputs/lm_eval \
    --wandb_offline=True \
    --rocm_version=6.4.2 \
    --rccl_installdir=/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.4.1/install/lib \
    --rccl_cfg=rdzv-lbann \
    --qos=pdebug \
    --bank=effml \
    --minutes=10 \
    --nodes=1 \
    --tasks_per_node=1 \
    --gpus_per_node=4 \
    --run_name=$RUN_NAME \
    --custom_invocation="lm-eval run $CONFIG $MODEL_ARGS $TASKS" \
    --pass_run_name=False
    





export RUN_NAME="lmeval-retrorec-baseline" && \
export MODEL_ARGS="--model_args pretrained=/p/vast1/kirchenb/singleshot-root/singleshot/checkpoints/meta-llama/Meta-Llama-3.1-8B,dtype=float32" && \
export CONFIG="--config config_hub/lm_eval/debug_eval_baseline.yaml" && \
export TASKS="--tasks=gsm8k_cot_retrorec" && \
python /p/vast1/$USER/llnl-tools/launch_tuo.py \
    --output_dir=/p/vast1/kirchenb/singleshot-root/singleshot/outputs/lm_eval \
    --wandb_offline=True \
    --rocm_version=6.4.2 \
    --rccl_installdir=/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.4.1/install/lib \
    --rccl_cfg=rdzv-lbann \
    --qos=pdebug \
    --bank=effml \
    --minutes=10 \
    --nodes=1 \
    --tasks_per_node=1 \
    --gpus_per_node=4 \
    --run_name=$RUN_NAME \
    --custom_invocation="lm-eval run $CONFIG $MODEL_ARGS $TASKS" \
    --pass_run_name=False && \
export RUN_NAME="lmeval-retrorec-mtp1" && \
export MODEL_ARGS="--model_args pretrained=tomg-group-umd/debug_metamath_full_rand_k2-8_ex_valk_baseline_latest,dtype=float32" && \
export CONFIG="--config config_hub/lm_eval/debug_eval_mtp1.yaml" && \
export TASKS="--tasks=gsm8k_cot_retrorec" && \
python /p/vast1/$USER/llnl-tools/launch_tuo.py \
    --output_dir=/p/vast1/kirchenb/singleshot-root/singleshot/outputs/lm_eval \
    --wandb_offline=True \
    --rocm_version=6.4.2 \
    --rccl_installdir=/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.4.1/install/lib \
    --rccl_cfg=rdzv-lbann \
    --qos=pdebug \
    --bank=effml \
    --minutes=10 \
    --nodes=1 \
    --tasks_per_node=1 \
    --gpus_per_node=4 \
    --run_name=$RUN_NAME \
    --custom_invocation="lm-eval run $CONFIG $MODEL_ARGS $TASKS" \
    --pass_run_name=False && \
export RUN_NAME="lmeval-retrorec-mtp3" && \
export MODEL_ARGS="--model_args pretrained=tomg-group-umd/debug_metamath_full_rand_k2-8_ex_valk_baseline_latest,dtype=float32" && \
export CONFIG="--config config_hub/lm_eval/debug_eval_mtp3.yaml" && \
export TASKS="--tasks=gsm8k_cot_retrorec" && \
python /p/vast1/$USER/llnl-tools/launch_tuo.py \
    --output_dir=/p/vast1/kirchenb/singleshot-root/singleshot/outputs/lm_eval \
    --wandb_offline=True \
    --rocm_version=6.4.2 \
    --rccl_installdir=/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.4.1/install/lib \
    --rccl_cfg=rdzv-lbann \
    --qos=pdebug \
    --bank=effml \
    --minutes=10 \
    --nodes=1 \
    --tasks_per_node=1 \
    --gpus_per_node=4 \
    --run_name=$RUN_NAME \
    --custom_invocation="lm-eval run $CONFIG $MODEL_ARGS $TASKS" \
    --pass_run_name=False


# different decoding strats

export MODEL_PATH="/p/vast1/kirchenb/singleshot-root/singleshot/outputs/debug_metamath_full_rand_k2-8_ex_valk/debug_metamath_full_rand_k2-8_ex_valk_baseline/latest" && \
python -u litgpt/generate/base_mtp.py \
    --checkpoint_dir=$MODEL_PATH \
    --max_new_tokens=256 \
    --num_samples=1 \
    --precision=32-true \
    --compile=False \
    --k_toks=8


# pulling some non-base models
some have no vocab size change so can be directly loaded
litgpt download meta-llama/Meta-Llama-3.1-8B-Instruct && \
litgpt download nvidia/OpenMath2-Llama3.1-8B --model_name Llama-3.1-8B-Instruct && \

others have modified the vocab/padding amount so need to be added as litgpt configs
litgpt download Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1 && \
litgpt download allenai/Llama-3.1-Tulu-3-8B-SFT && \

# debugging a new tokenizer extension utility

from transformers import AutoTokenizer
from tokenizers import AddedToken

# tokenizer = AutoTokenizer.from_pretrained("/capstor/store/cscs/userlab/lp98/jkirchen/singleshot-root/singleshot/checkpoints/Qwen/Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained("/capstor/store/cscs/userlab/lp98/jkirchen/singleshot-root/singleshot/checkpoints/meta-llama/Meta-Llama-3.1-8B")
# tokenizer
# # tokenizer.additional_special_tokens

# num_mtp_toks = 16+16
# tokenizer.add_tokens(
#    [AddedToken(f"<|mtp_special_token_{i}|>", lstrip=False, rstrip=False, single_word=False, normalized=False, special=True) for i in range(num_mtp_toks)]
# )
# tokenizer.add_special_tokens(special_tokens_dict={
# 'additional_special_tokens':[AddedToken(f"<|mtp_special_token_{i}|>", lstrip=False, rstrip=False, single_word=False, normalized=False, special=True) for i in range(num_mtp_toks)]
# }, replace_additional_special_tokens=False)

# mtp_toks_start, mtp_toks_end = 240, 247+1
# tokenizer.add_tokens(
#    [AddedToken(f"<|reserved_special_token_{i}|>", lstrip=True, rstrip=False, single_word=False, normalized=False, special=True) for i in range(mtp_toks_start, mtp_toks_end)]
# )

tokenizer

# TOK_SAVE_PATH = "/capstor/store/cscs/userlab/lp98/jkirchen/singleshot-root/singleshot/config_hub/models/Qwen/Qwen3-8B_modified"
TOK_SAVE_PATH = "/capstor/store/cscs/userlab/lp98/jkirchen/singleshot-root/singleshot/config_hub/models/meta-llama/Meta-Llama-3.1-8B_modified"

tokenizer.save_pretrained(TOK_SAVE_PATH)

# batching some extensions

python -u litgpt/scripts/add_mtp_tokens.py \
--model_path=checkpoints/meta-llama/Meta-Llama-3.1-8B \
--output_dir=checkpoints/extended/meta-llama/Meta-Llama-3.1-8B-MTPV128384 \
--model_name=Llama-3.1-8B-MTPV128384 \
&& \
python -u litgpt/scripts/add_mtp_tokens.py \
--model_path=checkpoints/meta-llama/Meta-Llama-3.1-8B-Instruct \
--output_dir=checkpoints/extended/meta-llama/Meta-Llama-3.1-8B-Instruct-MTPV128384 \
--model_name=Llama-3.1-8B-Instruct-MTPV128384 \
&& \
python -u litgpt/scripts/add_mtp_tokens.py \
--model_path=checkpoints/Qwen/Qwen3-8B \
--output_dir=checkpoints/extended/Qwen/Qwen3-8B-MTP \
--model_name=Qwen3-8B \
&& \
python -u litgpt/scripts/add_mtp_tokens.py \
--model_path=checkpoints/Qwen/Qwen3-8B-Base \
--output_dir=checkpoints/extended/Qwen/Qwen3-8B-Base-MTP \
--model_name=Qwen3-8B-Base \
&& \
python -u litgpt/scripts/add_mtp_tokens.py \
--model_path=checkpoints/allenai/Llama-3.1-Tulu-3-8B-SFT \
--output_dir=checkpoints/extended/allenai/Llama-3.1-Tulu-3-8B-SFT-MTPV128384 \
--model_name=Llama-3.1-Tulu-3-8B-SFT-MTPV128384 \
&& \
python -u litgpt/scripts/add_mtp_tokens.py \
--model_path=checkpoints/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1 \
--output_dir=checkpoints/extended/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384 \
--model_name=Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384 \
&& \
python -u litgpt/scripts/add_mtp_tokens.py \
--model_path=checkpoints/nvidia/OpenMath2-Llama3.1-8B \
--output_dir=checkpoints/extended/nvidia/OpenMath2-Llama3.1-8B-MTPV128384 \
--model_name=OpenMath2-Llama3.1-8B-MTPV128384 \
&& \
python -u litgpt/scripts/add_mtp_tokens.py \
--model_path=checkpoints/Qwen/Qwen3-4B-Instruct-2507 \
--output_dir=checkpoints/extended/Qwen/Qwen3-4B-Instruct-2507-MTP \
--model_name=Qwen3-4B-Instruct-2507 \
&& \
python -u litgpt/scripts/add_mtp_tokens.py \
--model_path=checkpoints/Qwen/Qwen3-30B-A3B-Instruct-2507 \
--output_dir=checkpoints/extended/Qwen/Qwen3-30B-A3B-Instruct-2507-MTP \
--model_name=Qwen3-30B-A3B-Instruct-2507 \
&& \
python -u litgpt/scripts/add_mtp_tokens.py \
--model_path=checkpoints/qywu/Qwen3-32B-Instruct \
--output_dir=checkpoints/extended/qywu/Qwen3-32B-Instruct-MTP \
--model_name=Qwen3-32B-Instruct \
&& \
python -u litgpt/scripts/add_mtp_tokens.py \
--model_path=checkpoints/Qwen/Qwen3-14B-Base \
--output_dir=checkpoints/extended/Qwen/Qwen3-14B-Base-MTP \
--model_name=Qwen3-14B-Base \
&& \


# testing a gemini generated wandb pusher

python -u litgpt/scripts/push_lmeval_metrics_to_wandb.py \
    --exp_dir ./outputs/debug_lm_eval_metrics/debug_lm_eval_metrics_gsm8k_cot_retrorec_gsm8k_mtp_lim10_ktoks8_stratnone_daint_prod_ift_mask_fix_1N4n_450c1bb8_step-00100160 \
    --dry_run True

python -u litgpt/scripts/push_lmeval_metrics_to_wandb.py \
--exp_dir ./outputs/debug_lm_eval_metrics/debug_lm_eval_metrics_gsm8k_cot_retrorec_gsm8k_mtp_lim10_ktoks8_stratnone_daint_prod_ift_mask_fix_1N4n_450c1bb8_step-00100160 \
--hf_tokenizer_path=/capstor/scratch/cscs/jkirchen/singleshot-root/singleshot/outputs/daint_prod_ift_mask_fix/daint_prod_ift_mask_fix_1N4n_450c1bb8/step-00100160 \
--wandb_args name=daint_prod_ift_mask_fix_1N4n_450c1bb8,project=singleshot-evals,step=100160,tags=daint+stepwise+manual_pusher \
--dry_run=False


python -u litgpt/scripts/push_lmeval_metrics_to_wandb.py \
--exp_dir ./outputs/debug_lm_eval_metrics/debug_lm_eval_metrics_gsm8k_cot_retrorec_gsm8k_mtp_ktoks16_stratconf_adapt@0.6_daint_prod_ift_mask_fix_1N4n_450c1bb8_step-00100160 \
--hf_tokenizer_path=/capstor/scratch/cscs/jkirchen/singleshot-root/singleshot/outputs/daint_prod_ift_mask_fix/daint_prod_ift_mask_fix_1N4n_450c1bb8/step-00100160 \
--wandb_args name=daint_prod_ift_mask_fix_1N4n_450c1bb8,project=singleshot-evals,step=100160,tags=daint+stepwise+manual_pusher \
--dry_run=False
