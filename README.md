## Multi-Token Prediction via Self-Distillation

This repository contains the code for the arXiv preprint: [[2602.06019] Multi-Token Prediction via Self-Distillation](https://arxiv.org/abs/2602.06019)

The main models behind the results in the paper are available in this huggingface hub collection: [huggingface.co/collections/tomg-group-umd/mtp-lm](https://huggingface.co/collections/tomg-group-umd/mtp-lm).
Each model is accompanied by a model card/README describing its usage. The only requirement for use is a reasonably modern version of the `transformers` library and use of the `trust_remote_code=True` setting to dynamically load the generation logic included in the model repos.

#### 🚧 Work in Progress! 👷

This codebase is still under active development and may be updated or refactored in the future.

#### Note: on the `singleshot` moniker

Early in this project, the approach was internally referred to as "single-shot language modeling" in reference to the generation of multiple tokens in a single shot, or forward pass. As a result the following terms and abbreviations appear throughout the codebase: `singleshot`,`ss`, `sslm`. To avoid breaking things and to minimize retroactive changes that could distance the release from the true experimental code's state, these terms have been left in the code as they are.

### Structure of the codebase

This repository is built on top of a stripped down and extended version of the `litgpt` framework, initially forked circa late 2025. The major modifications are in the `litgpt/pretrain.py` (monolith),`litgpt/model.py` files, with various other core additions like `litgpt/mtp.py` and `litgpt/parquet_dataset.py` to add block masking and custom dataset features. Many components of the CLI such as hf model conversion and litgpt model export are used with only minor modifications, and most of the method-specific logic is contained in the main training script.

### Reproducing the training runs

Our MTP model can be trained using a single node of 4xGH200 gpus or equivalent. Environment and initial checkpoint and data setups require you to run a short series of automated steps.

### Environment

The environment setup will vary depending on the system or cluster, but a workflow using conda + pip is provided in `install_torch_210_cuda_129_singleshot.sh`. The torch version requirement is relatively loose. Any torch 2.9 or 2.10 install on a cuda 12.8 or 12.9 system should work. Alternatively, for a AMD based system, `install_torch_291_rocm_642_singleshot.sh` provides a general template for a similar working environment (architecture target is MI300A in that example). Codebase should work without any src code changes on both types of systems.

For clarity in terms of purpose, original dependencies required by upstream `litgpt` are specified in `pyproject.toml` and then extra packages and local installs are specified in the install shell script itself.

### Initial Checkpoint Setup

To download the initial NTP checkpoints that we adapt into MTP models, we use the existing `litgpt` download and convert utilities.

```
litgpt download Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1

litgpt download Qwen/Qwen3-4B-Instruct-2507
```

By default, these commands will result in complete model config, weight, and tokenizer files being downloaded to the `checkpoints/` folder at the root of the repo. Next, we extend the tokenizers and embedding layers of the models to accomodate our MTP special token. The initialization scheme used for the embedding layer follows the huggingface default convention using the `resize_token_embeddings` class method for huggingface models. 

By default, the extension logic adds 32 special tokens and only actually extends the matrices if they are not large enough to accomodate these new elements (many models come with padded matrices for hardware efficiency reasons). In the final version of the methodology, only a single MTP mask token (the first one in the range) is actually used during training and inference, and the rest are used as a pool of special separators in some of the validation logic. However, we considered using a unique token per mask position in the `k`-token region, and this initialization scheme accomodates that.

```
python -u litgpt/scripts/add_mtp_tokens.py \
--model_path=checkpoints/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1 \
--output_dir=checkpoints/extended/Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384 \
--model_name=Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384

python -u litgpt/scripts/add_mtp_tokens.py \
--model_path=checkpoints/Qwen/Qwen3-4B-Instruct-2507 \
--output_dir=checkpoints/extended/Qwen/Qwen3-4B-Instruct-2507-MTP \
--model_name=Qwen3-4B-Instruct-2507
```

Note that while `Qwen3-4B-Instruct-2507` was an existing model definition included in the `litgpt/config.py` registry, we added `Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384` to the config list manually as this model requires an actual embedding extension to accomodate the new tokens. To apply this approach to other models on the hf hub, similar addtions to the registry file may be required.

#### To reproduce the Qwen results ...

The rest of the examples use the `Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384` model as the target, but for the other main model in the paper, replace the model name and the checkpoint path arguments with `Qwen3-4B-Instruct-2507` and `checkpoints/extended/Qwen/Qwen3-4B-Instruct-2507-MTP` respectively. Additionally, the examples for the `Llama-3.1-8B-Magpie` model use chat templated data, but we found that the `Qwen3-4B-Instruct-2507` based model worked better when trained with a simpler BOS only format, so replace the `metamath_strat_split_chat_<train/val>.yaml` file args in the data prep commands with `metamath_strat_split_bos_<train/val>.yaml`.

### Data Preparation

Litgpt comes equipped with a data processing and loading format. However, we opted to use a custom format due to prior familiarity with this setup and its flexibility wrt to training on HPC systems (SLURM) in multi-gpu and multi-node configurations. Use of this format is not required for any fundamental reason, but the training instructions below will assume it.

At training time, logs, intermediate checkpoints, and other I/O is all localized with respect to an `outputs/` folder at the root of the repository. To prepare the MetaMathQA data, grouped into train and validation splits, for the `Llama-3.1-8B-Magpie` based model, the commands are as follows:

```
export RUN_NAME=l3_magpie_metamath
export RUN_OUTPUT_DIR=outputs/$RUN_NAME
export MODEL_NAME=Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384
export INITIAL_CHECKPOINT=checkpoints/extended/Magpie-Align/$MODEL_NAME

# download and tokenize the training split
python litgpt/pull_raw_datasets.py \
--srcs_config_file=config_hub/data/metamath_strat_split_chat_train.yaml \
--script_configs=config_hub/data/p2p_generic_metamath.yaml \
--script_configs.num_raw_shards=32 \
--script_configs.target_shard_num=32 \
--script_configs.hf_tokenizer=$INITIAL_CHECKPOINT \
--script_configs.raw_dir=$RUN_OUTPUT_DIR/train/raw \
--script_configs.output_dir=$RUN_OUTPUT_DIR/train || exit 1

python litgpt/p2p_tokenizer.py \
--srcs_config_file=config_hub/data/metamath_strat_split_chat_train.yaml \
--script_configs=config_hub/data/p2p_generic_metamath.yaml \
--script_configs.num_raw_shards=32 \
--script_configs.target_shard_num=32 \
--script_configs.hf_tokenizer=$INITIAL_CHECKPOINT \
--script_configs.raw_dir=$RUN_OUTPUT_DIR/train/raw \
--script_configs.output_dir=$RUN_OUTPUT_DIR/train \
--script_configs.train_split_pct=1.0 || exit 1

# download and tokenize the validation split
python litgpt/pull_raw_datasets.py \
--srcs_config_file=config_hub/data/metamath_strat_split_chat_val.yaml \
--script_configs=config_hub/data/p2p_generic_metamath.yaml \
--script_configs.num_raw_shards=32 \
--script_configs.target_shard_num=32 \
--script_configs.hf_tokenizer=$INITIAL_CHECKPOINT \
--script_configs.raw_dir=$RUN_OUTPUT_DIR/val/raw \
--script_configs.output_dir=$RUN_OUTPUT_DIR/val || exit 1

python litgpt/p2p_tokenizer.py \
--srcs_config_file=config_hub/data/metamath_strat_split_chat_val.yaml \
--script_configs=config_hub/data/p2p_generic_metamath.yaml \
--script_configs.num_raw_shards=32 \
--script_configs.target_shard_num=32 \
--script_configs.hf_tokenizer=$INITIAL_CHECKPOINT \
--script_configs.raw_dir=$RUN_OUTPUT_DIR/val/raw \
--script_configs.output_dir=$RUN_OUTPUT_DIR/val \
--script_configs.train_split_pct=1.0 || exit 1

# consolidate the pair of datasets under one folder
mkdir -p $RUN_OUTPUT_DIR/processed
mv $RUN_OUTPUT_DIR/train/processed/train $RUN_OUTPUT_DIR/processed/train || exit 1
mv $RUN_OUTPUT_DIR/val/processed/train $RUN_OUTPUT_DIR/processed/val || exit 1
```

### Training

Rather than hiding the key experimental arguments in a yaml file, for clarity in what the key hyperparameters for the main experiments actually were, and to make further ablations strightforward, we show them as cli args below. (These can be consolidated into a yaml file or the default values can be adjusted in `litgpt/args.py` to reduce cli volume.)

To identify the right launching workflow for your system, refer to the [distributed launching docs](https://lightning.ai/docs/fabric/stable/fundamentals/launch.html) for pytorch lighting's fabric toolkit; this is what is used within the training script to managed distributed setups. Depending on the hardware and quirks of the specific torch version used, the main adjustment that might be required to control memory usage is to lower the batch size. 

The configuration below requires approximately 56/83 GB of memory (alloc/reserv) during the training run for this 8B llama3 model. If the model size is increased, then the number of devices and fsdp mesh may need adjustment as well. The code will scale to multi-node settings smoothly as long as SLURM or torchrun are used and WORLD_SIZE, RANK, LOCAL_RANK are set upon launch; torchrun does this automatically, else set those args using SLURM_NTASKS, SLURM_PROCID, SLURM_LOCALID respectively.

Please see `litgpt/args.py` for descriptions of what each cli argument for the training script does, and see `litgpt/args_data.py` for the data processing specific arguments.

```
export RUN_NAME=l3_magpie_metamath
export RUN_OUTPUT_DIR=outputs/$RUN_NAME
export MODEL_NAME=Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384
export INITIAL_CHECKPOINT=checkpoints/extended/Magpie-Align/$MODEL_NAME

python -u litgpt/pretrain.py \
--config=config_hub/pretrain/ss.yaml \
--model_name=$MODEL_NAME \
--tokenizer_dir=$INITIAL_CHECKPOINT \
--initial_checkpoint_dir=$INITIAL_CHECKPOINT \
--train.fabric_strategy=fsdp \
--train.fsdp_device_mesh=1x4 \
--train.micro_batch_size=32 \
--train.global_batch_size=128 \
--train.max_seq_length=160 \
--singleshot.mtp_special_token_pattern="<|mtp_special_token_{i}|>" \
--singleshot.truncation_length=160 \
--singleshot.k_toks=0-16 \
--singleshot.k_toks_min=0-2 \
--singleshot.k_toks_max=0-16 \
--singleshot.rand_rank_k_toks=True \
--singleshot.mask_region_ct=5 \
--singleshot.rollout_multiplier=4 \
--train.initial_save=True \
--train.save_latest_interval=1000 \
--train.save_interval=10000 \
--train.max_ckpts_to_keep=null \
--eval.interval=1000 \
--eval.max_iters=1 \
--train.lr_warmup_steps=2000 \
--train.lr_schedule=constant \
--train.peak_lr=1e-05 \
--train.min_lr=null \
--train.do_compile=True \
--train.dynamo_cache_size_limit=256 \
--pqds.dataset_script_config_file=config_hub/data/p2p_generic_metamath.yaml \
--pqds.dataset_sources_config_file=config_hub/data/metamath_strat_split_chat_train.yaml \
--train.max_tokens=2000000000  \
--data=pqds \
--pqds.train_dataset_folder_path=$RUN_OUTPUT_DIR/processed/train \
--pqds.val_dataset_folder_path=$RUN_OUTPUT_DIR/processed/val \
--out_dir=$RUN_OUTPUT_DIR \
--wandb.run_name=$RUN_NAME

```

### Reproducing the evaluation results

After training is complete, the run output dir will contain a series of checkpoints. To convert these checkpoints to huggingface format for interactive use and evaluation using the lm-eval-harness, a series of commands are required. The full results in the paper require thousands of evaluation runs when considering all model checkpoints, benchmark tasks, and MTP decoding strategies, and automating this is mostly cluster specific. Therefore, only a representative configuration is provided here. (However, several old automation scripts are left in the repo. For training: `misc_scripts_and_nbs/launch_exps_daint_q4-q1.py`, and for evaluation: `misc_scripts_and_nbs/launch_evals_daint.py`; they are not generalized for public use, but serve as examples of how the full set of results for the paper were automated.)

The example below corresponds to the `Llama-3.1-8B-Magpie` based model configured to achieve $3\times$ faster decoding at $<5\%$ drop in accuracy on GSM8K relative to the single token decoding performance of the same checkpoint. The cfg uses ConfAdapt @ $\tau=90\%$, k_max=16 decoding and targets the final saved checkpoint that should be yielded by the training command above.

### Export

```
export RUN_NAME=l3_magpie_metamath
export RUN_OUTPUT_DIR=outputs/$RUN_NAME
export CKPT_SUBDIR=step-00100160

litgpt convert_from_litgpt \
--checkpoint_dir=$RUN_OUTPUT_DIR/$CKPT_SUBDIR \
--output_dir=$RUN_OUTPUT_DIR/$CKPT_SUBDIR \
--output_name=pytorch_model.bin \
--skip_if_exists=True \
--config_class_path=litgpt.transformers_local.llama.configuration_llama.LlamaConfig \
--model_class_path=litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM
```

The `Llama-3.1-8B-Magpie` based model behind the main results in the paper was published to the hub using the following command:
```
litgpt push_to_hub \
--model_path=$RUN_OUTPUT_DIR/$CKPT_SUBDIR \
--model_class_path=litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM \
--org=jwkirchenbauer \
--private=False \
--model_name=L3-1-8B-Magpie-MTP \
--precision=bfloat16 \
--dry_run=False \
--update_existing=True \
--readme_path=litgpt/transformers_local/llama/README.md \
--readme_only=False
```

### Evaluation

To accomodate additional generation arguments and allow logging of MTP specific inference metrics, a lightly customized fork of the lm-eval-harness is required (see install script). This fork also includes the specific gsm8k eval configuration we use; the only important modification from the default task def is the addition of a "think step by step" prompt suffix. The example command below uses a hf accelerate config to do evaluation using 4 way data parallelism assuming the same 4xGH200 node as was used for training.

Finally, a small util for pushing the processed evaluation results to wandb is also required to accomodate the MTP metrics as well as organize the evaluation results on a training step-wise basis. 

```
export RUN_NAME=l3_magpie_metamath
export RUN_OUTPUT_DIR=outputs/$RUN_NAME
export STEP_NUM=100160
export CKPT_SUBDIR=step-00100160
export EVAL_OUTPUT_DIR=outputs/lm_eval_$RUN_DIR_$CKPT_SUBDIR

accelerate launch --config_file config_hub/lm_eval/accelerate_config_1N.yaml -m lm_eval run \
--config config_hub/lm_eval/default_mtp.yaml \
--model_args pretrained=$RUN_OUTPUT_DIR/$CKPT_SUBDIR,dtype=float32 \
--tasks gsm8k_cot_singleshot \
--apply_chat_template \
--fewshot_as_multiturn \
--gen_kwargs do_sample=False,do_mtp=True,include_prompt=True,return_mtp_result_dict=True,"until=Q:+</s>+<|end_of_text|>+<|eot_id|>+<|endoftext|>+<|im_end|>",mask_id=128259,eos_id=128009+128001,k_toks=1 \
--output_path $EVAL_OUTPUT_DIR

python -u litgpt/scripts/push_lmeval_metrics_to_wandb.py \
--run_dir $EVAL_OUTPUT_DIR \
--wandb_args name=$RUN_NAME,project=singleshot-evals,step=$STEP_NUM,tags=daint+stepwise+manual_pusher \
--dry_run=False
```

### Plotting

After the evaluation results across all configurations of interest are run and pushed to wandb, they can be collated to csv locally for presentation in the format of the paper tables and figures using the following utils:
```
python pull_wandb_eval_data.py

make_paper_figures_and_tables.ipynb
```


#### Citation

```bibtex
@article{kirchenbauer2026multi,
  title={Multi-Token Prediction via Self-Distillation},
  author={Kirchenbauer, John and Hans, Abhimanyu and Bartoldson, Brian and Goldblum, Micah and Panda, Ashwinee and Goldstein, Tom},
  journal={arXiv preprint arXiv:2602.06019},
  year={2026}
}
```

---

### original `litgpt` README below

---

<div align="center">


# ⚡ LitGPT

**20+ high-performance LLMs with recipes to pretrain, finetune, and deploy at scale.**

<pre>
✅ From scratch implementations      ✅ No abstractions         ✅ Beginner friendly
   ✅ Flash attention                   ✅ FSDP                    ✅ LoRA, QLoRA, Adapter
✅ Reduce GPU memory (fp4/8/16/32)   ✅ 1-1000+ GPUs/TPUs       ✅ 20+ LLMs         
</pre>


---


![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-lightning)
![cpu-tests](https://github.com/lightning-AI/lit-stablelm/actions/workflows/cpu-tests.yml/badge.svg) [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lit-stablelm/blob/master/LICENSE) [![Discord](https://img.shields.io/discord/1077906959069626439)](https://discord.gg/VptPCZkGNa)

<p align="center">
  <a href="#quick-start">Quick start</a> •
  <a href="#choose-from-20-llms">Models</a> •
  <a href="#finetune-an-llm">Finetune</a> •
  <a href="#deploy-an-llm">Deploy</a> •
  <a href="#all-workflows">All workflows</a> •
  <a href="#state-of-the-art-features">Features</a> •
  <a href="#training-recipes">Recipes (YAML)</a> •
  <a href="https://lightning.ai/">Lightning AI</a> •
    <a href="#tutorials">Tutorials</a>
</p>

&nbsp;

<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litgpt-quick-start">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/get-started-badge.svg" height="36px" alt="Get started"/>
</a>

&nbsp;

</div>

# Use, finetune, pretrain, and deploy LLMs Lightning fast ⚡⚡
Every LLM is implemented from scratch with **no abstractions** and **full control**, making them blazing fast, minimal, and performant at enterprise scale.

✅ **Enterprise ready -** Apache 2.0 for unlimited enterprise use.</br>
✅ **Developer friendly -** Easy debugging with no abstraction layers and single file implementations.</br>
✅ **Optimized performance -** Models designed to maximize performance, reduce costs, and speed up training.</br>
✅ **Proven recipes -** Highly-optimized training/finetuning recipes tested at enterprise scale.</br>

&nbsp;

# Quick start
Install LitGPT
```
pip install 'litgpt[extra]'
```

Load and use any of the [20+ LLMs](#choose-from-20-llms):
```python
from litgpt import LLM

llm = LLM.load("microsoft/phi-2")
text = llm.generate("Fix the spelling: Every fall, the family goes to the mountains.")
print(text)
# Corrected Sentence: Every fall, the family goes to the mountains.
```

&nbsp;

✅ Optimized for fast inference</br>
✅ Quantization</br>
✅ Runs on low-memory GPUs</br>
✅ No layers of internal abstractions</br>
✅ Optimized for production scale</br>

<details>
  <summary>Advanced install options</summary>

Install from source:

```bash
git clone https://github.com/Lightning-AI/litgpt
cd litgpt
pip install -e '.[all]'
```
</details>

[Explore the full Python API docs](tutorials/python-api.md).

&nbsp;

---
# Choose from 20+ LLMs
Every model is written from scratch to maximize performance and remove layers of abstraction:

| Model | Model size | Author | Reference |
|----|----|----|----|
| Llama 3, 3.1, 3.2, 3.3 | 1B, 3B, 8B, 70B, 405B | Meta AI | [Meta AI 2024](https://github.com/meta-llama/llama3)                                           |
| Code Llama | 7B, 13B, 34B, 70B | Meta AI | [Rozière et al. 2023](https://arxiv.org/abs/2308.12950)                                       |
| CodeGemma | 7B | Google | [Google Team, Google Deepmind](https://ai.google.dev/gemma/docs/codegemma)                                     |
| Gemma 2 | 2B, 9B, 27B | Google | [Google Team, Google Deepmind](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf)  |
| Phi 4 | 14B | Microsoft Research | [Abdin et al. 2024](https://arxiv.org/abs/2412.08905)                                                                            |
| Qwen2.5 | 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B | Alibaba Group | [Qwen Team 2024](https://qwenlm.github.io/blog/qwen2.5/)                                               |
| Qwen2.5 Coder | 0.5B, 1.5B, 3B, 7B, 14B, 32B | Alibaba Group | [Hui, Binyuan et al. 2024](https://arxiv.org/abs/2409.12186)                                          |
| R1 Distill Llama | 8B, 70B | DeepSeek AI | [DeepSeek AI 2025](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)                                                                                 |
| ... | ... | ... | ...   |

<details>
  <summary>See full list of 20+ LLMs</summary>

&nbsp;

#### All models

| Model | Model size | Author | Reference |
|----|----|----|----|
| CodeGemma | 7B | Google | [Google Team, Google Deepmind](https://ai.google.dev/gemma/docs/codegemma)                                                                 |
| Code Llama | 7B, 13B, 34B, 70B | Meta AI | [Rozière et al. 2023](https://arxiv.org/abs/2308.12950)                                                                   |
| Falcon | 7B, 40B, 180B | TII UAE | [TII 2023](https://falconllm.tii.ae)                                                                                              |
| Falcon 3 | 1B, 3B, 7B, 10B | TII UAE | [TII 2024](https://huggingface.co/blog/falcon3)                                                                                              |
| FreeWilly2 (Stable Beluga 2) | 70B | Stability AI | [Stability AI 2023](https://stability.ai/blog/stable-beluga-large-instruction-fine-tuned-models)                 |
| Function Calling Llama 2 | 7B | Trelis | [Trelis et al. 2023](https://huggingface.co/Trelis/Llama-2-7b-chat-hf-function-calling-v2)                                  |
| Gemma | 2B, 7B | Google | [Google Team, Google Deepmind](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)                                       |
| Gemma 2 | 9B, 27B | Google | [Google Team, Google Deepmind](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf)                                  |
| Gemma 3 | 1B, 4B, 12B, 27B | Google | [Google Team, Google Deepmind](https://arxiv.org/pdf/2503.19786)                                  |
| Llama 2 | 7B, 13B, 70B | Meta AI | [Touvron et al. 2023](https://arxiv.org/abs/2307.09288)                                                                           |
| Llama 3.1 | 8B, 70B | Meta AI | [Meta AI 2024](https://github.com/meta-llama/llama3)                                                                                 |
| Llama 3.2 | 1B, 3B | Meta AI | [Meta AI 2024](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)                                           |
| Llama 3.3 | 70B | Meta AI | [Meta AI 2024](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)                                                                                 |
| Mathstral | 7B | Mistral AI | [Mistral AI 2024](https://mistral.ai/news/mathstral/)                                                                                  |
| MicroLlama | 300M | Ken Wang | [MicroLlama repo](https://github.com/keeeeenw/MicroLlama)                                                                             |
| Mixtral MoE | 8x7B | Mistral AI | [Mistral AI 2023](https://mistral.ai/news/mixtral-of-experts/)                                                                     |
| Mistral | 7B, 123B | Mistral AI | [Mistral AI 2023](https://mistral.ai/news/announcing-mistral-7b/)                                                                  |
| Mixtral MoE | 8x22B | Mistral AI | [Mistral AI 2024](https://mistral.ai/news/mixtral-8x22b/)                                                                         |
| OLMo | 1B, 7B | Allen Institute for AI (AI2) | [Groeneveld et al. 2024](https://aclanthology.org/2024.acl-long.841/)    |
| OpenLLaMA | 3B, 7B, 13B | OpenLM Research | [Geng & Liu 2023](https://github.com/openlm-research/open_llama)                                                         |
| Phi 1.5 & 2 | 1.3B, 2.7B | Microsoft Research  | [Li et al. 2023](https://arxiv.org/abs/2309.05463)                                                                  |
| Phi 3 | 3.8B | Microsoft Research | [Abdin et al. 2024](https://arxiv.org/abs/2404.14219)                                                                            |
| Phi 4 | 14B | Microsoft Research | [Abdin et al. 2024](https://arxiv.org/abs/2412.08905)                                                                            |
| Phi 4 Mini Instruct | 3.8B | Microsoft Research | [Microsoft 2025](https://arxiv.org/abs/2503.01743)                                           |
| Phi 4 Mini Reasoning | 3.8B | Microsoft Research | [Xu, Peng et al. 2025](https://arxiv.org/abs/2504.21233)                                           |
| Phi 4 Reasoning | 3.8B | Microsoft Research | [Abdin et al. 2025](https://arxiv.org/abs/2504.21318)                                           |
| Phi 4 Reasoning Plus | 3.8B | Microsoft Research | [Abdin et al. 2025](https://arxiv.org/abs/2504.21318)                                           |
| Platypus | 7B, 13B, 70B |  Lee et al. | [Lee, Hunter, and Ruiz 2023](https://arxiv.org/abs/2308.07317)                                                               |
| Pythia | {14,31,70,160,410}M, {1,1.4,2.8,6.9,12}B | EleutherAI | [Biderman et al. 2023](https://arxiv.org/abs/2304.01373)                                            |
| Qwen2.5 | 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B | Alibaba Group | [Qwen Team 2024](https://qwenlm.github.io/blog/qwen2.5/)                                               |
| Qwen2.5 Coder | 0.5B, 1.5B, 3B, 7B, 14B, 32B | Alibaba Group | [Hui, Binyuan et al. 2024](https://arxiv.org/abs/2409.12186)                                          |
| Qwen2.5 1M (Long Context) | 7B, 14B | Alibaba Group | [Qwen Team 2025](https://qwenlm.github.io/blog/qwen2.5-1m/)                                          |
| Qwen2.5 Math | 1.5B, 7B, 72B | Alibaba Group | [An, Yang et al. 2024](https://arxiv.org/abs/2409.12122)                                          |
| QwQ | 32B | Alibaba Group | [Qwen Team 2025](https://qwenlm.github.io/blog/qwq-32b/)                                                                         |
| QwQ-Preview | 32B | Alibaba Group | [Qwen Team 2024](https://qwenlm.github.io/blog/qwq-32b-preview/)                                                                         |
| Qwen3 | 0.6B, 1.7B, 4B{Hybrid, Thinking-2507, Instruct-2507}, 8B, 14B, 32B | Alibaba Group | [Qwen Team 2025](https://arxiv.org/abs/2505.09388/)                                                                         |
| Qwen3 MoE | 30B{Hybrid, Thinking-2507, Instruct-2507}, 235B{Hybrid, Thinking-2507, Instruct-2507} | Alibaba Group | [Qwen Team 2025](https://arxiv.org/abs/2505.09388/)                                                                         |
| R1 Distill Llama | 8B, 70B | DeepSeek AI | [DeepSeek AI 2025](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)                                                                                 |
| SmolLM2 | 135M, 360M, 1.7B | Hugging Face | [Hugging Face 2024](https://github.com/huggingface/smollm)                                                               |
| Salamandra | 2B, 7B | Barcelona Supercomputing Centre | [BSC-LTC 2024](https://github.com/BSC-LTC/salamandra)                                                                         |
| StableCode | 3B | Stability AI | [Stability AI 2023](https://stability.ai/blog/stablecode-llm-generative-ai-coding)                                                  |
| StableLM  | 3B, 7B | Stability AI | [Stability AI 2023](https://github.com/Stability-AI/StableLM)                                                                    |
| StableLM Zephyr | 3B | Stability AI | [Stability AI 2023](https://stability.ai/blog/stablecode-llm-generative-ai-coding)                                             |
| TinyLlama | 1.1B | Zhang et al. | [Zhang et al. 2023](https://github.com/jzhang38/TinyLlama)                                                                         |


**Tip**: You can list all available models by running the `litgpt download list` command.


</details>

&nbsp;

---

# Workflows

<p align="center">
  <a href="#finetune-an-llm">Finetune</a> •
  <a href="#pretrain-an-llm">Pretrain</a> •
  <a href="#continue-pretraining-an-llm">Continued pretraining</a> •
    <a href="#evaluate-an-llm">Evaluate</a> •
    <a href="#deploy-an-llm">Deploy</a> •
    <a href="#test-an-llm">Test</a>
</p>

&nbsp;

Use the command line interface to run advanced workflows such as pretraining or finetuning on your own data.


## All workflows
After installing LitGPT, select the model and workflow to run (finetune, pretrain, evaluate, deploy, etc...):

```bash
# litgpt [action] [model]
litgpt  serve     meta-llama/Llama-3.2-3B-Instruct
litgpt  finetune  meta-llama/Llama-3.2-3B-Instruct
litgpt  pretrain  meta-llama/Llama-3.2-3B-Instruct
litgpt  chat      meta-llama/Llama-3.2-3B-Instruct
litgpt  evaluate  meta-llama/Llama-3.2-3B-Instruct
```

&nbsp;

----

## Finetune an LLM

<div align="center">
<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litgpt-finetune">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/run-on-studio.svg" height="36px" alt="Run on Studios"/>
</a>
</div>

&nbsp;

Finetuning is the process of taking a pretrained AI model and further training it on a smaller, specialized dataset tailored to a specific task or application.


&nbsp;

```bash
# 0) setup your dataset
curl -L https://huggingface.co/datasets/ksaw008/finance_alpaca/resolve/main/finance_alpaca.json -o my_custom_dataset.json

# 1) Finetune a model (auto downloads weights)
litgpt finetune microsoft/phi-2 \
  --data JSON \
  --data.json_path my_custom_dataset.json \
  --data.val_split_fraction 0.1 \
  --out_dir out/custom-model

# 2) Test the model
litgpt chat out/custom-model/final

# 3) Deploy the model
litgpt serve out/custom-model/final
```

[Read the full finetuning docs](tutorials/finetune.md)

&nbsp;

----

## Deploy an LLM

<div align="center">
<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litgpt-serve">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/deploy-on-studios.svg" height="36px" alt="Deploy on Studios"/>
</a>
</div>

&nbsp;

Deploy a pretrained or finetune LLM to use it in real-world applications. Deploy, automatically sets up a web server that can be accessed by a website or app.

```bash
# deploy an out-of-the-box LLM
litgpt serve microsoft/phi-2

# deploy your own trained model
litgpt serve path/to/microsoft/phi-2/checkpoint
```

<details>
  <summary>Show code to query server:</summary>

&nbsp;

Test the server in a separate terminal and integrate the model API into your AI product:
```python
# 3) Use the server (in a separate Python session)
import requests, json
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"prompt": "Fix typos in the following sentence: Example input"}
)
print(response.json()["output"])
```
</details>

[Read the full deploy docs](tutorials/deploy.md).

&nbsp;

----

## Evaluate an LLM
Evaluate an LLM to test its performance on various tasks to see how well it understands and generates text. Simply put, we can evaluate things like how well would it do in college-level chemistry, coding, etc... (MMLU, Truthful QA, etc...)

```bash
litgpt evaluate microsoft/phi-2 --tasks 'truthfulqa_mc2,mmlu'
```

[Read the full evaluation docs](tutorials/evaluation.md).

&nbsp;

----

##  Test an LLM

<div align="center">
<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litgpt-chat">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/run-on-studio.svg" height="36px" alt="Run on Studios"/>
</a>
</div>

&nbsp;

Test how well the model works via an interactive chat. Use the `chat` command to chat, extract embeddings, etc...

Here's an example showing how to use the Phi-2 LLM:
```bash
litgpt chat microsoft/phi-2

>> Prompt: What do Llamas eat?
```

<details>
  <summary>Full code:</summary>

&nbsp;

```bash
# 1) List all supported LLMs
litgpt download list

# 2) Use a model (auto downloads weights)
litgpt chat microsoft/phi-2

>> Prompt: What do Llamas eat?
```

The download of certain models requires an additional access token. You can read more about this in the [download](tutorials/download_model_weights.md#specific-models-and-access-tokens) documentation.

</details>

[Read the full chat docs](tutorials/inference.md).

&nbsp;

----

## Pretrain an LLM

<div align="center">
<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litgpt-pretrain">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/run-on-studio.svg" height="36px" alt="Run on Studios"/>
</a>
</div>

&nbsp;

Pretraining is the process of teaching an AI model by exposing it to a large amount of data before it is fine-tuned for specific tasks.

<details>
  <summary>Show code:</summary>

&nbsp;

```bash
mkdir -p custom_texts
curl https://www.gutenberg.org/cache/epub/24440/pg24440.txt --output custom_texts/book1.txt
curl https://www.gutenberg.org/cache/epub/26393/pg26393.txt --output custom_texts/book2.txt

# 1) Download a tokenizer
litgpt download EleutherAI/pythia-160m \
  --tokenizer_only True

# 2) Pretrain the model
litgpt pretrain EleutherAI/pythia-160m \
  --tokenizer_dir EleutherAI/pythia-160m \
  --data TextFiles \
  --data.train_data_path "custom_texts/" \
  --train.max_tokens 10_000_000 \
  --out_dir out/custom-model

# 3) Test the model
litgpt chat out/custom-model/final
```
</details>

[Read the full pretraining docs](tutorials/pretrain.md)

&nbsp;

----

## Continue pretraining an LLM

<div align="center">
<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litgpt-continue-pretraining">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/run-on-studio.svg" height="36px" alt="Run on Studios"/>
</a>
</div>

&nbsp;

Continued pretraining is another way of finetuning that specializes an already pretrained model by training on custom data:

<details>
  <summary>Show code:</summary>

&nbsp;

```bash
mkdir -p custom_texts
curl https://www.gutenberg.org/cache/epub/24440/pg24440.txt --output custom_texts/book1.txt
curl https://www.gutenberg.org/cache/epub/26393/pg26393.txt --output custom_texts/book2.txt

# 1) Continue pretraining a model (auto downloads weights)
litgpt pretrain EleutherAI/pythia-160m \
  --tokenizer_dir EleutherAI/pythia-160m \
  --initial_checkpoint_dir EleutherAI/pythia-160m \
  --data TextFiles \
  --data.train_data_path "custom_texts/" \
  --train.max_tokens 10_000_000 \
  --out_dir out/custom-model

# 2) Test the model
litgpt chat out/custom-model/final
```

</details>

[Read the full continued pretraining docs](tutorials/pretrain.md#continued-pretraining-on-custom-data)

&nbsp;

----

# State-of-the-art features

✅ State-of-the-art optimizations: Flash Attention v2, multi-GPU support via fully-sharded data parallelism, [optional CPU offloading](tutorials/oom.md#do-sharding-across-multiple-gpus), and [TPU and XLA support](extensions/xla).</br>
✅ [Pretrain](tutorials/pretrain.md), [finetune](tutorials/finetune.md), and [deploy](tutorials/inference.md)</br>
✅ Reduce compute requirements with low-precision settings: FP16, BF16, and FP16/FP32 mixed.</br>
✅ Lower memory requirements with [quantization](tutorials/quantize.md): 4-bit floats, 8-bit integers, and double quantization.</br>
✅ [Configuration files](config_hub) for great out-of-the-box performance.</br>
✅ Parameter-efficient finetuning: [LoRA](tutorials/finetune_lora.md), [QLoRA](tutorials/finetune_lora.md), [Adapter](tutorials/finetune_adapter.md), and [Adapter v2](tutorials/finetune_adapter.md).</br>
✅ [Exporting](tutorials/convert_lit_models.md) to other popular model weight formats.</br>
✅ Many popular datasets for [pretraining](tutorials/pretrain.md) and [finetuning](tutorials/prepare_dataset.md), and [support for custom datasets](tutorials/prepare_dataset.md#preparing-custom-datasets-for-instruction-finetuning).</br>
✅ Readable and easy-to-modify code to experiment with the latest research ideas.</br>

&nbsp;

---

# Training recipes

LitGPT comes with validated recipes (YAML configs) to train models under different conditions.  We've generated these recipes based on the parameters we found to perform the best for different training conditions.

Browse all training recipes [here](config_hub).

### Example

```bash
litgpt finetune \
  --config https://raw.githubusercontent.com/Lightning-AI/litgpt/main/config_hub/finetune/llama-2-7b/lora.yaml
```
<details>
  <summary>✅ Use configs to customize training</summary>

Configs let you customize training for all granular parameters like:

```yaml
# The path to the base model's checkpoint directory to load for finetuning. (type: <class 'Path'>, default: checkpoints/stabilityai/stablelm-base-alpha-3b)
checkpoint_dir: checkpoints/meta-llama/Llama-2-7b-hf

# Directory in which to save checkpoints and logs. (type: <class 'Path'>, default: out/lora)
out_dir: out/finetune/qlora-llama2-7b

# The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true". (type: Optional[str], default: null)
precision: bf16-true

...
```
</details>

<details>
  <summary>✅ Example: LoRA finetuning config</summary>

&nbsp;

```yaml
# The path to the base model's checkpoint directory to load for finetuning. (type: <class 'Path'>, default: checkpoints/stabilityai/stablelm-base-alpha-3b)
checkpoint_dir: checkpoints/meta-llama/Llama-2-7b-hf

# Directory in which to save checkpoints and logs. (type: <class 'Path'>, default: out/lora)
out_dir: out/finetune/qlora-llama2-7b

# The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true". (type: Optional[str], default: null)
precision: bf16-true

# If set, quantize the model with this algorithm. See ``tutorials/quantize.md`` for more information. (type: Optional[Literal['nf4', 'nf4-dq', 'fp4', 'fp4-dq', 'int8-training']], default: null)
quantize: bnb.nf4

# How many devices/GPUs to use. (type: Union[int, str], default: 1)
devices: 1

# How many nodes to use. (type: int, default: 1)
num_nodes: 1

# The LoRA rank. (type: int, default: 8)
lora_r: 32

# The LoRA alpha. (type: int, default: 16)
lora_alpha: 16

# The LoRA dropout value. (type: float, default: 0.05)
lora_dropout: 0.05

# Whether to apply LoRA to the query weights in attention. (type: bool, default: True)
lora_query: true

# Whether to apply LoRA to the key weights in attention. (type: bool, default: False)
lora_key: false

# Whether to apply LoRA to the value weights in attention. (type: bool, default: True)
lora_value: true

# Whether to apply LoRA to the output projection in the attention block. (type: bool, default: False)
lora_projection: false

# Whether to apply LoRA to the weights of the MLP in the attention block. (type: bool, default: False)
lora_mlp: false

# Whether to apply LoRA to output head in GPT. (type: bool, default: False)
lora_head: false

# Data-related arguments. If not provided, the default is ``litgpt.data.Alpaca``.
data:
  class_path: litgpt.data.Alpaca2k
  init_args:
    mask_prompt: false
    val_split_fraction: 0.05
    prompt_style: alpaca
    ignore_index: -100
    seed: 42
    num_workers: 4
    download_dir: data/alpaca2k

# Training-related arguments. See ``litgpt.args.TrainArgs`` for details
train:

  # Number of optimizer steps between saving checkpoints (type: Optional[int], default: 1000)
  save_interval: 200

  # Number of iterations between logging calls (type: int, default: 1)
  log_interval: 1

  # Number of samples between optimizer steps across data-parallel ranks (type: int, default: 128)
  global_batch_size: 8

  # Number of samples per data-parallel rank (type: int, default: 4)
  micro_batch_size: 2

  # Number of iterations with learning rate warmup active (type: int, default: 100)
  lr_warmup_steps: 10

  # Number of epochs to train on (type: Optional[int], default: 5)
  epochs: 4

  # Total number of tokens to train on (type: Optional[int], default: null)
  max_tokens:

  # Limits the number of optimizer steps to run (type: Optional[int], default: null)
  max_steps:

  # Limits the length of samples (type: Optional[int], default: null)
  max_seq_length: 512

  # Whether to tie the embedding weights with the language modeling head weights (type: Optional[bool], default: null)
  tie_embeddings:

  #   (type: float, default: 0.0003)
  learning_rate: 0.0002

  #   (type: float, default: 0.02)
  weight_decay: 0.0

  #   (type: float, default: 0.9)
  beta1: 0.9

  #   (type: float, default: 0.95)
  beta2: 0.95

  #   (type: Optional[float], default: null)
  max_norm:

  #   (type: float, default: 6e-05)
  min_lr: 6.0e-05

# Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details
eval:

  # Number of optimizer steps between evaluation calls (type: int, default: 100)
  interval: 100

  # Number of tokens to generate (type: Optional[int], default: 100)
  max_new_tokens: 100

  # Number of iterations (type: int, default: 100)
  max_iters: 100

# The name of the logger to send metrics to. (type: Literal['wandb', 'tensorboard', 'csv'], default: csv)
logger_name: csv

# The random seed to use for reproducibility. (type: int, default: 1337)
seed: 1337
```
</details>

<details>
  <summary>✅ Override any parameter in the CLI:</summary>

```bash
litgpt finetune \
  --config https://raw.githubusercontent.com/Lightning-AI/litgpt/main/config_hub/finetune/llama-2-7b/lora.yaml \
  --lora_r 4
```
</details>

&nbsp;

----

# Project highlights

LitGPT powers many great AI projects, initiatives, challenges and of course enterprises. Please submit a pull request to be considered for a feature.

<details>
  <summary>📊 SAMBA: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling</summary>

The [Samba](https://github.com/microsoft/Samba) project by researchers at Microsoft is built on top of the LitGPT code base and combines state space models with sliding window attention, which outperforms pure state space models.

</details>

<details>
  <summary>🏆 NeurIPS 2023 Large Language Model Efficiency Challenge: 1 LLM + 1 GPU + 1 Day</summary>

The LitGPT repository was the official starter kit for the [NeurIPS 2023 LLM Efficiency Challenge](https://llm-efficiency-challenge.github.io), which is a competition focused on finetuning an existing non-instruction tuned LLM for 24 hours on a single GPU.

</details>

<details>
  <summary>🦙 TinyLlama: An Open-Source Small Language Model</summary>


LitGPT powered the [TinyLlama project](https://github.com/jzhang38/TinyLlama) and [TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385) research paper.

</details>

<details>
  <summary>🍪 MicroLlama: MicroLlama-300M</summary>

[MicroLlama](https://github.com/keeeeenw/MicroLlama) is a 300M Llama model pretrained on 50B tokens powered by TinyLlama and LitGPT.
</details>

<details>
  <summary>🔬 Pre-training Small Base LMs with Fewer Tokens</summary>

The research paper ["Pre-training Small Base LMs with Fewer Tokens"](https://arxiv.org/abs/2404.08634), which utilizes LitGPT, develops smaller base language models by inheriting a few transformer blocks from larger models and training on a tiny fraction of the data used by the larger models. It demonstrates that these smaller models can perform comparably to larger models despite using significantly less training data and resources.

</details>

&nbsp;

----

# Community

We welcome all individual contributors, regardless of their level of experience or hardware. Your contributions are valuable, and we are excited to see what you can accomplish in this collaborative and supportive environment.

- [Request a feature](https://github.com/Lightning-AI/litgpt/issues)
- [Submit your first contribution](https://lightning.ai/pages/community/tutorial/how-to-contribute-to-litgpt/)
- [Join our Discord](https://discord.gg/VptPCZkGNa)

&nbsp;

# Tutorials

🚀 [Get started](tutorials/0_to_litgpt.md)</br>
⚡️ [Finetuning, incl. LoRA, QLoRA, and Adapters](tutorials/finetune.md)</br>
🤖 [Pretraining](tutorials/pretrain.md)</br>
💬 [Model evaluation](tutorials/evaluation.md)</br>
📘 [Supported and custom datasets](tutorials/prepare_dataset.md)</br>
🧹 [Quantization](tutorials/quantize.md)</br>
🤯 [Tips for dealing with out-of-memory (OOM) errors](tutorials/oom.md)</br>
🧑🏽‍💻 [Using cloud TPUs](extensions/xla)</br>

&nbsp;

----

### Acknowledgments

This implementation extends on [Lit-LLaMA](https://github.com/lightning-AI/lit-llama) and [nanoGPT](https://github.com/karpathy/nanoGPT), and it's **powered by [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) ⚡**.

- [@karpathy](https://github.com/karpathy) for [nanoGPT](https://github.com/karpathy/nanoGPT)
- [@EleutherAI](https://github.com/EleutherAI) for [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) and the [Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [@TimDettmers](https://github.com/TimDettmers) for [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [@Microsoft](https://github.com/microsoft) for [LoRA](https://github.com/microsoft/LoRA)
- [@tridao](https://github.com/tridao) for [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)

### License

LitGPT is released under the [Apache 2.0](https://github.com/Lightning-AI/litgpt/blob/main/LICENSE) license.

### Citation

If you use LitGPT in your research, please cite the following work:

```bibtex
@misc{litgpt-2023,
  author       = {Lightning AI},
  title        = {LitGPT},
  howpublished = {\url{https://github.com/Lightning-AI/litgpt}},
  year         = {2023},
}
```

&nbsp;
