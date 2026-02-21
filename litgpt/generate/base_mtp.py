# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from jsonargparse import CLI, set_parsing_settings

import sys
import time
import warnings
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import lightning as L
import torch
import torch._dynamo.config
import torch._inductor.config
from lightning.fabric.plugins import BitsandbytesPrecision

from litgpt.config import Config
from litgpt.model import GPT, Block
from litgpt.prompts import PromptStyle, has_prompt_style, load_prompt_style
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    _BITANDBYTES_AVAILABLE_NOT_EQUAL_0_42_0,
    check_file_size_on_cpu_and_warn,
    check_valid_checkpoint_dir,
    extend_checkpoint_dir,
    get_default_supported_precision,
    load_checkpoint,
    check_nvlink_connectivity,
    _TORCH_EQUAL_2_7,
    _TORCH_EQUAL_2_8,
)
from litgpt.pretrain import ent_and_top1_confidence


def extend_w_mask(input_ids=None, k_toks=None, mask_id=None):
    bsz, _ = input_ids.shape

    if k_toks - 1 > 0:
        mask_tensor = (
            torch.ones((bsz, k_toks - 1), dtype=torch.int64, device=input_ids.device)
            * mask_id
        )
        return torch.cat([input_ids, mask_tensor], dim=-1)

    return input_ids


def mtp_next_tokens(
    model: GPT,
    input_pos: torch.Tensor,
    x: torch.Tensor,
    input_pos_maxp1: Optional[int] = None,
    k_toks: int = 1,
    strategy: Optional[list] = None,
) -> torch.Tensor:
    logits = model(x, input_pos, input_pos_maxp1=input_pos_maxp1)
    logits = logits[0, -k_toks:]
    if strategy is None:
        _next = torch.argmax(logits, dim=-1, keepdim=False)
    elif strategy[0] == "conf_adapt" or strategy[0] == "conf_adapt_sample@1":
        # we compute the position wise confidences using the ent_and_top1_confidence function
        ent, top1_conf = ent_and_top1_confidence(logits)
        # print(f"top1_conf: {top1_conf}", flush=True)
        # now we compute the position of the farthest token geq the threshold
        # but contiguously, so if we have [0.95, 0.92, 0.85, 0.97] and threshold 0.9
        # we want to get position 1 not 3, since position 2 is below the threshold
        # also being careful of situation like [0.85, 0.88, 0.95] where nothing meets the threshold
        # falling back to the first token in that case
        threshold = strategy[1]
        lt_thresh_mask = top1_conf < threshold
        # now we find the first case where the mask is true, and go back one position
        if torch.all(~lt_thresh_mask):
            # then all positions are above the threshold, we take the last position
            last_pos = k_toks - 1
        else:
            last_pos = torch.argmax(lt_thresh_mask.int()).item() - 1
            if last_pos < 0:
                last_pos = 0

        # print(f"last_pos: {last_pos}", flush=True)

        # then we slice the logits to only keep up to that position
        logits = logits[: last_pos + 1]

        if last_pos == 0 and strategy[0] == "conf_adapt_sample@1":
            # if k is 1 this step, then draw from the distribution
            probs = torch.softmax(logits, dim=-1)
            # print(f"Sampling from probs at k={logits.size(0)}: {probs.shape}", flush=True)
            temperature = strategy[2]
            if 0.0 < temperature < 1.0:
                probs = probs.pow(1.0 / temperature)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            else:
                print(f"Using temperature={temperature} has no effect.", flush=True)
            _next = torch.multinomial(probs, num_samples=1).squeeze(0)
            
        else:
            _next = torch.argmax(logits, dim=-1, keepdim=False)
    elif strategy[0] == "random":
        sampling_weights = strategy[1]
        # we sample k for this step according to the provided weights
        k_toks = int(
            torch.multinomial(torch.tensor(sampling_weights), num_samples=1).item()
        )
        logits = logits[: k_toks + 1]
        _next = torch.argmax(logits, dim=-1, keepdim=False)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return _next


@torch.inference_mode()
def generate_fn(
    model: GPT,
    prompt: torch.Tensor,
    max_returned_tokens: int,
    *,
    k_toks: int = 1,
    mask_id: int = 128002,
    stop_tokens: Tuple[List[int], ...] = (),
    include_prompt: bool,
    include_eos: bool,
    tokenizer: Optional[Tokenizer] = None,
    streamer=None,
    strategy: Optional[list] = None,
) -> Iterator[torch.Tensor]:
    """
    Generates tokens for a single prompt.

    Args:
        model: The model to use.
        prompt: The tokenized prompt to generate from.
        max_returned_tokens: The maximum number of new tokens to return. Does not include the prompt tokens.
        stop_tokens: A tuple of stop sequences. If any of the sequences are generated, the generation stops early before max_returned_tokens.
        include_prompt: Whether to output the prompt tokens.
        include_eos: Whether to output the stop tokens if generation stops early.
    """

    prompt_size = prompt.size(0)
    device = prompt.device

    assert (
        max_returned_tokens > prompt_size
    ), f"Not enough space for {prompt_size} prompt tokens in a context length of {max_returned_tokens}."
    if model.max_seq_length < max_returned_tokens - 1:
        raise NotImplementedError(
            f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}"
        )

    # Yield the prompt if include_prompt is True
    if include_prompt:
        # print(f"\r{tokenizer.decode(torch.tensor(prompt))}".replace("\n","\\n"),end='\n', flush=True)
        if streamer is not None:
            print(f"\n<BEGIN Streaming Prompt>", flush=True)
            streamer.put(prompt)
            print(f"\n<END Streaming Prompt>", flush=True)
        yield from prompt.tolist()

    if streamer is not None:
        print(f"\n<BEGIN Streaming Generation>", flush=True)

    stop_progress = [0] * len(stop_tokens)
    yielded_idx = 0

    # Generate output tokens.
    # The first token generated is the prefill token.
    # The input_pos for this token is the width of the entire prompt.
    # For subsequent iterations, it's the index in the context for the token that we're generating.
    # print(f"\n<BEGIN Streaming Generation>", flush=True)

    # k_toks = 1
    # k_toks = 3
    # k_toks = 8
    # k_toks = 16
    # mask_id = 128002
    t0_prefill = time.perf_counter()
    t1_prefill = None
    t0_gen = None
    t1_gen = None
    toks_post_prefill = None

    tokens = []
    token = prompt
    prefill_token = True

    if k_toks > 1:
        input_pos = torch.arange(
            0, prompt_size + (k_toks - 1), device=device, dtype=torch.int64
        )
        input_pos_maxp1 = prompt_size + (k_toks - 1)
    else:
        input_pos = torch.arange(0, prompt_size, device=device, dtype=torch.int64)
        # input_pos_maxp1 introduces data-dependent shapes and control flow.
        # We want to skip if ThunderModules are involved, either directly or wrapped in LightningModule etc.
        input_pos_maxp1 = (
            prompt_size
            if all(m.__class__.__name__ != "ThunderModule" for m in model.modules())
            else None
        )
    # for current_idx in range(max_returned_tokens - prompt_size):
    # for current_idx in range(0, max_returned_tokens - prompt_size, k_toks):
    current_idx = token.size(0)
    num_fwd_evals = 0
    effective_k_values = []
    while current_idx + k_toks <= max_returned_tokens:
        # print(f"Current idx: {current_idx}", flush=True)
        # print(f"Num fwd evals: {num_fwd_evals}", flush=True)
        # print(f"Len of input array: {token.shape[0]}", flush=True)
        # print(f"Array len after this step would be at most: {token.shape[0] + k_toks}", flush=True)
        if (
            # t0_gen is None and current_idx == 2 * k_toks
            t0_gen is None
            and num_fwd_evals == 2
        ):  # 0 is prefill, 1 is first step which can include compile time, then 2 is steady state
            t1_prefill = time.perf_counter()
            t0_gen = time.perf_counter()
            toks_post_prefill = len(tokens)
        # Generate the token
        # print(
        #     f"Prefill: {prefill_token}, input_pos: {input_pos}, input_pos_maxp1: {input_pos_maxp1}",
        #     flush=True,
        # )
        # if not prefill_token:
        #     token = token[
        #         -1:
        #     ]  # when using the real result of mtp fwd as kv cache, only need last token
        # print(f"Input pos: {input_pos}, input_pos_maxp1: {input_pos_maxp1}", flush=True)
        token = token.view(1, -1)
        token = extend_w_mask(input_ids=token, k_toks=k_toks, mask_id=mask_id)
        # print(f"token: {token}", flush=True)
        # print(f"Pre-forward token.shape: {token.shape}")
        # print(f"Pre-forward token: {token}")
        # breakpoint()
        token = mtp_next_tokens(
            model,
            input_pos,
            token,
            input_pos_maxp1=input_pos_maxp1,
            k_toks=k_toks,
            strategy=strategy,
        )
        effective_k_values.append(token.shape[0])
        # print(f"Post-forward token.shape: {token.shape}")
        # print(f"Post-forward token: {token}")
        # breakpoint()
        # tokens.append(token)
        tokens.extend(token.tolist())
        # print(f"Post-forward tokens: {tokens}")
        # breakpoint()
        # print(f"\r{tokenizer.decode(torch.tensor(tokens))}".replace("\n","\\n"),end='', flush=True)
        # int_token = token.item()
        int_token = token.tolist()

        # Check for stop sequences
        # For each stop sequence, we keep a running total of how many are matched in stop_progress.
        # If the current token matches the next token in the stop sequence, we increment the
        # running total and hold off on yielding the token.
        for int_tok in int_token:
            for i, seq in enumerate(stop_tokens):
                if int_tok == seq[stop_progress[i]]:
                    stop_progress[i] += 1
                    if stop_progress[i] == len(seq):
                        if include_eos:
                            yield from tokens[yielded_idx:]
                            # print(f"\r{tokenizer.decode(torch.tensor(tokens))}".replace("\n","\\n"),end='\n', flush=True)
                            if streamer is not None:
                                streamer.put(torch.tensor(tokens[yielded_idx:]))
                                streamer.end()
                                print(f"\n<END Streaming Generation>", flush=True)
                        t1_gen = time.perf_counter()
                        t_gen = t1_gen - t0_gen
                        t_prefill = t1_prefill - t0_prefill
                        tokens_generated = len(tokens) - toks_post_prefill
                        print(
                            f"Time for prefill plus first/compilation step was {t_prefill:.02f} sec, generation time was {t_gen:.02f} sec @ {tokens_generated / t_gen:.02f} tokens/sec over {tokens_generated} tokens.",
                            flush=True,
                        )
                        # if strategy is not None:
                        print(f"Strategy used: {strategy}", flush=True)
                        avg_effective_k = sum(effective_k_values) / len(
                            effective_k_values
                        )
                        print(
                            f"Average effective k_toks over generation: {avg_effective_k:.02f}, full array of effective k_toks: {effective_k_values}",
                            flush=True,
                        )
                        return
                else:
                    stop_progress[i] = 0

        # Yield tokens that are not part of a stop sequence in progress.
        # If there are no stop sequences, then that's all of them.
        if stop_tokens:
            safe_idx = len(tokens) - max(stop_progress)
            # print(f"safe_idx: {safe_idx}", flush=True)
        else:
            # safe_idx = current_idx + 1  # include the token just generated
            # safe_idx = current_idx + k_toks
            safe_idx = current_idx + token.shape[0]
            # print(f"safe_idx: {safe_idx}", flush=True)

        if yielded_idx < safe_idx:
            y_tokens = tokens[yielded_idx:safe_idx]
            yield from y_tokens
            # print(f"\r{tokenizer.decode(torch.tensor(tokens))}".replace("\n","\\n"),end='', flush=True)
            if streamer is not None:
                streamer.put(torch.tensor(y_tokens))
            yielded_idx = safe_idx

        # Update input_pos for the next iteration.
        if strategy is None:
            if prefill_token:
                prefill_token = False
                if k_toks > 1:
                    input_pos = torch.arange(
                        prompt_size,
                        prompt_size + k_toks + (k_toks - 1),
                        # prompt_size + (k_toks - 1),
                        # prompt_size + k_toks + (k_toks - 1),
                        device=device,
                        dtype=torch.int64,
                    )
                else:
                    input_pos = torch.tensor(
                        [prompt_size], device=device, dtype=torch.int64
                    )
            else:
                if k_toks > 1:
                    input_pos.add_(k_toks)
                else:
                    input_pos.add_(1)
            if input_pos_maxp1 is not None:
                if k_toks > 1:
                    input_pos_maxp1 += k_toks
                else:
                    input_pos_maxp1 += 1
        else:  # we can assume that all strats produce variable number of tokens
            # in this case, the number of tokens generated is data dependent
            num_new_tokens = token.shape[0]
            if prefill_token:
                prefill_token = False
                # the position setup is always the previous number of tokens generated
                # and then k_toks - 1 more after that
                if k_toks > 1:
                    input_pos = torch.arange(
                        prompt_size,
                        prompt_size + num_new_tokens + (k_toks - 1),
                        device=device,
                        dtype=torch.int64,
                    )
                else:
                    input_pos = torch.tensor(
                        [prompt_size], device=device, dtype=torch.int64
                    )
            else:
                if k_toks > 1:
                    # the update logic is more complex here. we first need to identify
                    # which of the positions we were "recomputing" in the previous step
                    # which is the full tensor minus the trailing (k_toks - 1) positions.
                    # then if we take the first element of that, and increment it by
                    # that number of tokens prev generated, we get the new starting position.
                    # then we just recreate the mask like in the starting case.
                    recomputation_positions = input_pos[: -(k_toks - 1)]
                    previous_num_new_tokens = recomputation_positions.size(0)
                    new_start_pos = recomputation_positions[0] + previous_num_new_tokens
                    input_pos = torch.arange(
                        new_start_pos,
                        new_start_pos + num_new_tokens + (k_toks - 1),
                        device=device,
                        dtype=torch.int64,
                    )
                else:
                    input_pos.add_(1)
            if input_pos_maxp1 is not None:
                # the maxp1 is just the farthest position plus one
                input_pos_maxp1 = input_pos[-1] + 1

        current_idx += token.shape[0]
        num_fwd_evals += 1

    # Yield any remaining tokens
    if yielded_idx < len(tokens):
        yield from tokens[yielded_idx:]
        # print(f"\r{tokenizer.decode(torch.tensor(tokens))}".replace("\n","\\n"),end='\n', flush=True)
        if streamer is not None:
            streamer.put(torch.tensor(tokens[yielded_idx:]))
    if streamer is not None:
        streamer.end()
        print(f"\n<END Streaming Generation>", flush=True)
    t1_gen = time.perf_counter()
    t_gen = t1_gen - t0_gen
    t_prefill = t1_prefill - t0_prefill
    tokens_generated = len(tokens) - toks_post_prefill
    print(
        f"Time for prefill plus first/compilation step was {t_prefill:.02f} sec, generation time was {t_gen:.02f} sec @ {tokens_generated / t_gen:.02f} tokens/sec over {tokens_generated} tokens.",
        flush=True,
    )
    # if strategy is not None:
    print(f"Strategy used: {strategy}", flush=True)
    avg_effective_k = sum(effective_k_values) / len(effective_k_values)
    print(
        f"Average effective k_toks over generation: {avg_effective_k:.02f}, full array of effective k_toks: {effective_k_values}",
        flush=True,
    )


@torch.inference_mode()
def generate_mtp(
    model: GPT = None,
    prompt: torch.Tensor = None,
    max_returned_tokens: int = None,
    k_toks: int = 1,
    mask_id: int = 128002,
    eos_id: Optional[int] = None,
    include_prompt: bool = True,
    tokenizer: Optional[Tokenizer] = None,
    streamer=None,
    strategy: Optional[list] = None,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        prompt: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
        include_prompt: If true (default) prepends the prompt (after applying the prompt style) to the output.
    """

    token_list = list(
        generate_fn(
            include_prompt=include_prompt,
            include_eos=True,
            model=model,
            prompt=prompt,
            max_returned_tokens=max_returned_tokens,
            k_toks=k_toks,
            stop_tokens=(([eos_id],) if eos_id is not None else ()),
            tokenizer=tokenizer,
            streamer=streamer,
            strategy=strategy,
        )
    )
    if len(token_list) > 0 and isinstance(token_list[0], int):
        return torch.tensor(token_list, dtype=torch.int64)
    else:
        return torch.cat(token_list) if not len(token_list) == 0 else torch.Tensor()


def reset_model_kv_cache(
    model=None,
    batch_size: int = None,
    max_returned_tokens: int = None,
    device: torch.device = None,
    fabric: L.Fabric = None,
):
    # following some api.py commands to set up kv cache and things
    with fabric.init_tensor():

        model.clear_kv_cache()

        model.max_seq_length = max_returned_tokens

        model.set_kv_cache(
            batch_size=batch_size, max_seq_length=max_returned_tokens, device=device
        )

        for block in model.transformer.h:
            block.attn.kv_cache.reset_parameters()


@torch.inference_mode()
def main(
    checkpoint_dir: Path = None,
    prompt: str = "What food do llamas eat?",
    include_prompt: bool = True,
    *,
    sys_prompt: Optional[str] = None,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    k_toks: int = 1,
    mask_id: int = 128002,
    quantize: Optional[
        Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]
    ] = None,
    precision: Optional[str] = None,
    compile: bool = False,
    strategy: Optional[list] = ["conf_adapt", 0.9],
) -> None:
    """Default generation option.

    Generates text samples based on a pre-trained model and tokenizer.

    Args:
        checkpoint_dir: The checkpoint directory to load.
        prompt: The prompt string to use for generating the samples.
        sys_prompt: The system prompt to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        top_p: If specified, it represents the cumulative probability threshold to consider in the sampling process.
            In top-p sampling, the next token is sampled from the highest probability tokens
            whose cumulative probability exceeds the threshold `top_p`. When specified,
            it must be `0 <= top_p <= 1`. Here, `top_p=0` is equivalent
            to sampling the most probable token, while `top_p=1` samples from the whole distribution.
            It can be used in conjunction with `top_k` and `temperature` with the following order
            of application:

            1. `top_k` sampling
            2. `temperature` scaling
            3. `top_p` sampling

            For more details, see https://arxiv.org/abs/1904.09751
            or https://huyenchip.com/2024/01/16/sampling.html#top_p
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/litgpt/blob/main/tutorials/quantize.md
        precision: Indicates the Fabric precision setting to use.
        compile: Whether to compile the model.
    """
    #     prompt=f"""\
    # <|begin_of_text|>What is the units digit of the product of all the odd positive integers between 10 and 110?

    # d positive integers between 10 and 110 are 1, 3, 5, 7, and 9.
    # The units digits of their product is the units digit of $1 \times 3 \times \
    # """
    #     prompt = f"""\
    # <|begin_of_text|>Given $g(x) = x^2$ and $f(x) = 2x - 1$, what is the value of $f(g(2))$?

    # First, we need to find $g(2)$.
    # Since $g(x) = x^2$, we have $g(2) = \
    # """
    prompt = f"""\
<|begin_of_text|>Sandy plans to paint one wall in her bedroom. The wall is 9 feet high and 12 feet long. There is a 2-foot by 4-foot area on that wall that she will not have to paint due to the window. How many square feet will she need to paint?

The total area of the wall is\
"""
    #     prompt = f"""\
    # <|begin_of_text|>Hash has nine more than half as many toys as Bill has. If Bill has x toys, The boys have 99 total toys. What is the value of unknown variable x?

    # To solve this problem, we need to determine the value of x, which represents the number of toys that Bill has.
    # Let's break down the information given:
    # Number of toys that Hash has:\
    # """
    #     prompt = f"""\
    # <|begin_of_text|>A man used to have 39 cows but last year 25 of them died and he sold 6 of them. This year the number of the cows increased by 24 and the man bought 43 more. His friend gave him 8 cows as a gift. How many cows does the man have now?

    # The man started with 39 cows.
    # Last year, 25 cows died, so\
    # """

    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    pprint(locals())

    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        if _BITANDBYTES_AVAILABLE_NOT_EQUAL_0_42_0:
            warnings.warn(
                "LitGPT only supports bitsandbytes v0.42.0. This may result in errors when using quantization."
            )
        dtype = {
            "16-true": torch.float16,
            "bf16-true": torch.bfloat16,
            "32-true": torch.float32,
        }[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)

    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    checkpoint_path = checkpoint_dir / "lit_model.pth"
    check_file_size_on_cpu_and_warn(checkpoint_path, fabric.device)

    tokenizer = Tokenizer(checkpoint_dir)
    prompt_style = (
        load_prompt_style(checkpoint_dir)
        if has_prompt_style(checkpoint_dir)
        else PromptStyle.from_config(config)
    )

    prompt = prompt_style.apply(prompt, sys_prompt=sys_prompt)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens
    print(f"Prompt length (tokens): {prompt_length}", flush=True)
    print(f"Max returned tokens: {max_returned_tokens}", flush=True)

    fabric.print(
        f"Loading model {str(checkpoint_path)!r} with {config.__dict__}",
        file=sys.stderr,
    )
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        # model = GPT(config)
        model = GPT(config, hparams=None, use_block_mask=False)
    fabric.print(
        f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.",
        file=sys.stderr,
    )
    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    model.eval()

    if compile:
        # torch._dynamo.config.automatic_dynamic_shapes = True
        # torch._inductor.config.triton.unique_kernel_names = True
        # torch._inductor.config.coordinate_descent_tuning = True
        # global next_token
        # next_token = torch.compile(next_token, mode="reduce-overhead")
        torch._dynamo.config.cache_size_limit = 256
        model = torch.compile(model)

    model = fabric.setup_module(model)

    t0 = time.perf_counter()
    load_checkpoint(fabric, model, checkpoint_path)

    # work around PyTorch issue https://github.com/pytorch/pytorch/issues/152162
    # which does not like the lazy initialization to be called in dynamo.
    # TODO: Happens with PyTorch 2.7+
    if (
        (_TORCH_EQUAL_2_7 or _TORCH_EQUAL_2_8)
        and (model._forward_module.__class__.__name__ == "OptimizedModule")
        and (
            model._forward_module._orig_mod.__class__.__name__
            == "FullyShardedDataParallel"
        )
    ):
        from torch.distributed.fsdp._runtime_utils import _root_pre_forward

        _root_pre_forward(
            model._forward_module._orig_mod, model._forward_module._orig_mod, [], {}
        )
        _root_pre_forward(
            model_teacher._forward_module._orig_mod,
            model_teacher._forward_module._orig_mod,
            [],
            {},
        )

    fabric.print(
        f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.",
        file=sys.stderr,
    )

    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(fabric.device)

    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        reset_model_kv_cache(
            model=model,
            batch_size=1,
            max_returned_tokens=max_returned_tokens,
            device=fabric.device,
            fabric=fabric,
        )
        y = generate_mtp(
            model=model,
            prompt=encoded,
            max_returned_tokens=max_returned_tokens,
            k_toks=k_toks,
            mask_id=mask_id,
            eos_id=tokenizer.eos_id,
            include_prompt=include_prompt,
            tokenizer=tokenizer,
            strategy=strategy,
        )
        t = time.perf_counter() - t0

        if include_prompt:
            y = y[prompt_length:]

        tokens_generated = y.size(0)

        fabric.print(f"\n\n#### Complete Output ####")
        fabric.print(f"\n\n#### BEGIN Prompt ####")
        fabric.print(f"{prompt}")
        fabric.print(f"########")
        fabric.print(f"tokenized as:\n{encoded.tolist()}")
        fabric.print(f"#### END Prompt ####")
        fabric.print(f"#### BEGIN Generation ####")
        fabric.print(f"{tokenizer.decode(y)}")
        fabric.print(f"########")
        fabric.print(f"tokenized as:\n{y.tolist()}")
        fabric.print(f"#### END Generation ####")
        fabric.print(
            f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec over {tokens_generated} tokens.",
            file=sys.stderr,
        )

    if fabric.device.type == "cuda":
        max_memory_allocated_per_gpu = (
            torch.cuda.max_memory_allocated(fabric.device) / 1024**3
        )
        max_memory_reserved_per_gpu = (
            torch.cuda.max_memory_reserved(fabric.device) / 1024**3
        )
        torch.cuda.reset_peak_memory_stats(fabric.device)
        fabric.print(
            f"Memory alloc/reserved: {max_memory_allocated_per_gpu:.02f} GB / {max_memory_reserved_per_gpu:.02f} GB",
            file=sys.stderr,
        )


if __name__ == "__main__":

    set_parsing_settings(
        config_read_mode_urls_enabled=True,
        docstring_parse_attribute_docstrings=True,
    )

    torch.set_float32_matmul_precision("high")

    CLI(main)
