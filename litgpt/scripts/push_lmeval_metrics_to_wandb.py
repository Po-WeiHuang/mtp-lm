import time
import json
import os
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List

from transformers import AutoTokenizer

import pandas as pd
import numpy as np

import wandb
from jsonargparse import CLI, set_parsing_settings


def flatten_results(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Recursively flattens nested dicts into key1/key2: value format."""
    items = {}
    for key, value in data.items():
        new_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            items.update(flatten_results(value, new_key))
        else:
            items[new_key] = value
    return items


def match_tok_id_prefix_to_resp(
    token_ids: List[int], response: str, tokenizer, min_match_len=None, do_retok=False
) -> int:
    """
    Matches the token ID prefix to the response string using the provided tokenizer.
    Returns the length of the matched token IDs.
    """
    if tokenizer is None:
        raise ValueError(
            "Tokenizer must be provided for matching token IDs to response."
        )

    response_tokens = tokenizer.encode(response, add_special_tokens=False)
    match_length = 0

    for tok_id in token_ids:
        if (
            match_length < len(response_tokens)
            and tok_id == response_tokens[match_length]
        ):
            match_length += 1
        else:
            break

    matched_tok_prefix = token_ids[:match_length]

    if (min_match_len is not None and match_length < min_match_len) or do_retok:
        # print(f"Warning: Matched token prefix length is very short ({match_length}). Possible mismatch.")
        # try bouncing the token ids off of string space and re call the match function
        retokd_token_ids = tokenizer.encode(
            tokenizer.decode(token_ids, skip_special_tokens=False),
            add_special_tokens=False,
        )
        # breakpoint()
        return match_tok_id_prefix_to_resp(retokd_token_ids, response, tokenizer)

    return match_length, matched_tok_prefix


def trim_eff_k_toks_to_matched_prefix(
    eff_k_toks: List[int], match_length: int
) -> List[int]:

    # effective_k_tokens is a list of nums of tok's predicted at each step
    # so find the palce where cumulative sum reaches match_length using numpy
    eff_k_toks_array = np.array(eff_k_toks)
    cumulative_sum = np.cumsum(eff_k_toks_array)
    trim_index = np.searchsorted(cumulative_sum, match_length, side="right")
    return eff_k_toks[:trim_index]


def process_samples(
    samples_path: Path,
    task_name: str = None,
    mtp_rows_only: bool = True,
    tokenizer=None,
    min_match_len: int = None,
    do_retok: bool = False,
) -> Dict[str, Any]:
    """
    Processes samples to count total vs processed rows.
    """
    total_raw_rows = 0
    processed_rows_count = 0

    valid_rows = []

    with open(samples_path, "r") as f:
        for line in f:
            total_raw_rows += 1
            row = json.loads(line)

            if mtp_rows_only and not row.get("mtp_results"):
                continue

            valid_rows.append(row)
            processed_rows_count += 1

    df = pd.DataFrame.from_records(valid_rows)
    if len(df) == 0:
        warnings.warn("No valid rows found in samples after filtering.")
        return {}
    # check uniqueness of the doc_id column
    if "doc_id" in df.columns:
        if df["doc_id"].nunique() != len(df):
            warnings.warn("doc_id column is not unique in the final samples data.")

    # flatted the mtp_results column into multiple columns for each of its keys
    mtp_subcols = (
        df.iloc[0]["mtp_results"][0][0].keys() if processed_rows_count > 0 else []
    )
    for subcol in mtp_subcols:
        df[f"mtp_{subcol}"] = df["mtp_results"].apply(
            lambda x: (
                x[0][0].get(subcol) if x and isinstance(x, list) and x[0] else None
            )
        )

    # now add any more derived cols as necessary.
    # add the matched token ids prefix column using the fn above
    if tokenizer is not None:

        def compute_matched_tok_prefix(row):
            token_ids = row.get("mtp_token_ids", [])
            resp = row.get("resps", [[]])
            if not (
                isinstance(resp, list)
                and isinstance(resp[0], list)
                and len(resp) == 1
                and len(resp[0]) == 1
            ):
                # if it's just duplicates we let it slide, since we still dont really know what this nesting is for
                if len(set(resp[0])) == 1:
                    resp = [[resp[0][0]]]
            assert (
                isinstance(resp, list)
                and isinstance(resp[0], list)
                and len(resp) == 1
                and len(resp[0]) == 1
            ), "Expected resp to be a nested list with a single element."
            response = resp[0][0]
            match_length, matched_tok_prefix = match_tok_id_prefix_to_resp(
                token_ids,
                response,
                tokenizer,
                min_match_len=min_match_len,
                do_retok=do_retok,
            )
            return match_length, matched_tok_prefix

        def compute_effective_k_trimmed(row):
            eff_k_toks = row.get("mtp_effective_k_values", [])
            match_length = row.get("mtp_matched_tok_prefix_length", 0)
            trimmed_eff_k_toks = trim_eff_k_toks_to_matched_prefix(
                eff_k_toks, match_length
            )
            return trimmed_eff_k_toks

        df[["mtp_matched_tok_prefix_length", "mtp_matched_tok_prefix"]] = df.apply(
            compute_matched_tok_prefix, axis=1, result_type="expand"
        )
        df["mtp_effective_k_tokens_trimmed"] = df.apply(
            compute_effective_k_trimmed, axis=1
        )
        df["mtp_avg_effective_k_trimmed"] = df["mtp_effective_k_tokens_trimmed"].apply(
            lambda x: np.mean(x) if x else 0
        )

    # compute mean and std and quantiles for each of the mtp subcolumns we want
    reduction_cols = [
        "mtp_toks_gend_incl_prefillplus1",
        "mtp_avg_effective_k",
        "mtp_matched_tok_prefix_length",
        "mtp_avg_effective_k_trimmed",
        "mtp_num_fwd_evals",
        "mtp_t_prefill",
        "mtp_t_gen",
        "mtp_tps",
    ]
    # red_dict = {}
    # red_dict.update(df[reduction_cols].agg(['mean', 'std']).to_dict())
    # quantiles = [0.1,0.25, 0.5, 0.75, 0.9]
    # red_dict.update(df[reduction_cols].quantile(quantiles).to_dict())
    # # add min and max
    # red_dict.update(df[reduction_cols].agg(['min', 'max']).to_dict())
    # red_df = pd.DataFrame.from_dict(red_dict)
    # do this correctly
    red_dict = {}
    red_dict["total_raw_rows"] = total_raw_rows
    red_dict["processed_rows_count"] = processed_rows_count

    for col in reduction_cols:
        red_dict[f"{col}/mean"] = df[col].mean()
        red_dict[f"{col}/median"] = df[col].median()
        red_dict[f"{col}/std"] = df[col].std()
        red_dict[f"{col}/min"] = df[col].min()
        red_dict[f"{col}/max"] = df[col].max()
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            red_dict[f"{col}/quantile_{q}"] = df[col].quantile(q)

    red_df = pd.DataFrame.from_dict(red_dict, orient="index", columns=["value"])
    # print(red_df)
    task_header = task_name + "/" if task_name is not None else ""
    return {f"{task_header}samples/{k}": v for k, v in red_dict.items()}


def parse_eval_results(
    dry_run: bool = True,
    run_dir: Path = None,
    wandb_args: str = "entity=tomg-group-umd,project=singleshot-evals,run_name=debug-lmeval-pusher,step=0,tags=debug+stepwise+manual_pusher",
    hf_tokenizer_path=None,
    mtp_rows_only: bool = True,
    min_match_len: int = None,
    do_retok: bool = True,
    samples_only: bool = False,
):
    """
    Parses the latest lm-eval-harness results and logs them to WandB.

    Args:
        run_dir: Directory containing the evaluation subdirectories.
        wandb_args: Comma-separated WandB arguments (entity, project, run_name, etc.).
        dry_run: If True, only prints metrics without logging to WandB.
        mtp_rows_only: If True, only processes rows with MTP results.
    """
    assert run_dir is not None, "run_dir must be specified"

    wandb_args_dict = {}
    for arg in wandb_args.split(","):
        key, value = arg.split("=")
        if "+" in value:
            value = value.split("+")
        elif key == "step":
            value = int(value)
        else:
            value = value.strip()
        wandb_args_dict[key.strip()] = value

    result_files = sorted(list(run_dir.rglob("results_*.json")))
    sample_files = sorted(list(run_dir.rglob("samples_*.jsonl")))

    if not result_files:
        print(f"No result files found in {run_dir}")
        return

    latest_results_path = result_files[-1]
    latest_samples_path = sample_files[-1] if sample_files else None

    print(f"Processing latest results: {latest_results_path.name}")

    with open(latest_results_path, "r") as f:
        raw_data = json.load(f)

    results_payload = raw_data.pop("results", {})
    metrics_to_log = flatten_results(results_payload)
    # assert len(results_payload.keys()) == 1, "Expected results payload to contain exactly one key (the task name)."
    # task_name = list(results_payload.keys())[0]

    # lets handle multiple tasks now

    if len(results_payload.keys()) == 1:
        task_names = [list(results_payload.keys())[0]]
        latest_samples_paths = [latest_samples_path]
    else:
        task_names = list(results_payload.keys())
        latest_samples_paths = []
        for tname in task_names:
            candidate_sample_files = [
                f for f in sample_files if f.name.startswith(f"samples_{tname}_")
            ]
            if candidate_sample_files:
                # special case where the task name is a base for a series of tasks
                # then we want to cat the whole set of samples into a new file and append that
                if len(candidate_sample_files) > 1 and (len(
                    candidate_sample_files
                )+1) == len(results_payload.keys()):
                    # concatenate all files into a new file
                    concat_samples_path = run_dir / f"samples_{tname}_concat.jsonl"
                    print(
                        f"SPECIAL CASE: Concatenating samples for task {tname} into {concat_samples_path.name}"
                    )
                    # rm if it exists
                    if concat_samples_path.exists():
                        os.remove(concat_samples_path)
                    with open(concat_samples_path, "w") as outfile:
                        for fname in sorted(candidate_sample_files):
                            with open(fname) as infile:
                                for line in infile:
                                    outfile.write(line)
                    latest_samples_paths.append(concat_samples_path)
                else:  # latest
                    latest_samples_paths.append(sorted(candidate_sample_files)[-1])
            else:
                latest_samples_paths.append(None)
    assert len(task_names) == len(
        latest_samples_paths
    ), "Mismatch in number of tasks and sample files."

    if hf_tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_path)
        # Example usage of tokenizer if needed
        # e.g., decoding or encoding operations can be performed here

    for task_name, latest_samples_path in zip(task_names, latest_samples_paths):
        print(f"Processing task: {task_name}")

        if latest_samples_path:
            print(f"Found corresponding samples: {latest_samples_path.name}")

            if samples_only:
                print("Samples only flag is set. Logging only sample metrics.")
                metrics_to_log = {}
            sample_metrics = process_samples(
                latest_samples_path,
                task_name=task_name,
                mtp_rows_only=mtp_rows_only,
                tokenizer=tokenizer,
                min_match_len=min_match_len,
                do_retok=do_retok,
            )
            metrics_to_log.update(sample_metrics)

    if dry_run:
        print("\n--- DRY RUN: Metrics that would be logged ---")
        for k, v in metrics_to_log.items():
            print(f"{k}: {v}")
        print("-------------------------------------------\n")
        # print(f"WandB Args: {json.dumps(wandb_args_dict, indent=2)}")
        # print(f"Wandb 'Config': {json.dumps(raw_data, indent=2)}")
        print("-------------------------------------------\n")
    else:
        step = wandb_args_dict.pop("step", None)
        run_name = wandb_args_dict["name"]
        run = wandb.init(
            **wandb_args_dict,
            config=raw_data,  # remaining data is treated as config
        )
        metrics_to_log["run_name"] = (
            run_name  # this should be a better behaved field in the gui
        )
        try:
            metrics_to_log["config"]["gen_kwargs"]["strategy"] = str(
                metrics_to_log["config"]["gen_kwargs"]["strategy"]
            )
        except:
            pass  # might be buggy
        run.log(metrics_to_log, step=step)
        run.finish()

        # # wait for a couple minutes
        # print("Waiting for 2 minutes to ensure WandB logging is complete...")
        # time.sleep(120)
        # print("Exiting.")


if __name__ == "__main__":
    set_parsing_settings(docstring_parse_attribute_docstrings=True)
    CLI(parse_eval_results)
