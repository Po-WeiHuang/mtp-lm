# Originally from https://github.com/yxuansu/SimCTG/blob/main/simctg/evaluation.py
# and used in Contrastive Decoding https://github.com/XiangLi1999/ContrastiveDecoding
# ported in 2023 for use in https://github.com/jwkirchenbauer/lm-watermarking/blob/main/watermark_reliability_release/metrics/repetition_diversity.py
# Again ported in 2025 and edited by Gemini to use pretokenized input vs strings/whitespace before other minor changes by yours truly.

import math

def eval_text(token_list, ngram):
    """
    Evaluates unique and total n-grams in a list of tokens (integers).
    token_list: a list of integers representing tokens
    ngram: the size of the n-gram (e.g., 1, 2, 3, 4)
    """
    if ngram == 1:
        # 1-gram (unigram) case
        total_num = len(token_list)
        ngram_set = set(token_list)
        return len(ngram_set), total_num

    # N-gram case for N > 1
    start_idx, end_idx = 0, ngram
    total_num = 0
    ngram_set = set() # Set to store unique n-grams (as tuples of integers)

    while end_idx <= len(token_list):
        one_ngram_list = token_list[start_idx:end_idx]
        assert len(one_ngram_list) == ngram
        
        # Convert the list of tokens to a tuple to be used in a set (tuples are hashable)
        one_ngram = tuple(one_ngram_list)
        
        total_num += 1
        ngram_set.add(one_ngram)
        start_idx += 1
        end_idx += 1

    return len(ngram_set), total_num


def eval_one_instance(token_list, ngram_list):
    """
    Computes n-gram stats for an instance.
    token_list: a list of integers representing tokens
    ngram_list: a list of n-gram sizes to evaluate (e.g., [1, 2, 3, 4])
    """
    res_dict = {}
    
    # Pre-calculate unique token set, which is necessary for 1-gram
    unique_token_set = set(token_list)
    
    for n in ngram_list:
        if n == 1:
            # 1-gram is special: unique is len(set), total is len(list)
            res_dict[1] = {"unique": len(unique_token_set), "total": len(token_list)}
        else:
            n_unique, n_total = eval_text(token_list, n)
            res_dict[n] = {"unique": n_unique, "total": n_total}

    return res_dict, unique_token_set


def measure_repetition_and_diversity(input_tokens):
    """
    input_tokens: a list of integers representing the tokens of the text
    """
    ngram_list = [1, 2, 3, 4] 
    pred_res_dict = {}
    for n in ngram_list:
        pred_res_dict[n] = {"unique": 0, "total": 0}

    pred_unique_token_set = set()
    
    one_pred_res_dict, one_pred_uni_token_set = eval_one_instance(input_tokens, ngram_list)

    # unique token set
    pred_unique_token_set = pred_unique_token_set.union(one_pred_uni_token_set)
    
    # ngram statistic
    for n in ngram_list:
        pred_res_dict[n]["unique"] += one_pred_res_dict[n]["unique"]
        pred_res_dict[n]["total"] += one_pred_res_dict[n]["total"]

    # Calculate repetition: 1 - (unique_ngrams / total_ngrams)
    # Handle the case where total_ngrams is 0 (i.e., sequence is too short for n-gram)
    
    # Unique and Repetition rates for N-grams, plus summary "Diversity" measure
    diversity = 1.0 # base case
    unique_1,unique_2,unique_3,unique_4 = 1.0,1.0,1.0,1.0
    repetition_1,repetition_2,repetition_3,repetition_4 = 0.0,0.0,0.0,0.0
    if pred_res_dict[1]["total"] > 0:
        unique_1 = (pred_res_dict[1]["unique"] / pred_res_dict[1]["total"])
        diversity *= unique_1
        repetition_1 = 1.0 - unique_1
    if pred_res_dict[2]["total"] > 0:
        unique_2 = (pred_res_dict[2]["unique"] / pred_res_dict[2]["total"])
        diversity *= unique_2
        repetition_2 = 1.0 - unique_2
    if pred_res_dict[3]["total"] > 0:
        unique_3 = (pred_res_dict[3]["unique"] / pred_res_dict[3]["total"])
        diversity *= unique_3
        repetition_3 = 1.0 - unique_3
    if pred_res_dict[4]["total"] > 0:
        unique_4 = (pred_res_dict[4]["unique"] / pred_res_dict[4]["total"])
        diversity *= unique_4
        repetition_4 = 1.0 - unique_4
    
    # Calculate log-diversity, a questionably principled metric from https://github.com/jwkirchenbauer/lm-watermarking
    log_diversity = -math.log(max(1 - diversity, math.exp(-20)))

    # return a dictionary with the ngram repetition levels and diversity
    return {
        "unique_1": unique_1, 
        "unique_2": unique_2, 
        "unique_3": unique_3, 
        "unique_4": unique_4, 
        "repetition_1": repetition_1, 
        "repetition_2": repetition_2,
        "repetition_3": repetition_3,
        "repetition_4": repetition_4,
        "diversity": diversity,
        "log_diversity": log_diversity,
    }

dummy_rep_div_result = {
    "unique_1": float("nan"), 
    "unique_2": float("nan"), 
    "unique_3": float("nan"), 
    "unique_4": float("nan"), 
    "repetition_1": float("nan"), 
    "repetition_2": float("nan"),
    "repetition_3": float("nan"),
    "repetition_4": float("nan"),
    "diversity": float("nan"),
    "log_diversity": float("nan"),
}