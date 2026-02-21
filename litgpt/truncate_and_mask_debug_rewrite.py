import torch

def truncate_and_mask_select_with_offset(input_ids=None, target_ids=None, k_toks=None, mask_id=None, truncation_length=None, mask_region_ct=1, offset=0, verbose=False):
   
    B, og_slen = input_ids.shape
    S = truncation_length

    assert S == og_slen, "For this debug rewrite, truncation_length must equal input sequence length."

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

    # special roll slide thing for offset
    if offset < 0:
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

    # repeat the masks in the batch dim to match the new prepared inputs and targets
    mask_id_mask = mask_id_mask.unsqueeze(0).repeat(B, 1)
    pred_pos_mask = pred_pos_mask.unsqueeze(0).repeat(B, 1)

    # overwrite the mask positions in the input using mask_id_mask
    prepared_input_ids[mask_id_mask] = mask_id

    return prepared_input_ids, prepared_target_ids, mask_id_mask, pred_pos_mask

# ============================================================================
# --- Test and Verification ---
# ============================================================================

# --- Example Settings ---
mask_id = -1
# Input sequence [0, 1, ..., 31]
input_ids = torch.tensor([[i for i in range(32)]]) + 100
# Target sequence [1, 2, ..., 32]
target_ids = input_ids.clone() + 1

k_toks = 3
truncation_length = 32
mask_region_ct = 4

# --- 1. Run the new function with offset=0 (Baseline) ---
print("--- Running new function (offset=0) ---")
prep_input_ids_new_0, prep_target_ids_new_0, mask_id_mask_0, pred_pos_mask_0 = truncate_and_mask_select_with_offset(
# prep_input_ids_new_0, prep_target_ids_new_0 = truncate_and_mask_select_with_offset(
    input_ids=input_ids, target_ids=target_ids, k_toks=k_toks, mask_id=mask_id,
    truncation_length=truncation_length, mask_region_ct=mask_region_ct, offset=0,
    verbose=True
)
print("Prepared Input IDs (Offset=0):")
print(prep_input_ids_new_0)
print("\nPrepared Target IDs (Offset=0):")
print(prep_target_ids_new_0)
print("\nMask ID Mask (Offset=0):")
print(mask_id_mask_0)
print("\nPred Pos Mask (Offset=0):")
print(pred_pos_mask_0)
print("\n" + "="*80)


# --- 2. Run with your negative offsets: -1 and -3 ---
print("--- Running new function (offset=-1) ---")
prep_input_ids_new_neg1, prep_target_ids_new_neg1, mask_id_mask_neg1, pred_pos_mask_neg1 = truncate_and_mask_select_with_offset(
    input_ids=input_ids, target_ids=target_ids, k_toks=k_toks, mask_id=mask_id,
    truncation_length=truncation_length, mask_region_ct=mask_region_ct, offset=-1,
    verbose=True
)
print("Prepared Input IDs (Offset=-1):")
print(prep_input_ids_new_neg1)
print("\nPrepared Target IDs (Offset=-1):")
print(prep_target_ids_new_neg1)
print("\nMask ID Mask (Offset=-1):")
print(mask_id_mask_neg1)
print("\nPred Pos Mask (Offset=-1):")
print(pred_pos_mask_neg1)
print("\n" + "="*80)

print("--- Running new function (offset=-3) ---")
prep_input_ids_new_neg3, prep_target_ids_new_neg3, mask_id_mask_neg3, pred_pos_mask_neg3 = truncate_and_mask_select_with_offset(
    input_ids=input_ids,
    target_ids=target_ids, k_toks=k_toks, mask_id=mask_id,
    truncation_length=truncation_length, mask_region_ct=mask_region_ct, offset=-3,
    verbose=True
)
print("Prepared Input IDs (Offset=-3):")
print(prep_input_ids_new_neg3)
print("\nPrepared Target IDs (Offset=-3):")
print(prep_target_ids_new_neg3)
print("\nMask ID Mask (Offset=-3):")
print(mask_id_mask_neg3)
print("\nPred Pos Mask (Offset=-3):")
print(pred_pos_mask_neg3)
print("\n" + "="*80)

# --- 2. Run with your positive offsets: 1 and 3 ---
print("--- Running new function (offset=1) ---")
prep_input_ids_new_pos1, prep_target_ids_new_pos1, mask_id_mask_pos1, pred_pos_mask_pos1 = truncate_and_mask_select_with_offset(
    input_ids=input_ids, target_ids=target_ids, k_toks=k_toks, mask_id=mask_id,
    truncation_length=truncation_length, mask_region_ct=mask_region_ct, offset=1,
    verbose=True
)
print("Prepared Input IDs (Offset=1):")
print(prep_input_ids_new_pos1)
print("\nPrepared Target IDs (Offset=1):")
print(prep_target_ids_new_pos1)
print("\nMask ID Mask (Offset=1):")
print(mask_id_mask_pos1)
print("\nPred Pos Mask (Offset=1):")
print(pred_pos_mask_pos1)
print("\n" + "="*80)

print("--- Running new function (offset=3) ---")
prep_input_ids_new_pos3, prep_target_ids_new_pos3, mask_id_mask_pos3, pred_pos_mask_pos3 = truncate_and_mask_select_with_offset(
    input_ids=input_ids, target_ids=target_ids, k_toks=k_toks, mask_id=mask_id,
    truncation_length=truncation_length, mask_region_ct=mask_region_ct, offset  =3,
    verbose=True
)
print("Prepared Input IDs (Offset=3):")
print(prep_input_ids_new_pos3)
print("\nPrepared Target IDs (Offset=3):")
print(prep_target_ids_new_pos3)
print("\nMask ID Mask (Offset=3):")
print(mask_id_mask_pos3)
print("\nPred Pos Mask (Offset=3):")
print(pred_pos_mask_pos3)
print("\n" + "="*80)

print("--- Running new function (offset=2) ---")
prep_input_ids_new_pos2, prep_target_ids_new_pos2, mask_id_mask_pos2, pred_pos_mask_pos2 = truncate_and_mask_select_with_offset(
    input_ids=input_ids, target_ids=target_ids, k_toks=k_toks, mask_id=mask_id,
    truncation_length=truncation_length, mask_region_ct=mask_region_ct, offset=2,
    verbose=True
)
print("Prepared Input IDs (Offset=2):")
print(prep_input_ids_new_pos2)
print("\nPrepared Target IDs (Offset=2):")
print(prep_target_ids_new_pos2)
print("\nMask ID Mask (Offset=2):")
print(mask_id_mask_pos2)
print("\nPred Pos Mask (Offset=2):")
print(pred_pos_mask_pos2)
print("\n" + "="*80)    