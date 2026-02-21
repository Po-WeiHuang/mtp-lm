"""Mask Mod for Multi-Token_prediction a la https://arxiv.org/abs/2507.11851"""

import torch
from attn_gym import visualize_attention_scores


def interleaved_mtp_mask_mod_factory(prefix_length: int, K: int, offset: int = 0, bidirect_ss_attn: bool = False):
    # B is the total block size: P (prefix) + K (window)
    B = prefix_length + K
    # P is the length of the prefix in a standard block
    P = prefix_length

    # The offset affects the effective starting position for the block calculation
    Ofs = offset

    # single bit to use to toggle conditional bidirectional MTP attention
    BD_MTP = bidirect_ss_attn

    assert abs(Ofs) < P, "Offset magnitude must be less than prefix length P."

    def block_idx(pos):
        pos = torch.as_tensor(pos)
        
        # 1. Apply the offset:
        # A position 'pos' now behaves like 'pos - offset'.
        # Positions [0, offset - 1] will now map to negative indices,
        # effectively making them part of the *first* block prefix.
        pos_shifted = pos - Ofs
        
        # 2. Calculate the block remainder (off) and block index (win)
        # using the shifted position.
        # This repeats the [P, K] pattern starting from 'offset'.
        off = pos_shifted.remainder(B)
        win = torch.div(pos_shifted, B, rounding_mode='floor')
        
        # 3. Determine if the position is in the prefix (ntp) or a block (mtp).
        # A position is considered part of a prefix (-1) if its remainder
        # (after shifting) is less than P OR if the shifted position itself
        # is negative (which happens for positions [0, offset - 1]).
        is_prefix_after_shift = off < P
        is_before_offset = pos_shifted < 0
        
        # Positions before the offset are treated as part of the initial prefix (-1).
        # For positions >= offset, the standard interleaved logic applies.
        return torch.where(
            is_before_offset | is_prefix_after_shift, 
            torch.full_like(win, -1), # Maps to prefix
            win                      # Maps to block index
        )

    def mask_mod(b, h, q, k):
        q = torch.as_tensor(q); k = torch.as_tensor(k)

        qb = block_idx(q)
        kb = block_idx(k)

        causal = k <= q
        q_is_ntp = qb.eq(-1)
        k_is_ntp = kb.eq(-1)
        same_block = qb.eq(kb)

        # during bidirect, we also need to allow the last prefix q to look forward to the
        # mtp region that comes right after it, so we compute whether q+1 is *not* in prefix
        if BD_MTP:
            q_plus_1 = q + 1
            qb_plus_1 = block_idx(q_plus_1)
            q_plus_1_is_ntp = qb_plus_1.eq(-1)
            qb_plus_1_same_block = qb_plus_1.eq(kb)
        else:
            # particular values don't matter here since we & with the false BD_MTP later
            q_plus_1_is_ntp = torch.zeros_like(q_is_ntp, dtype=torch.bool)
            qb_plus_1_same_block = torch.zeros_like(same_block, dtype=torch.bool)


        # Allow N-to-P (prefix) attention:
        # Query in prefix & Key in prefix & causal
        # allow_ntp = q_is_ntp & k_is_ntp & causal
        
        # Allow N-to-P (prefix) attention but with lookahead option to complete bidirect MTP:
        allow_ntp = (q_is_ntp & k_is_ntp & causal) | (BD_MTP & (~q_plus_1_is_ntp) & (~k_is_ntp) & qb_plus_1_same_block)
        
        # Allow M-to-P (multi-block/standard) attention:
        # Query *not* in prefix & causal & (Key in prefix OR same block)
        # allow_mtp = (~q_is_ntp) & causal & (k_is_ntp | same_block)
        
        # Allow M-to-P (multi-block/standard) attention but with bidirect MTP option:
        allow_mtp = ((~q_is_ntp) & causal & (k_is_ntp | same_block)) | (BD_MTP & (~q_is_ntp) & (~k_is_ntp) & same_block)
        
        allow = allow_ntp | allow_mtp
        return allow

    return mask_mod

