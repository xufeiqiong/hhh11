import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Union, List

from flash_attn.layers.rotary import apply_rotary_emb 

import time

def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def apply_rotary_pos_emb_flashatt(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    q_embed = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
    k_embed = apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)
    return q_embed, k_embed

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    
    cos = cos.to(q.device)
    sin = sin.to(q.device)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def token_merging(image_embeds, keep_indices, scaling=1):
    """
    Merges non-retained tokens with their nearest retained tokens based on cosine similarity.

    Args:
        image_embeds (Tensor): Tensor of shape (N, D) where N is the number of tokens, and D is the feature dimension.
        keep_indices (Tensor): Tensor of shape (T, ), where T is the number of retained tokens.

    Returns:
        merged_features (Tensor): Tensor of shape (T, D) where T is the number of retained tokens
                                and D is the feature dimension. The merged features are
                                the average of the retained token and the non-retained tokens.
    """
    N, D = image_embeds.shape
    T = len(keep_indices)
    
    keep_index_mask = torch.zeros(N, dtype=torch.bool, device=image_embeds.device)
    keep_index_mask[keep_indices] = True
    
    retained_tokens = image_embeds[keep_index_mask, :] # [T, D]
    non_retained_tokens = image_embeds[~keep_index_mask, :] # [N - T, D]
    # print(retained_tokens.shape, non_retained_tokens.shape)
    
    cosine_sim = F.cosine_similarity(non_retained_tokens.unsqueeze(1), retained_tokens.unsqueeze(0), dim=2)
    nearest_token_indices = cosine_sim.argmax(dim=1) # [N - T]
    # print(nearest_token_indices)
    
    merged_features = torch.zeros_like(retained_tokens) # [T, D]
    merged_features += retained_tokens * scaling
    
    expanded_indices = nearest_token_indices # [N - T]
    merged_features.scatter_add_(0, expanded_indices.unsqueeze(-1).expand(-1, D), non_retained_tokens)
    
    merge_count = torch.zeros(T, device=image_embeds.device, dtype=torch.int) # [T]
    merge_count.scatter_add_(0, expanded_indices, torch.ones_like(expanded_indices, dtype=merge_count.dtype))
    merged_features /= (scaling + merge_count.unsqueeze(1))
    
    return merged_features

def attn_dpp_select(
    proj_win: torch.Tensor,   # [M, D]  该图的窗口级特征（已是 merger 后的 token）
    qual_win: torch.Tensor,   # [M]     该图的窗口级“质量分”（来自深层注意力）
    T: int,                   # 目标保留窗口数
    tau_attn: float = 0.7,
    gamma_q: float = 0.5,
    cand_ratio: float = 2.0,
    neg_small: float = -1e4,  # fp16-safe sentinel
):
    """
    线性核 DPP（贪心近似）：Z = sqrt(q) * normalize(proj)
    输入为“单张图”的窗口级特征/质量分；返回 keep_idx（窗口下标，长度 T）
    """
    M = proj_win.size(0)
    if T >= M:
        return torch.arange(M, device=proj_win.device)

    # 质量分 q：softmax( A / tau )，再幂次 gamma，归一到均值=1
    q = torch.softmax(qual_win / max(tau_attn, 1e-2), dim=0)
    q = q / (q.mean() + 1e-12)
    if gamma_q != 1.0:
        q = torch.pow(q, gamma_q)

    # 候选子集（加速）
    if cand_ratio is None or cand_ratio <= 1.0:
        cand = torch.arange(M, device=proj_win.device)
    else:
        Mc = min(M, int(max(T, round(T * float(cand_ratio)))))
        cand = torch.topk(q, k=Mc, dim=0).indices  # [Mc]

    # 线性核 DPP：Z = sqrt(qc) * normalize(Xc)
    Xc = F.normalize(proj_win[cand], dim=-1).float()   # [Mc, D] fp32
    qc = q[cand].float()                               # [Mc]
    Z  = Xc * torch.sqrt(qc).unsqueeze(1)              # [Mc, D]

    Mc = Z.size(0)
    if Mc <= T:
        keep_local = torch.arange(Mc, device=Z.device)
    else:
        # 贪心 Gram-Schmidt 残差近似 log-det
        res2 = (Z * Z).sum(dim=1).clone()              # ||Z_i||^2
        used = torch.zeros(Mc, dtype=torch.bool, device=Z.device)
        keep = []

        # first
        i0 = torch.argmax(res2).item()
        keep.append(i0); used[i0] = True
        u = Z[i0] / torch.sqrt(res2[i0] + 1e-12)
        proj_u = Z @ u
        res2 = torch.clamp(res2 - proj_u * proj_u, min=0.0)

        # next T-1
        for _ in range(T - 1):
            res2_masked = res2.masked_fill(used, neg_small)
            i = torch.argmax(res2_masked).item()
            if res2_masked[i] <= 0:
                rest = (~used).nonzero(as_tuple=False).squeeze(1)
                if rest.numel() > 0:
                    k = min(T - len(keep), rest.numel())
                    extra = torch.topk(qc[rest], k=k, dim=0).indices
                    keep.extend(rest[extra].tolist())
                break
            keep.append(i); used[i] = True
            u = Z[i] / torch.sqrt(res2[i] + 1e-12)
            proj_u = Z @ u
            res2 = torch.clamp(res2 - proj_u * proj_u, min=0.0)

        keep_local = torch.tensor(keep, device=Z.device)
        if keep_local.numel() > T:  # 极少数超选，按 qc 截断
            order = torch.argsort(-qc[keep_local])[:T]
            keep_local = keep_local[order]

    keep_idx = cand[keep_local]  # map 回原窗口下标
    # 若不足 T，再用 q 补齐
    if keep_idx.numel() < T:
        need = T - keep_idx.numel()
        mask = torch.ones(M, dtype=torch.bool, device=proj_win.device)
        mask[keep_idx] = False
        add = torch.topk(q.masked_fill(~mask, neg_small), k=need).indices
        keep_idx = torch.cat([keep_idx, add], dim=0)

    # 超出 T 的保护（基本不会发生）
    if keep_idx.numel() > T:
        order = torch.argsort(-q[keep_idx])[:T]
        keep_idx = keep_idx[order]
    return keep_idx.sort().values
    
    
    
    
    
