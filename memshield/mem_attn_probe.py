"""Memory-attention probe: extract differentiable attention-mass distributions
over FIFO memory-bank slots, for MemoryShield's L_stale regularizer.

Background
----------
SAM2 computes cross-attention from the current frame's queries to a concatenated
memory tensor with layout:

    memory = [ cond-frame spatial tokens (HW) |
               FIFO slot_0 spatial tokens (HW) |
               FIFO slot_1 spatial tokens (HW) |
               ...
               FIFO slot_{N-1} spatial tokens (HW) |
               object-pointer tokens (num_obj_ptr_tokens) ]

The cross-attention is implemented with `F.scaled_dot_product_attention`, which
is a fused kernel that does NOT return attention weights. This module replaces
that call at forward time on selected `MemoryAttentionLayer.cross_attn_image`
modules with an explicit `softmax(q @ k^T / sqrt(d)) @ v` computation that
yields a differentiable attention-weight tensor of shape `[B, H, Nq, Nk]`.

The weight tensor is large (Nq·Nk ≈ 10^8 floats for 1024-res). To avoid OOM we
do NOT store the full weight tensor. Instead we reduce it on-the-fly inside
the patched forward to a 3-bin vector `P_u = [A^ins, A^recent, A^other]`
using pre-set foreground-query mask and per-key slot-provenance tag.

Usage
-----
    probe = MemAttnProbe(sam2_model.memory_attention)
    probe.set_targets(fg_query_mask_per_frame,   # dict: frame_idx -> [Nq] bool
                      slot_tag_per_frame)        # dict: frame_idx -> [Nk] long {0,1,2}
    with probe:
        outs = sam2_surrogate.forward_video(frames, gt_mask)
        # probe.P_u_by_frame[frame_idx][layer_idx] is a [3] differentiable tensor
        # use in L_stale = KL(Q || mean_u P_u[last_layer])

Design notes
------------
- Patches only the `cross_attn_image` module of each `MemoryAttentionLayer`
  (not self-attn, not SAM-head attention). Restores originals on __exit__.
- Captures `weights` INSIDE the forward and eagerly reduces to `P_u`. The
  reduction MUST stay in the autograd graph for L_stale gradients to flow.
- Frame tagging is keyed by a mutable frame-counter that the caller must
  advance between frames (see `advance_frame`). This mirrors SAM2's
  sequential-frame inference API.
- "Final memory-attention block" per the paper spec = last layer index.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MemAttnProbe:
    """Context manager that instruments SAM2's memory cross-attention.

    After `__enter__`, every call to the patched `cross_attn_image.forward`
    reduces its attention-weight tensor into a per-layer 3-bin `P_u` using
    the foreground-query mask + slot-provenance tag registered for the
    current frame. On `__exit__`, originals are restored.

    Args:
        memory_attention: the `MemoryAttention` module (has `.layers`).
        layer_indices: which layers to probe. Default = [last layer only],
            matching the paper spec "final memory-attention block".
        store_frames: if True, store P_u per frame (dict). If False, only
            store the most-recent frame's P_u. Default True.
    """

    def __init__(
        self,
        memory_attention: nn.Module,
        layer_indices: Optional[List[int]] = None,
        store_frames: bool = True,
    ):
        self.memory_attention = memory_attention
        all_layers = list(memory_attention.layers)
        if layer_indices is None:
            # Default: final layer only (paper spec).
            layer_indices = [len(all_layers) - 1]
        for i in layer_indices:
            if not (0 <= i < len(all_layers)):
                raise ValueError(
                    f"layer_index {i} out of range for {len(all_layers)} layers")
        self.layer_indices = layer_indices
        self.store_frames = store_frames

        # Per-frame state, set by caller before each SAM2 forward.
        self.current_frame_id: Optional[int] = None
        self.fg_mask_by_frame: Dict[int, Tensor] = {}
        self.slot_tag_by_frame: Dict[int, Tensor] = {}

        # Collected outputs.
        # P_u_by_frame[frame_id][layer_idx] = [3] differentiable tensor.
        self.P_u_by_frame: Dict[int, Dict[int, Tensor]] = {}

        # Backup of original forward bound methods (one per patched attn module).
        self._saved_forwards: Dict[int, callable] = {}
        # Keep attn-module references for restore.
        self._attn_modules: Dict[int, nn.Module] = {}

    # -- target registration ------------------------------------------------
    def set_frame(self, frame_id: int) -> None:
        """Advance to frame_id; subsequent forward uses its fg_mask/slot_tag."""
        self.current_frame_id = frame_id

    def set_targets(
        self,
        fg_mask_by_frame: Dict[int, Tensor],
        slot_tag_by_frame: Dict[int, Tensor],
    ) -> None:
        """Register the foreground-query mask and slot-provenance tag per frame.

        fg_mask_by_frame: frame_id -> [Nq] bool tensor on the SAME device as
            q. Nq is the number of spatial query tokens of the current frame
            (= H*W of the image-feature map; typically 64*64 for 1024 input).
        slot_tag_by_frame: frame_id -> [Nk] long tensor with values in
            {0: insert-bank, 1: recent-clean-bank, 2: other (cond+ptrs)}.
            Nk = total key count = HW*num_bank_slots + num_obj_ptr_tokens.
        """
        self.fg_mask_by_frame = fg_mask_by_frame
        self.slot_tag_by_frame = slot_tag_by_frame

    def reset(self) -> None:
        """Clear collected P_u state (call between optimization steps)."""
        self.P_u_by_frame = {}

    # -- context manager ----------------------------------------------------
    def __enter__(self) -> "MemAttnProbe":
        for layer_idx in self.layer_indices:
            layer = self.memory_attention.layers[layer_idx]
            attn = layer.cross_attn_image  # Attention or RoPEAttention
            self._attn_modules[layer_idx] = attn
            self._saved_forwards[layer_idx] = attn.forward

            # Build a patched forward bound to this attn instance + layer_idx.
            patched = self._make_patched_forward(layer_idx, attn)
            # Bind to the module instance (replacing the bound method).
            attn.forward = patched
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for layer_idx, saved in self._saved_forwards.items():
            self._attn_modules[layer_idx].forward = saved
        self._saved_forwards = {}
        self._attn_modules = {}

    # -- patched forward ----------------------------------------------------
    def _make_patched_forward(self, layer_idx: int, attn_module: nn.Module):
        """Return a patched forward(q, k, v, [num_k_exclude_rope=0]) for attn_module.

        Replicates `Attention.forward` / `RoPEAttention.forward` up to and
        including the q/k/v projections + head separation + RoPE (if
        applicable), then replaces the fused SDPA with explicit
        softmax(q k^T / sqrt(d)) @ v. The explicit softmax yields a
        differentiable weights tensor [B, H, Nq, Nk] which is reduced in
        place to a 3-bin P_u for the current frame using the registered
        fg_mask + slot_tag. The reduction stays in the autograd graph.
        """
        # Import lazily to avoid hard dependency on sam2 at module import time.
        from sam2.modeling.sam.transformer import RoPEAttention, apply_rotary_enc

        is_rope = isinstance(attn_module, RoPEAttention)
        probe_self = self

        if is_rope:
            def patched_forward(q_in: Tensor, k_in: Tensor, v_in: Tensor,
                                num_k_exclude_rope: int = 0) -> Tensor:
                # Projections
                q = attn_module.q_proj(q_in)
                k = attn_module.k_proj(k_in)
                v = attn_module.v_proj(v_in)
                # Heads: [B, H, N, d_head]
                q = attn_module._separate_heads(q, attn_module.num_heads)
                k = attn_module._separate_heads(k, attn_module.num_heads)
                v = attn_module._separate_heads(v, attn_module.num_heads)

                # RoPE
                w_sz = h_sz = math.sqrt(q.shape[-2])
                attn_module.freqs_cis = attn_module.freqs_cis.to(q.device)
                if attn_module.freqs_cis.shape[0] != q.shape[-2]:
                    attn_module.freqs_cis = attn_module.compute_cis(
                        end_x=w_sz, end_y=h_sz).to(q.device)
                if q.shape[-2] != k.shape[-2]:
                    assert attn_module.rope_k_repeat

                num_k_rope = k.size(-2) - num_k_exclude_rope
                q, k[:, :, :num_k_rope] = apply_rotary_enc(
                    q, k[:, :, :num_k_rope],
                    freqs_cis=attn_module.freqs_cis,
                    repeat_freqs_k=attn_module.rope_k_repeat,
                )
                # Explicit attention with differentiable weights.
                out = _explicit_attention_and_reduce(
                    q, k, v, layer_idx, probe_self,
                )
                out = attn_module._recombine_heads(out)
                out = attn_module.out_proj(out)
                return out
        else:
            def patched_forward(q_in: Tensor, k_in: Tensor, v_in: Tensor) -> Tensor:
                q = attn_module.q_proj(q_in)
                k = attn_module.k_proj(k_in)
                v = attn_module.v_proj(v_in)
                q = attn_module._separate_heads(q, attn_module.num_heads)
                k = attn_module._separate_heads(k, attn_module.num_heads)
                v = attn_module._separate_heads(v, attn_module.num_heads)
                out = _explicit_attention_and_reduce(
                    q, k, v, layer_idx, probe_self,
                )
                out = attn_module._recombine_heads(out)
                out = attn_module.out_proj(out)
                return out

        return patched_forward


def _explicit_attention_and_reduce(
    q: Tensor,  # [B, H, Nq, d_head]
    k: Tensor,  # [B, H, Nk, d_head]
    v: Tensor,  # [B, H, Nk, d_head]
    layer_idx: int,
    probe: MemAttnProbe,
) -> Tensor:
    """Explicit softmax attention + in-place 3-bin P_u reduction.

    Returns: out = weights @ v, shape [B, H, Nq, d_head].
    Side effect: populates probe.P_u_by_frame[frame_id][layer_idx] = [3] tensor.

    The [3] tensor sums attention mass over all heads and all foreground
    queries, normalized so the three bins sum to 1 (they partition the keys).
    """
    d_k = q.size(-1)
    scale = 1.0 / math.sqrt(d_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, Nq, Nk]
    weights = F.softmax(scores, dim=-1)  # [B, H, Nq, Nk]; differentiable

    frame_id = probe.current_frame_id
    if frame_id is not None:
        fg_mask = probe.fg_mask_by_frame.get(frame_id)
        slot_tag = probe.slot_tag_by_frame.get(frame_id)
        if fg_mask is not None and slot_tag is not None:
            # weights: [B, H, Nq, Nk]
            # Average over heads first.
            w_mean_h = weights.mean(dim=1)  # [B, Nq, Nk]
            # Restrict to foreground queries.
            fg_mask = fg_mask.to(w_mean_h.device)
            if fg_mask.dtype != torch.bool:
                fg_mask = fg_mask.bool()
            if fg_mask.shape[-1] != w_mean_h.shape[1]:
                raise ValueError(
                    f"fg_mask length {fg_mask.shape[-1]} != Nq {w_mean_h.shape[1]} "
                    f"at frame {frame_id} layer {layer_idx}")
            w_fg = w_mean_h[:, fg_mask, :]  # [B, Nq_fg, Nk]
            if w_fg.shape[1] == 0:
                # No foreground queries for this frame → skip recording.
                return torch.matmul(weights, v)
            w_fg_mean = w_fg.mean(dim=1)  # [B, Nk]; mean over fg queries
            slot_tag = slot_tag.to(w_fg_mean.device)
            if slot_tag.shape[-1] != w_fg_mean.shape[-1]:
                raise ValueError(
                    f"slot_tag length {slot_tag.shape[-1]} != Nk "
                    f"{w_fg_mean.shape[-1]} at frame {frame_id} layer {layer_idx}")
            # 3-bin accumulation; scatter-add over Nk by slot category.
            P = torch.zeros(3, device=w_fg_mean.device, dtype=w_fg_mean.dtype)
            for b_idx in range(3):
                mask = (slot_tag == b_idx)
                if mask.any():
                    # Sum attention mass in this bin across batch (mean over B).
                    P = P.clone()  # avoid in-place aliasing across bins
                    P[b_idx] = w_fg_mean[:, mask].sum(dim=-1).mean(dim=0)
            # Ensure P sums to ~1 (the three bins partition Nk). Don't renormalize
            # unless all three bins are defined; otherwise report as-is.
            if probe.store_frames:
                probe.P_u_by_frame.setdefault(frame_id, {})[layer_idx] = P
            else:
                probe.P_u_by_frame = {frame_id: {layer_idx: P}}

    out = torch.matmul(weights, v)  # [B, H, Nq, d_head]
    return out


# ───────────────────────── Slot-tag builders ────────────────────────────────


def build_slot_tag(
    num_bank_slots: int,
    hw_per_slot: int,
    num_obj_ptr_tokens: int,
    is_insert_per_slot: List[bool],
    cond_is_first: bool = True,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Build a [Nk] long tensor tagging each key token as 0=ins / 1=recent / 2=other.

    Matches SAM2's memory layout (see sam2_base.py _prepare_memory_conditioned_features):
        [cond-frame HW tokens | FIFO slot_0 HW | ... | FIFO slot_{N-1} HW | obj_ptr tokens]
    where cond-frame is at bank-slot index 0 iff `cond_is_first`, otherwise
    all N slots are FIFO.

    Args:
        num_bank_slots: total bank slot count including conditioning if present.
            Equals len(is_insert_per_slot).
        hw_per_slot: HW spatial tokens per bank slot (same for all slots).
        num_obj_ptr_tokens: number of obj_ptr tokens appended at end.
        is_insert_per_slot: bool list of length num_bank_slots; True iff the
            slot's source frame was one of our inserted frames. The
            conditioning slot should be False (never our insert).
        cond_is_first: whether slot 0 is the privileged conditioning frame.
            If True, slot 0 is tagged as 'other' (bin 2) regardless of
            is_insert_per_slot[0]; this is defensive because the user should
            pass False for conditioning anyway.

    Returns:
        [Nk] long tensor where Nk = num_bank_slots * hw_per_slot + num_obj_ptr_tokens.
    """
    if len(is_insert_per_slot) != num_bank_slots:
        raise ValueError(
            f"is_insert_per_slot len {len(is_insert_per_slot)} != num_bank_slots "
            f"{num_bank_slots}")
    total = num_bank_slots * hw_per_slot + num_obj_ptr_tokens
    tag = torch.full((total,), 2, dtype=torch.long,
                     device=device)  # default = other
    for s in range(num_bank_slots):
        start = s * hw_per_slot
        end = start + hw_per_slot
        if cond_is_first and s == 0:
            # conditioning stays as 'other'
            tag[start:end] = 2
        elif is_insert_per_slot[s]:
            tag[start:end] = 0  # insert
        else:
            tag[start:end] = 1  # recent-clean
    # obj_ptr tokens at the end are already 2 (other) from the default fill.
    return tag


def build_fg_query_mask(
    true_region_mask: Tensor,  # [H, W] bool or {0,1}
    q_token_hw: tuple,         # (H_q, W_q) of the query token grid
) -> Tensor:
    """Downsample a true-region mask at image resolution to the query-token grid.

    Args:
        true_region_mask: [H, W] on any device; foreground region `C_u` after
            erode/flow-warp.
        q_token_hw: (H_q, W_q) of the memory-attention query feature map.

    Returns:
        [H_q*W_q] bool tensor, flattened in row-major order matching SAM2's
        `.flatten(2).permute(2, 0, 1)` convention on feature maps.
    """
    if true_region_mask.dim() != 2:
        raise ValueError(
            f"true_region_mask must be 2D [H,W], got shape {true_region_mask.shape}")
    H_q, W_q = q_token_hw
    m = true_region_mask.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    m_down = F.interpolate(m, size=(H_q, W_q), mode="nearest").squeeze()
    fg = (m_down > 0.5).view(-1)
    return fg
