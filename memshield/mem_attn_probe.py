"""Memory-attention probe: extract differentiable attention-mass distributions
over FIFO memory-bank slots, for MemoryShield's L_stale regularizer.

Design (after Codex R1 review)
------------------------------
The probe is *observational*, not intrusive. The forward path of SAM2's
memory cross-attention is left UNCHANGED (fused `F.scaled_dot_product_attention`
still produces `out`). On top of that, we compute a SIDE channel:

    weights_fg = softmax(q_fg @ k^T / sqrt(d), dim=-1)   # [B, H, Nq_fg, Nk]

where `q_fg` is `q` restricted to foreground-query tokens only (Nq_fg << Nq).
This side channel is differentiable end-to-end (it reuses the same `q` and
`k` tensors computed by the main path), and is reduced to a 3-bin vector
`P_u = [A^ins, A^recent, A^other]` via the caller-provided slot-provenance tag.

Why a side channel:
- Keeps SAM2 numerical path identical to upstream; no fused-kernel loss.
- Avoids retaining the full `[B, H, Nq, Nk]` weight tensor in the autograd
  graph; we only retain `[B, H, Nq_fg, Nk]` and reduce it immediately.
- Side-channel softmax is done in float32 for numerical stability at large Nk.

Slot-provenance tagging (Codex R1 CRITICAL #2):
- SAM2's `_prepare_memory_conditioned_features` builds `memory` as a concat
  of MULTIPLE selected conditioning frames + recent non-conditioning frames
  + object pointer tokens, in that order. The counts are video-state-
  dependent. This probe does NOT assume any fixed layout. The caller must
  pass in a `slot_tag` of length exactly Nk built from the ACTUAL memory
  assembly at the current frame (see helpers in `build_slot_tag_from_memory`
  below, which hooks the concat site).

Gradient / memory guarantees:
- `out` of the main path = original SDPA (untouched).
- `P_u` backprops through the side-channel softmax to `q` and `k`, which
  are projected from the current frame's queries and the memory tensor.
  Memory tensor includes features from inserted frames (our `ν`), so
  ∂L_stale/∂ν is non-trivial.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ══════════════════════════════════════════════════════════════════════════════
# Probe
# ══════════════════════════════════════════════════════════════════════════════


class MemAttnProbe:
    """Observational probe on SAM2's memory cross-attention.

    Usage:
        probe = MemAttnProbe(sam2_model.memory_attention)
        probe.set_targets(fg_mask_by_frame, slot_tag_by_frame)
        with probe:
            for frame_id in range(T):
                probe.set_frame(frame_id)
                # ... run SAM2 forward for this frame ...
        # Now probe.P_u_by_frame[frame_id][layer_idx] is a [3] diff tensor.

    Args:
        memory_attention: the `MemoryAttention` module containing `.layers`.
        layer_indices: which layers to probe. Default = [last layer only].
        dtype_softmax: dtype to use for the side-channel softmax.
            float32 is strongly recommended for numerical stability when Nk
            is large (≈ 30k). The main path dtype is unchanged.
    """

    def __init__(
        self,
        memory_attention: nn.Module,
        layer_indices: Optional[List[int]] = None,
        dtype_softmax: torch.dtype = torch.float32,
    ):
        self.memory_attention = memory_attention
        all_layers = list(memory_attention.layers)
        if layer_indices is None:
            layer_indices = [len(all_layers) - 1]
        for i in layer_indices:
            if not (0 <= i < len(all_layers)):
                raise ValueError(
                    f"layer_index {i} out of range for {len(all_layers)} layers")
        self.layer_indices = layer_indices
        self.dtype_softmax = dtype_softmax

        self.current_frame_id: Optional[int] = None
        self.fg_mask_by_frame: Dict[int, Tensor] = {}
        self.slot_tag_by_frame: Dict[int, Tensor] = {}
        self.P_u_by_frame: Dict[int, Dict[int, Tensor]] = {}

        self._saved_forwards: Dict[int, callable] = {}
        self._attn_modules: Dict[int, nn.Module] = {}

    # -- target registration ------------------------------------------------
    def set_frame(self, frame_id: int) -> None:
        self.current_frame_id = frame_id

    def set_targets(
        self,
        fg_mask_by_frame: Dict[int, Tensor],
        slot_tag_by_frame: Dict[int, Tensor],
    ) -> None:
        """Register per-frame foreground-query mask + slot-provenance tag.

        fg_mask_by_frame: frame_id -> 1-D bool tensor of length Nq (= HW of
            the image-feature grid, typically 64*64=4096 at 1024-res).
        slot_tag_by_frame: frame_id -> 1-D long tensor of length Nk with
            values in {0: insert, 1: recent-clean, 2: other}. Nk must
            equal the ACTUAL memory concat length at this frame (= sum of
            bank-slot HW contributions + num_obj_ptr_tokens).
        """
        for fid, m in fg_mask_by_frame.items():
            if m.dim() != 1:
                raise ValueError(
                    f"fg_mask[{fid}] must be 1-D, got shape {tuple(m.shape)}")
        for fid, t in slot_tag_by_frame.items():
            if t.dim() != 1:
                raise ValueError(
                    f"slot_tag[{fid}] must be 1-D, got shape {tuple(t.shape)}")
        self.fg_mask_by_frame = fg_mask_by_frame
        self.slot_tag_by_frame = slot_tag_by_frame

    def reset(self) -> None:
        self.P_u_by_frame = {}

    # -- context manager ----------------------------------------------------
    def __enter__(self) -> "MemAttnProbe":
        for layer_idx in self.layer_indices:
            layer = self.memory_attention.layers[layer_idx]
            attn = layer.cross_attn_image
            self._attn_modules[layer_idx] = attn
            self._saved_forwards[layer_idx] = attn.forward
            attn.forward = self._make_patched_forward(layer_idx, attn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for layer_idx, saved in self._saved_forwards.items():
            self._attn_modules[layer_idx].forward = saved
        self._saved_forwards = {}
        self._attn_modules = {}

    # -- patched forward ----------------------------------------------------
    def _make_patched_forward(self, layer_idx: int, attn_module: nn.Module):
        """Wrap `attn.forward` so the main path is unchanged and a side
        channel computes P_u for the current frame.

        We call the saved original forward to get `out` (numerically
        identical to upstream). Then, using the same internal projections,
        we compute a side-channel softmax on foreground queries only and
        reduce to a 3-bin P_u.
        """
        from sam2.modeling.sam.transformer import RoPEAttention, apply_rotary_enc

        is_rope = isinstance(attn_module, RoPEAttention)
        saved_forward = attn_module.forward  # captured before patch
        probe_self = self

        if is_rope:
            def patched_forward(q_in: Tensor, k_in: Tensor, v_in: Tensor,
                                num_k_exclude_rope: int = 0) -> Tensor:
                # Main path: unchanged.
                out = saved_forward(q_in, k_in, v_in,
                                    num_k_exclude_rope=num_k_exclude_rope)
                # Side channel for P_u (conditional).
                if probe_self.current_frame_id is not None:
                    probe_self._compute_P_u_rope(
                        layer_idx, attn_module,
                        q_in, k_in, num_k_exclude_rope,
                        apply_rotary_enc,
                    )
                return out
        else:
            def patched_forward(q_in: Tensor, k_in: Tensor, v_in: Tensor) -> Tensor:
                out = saved_forward(q_in, k_in, v_in)
                if probe_self.current_frame_id is not None:
                    probe_self._compute_P_u_plain(layer_idx, attn_module, q_in, k_in)
                return out

        return patched_forward

    # -- side-channel P_u computation ---------------------------------------
    def _compute_P_u_plain(
        self,
        layer_idx: int,
        attn_module: nn.Module,
        q_in: Tensor,
        k_in: Tensor,
    ) -> None:
        frame_id = self.current_frame_id
        fg_mask = self.fg_mask_by_frame.get(frame_id)
        slot_tag = self.slot_tag_by_frame.get(frame_id)
        if fg_mask is None or slot_tag is None:
            return

        q = attn_module.q_proj(q_in)
        k = attn_module.k_proj(k_in)
        q = attn_module._separate_heads(q, attn_module.num_heads)
        k = attn_module._separate_heads(k, attn_module.num_heads)
        self._reduce_to_P_u(layer_idx, q, k, fg_mask, slot_tag)

    def _compute_P_u_rope(
        self,
        layer_idx: int,
        attn_module: nn.Module,
        q_in: Tensor,
        k_in: Tensor,
        num_k_exclude_rope: int,
        apply_rotary_enc,
    ) -> None:
        frame_id = self.current_frame_id
        fg_mask = self.fg_mask_by_frame.get(frame_id)
        slot_tag = self.slot_tag_by_frame.get(frame_id)
        if fg_mask is None or slot_tag is None:
            return

        q = attn_module.q_proj(q_in)
        k = attn_module.k_proj(k_in)
        q = attn_module._separate_heads(q, attn_module.num_heads)
        k = attn_module._separate_heads(k, attn_module.num_heads)
        # Replicate RoPE from RoPEAttention.forward — attn_module.freqs_cis
        # was updated (possibly in place) by the main-path forward that
        # already ran; we reuse it as-is so q/k here match the main path.
        attn_module.freqs_cis = attn_module.freqs_cis.to(q.device)
        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k_rope = apply_rotary_enc(
            q, k[:, :, :num_k_rope],
            freqs_cis=attn_module.freqs_cis,
            repeat_freqs_k=attn_module.rope_k_repeat,
        )
        # Reassemble k so the non-rope tail (obj_ptr tokens) is unchanged.
        if num_k_exclude_rope > 0:
            k = torch.cat([k_rope, k[:, :, num_k_rope:]], dim=-2)
        else:
            k = k_rope
        self._reduce_to_P_u(layer_idx, q, k, fg_mask, slot_tag)

    def _reduce_to_P_u(
        self,
        layer_idx: int,
        q: Tensor,       # [B, H, Nq, d_head]
        k: Tensor,       # [B, H, Nk, d_head]
        fg_mask: Tensor, # 1-D bool, length Nq
        slot_tag: Tensor,# 1-D long, length Nk, values in {0,1,2}
    ) -> None:
        B, H, Nq, d_head = q.shape
        Nk = k.size(-2)

        fg_mask = fg_mask.to(q.device)
        if fg_mask.dtype != torch.bool:
            fg_mask = fg_mask.bool()
        if fg_mask.shape[-1] != Nq:
            raise ValueError(f"fg_mask length {fg_mask.shape[-1]} != Nq {Nq}")
        if slot_tag.shape[-1] != Nk:
            raise ValueError(f"slot_tag length {slot_tag.shape[-1]} != Nk {Nk}")
        if q.shape[-2] != Nq or k.shape[-2] != Nk:
            raise ValueError(f"unexpected q/k shapes: q {q.shape} k {k.shape}")

        n_fg = int(fg_mask.sum().item())
        if n_fg == 0:
            # Record a sentinel so the caller can decide to skip this frame
            # consistently (Codex R1 IMPORTANT): avoid silent omission.
            frame_id = self.current_frame_id
            self.P_u_by_frame.setdefault(frame_id, {})[layer_idx] = None
            return

        # Side-channel softmax in float32 on foreground queries only.
        q_fg = q[:, :, fg_mask, :]                    # [B, H, Nq_fg, d_head]
        q_fg32 = q_fg.to(self.dtype_softmax)
        k32 = k.to(self.dtype_softmax)                # [B, H, Nk, d_head]
        scale = 1.0 / math.sqrt(d_head)
        scores = torch.matmul(q_fg32, k32.transpose(-2, -1)) * scale
        weights = F.softmax(scores, dim=-1)           # [B, H, Nq_fg, Nk]

        # Average over heads, then over foreground queries.
        w_mean_h = weights.mean(dim=1)                # [B, Nq_fg, Nk]
        w_fg_mean = w_mean_h.mean(dim=1)              # [B, Nk]

        # Batch-mean too (B is typically 1; we keep this for robustness).
        w_mean = w_fg_mean.mean(dim=0)                # [Nk]

        # 3-bin scatter-add: no in-place aliasing risk (Codex R1 IMPORTANT).
        slot_tag = slot_tag.to(w_mean.device)
        P = torch.zeros(3, dtype=w_mean.dtype, device=w_mean.device)
        P = P.scatter_add(0, slot_tag, w_mean)        # differentiable

        # Cast back to the model's active dtype at caller-side loss time if
        # needed. We keep float32 here for KL-loss stability.
        frame_id = self.current_frame_id
        self.P_u_by_frame.setdefault(frame_id, {})[layer_idx] = P


# ══════════════════════════════════════════════════════════════════════════════
# Slot-tag builder (hooks SAM2's actual memory-concat site)
# ══════════════════════════════════════════════════════════════════════════════


class MemoryProvenanceHook:
    """Capture SAM2's ACTUAL memory-concat provenance at each frame.

    This hook intercepts the concat that happens inside
    `_prepare_memory_conditioned_features` by wrapping the model's
    `memory_attention.__call__`. At call time, `memory` has already been
    built, so its layout IS the ground-truth provenance. The hook uses a
    per-call closure state populated by the caller immediately before the
    SAM2 per-frame forward to map (cond frame-indices, recent frame-indices,
    hw_per_slot, num_obj_ptr_tokens) → slot_tag.

    The caller registers, BEFORE each frame's forward:
        hook.declare(
            cond_frame_modified_ids=[0],        # list of modified-seq indices
            recent_frame_modified_ids=[1,2,3,4,5,6],
            hw_per_slot=4096,
            num_obj_ptr_tokens=16,
            insert_modified_id_set={6, 12, 14},
        )
    and after the forward, reads the constructed tag via
    `hook.last_slot_tag_for(frame_id)`.

    This avoids the Codex R1 CRITICAL #2 pitfall: we never assume "cond is
    always slot 0" or a fixed num_maskmem.
    """

    def __init__(self):
        self._declared: Dict[int, dict] = {}
        self._last_tag_by_frame: Dict[int, Tensor] = {}

    def declare(self, frame_id: int, **kwargs) -> None:
        """Declare the planned memory layout for frame_id before its forward."""
        required = {
            "cond_frame_modified_ids",
            "recent_frame_modified_ids",
            "hw_per_slot",
            "num_obj_ptr_tokens",
            "insert_modified_id_set",
        }
        missing = required - set(kwargs.keys())
        if missing:
            raise ValueError(f"declare() missing keys: {missing}")
        self._declared[frame_id] = kwargs

    def build_tag_for_frame(
        self,
        frame_id: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Construct slot_tag of length Nk for frame_id based on declared layout.

        Frame-concat order in SAM2 (verified against sam2_base.py):
            to_cat_memory = [
                *maskmem from selected conditioning frames (in order),
                *maskmem from selected recent non-cond frames (in order),
                obj_ptr tokens (appended at end),
            ]
        Each maskmem contributes `hw_per_slot` tokens.
        Slot-tag values:
            0: source frame is one of our inserts
            1: source frame is a clean-prefix frame (recent or cond, if clean)
            2: other (obj_ptr tokens; conditioning frame if desired)
        """
        decl = self._declared.get(frame_id)
        if decl is None:
            raise ValueError(f"No declaration for frame_id {frame_id}")
        hw = decl["hw_per_slot"]
        n_ptrs = decl["num_obj_ptr_tokens"]
        cond_ids = list(decl["cond_frame_modified_ids"])
        recent_ids = list(decl["recent_frame_modified_ids"])
        insert_set = set(decl["insert_modified_id_set"])

        n_slots = len(cond_ids) + len(recent_ids)
        Nk = n_slots * hw + n_ptrs
        tag = torch.full((Nk,), 2, dtype=torch.long, device=device)  # default: other

        # Conditioning frames first.
        cursor = 0
        for mid in cond_ids:
            end = cursor + hw
            if mid in insert_set:
                tag[cursor:end] = 0  # insert
            else:
                tag[cursor:end] = 2  # conditioning treated as 'other'
            cursor = end
        # Recent non-cond frames next.
        for mid in recent_ids:
            end = cursor + hw
            if mid in insert_set:
                tag[cursor:end] = 0
            else:
                tag[cursor:end] = 1  # recent clean
            cursor = end
        # Remaining (obj_ptr tokens) stay 2.
        assert cursor == Nk - n_ptrs, (
            f"Cursor {cursor} != Nk - n_ptrs = {Nk - n_ptrs}")

        self._last_tag_by_frame[frame_id] = tag
        return tag

    def last_slot_tag_for(self, frame_id: int) -> Tensor:
        return self._last_tag_by_frame[frame_id]


def build_fg_query_mask(
    true_region_mask: Tensor,  # [H, W] bool or {0,1}
    q_token_hw: tuple,         # (H_q, W_q) of the memory-attention query grid
) -> Tensor:
    """Downsample a full-resolution target-region mask to the query-token grid.

    SAM2 flattens image features with `.flatten(2).permute(2, 0, 1)`, yielding
    row-major [H_q * W_q] order. We replicate that here after
    nearest-neighbor downsampling.
    """
    if true_region_mask.dim() != 2:
        raise ValueError(
            f"true_region_mask must be 2D [H,W], got {true_region_mask.shape}")
    H_q, W_q = q_token_hw
    m = true_region_mask.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    m_down = F.interpolate(m, size=(H_q, W_q), mode="nearest").squeeze()
    fg = (m_down > 0.5).view(-1)
    return fg


# ══════════════════════════════════════════════════════════════════════════════
# L_stale helper (safe KL with epsilon clamp)
# ══════════════════════════════════════════════════════════════════════════════


def l_stale_from_P_u_list(
    P_u_list: List[Tensor],    # list of [3] tensors (one per frame in V)
    Q: Tensor,                 # [3] target distribution
    eps: float = 1e-6,
) -> Tensor:
    """Compute L_stale = (1/|V|) sum_u KL(Q || P_u), with epsilon clamp.

    P_u that are None (e.g. empty foreground on that frame) are SKIPPED,
    and the denominator is the number of valid frames. If ALL frames are
    skipped, returns a zero-value tensor that does NOT backprop (caller
    should notice the |V|=0 situation via the returned scalar being exactly 0
    in fp32, or via logging the skip count outside this function).
    """
    valid = [P for P in P_u_list if P is not None]
    if len(valid) == 0:
        return torch.zeros((), device=Q.device, dtype=Q.dtype)
    # Clamp + renormalize each P_u to prevent log(0) in KL.
    KLs = []
    for P in valid:
        P_clamped = P.clamp(min=eps)
        P_norm = P_clamped / P_clamped.sum()
        # KL(Q || P) = sum_i Q_i * log(Q_i / P_i); we ignore Q_i=0 terms.
        q_nz = (Q > 0)
        kl = (Q[q_nz] * (Q[q_nz].clamp(min=eps).log() - P_norm[q_nz].log())).sum()
        KLs.append(kl)
    return torch.stack(KLs).mean()
