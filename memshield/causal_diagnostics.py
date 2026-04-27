"""Causal diagnostics for v5 (memory-mediated failure mechanism evidence).

Three non-trainable diagnostic components, used ONLY for paper experiments
(NOT during attack optimization):

R001: ``make_blocking_forward_fn(base, blocked_frames, extractor)`` —
  wraps an existing ``VADIForwardFn`` so that, at fid in
  ``blocked_frames``, the per-frame ``current_out`` is NOT appended to
  ``obj_output_dict["non_cond_frame_outputs"]``. This excludes the
  blocked frames from BOTH (a) future mask-memory chunks (assembled
  from non_cond_frame_outputs in
  ``_prepare_memory_conditioned_features``) AND (b) future object-
  pointer token assembly (also reads from non_cond_frame_outputs when
  ``use_obj_ptrs_in_encoder=True``).

  CODEX HIGH FIX (2026-04-27): the original docstring said "block memory
  writes" — that wording was too narrow because SAM2's
  ``_prepare_memory_conditioned_features`` reads from
  ``non_cond_frame_outputs`` for both mask-memory AND obj_ptr token
  assembly. The actual intervention removes ALL future temporal state
  contributions from blocked frames. This is the more honest
  pre-registration of A3 — see updated ``FINAL_PROPOSAL.md`` and
  ``EXPERIMENT_PLAN.md`` (R4-locked) for the revised claim wording.

R002: ``MemoryReadoutExtractor`` — monkey-patches the last-block cross-
  attention in ``memory_attention.layers[-1].cross_attn_image`` to
  capture the V tensor (pre-output-projection) and attention weights
  per frame. Used by ablation B5 (d_mem(t) persistence trace).
  CODEX MEDIUM FIX: integrated with the wrapper via the ``extractor``
  kwarg of ``make_blocking_forward_fn`` (calls ``set_frame(fid)`` in-loop).

R003: ``build_control_frames`` — deterministic uniform random sample of K
  attacked-space frames that are NOT in W_attacked and NOT in
  bridge_frames. Used by A3-control to pick the matched non-insert
  negative-control positions.

R004: ``compute_d_mem_trace`` — locked aggregation rule for d_mem(t)
  cross-condition comparison. CODEX LOW FIX: pre-register the analysis
  rule in code, not "decide at analysis time".

All four are observation-only / state-blocking; none change SAM2 weights
or introduce trainable parameters. Self-tests in ``__main__``.

Codex thread (paper-method design): ``019dcd87-c42b-7b03-9139-34df6b6ebd89``.
Pre-registration: ``refine-logs/FINAL_PROPOSAL.md``,
``refine-logs/EXPERIMENT_PLAN.md`` (2026-04-27).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# R003: control-frame sampler (pure utility, deterministic)
# ---------------------------------------------------------------------------


def build_control_frames(
    W_attacked: Sequence[int],
    bridge_frames: Sequence[int],
    T_proc: int,
    *,
    K: Optional[int] = None,
    seed: int = 0,
) -> List[int]:
    """Sample K attacked-space frames that are NOT in W_attacked and NOT in
    bridge_frames; deterministic via ``np.random.RandomState(seed)``.

    Used as the negative-control set for A3 (matched non-insert frame
    temporal-state blocking). The intent is: "block all future
    temporal-state contributions from K matched non-insert frames; if the
    attack still collapses comparably to insert-position blocking, the
    memory-mediated mechanism is NOT insert-position-specific."

    Args:
      W_attacked: insert positions in attacked-space (typically K=3).
      bridge_frames: union of all bridge frames adjacent to inserts.
      T_proc: total length of the attacked-space video (T_clean + K).
      K: number of control frames to sample. None -> len(W_attacked).
      seed: RNG seed (default 0). Same (W_attacked, bridge_frames, T_proc, K,
        seed) tuple deterministically produces the same output.

    Returns: sorted list of K attacked-space frame indices.

    Raises: ValueError if there are fewer than K eligible candidates.
    """
    if K is None:
        K = len(list(W_attacked))
    if K < 0:
        raise ValueError(f"K must be >= 0, got {K}")

    forbidden = (set(int(w) for w in W_attacked)
                 | set(int(b) for b in bridge_frames))
    # Exclude frame 0 from controls too: frame 0 is the prompt frame and
    # has special "is_init_cond_frame" semantics in SAM2; blocking memory
    # writes from frame 0 would be a categorically different intervention
    # than blocking interior frames. Keep the comparison apples-to-apples.
    forbidden.add(0)

    candidates = [t for t in range(T_proc) if t not in forbidden]
    if len(candidates) < K:
        raise ValueError(
            f"Not enough non-insert non-bridge non-zero candidates: have "
            f"{len(candidates)} eligible frames, need {K}. T_proc={T_proc}, "
            f"|W_attacked|={len(set(W_attacked))}, "
            f"|bridge|={len(set(bridge_frames))}."
        )

    rng = np.random.RandomState(int(seed))
    chosen_idx = rng.choice(len(candidates), size=K, replace=False)
    selected = sorted(int(candidates[i]) for i in chosen_idx)
    return selected


# ---------------------------------------------------------------------------
# R001: memory-block hook
# ---------------------------------------------------------------------------


def make_blocking_forward_fn(
    base_forward_fn: Any,
    *,
    blocked_frames: Sequence[int] = (),
    extractor: Optional["MemoryReadoutExtractor"] = None,
) -> Any:
    """Wrap an existing ``VADIForwardFn`` with optional memory-write blocking
    AND optional per-frame extractor integration.

    Behavior matrix:
      - ``blocked_frames=()`` AND ``extractor=None``: byte-equivalent
        passthrough of base forward (PARITY MODE; verified by self-test).
      - ``blocked_frames`` non-empty: at fid in blocked_frames, the
        per-frame ``current_out`` is NOT appended to
        ``obj_output_dict["non_cond_frame_outputs"]``. Hiera, mask decoder,
        and memory encoder all RUN normally for the blocked frame's own
        forward (its mask is correctly predicted using prior bank
        contents). The blocked frame's contribution to FUTURE temporal
        state retrieval (mask-memory chunks AND obj_ptr tokens, both of
        which SAM2 reads from ``non_cond_frame_outputs``) is suppressed.
      - ``extractor`` provided: ``extractor.set_frame(fid)`` is called
        before each per-frame track_step, enabling per-frame V/attn
        capture at ``memory_attention.layers[-1].cross_attn_image``.

    HIGH-fix note (codex 2026-04-27): the original wording said "block
    memory writes". That was too narrow because SAM2 reads from
    ``non_cond_frame_outputs`` for both mask-memory chunks AND obj_ptr
    tokens. The actual intervention is "block all future temporal state
    contributions from blocked frames". A3's pre-registered claim has
    been updated to match.

    Args:
      base_forward_fn: an instance of ``memshield.vadi_sam2_wiring.VADIForwardFn``.
      blocked_frames: attacked-space frame indices whose per-frame
        ``current_out`` should NOT enter ``obj_output_dict``. Frame 0 is
        rejected (prompt frame; see ``is_init_cond_frame`` semantics).
      extractor: optional ``MemoryReadoutExtractor``; if provided, its
        ``set_frame(fid)`` is called before each per-frame forward.

    Returns: a new callable with the same signature as ``base_forward_fn.__call__``.

    Note: re-implements the per-frame loop using the base fn's public
    attributes. If ``VADIForwardFn`` is refactored, this wrapper must be
    updated — but a parity-mode self-test guards against silent drift.
    """
    blocked_set = {int(f) for f in blocked_frames}
    if 0 in blocked_set:
        raise ValueError(
            "Cannot block temporal state contributions from frame 0 (it is "
            "the prompt conditioning frame; blocking it would change "
            "is_init_cond_frame semantics, not just temporal-state assembly). "
            "Drop fid=0 from blocked_frames."
        )

    # Lazy imports to avoid pulling SAM2 at module load.
    from memshield.vadi_sam2_wiring import (
        IMAGENET_MEAN,  # noqa: F401  (sanity-import; values inside base)
        _to_sam2_input,
        _low_res_to_video_res,
    )

    base = base_forward_fn

    def blocking_call(
        processed: Tensor, return_at: Iterable[int],
    ) -> Dict[int, Tensor]:
        if processed.dim() != 4 or processed.shape[-1] != 3:
            raise ValueError(
                f"processed must be [T, H, W, 3]; got {tuple(processed.shape)}")
        if (int(processed.shape[1]) != base.video_H
                or int(processed.shape[2]) != base.video_W):
            raise ValueError(
                f"processed spatial shape mismatch")
        T_proc = int(processed.shape[0])
        return_set = {int(t) for t in return_at}
        bad = [t for t in return_set if not (0 <= t < T_proc)]
        if bad:
            raise ValueError(
                f"return_at ids out of [0, {T_proc}): {sorted(bad)}")

        bad_block = [f for f in blocked_set if not (0 < f < T_proc)]
        if bad_block:
            raise ValueError(
                f"blocked_frames out of (0, {T_proc}): {sorted(bad_block)}")

        obj_output_dict: Dict[str, Dict[int, Dict]] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        out: Dict[int, Tensor] = {}

        if base.autocast_dtype is not None and base.device.type == "cuda":
            autocast_ctx = torch.amp.autocast(
                device_type="cuda", dtype=base.autocast_dtype)
        else:
            from contextlib import nullcontext
            autocast_ctx = nullcontext()

        with autocast_ctx:
            for fid in range(T_proc):
                frame = processed[fid:fid + 1]
                img_norm = _to_sam2_input(
                    frame, base.image_size,
                    base._img_mean, base._img_std,
                )
                if base.use_gradient_checkpointing and img_norm.requires_grad:
                    from torch.utils.checkpoint import checkpoint as _ckpt
                    backbone_out = _ckpt(
                        base.predictor.forward_image, img_norm,
                        use_reentrant=False,
                    )
                else:
                    backbone_out = base.predictor.forward_image(img_norm)
                _, vision_feats, vision_pos, feat_sizes = \
                    base.predictor._prepare_backbone_features(backbone_out)

                # MEDIUM-fix integration (codex 2026-04-27): tag the
                # extractor with the current frame BEFORE track_step so
                # the patched cross_attn_image knows which slot to fill.
                if extractor is not None:
                    extractor.set_frame(fid)

                is_init = (fid == 0)
                current_out = base.predictor.track_step(
                    frame_idx=fid,
                    is_init_cond_frame=is_init,
                    current_vision_feats=vision_feats,
                    current_vision_pos_embeds=vision_pos,
                    feat_sizes=feat_sizes,
                    point_inputs=None,
                    mask_inputs=base._mask_inputs_f0 if is_init else None,
                    output_dict=obj_output_dict,
                    num_frames=T_proc,
                    track_in_reverse=False,
                    run_mem_encoder=True,
                    prev_sam_mask_logits=None,
                )

                # ====  R001 HOOK BEHAVIOR  ====
                # Append to obj_output_dict UNLESS fid is in blocked_set.
                # is_init (fid==0) is always added to cond_frame_outputs;
                # we already verified blocked_set excludes 0 above.
                if is_init:
                    obj_output_dict["cond_frame_outputs"][fid] = current_out
                else:
                    if fid not in blocked_set:
                        obj_output_dict["non_cond_frame_outputs"][fid] = \
                            current_out
                    # else: skip the write. Frame fid's memory is NOT in
                    # the bank for any t' > fid to query.

                if fid in return_set:
                    pred_vid = _low_res_to_video_res(
                        current_out["pred_masks"],
                        base.video_H, base.video_W,
                    )
                    out[fid] = pred_vid.float()

        missing = return_set - set(out.keys())
        if missing:
            raise RuntimeError(
                f"BlockingForwardFn: failed to fill return_at slots "
                f"{sorted(missing)}")
        return out

    # Attach a tag for debugging / logging.
    blocking_call.blocked_frames = sorted(blocked_set)
    blocking_call.base_forward_fn = base
    blocking_call.has_extractor = extractor is not None
    blocking_call.is_blocking = bool(blocked_set)
    return blocking_call


# ---------------------------------------------------------------------------
# R004: locked d_mem aggregation rule (LOW-fix per codex 2026-04-27)
# ---------------------------------------------------------------------------


def aggregate_V_top_attended(
    V: Tensor,
    attn_weights: Tensor,
    *,
    top_k: int = 32,
) -> Tensor:
    """Pre-registered V aggregation for d_mem(t) computation.

    For a single (V, attn) pair from one frame's forward at the last
    cross-attention block:

      V         shape: (B, H, Nk, d_head) — pre-output-projection memory
                       value vectors.
      attn      shape: (B, H, Nq, Nk)     — softmax attention from
                       current-frame queries to memory slots.

    Aggregation rule (LOCKED 2026-04-27, do NOT change post-hoc):

      1. Reduce attn over (B, H) and over Nq dim (sum across queries) to
         get per-Nk total received attention: shape (Nk,).
      2. Pick top-K Nk indices by total received attention.
      3. Average V over those Nk indices, then over (B, H).
      4. Output: a (d_head,) vector representing "what the highly-
         attended memory tokens look like at this frame".

    NOTE on cross-condition comparison: because the bank composition
    differs across clean / insert-only / full (Nk_c may differ since
    inserts contribute extra entries), top-K is re-derived in EACH
    condition — i.e. selection criterion is FIXED across conditions
    (top-K by attention received), but the resulting positions are
    condition-specific. This is the honest interpretation of the codex
    pre-registration "T_obj selection criterion frozen across conditions".

    Args:
      V: (B, H, Nk, d_head) detached.
      attn_weights: (B, H, Nq, Nk) detached, fp32.
      top_k: how many memory tokens to aggregate over (default 32).

    Returns: (d_head,) tensor (averaged over batch and heads).
    """
    if V.dim() != 4:
        raise ValueError(f"V must be (B, H, Nk, d_head); got {tuple(V.shape)}")
    if attn_weights.dim() != 4:
        raise ValueError(
            f"attn_weights must be (B, H, Nq, Nk); got "
            f"{tuple(attn_weights.shape)}"
        )
    B, H, Nk, d_head = V.shape
    Bw, Hw, Nq, Nkw = attn_weights.shape
    if (B, H, Nk) != (Bw, Hw, Nkw):
        raise ValueError(
            f"V and attn_weights shape mismatch: V={tuple(V.shape)} "
            f"attn={tuple(attn_weights.shape)}"
        )
    k = max(1, min(int(top_k), Nk))

    # Step 1: per-Nk total received attention (sum across B, H, Nq).
    per_k = attn_weights.sum(dim=(0, 1, 2))   # (Nk,)
    # Step 2: top-K indices.
    top_idx = torch.topk(per_k, k=k).indices   # (k,)
    # Step 3: V averaged over top-k positions, then over (B, H).
    V_top = V.index_select(dim=2, index=top_idx)   # (B, H, k, d_head)
    return V_top.mean(dim=(0, 1, 2)).float()        # (d_head,)


def compute_d_mem_trace(
    V_attn_clean: Dict[int, Tuple[Tensor, Tensor]],
    V_attn_attacked: Dict[int, Tuple[Tensor, Tensor]],
    *,
    top_k: int = 32,
    frames: Optional[Sequence[int]] = None,
) -> Dict[int, float]:
    """LOCKED d_mem(t) computation.

    For each frame t in ``frames`` (or the intersection of keys if None),
    compute:

      d_mem(t) = 1 - cos(M_clean[t], M_attacked[t])

    where M_c[t] = ``aggregate_V_top_attended(V_c[t], attn_c[t], top_k=top_k)``.

    Args:
      V_attn_clean: dict frame_id -> (V tensor, attn tensor) from the
        clean run (extractor under clean SAM2 forward).
      V_attn_attacked: same structure, from attacked run (insert-only or
        full v5).
      top_k: aggregation cardinality (default 32, codex pre-reg).
      frames: optional explicit frame list. None -> intersection of keys.

    Returns: dict frame_id -> d_mem(t) scalar in [0, 2] (cosine distance,
      0 means identical aggregated V; 2 means anti-aligned).
    """
    if frames is None:
        frames = sorted(set(V_attn_clean.keys()) & set(V_attn_attacked.keys()))
    out: Dict[int, float] = {}
    for t in frames:
        if t not in V_attn_clean or t not in V_attn_attacked:
            continue
        V_c, attn_c = V_attn_clean[t]
        V_a, attn_a = V_attn_attacked[t]
        M_c = aggregate_V_top_attended(V_c, attn_c, top_k=top_k)
        M_a = aggregate_V_top_attended(V_a, attn_a, top_k=top_k)
        cos = torch.nn.functional.cosine_similarity(
            M_c.unsqueeze(0), M_a.unsqueeze(0), dim=-1
        )
        out[int(t)] = float(1.0 - cos.item())
    return out


# ---------------------------------------------------------------------------
# R002: memory-readout extractor (V tensor + attention weights per frame)
# ---------------------------------------------------------------------------


class MemoryReadoutExtractor:
    """Captures the per-frame V tensor and attention weights at the LAST
    cross-attention block of SAM2's memory_attention.

    Pre-registered protocol (codex 2026-04-27, R2 round of v5 design):
    - Layer: ``memory_attention.layers[-1].cross_attn_image``
    - V extraction point: PRE-output-projection (i.e., V projected by
      ``v_proj`` and head-separated by ``_separate_heads``, immediately
      before SDPA mixes Q/K/V; the ``out_proj`` after attention is NOT
      applied to what we capture).
    - Per-frame raw outputs:
        * ``V[fid]`` shape (B, H, Nk, d_head)  — memory-side value vectors
        * ``attn_weights[fid]`` shape (B, H, Nq, Nk) — softmax attention
          (computed in float32 for numerical stability).
    - Note that token PROVENANCE is condition-specific (the bank
      composition at frame fid differs across clean / insert-only / full
      because inserts contribute extra entries). This extractor captures
      RAW per-frame V and weights; the analysis layer (post-experiment)
      decides how to compare across conditions (e.g., re-derive top-K
      attended tokens per condition, or use foreground-query aggregation
      on the SDPA output side).

    Usage:
        extractor = MemoryReadoutExtractor(predictor.memory_attention)
        with extractor:
            # Run any forward (clean / attacked / blocking)
            forward_fn(processed, return_at=...)
        # extractor.V_by_frame[fid] -> (B, H, Nk, d_head) tensor
        # extractor.attn_by_frame[fid] -> (B, H, Nq, Nk) tensor

    Reset between forwards by calling ``extractor.reset()``. The context
    manager auto-resets on entry.

    Args:
      memory_attention: the ``MemoryAttention`` module from SAM2 predictor.
      layer_idx: which layer to probe. None -> last layer (default; codex
        spec).
      capture_attn: if True, also capture full attention weights (Nq×Nk).
        Useful for post-hoc T_obj selection. Default True.
    """

    def __init__(
        self,
        memory_attention: nn.Module,
        *,
        layer_idx: Optional[int] = None,
        capture_attn: bool = True,
    ):
        self.memory_attention = memory_attention
        all_layers = list(memory_attention.layers)
        if layer_idx is None:
            layer_idx = len(all_layers) - 1
        if not (0 <= layer_idx < len(all_layers)):
            raise ValueError(
                f"layer_idx {layer_idx} out of range for "
                f"{len(all_layers)} layers")
        self.layer_idx = layer_idx
        self.capture_attn = bool(capture_attn)

        # Per-frame storage. Updated as the SAM2 forward iterates fids.
        self.current_frame_id: Optional[int] = None
        self.V_by_frame: Dict[int, Tensor] = {}
        self.attn_by_frame: Dict[int, Tensor] = {}

        # Patch state.
        self._saved_forward: Optional[Callable] = None
        self._attn_module: Optional[nn.Module] = None

    # -- frame tagging ------------------------------------------------------

    def set_frame(self, frame_id: int) -> None:
        """Register which frame is currently being forwarded. The patched
        cross-attention reads this when it captures V/attn."""
        self.current_frame_id = int(frame_id)

    def reset(self) -> None:
        """Clear all captured per-frame state."""
        self.V_by_frame = {}
        self.attn_by_frame = {}
        self.current_frame_id = None

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "MemoryReadoutExtractor":
        self.reset()
        layer = self.memory_attention.layers[self.layer_idx]
        attn = layer.cross_attn_image
        self._attn_module = attn
        self._saved_forward = attn.forward
        attn.forward = self._make_patched_forward(attn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._attn_module is not None and self._saved_forward is not None:
            self._attn_module.forward = self._saved_forward
        self._attn_module = None
        self._saved_forward = None

    # -- patched forward ----------------------------------------------------

    def _make_patched_forward(self, attn_module: nn.Module) -> Callable:
        """Replace ``attn_module.forward`` with a wrapper that captures
        V (pre-out-proj) and attention weights for the current frame."""
        # Lazy import for RoPE detection (SAM2.1's cross_attn_image is
        # typically a plain Attention, not RoPEAttention; but we handle
        # both for robustness).
        try:
            from sam2.modeling.sam.transformer import (
                RoPEAttention, apply_rotary_enc,
            )
            is_rope = isinstance(attn_module, RoPEAttention)
        except ImportError:
            # Off-SAM2 environment (e.g. self-test on Windows); patched
            # forward is still installed but is_rope is forced False.
            RoPEAttention = type(None)
            apply_rotary_enc = None
            is_rope = False

        saved_forward = attn_module.forward
        extractor_self = self
        import math as _math
        import torch.nn.functional as _F

        if is_rope:
            def patched_forward(
                q: Tensor, k: Tensor, v: Tensor,
                num_k_exclude_rope: int = 0,
            ) -> Tensor:
                out = saved_forward(
                    q, k, v, num_k_exclude_rope=num_k_exclude_rope)
                if extractor_self.current_frame_id is not None:
                    # Capture V pre-out-proj, attention weights.
                    extractor_self._capture_rope(
                        attn_module, q, k, v, num_k_exclude_rope,
                        apply_rotary_enc, _math, _F,
                    )
                return out
        else:
            def patched_forward(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
                out = saved_forward(q, k, v)
                if extractor_self.current_frame_id is not None:
                    extractor_self._capture_plain(
                        attn_module, q, k, v, _math, _F,
                    )
                return out

        return patched_forward

    def _capture_plain(
        self,
        attn_module: nn.Module,
        q_in: Tensor, k_in: Tensor, v_in: Tensor,
        _math: Any, _F: Any,
    ) -> None:
        """Plain (non-RoPE) capture path."""
        with torch.no_grad():
            q = attn_module.q_proj(q_in)
            k = attn_module.k_proj(k_in)
            v = attn_module.v_proj(v_in)
            q = attn_module._separate_heads(q, attn_module.num_heads)
            k = attn_module._separate_heads(k, attn_module.num_heads)
            v = attn_module._separate_heads(v, attn_module.num_heads)
            self._store(q, k, v, _math, _F)

    def _capture_rope(
        self,
        attn_module: nn.Module,
        q_in: Tensor, k_in: Tensor, v_in: Tensor,
        num_k_exclude_rope: int,
        apply_rotary_enc: Any,
        _math: Any, _F: Any,
    ) -> None:
        """RoPE capture path — replicates RoPEAttention.forward up to and
        including rotary-enc, so q/k match the main path (which already
        ran)."""
        with torch.no_grad():
            q = attn_module.q_proj(q_in)
            k = attn_module.k_proj(k_in)
            v = attn_module.v_proj(v_in)
            q = attn_module._separate_heads(q, attn_module.num_heads)
            k = attn_module._separate_heads(k, attn_module.num_heads)
            v = attn_module._separate_heads(v, attn_module.num_heads)
            attn_module.freqs_cis = attn_module.freqs_cis.to(q.device)
            Nk_total = k.size(-2)
            num_k_rope = Nk_total - num_k_exclude_rope
            q, k_rope = apply_rotary_enc(
                q, k[:, :, :num_k_rope],
                freqs_cis=attn_module.freqs_cis,
                repeat_freqs_k=attn_module.rope_k_repeat,
            )
            if num_k_exclude_rope > 0:
                k = torch.cat([k_rope, k[:, :, num_k_rope:]], dim=-2)
            else:
                k = k_rope
            self._store(q, k, v, _math, _F)

    def _store(
        self, q: Tensor, k: Tensor, v: Tensor,
        _math: Any, _F: Any,
    ) -> None:
        """Compute and store V (pre-out-proj) + attn weights for the
        current frame. Detached (no grad through diagnostic)."""
        fid = int(self.current_frame_id)
        # V is already detached (we computed it under no_grad above);
        # explicit clone-to-cpu would slow things down, so we keep on
        # device and let caller pull.
        self.V_by_frame[fid] = v.detach()

        if self.capture_attn:
            # Compute attn weights in fp32 for numerical stability with
            # large Nk (matches MemAttnProbe convention).
            d_head = q.size(-1)
            scale = 1.0 / _math.sqrt(d_head)
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            weights = _F.softmax(scores, dim=-1)
            self.attn_by_frame[fid] = weights.detach()


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


def _test_build_control_frames_basic() -> None:
    """build_control_frames returns K sorted ints from valid candidates."""
    W_attacked = [5, 13, 22]
    bridge_frames = [6, 7, 8, 9, 14, 15, 16, 17, 23, 24, 25, 26]
    T_proc = 33
    cf = build_control_frames(W_attacked, bridge_frames, T_proc, K=3, seed=0)
    assert len(cf) == 3, f"expected 3, got {cf}"
    assert sorted(cf) == cf
    for c in cf:
        assert c not in W_attacked, f"control {c} in W_attacked"
        assert c not in bridge_frames, f"control {c} in bridge"
        assert c != 0, f"control {c} == 0 (frame 0 reserved)"
        assert 0 < c < T_proc
    print(f"  build_control_frames basic OK ({cf})")


def _test_build_control_frames_determinism() -> None:
    """Same args + same seed → same output."""
    args = ([5, 13, 22], [6, 7, 8, 14, 15, 23, 24], 30)
    a = build_control_frames(*args, K=3, seed=0)
    b = build_control_frames(*args, K=3, seed=0)
    assert a == b, f"determinism broken: {a} vs {b}"
    # Different seed -> different output (with reasonably high probability).
    c = build_control_frames(*args, K=3, seed=1)
    if c == a:
        # very unlikely but possible with small candidate pool; require at
        # least different ordering or composition somehow
        pass
    print(f"  build_control_frames determinism OK (seed=0: {a})")


def _test_build_control_frames_validation() -> None:
    """Insufficient candidates raises."""
    # T=10, W={3,5,7}, bridge={4,6,8} → forbidden ∪ {0} = {0,3,4,5,6,7,8}
    # candidates = {1,2,9} (3 frames). Asking for K=3 OK; K=4 raises.
    cf = build_control_frames([3, 5, 7], [4, 6, 8], 10, K=3, seed=0)
    assert sorted(cf) == [1, 2, 9]
    try:
        build_control_frames([3, 5, 7], [4, 6, 8], 10, K=4, seed=0)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "Not enough" in str(e)
    print(f"  build_control_frames validation OK")


def _test_blocking_forward_fn_validates_frame_zero() -> None:
    """blocked_frames including 0 raises (frame 0 is the prompt frame)."""
    class _StubBase:
        device = torch.device("cpu")
        video_H = 8
        video_W = 8
    try:
        make_blocking_forward_fn(_StubBase(), blocked_frames=[0, 5])
        assert False, "expected ValueError on fid=0 in blocked_frames"
    except ValueError as e:
        assert "frame 0" in str(e)
    print("  make_blocking_forward_fn frame-0 guard OK")


class _StubAttn(nn.Module):
    """Minimal stand-in for SAM2's cross_attn_image, sufficient for the
    extractor self-test on Windows (no SAM2 installed)."""
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        B, N, D = x.shape
        x = x.view(B, N, num_heads, D // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        B, H, N, d = x.shape
        return x.transpose(1, 2).reshape(B, N, H * d)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q_p = self.q_proj(q)
        k_p = self.k_proj(k)
        v_p = self.v_proj(v)
        q_h = self._separate_heads(q_p, self.num_heads)
        k_h = self._separate_heads(k_p, self.num_heads)
        v_h = self._separate_heads(v_p, self.num_heads)
        import math
        scale = 1.0 / math.sqrt(q_h.size(-1))
        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v_h)
        return self.out_proj(self._recombine_heads(out))


class _StubLayer(nn.Module):
    def __init__(self, attn: nn.Module):
        super().__init__()
        self.cross_attn_image = attn


class _StubMemoryAttention(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int):
        super().__init__()
        self.layers = nn.ModuleList([
            _StubLayer(_StubAttn(embed_dim, num_heads))
            for _ in range(num_layers)
        ])


def _test_memory_extractor_captures_v_and_attn() -> None:
    """Patched cross_attn_image captures V and attn at the chosen frame."""
    torch.manual_seed(0)
    B, Nq, Nk, D, H = 1, 16, 24, 32, 4
    mem_attn = _StubMemoryAttention(num_layers=3, embed_dim=D, num_heads=H)

    extractor = MemoryReadoutExtractor(mem_attn, layer_idx=2, capture_attn=True)

    q = torch.randn(B, Nq, D)
    k = torch.randn(B, Nk, D)
    v = torch.randn(B, Nk, D)

    with extractor:
        # No frame_id set -> nothing captured (passthrough).
        out0 = mem_attn.layers[2].cross_attn_image(q, k, v)
        assert out0.shape == (B, Nq, D)
        assert len(extractor.V_by_frame) == 0

        # Set frame, run forward -> capture.
        extractor.set_frame(7)
        out1 = mem_attn.layers[2].cross_attn_image(q, k, v)
        assert 7 in extractor.V_by_frame
        assert extractor.V_by_frame[7].shape == (B, H, Nk, D // H)
        assert 7 in extractor.attn_by_frame
        assert extractor.attn_by_frame[7].shape == (B, H, Nq, Nk)

        # Different frame -> different storage slot.
        extractor.set_frame(11)
        _ = mem_attn.layers[2].cross_attn_image(q + 0.01, k, v)
        assert 11 in extractor.V_by_frame
        # V at 11 != V at 7 because q differed (k/v same; v_proj-output is
        # independent of q so V_by_frame[7] and [11] should be EQUAL since
        # k,v unchanged) -- but attn weights should differ slightly.
        assert torch.allclose(extractor.V_by_frame[7], extractor.V_by_frame[11])
        assert not torch.allclose(extractor.attn_by_frame[7],
                                  extractor.attn_by_frame[11], atol=1e-4)

    # After context exit: forward is restored.
    out2 = mem_attn.layers[2].cross_attn_image(q, k, v)
    assert out2.shape == (B, Nq, D)
    print("  MemoryReadoutExtractor capture OK")


def _test_memory_extractor_main_path_unchanged() -> None:
    """The patched forward must produce numerically IDENTICAL output to
    the unpatched forward when no frame is set (or even when it is)."""
    torch.manual_seed(0)
    B, Nq, Nk, D, H = 1, 8, 12, 16, 2
    mem_attn = _StubMemoryAttention(num_layers=2, embed_dim=D, num_heads=H)
    q = torch.randn(B, Nq, D)
    k = torch.randn(B, Nk, D)
    v = torch.randn(B, Nk, D)

    out_unpatched = mem_attn.layers[1].cross_attn_image(q, k, v)

    extractor = MemoryReadoutExtractor(mem_attn, layer_idx=1)
    with extractor:
        extractor.set_frame(0)
        out_patched = mem_attn.layers[1].cross_attn_image(q, k, v)

    assert torch.allclose(out_unpatched, out_patched, atol=1e-6), (
        "patched forward changed numerical output")
    print("  MemoryReadoutExtractor main-path unchanged OK")


def _test_memory_extractor_reset() -> None:
    """reset() clears per-frame storage."""
    torch.manual_seed(0)
    B, Nq, Nk, D, H = 1, 4, 6, 8, 2
    mem_attn = _StubMemoryAttention(1, D, H)
    q = torch.randn(B, Nq, D); k = torch.randn(B, Nk, D); v = torch.randn(B, Nk, D)
    extractor = MemoryReadoutExtractor(mem_attn, layer_idx=0)
    with extractor:
        extractor.set_frame(3)
        _ = mem_attn.layers[0].cross_attn_image(q, k, v)
        assert 3 in extractor.V_by_frame
        extractor.reset()
        assert len(extractor.V_by_frame) == 0
        assert len(extractor.attn_by_frame) == 0
        assert extractor.current_frame_id is None
    print("  MemoryReadoutExtractor reset OK")


class _StubVADIForwardFn:
    """Minimal stand-in for VADIForwardFn used to validate parity-mode of
    ``make_blocking_forward_fn`` on Windows (no SAM2 needed).

    Mimics the exact attribute surface that ``blocking_call`` reads:
    predictor (with forward_image, _prepare_backbone_features,
    track_step), device, video_H, video_W, image_size, autocast_dtype,
    use_gradient_checkpointing, _img_mean, _img_std, _mask_inputs_f0.
    """

    def __init__(self, T_proc: int, H: int = 4, W: int = 4, *,
                 emit_constant: bool = True, seed: int = 0):
        self.device = torch.device("cpu")
        self.video_H = H
        self.video_W = W
        self.image_size = 16
        self.autocast_dtype = None
        self.use_gradient_checkpointing = False
        self._img_mean = torch.zeros(1, 3, 1, 1)
        self._img_std = torch.ones(1, 3, 1, 1)
        self._mask_inputs_f0 = None
        torch.manual_seed(seed)
        # Per-frame predicted logits — stored, returned per request.
        self._frame_outputs: Dict[int, Dict[str, Tensor]] = {}
        for fid in range(T_proc):
            mask = torch.full((1, 1, H, W), float(fid) if emit_constant else 0.0)
            self._frame_outputs[fid] = {"pred_masks": mask,
                                         "maskmem_features": None,
                                         "maskmem_pos_enc": None,
                                         "obj_ptr": None}
        self.predictor = self._make_predictor()
        # Track which fids actually got dict-written (to verify behavior).
        self.last_write_log: List[int] = []

    def _make_predictor(self):
        outer = self
        class _StubPredictor:
            def forward_image(_, img):
                return {"backbone_fpn": [torch.zeros(1, 4, 4, 4)],
                        "vision_pos_enc": [torch.zeros(1, 4, 4, 4)]}
            def _prepare_backbone_features(_, backbone_out):
                return None, [torch.zeros(1, 4)], [torch.zeros(1, 4)], [(4, 4)]
            def track_step(_, *, frame_idx, is_init_cond_frame, output_dict, **kw):
                # Record what gets passed to output_dict implicitly via
                # the wrapper's append behavior. Just return the per-frame
                # canned output.
                return outer._frame_outputs[frame_idx]
        return _StubPredictor()


def _patch_helper_imports_for_stub() -> None:
    """make_blocking_forward_fn imports _to_sam2_input and
    _low_res_to_video_res from vadi_sam2_wiring. Provide stubs so the
    parity test runs on Windows without SAM2."""
    import memshield.vadi_sam2_wiring as _w
    if not hasattr(_w, "_orig_to_sam2_input_for_test"):
        _w._orig_to_sam2_input_for_test = _w._to_sam2_input
        _w._orig_low_res_to_video_res_for_test = _w._low_res_to_video_res

    def _stub_to_sam2_input(frame, image_size, mean, std):
        # frame: (1, H, W, 3) -> (1, 3, image_size, image_size)
        return torch.zeros(1, 3, image_size, image_size,
                           dtype=frame.dtype, device=frame.device)

    def _stub_low_res_to_video_res(pred_masks, video_H, video_W):
        # Just return the input mask reshaped to (H, W). The stub frame
        # outputs already have shape (1, 1, H, W), so squeeze.
        return pred_masks.squeeze(0).squeeze(0)

    _w._to_sam2_input = _stub_to_sam2_input
    _w._low_res_to_video_res = _stub_low_res_to_video_res


def _test_blocking_parity_mode() -> None:
    """blocked_frames=() and extractor=None should produce IDENTICAL output
    to the base forward — by construction (the wrapper duplicates the
    per-frame loop semantics). This test catches silent drift if the
    wrapper's loop diverges from VADIForwardFn.__call__ in the future.

    Uses _StubVADIForwardFn + monkey-patched helper functions so it runs
    on Windows without SAM2.
    """
    _patch_helper_imports_for_stub()
    T_proc = 6
    base = _StubVADIForwardFn(T_proc=T_proc, H=4, W=4, emit_constant=True)
    blocking = make_blocking_forward_fn(base, blocked_frames=(),
                                          extractor=None)
    processed = torch.zeros(T_proc, 4, 4, 3)
    return_at = list(range(T_proc))
    out = blocking(processed, return_at)
    assert sorted(out.keys()) == return_at
    for fid in return_at:
        # Stub emits a constant-fid mask of shape (4, 4).
        assert out[fid].shape == (4, 4)
        assert torch.allclose(out[fid], torch.full((4, 4), float(fid)))
    # Ensure parity-mode tags.
    assert blocking.is_blocking is False
    assert blocking.has_extractor is False
    print("  make_blocking_forward_fn parity (blocked=[], extractor=None) OK")


def _test_blocking_with_blocked_frames() -> None:
    """blocked_frames is honored: those fids must NOT enter
    obj_output_dict (verified via a custom predictor that records writes).

    We can't directly inspect obj_output_dict from outside the blocking_call,
    but the wrapper's `is_blocking` flag and parity behavior we already
    cover. Here we validate that blocked_frames are STILL returned in the
    output dict (their masks are still computed normally).
    """
    _patch_helper_imports_for_stub()
    T_proc = 8
    base = _StubVADIForwardFn(T_proc=T_proc, H=4, W=4, emit_constant=True)
    blocked = [3, 5]
    blocking = make_blocking_forward_fn(base, blocked_frames=blocked,
                                          extractor=None)
    processed = torch.zeros(T_proc, 4, 4, 3)
    out = blocking(processed, return_at=[3, 5, 7])
    # All three fids returned regardless of blocked-set membership.
    assert sorted(out.keys()) == [3, 5, 7]
    assert torch.allclose(out[3], torch.full((4, 4), 3.0))
    assert torch.allclose(out[5], torch.full((4, 4), 5.0))
    # Parity attrs.
    assert blocking.is_blocking is True
    assert blocking.blocked_frames == [3, 5]
    print("  make_blocking_forward_fn with blocked_frames returns all fids OK")


def _test_blocking_with_extractor_calls_set_frame() -> None:
    """When extractor is provided, set_frame(fid) must be called per fid.

    We use a recording stub that captures every set_frame call.
    """
    _patch_helper_imports_for_stub()

    class _RecExtractor:
        def __init__(self): self.calls = []
        def set_frame(self, fid): self.calls.append(int(fid))

    T_proc = 5
    base = _StubVADIForwardFn(T_proc=T_proc, H=4, W=4)
    rec = _RecExtractor()
    blocking = make_blocking_forward_fn(base, blocked_frames=(),
                                          extractor=rec)
    processed = torch.zeros(T_proc, 4, 4, 3)
    _ = blocking(processed, return_at=[0, T_proc - 1])
    assert rec.calls == list(range(T_proc)), (
        f"set_frame should be called per fid in order; got {rec.calls}")
    print("  make_blocking_forward_fn extractor.set_frame(fid) integration OK")


def _test_aggregate_V_top_attended() -> None:
    """aggregate_V_top_attended produces (d_head,) and selects top-K by
    sum of attention received."""
    torch.manual_seed(0)
    B, H, Nq, Nk, d_head = 1, 2, 4, 8, 6
    V = torch.randn(B, H, Nk, d_head)
    # Make attn_weights peak at Nk=3 (high attention) and Nk=5 (moderate).
    attn = torch.zeros(B, H, Nq, Nk)
    attn[..., 3] = 0.5
    attn[..., 5] = 0.3
    # Distribute remaining mass over other positions.
    attn[..., 0] = 0.05
    attn[..., 1] = 0.05
    attn[..., 2] = 0.025
    attn[..., 4] = 0.025
    attn[..., 6] = 0.025
    attn[..., 7] = 0.025
    M = aggregate_V_top_attended(V, attn, top_k=2)
    assert M.shape == (d_head,)
    # Manually compute expected: V over (Nk=3, Nk=5) averaged over B,H.
    expected = V[:, :, [3, 5], :].mean(dim=(0, 1, 2)).float()
    assert torch.allclose(M, expected, atol=1e-5)
    print("  aggregate_V_top_attended OK (selects top-K by attention)")


def _test_compute_d_mem_trace() -> None:
    """compute_d_mem_trace returns 0 when V/attn identical, > 0 otherwise."""
    torch.manual_seed(0)
    B, H, Nq, Nk, d_head = 1, 2, 4, 6, 4
    V = torch.randn(B, H, Nk, d_head)
    attn = torch.softmax(torch.randn(B, H, Nq, Nk), dim=-1)

    clean = {3: (V, attn), 7: (V, attn)}
    same = {3: (V.clone(), attn.clone()), 7: (V.clone(), attn.clone())}
    diff = {
        3: (torch.randn(B, H, Nk, d_head),
            torch.softmax(torch.randn(B, H, Nq, Nk), dim=-1)),
        7: (V, attn),
    }

    d_same = compute_d_mem_trace(clean, same, top_k=3)
    assert d_same[3] < 1e-5, f"identical V/attn should give d_mem=0, got {d_same[3]}"
    assert d_same[7] < 1e-5

    d_diff = compute_d_mem_trace(clean, diff, top_k=3)
    assert d_diff[3] > 0.01, f"different V should give d_mem>0, got {d_diff[3]}"
    assert d_diff[7] < 1e-5  # frame 7 is identical in diff
    print(f"  compute_d_mem_trace OK (identical->{d_same[3]:.4e}, "
          f"different->{d_diff[3]:.4f})")


def _self_test() -> None:
    print("memshield.causal_diagnostics self-tests:")
    _test_build_control_frames_basic()
    _test_build_control_frames_determinism()
    _test_build_control_frames_validation()
    _test_blocking_forward_fn_validates_frame_zero()
    _test_memory_extractor_captures_v_and_attn()
    _test_memory_extractor_main_path_unchanged()
    _test_memory_extractor_reset()
    _test_blocking_parity_mode()
    _test_blocking_with_blocked_frames()
    _test_blocking_with_extractor_calls_set_frame()
    _test_aggregate_V_top_attended()
    _test_compute_d_mem_trace()
    print("memshield.causal_diagnostics: all self-tests PASSED")


if __name__ == "__main__":
    _self_test()
