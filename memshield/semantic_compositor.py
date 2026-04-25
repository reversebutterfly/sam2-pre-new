"""Semantic decoy compositor for VADI Bundle B (codex Round 5, 2026-04-25).

Replaces the proxy `memshield.decoy_seed.build_duplicate_object_decoy_frame`
for Stage 14 (oracle-trajectory PGD). Provides a silhouette alpha-feather
paste compositor that fixes the dark-halo artifact present in the legacy
function's boundary handling.

## Background

Bundle B sub-session 3 was originally scoped as "inpainting model integration"
(LaMa or Stable Diffusion). gpt-5.4 xhigh review (thread
`019dc51a-c71a-7971-bece-116a592de2f5`, see `BUNDLE_B_INPAINTER_REVIEW.md`)
showed that under Stage 14's overlay math:

    x_edited = (1 - α·soft_decoy) · x_warped + (α·soft_decoy) · duplicate
    α_max = 0.35

the duplicate frame's pixels at the *true-object region* are dropped by
soft_decoy ≈ 0 and never reach SAM2's input. Inpainting only changes pixels
in that dropped region. Therefore B (alpha-feather paste, no model) ≡ C (LaMa)
≡ D (SD) pixel-for-pixel at SAM2's input in the disjoint-region regime.
Inpainting buys no attack-effectiveness uplift; it only adds 200MB-5GB of
checkpoints, ~1-45s/polish overhead, and LPIPS spill risk.

## What this module fixes

The legacy `build_duplicate_object_decoy_frame` does silhouette alpha-feather
paste, NOT Poisson seamless cloning (despite some comments suggesting otherwise).
Its actual formula:

    decoy = (1 - alpha) * x_ref + alpha * shifted_obj_pixels

where `shifted_obj_pixels = shift(x_ref * hard_mask, dy, dx)` (zero outside
the shifted silhouette) and `alpha = gaussian_blur(shifted_hard_mask)`.

**Bug**: at the feathered boundary *outside* the shifted silhouette, the
shifted object pixels are 0 while alpha > 0 (Gaussian leaks the support).
The convex combination then produces `(1-α)·x_ref + α·0 = darkened x_ref`,
creating a dark-halo ring around the decoy's silhouette. After Stage 14's
35% blend, this is the dominant LPIPS-catching artifact.

**Fix** (this module): fill `shifted_obj_pixels` with `x_ref` at pixels
outside the shifted silhouette, so the convex combination becomes
`(1-α)·x_ref + α·x_ref = x_ref` (background preserved) at boundary
exterior, and `(1-α)·x_ref + α·obj_pixel` at boundary interior. No halo.

## When this module's choice expires

This compositor's "no inpainter needed" rationale depends on Stage 14's
overlay math (`α_max ≤ 0.35` + soft_decoy gating). If Stage 14 is later
redesigned (α_max → 1, hard replacement, background-first composition,
etc.), the disjoint-equivalence argument no longer holds and inpainting
must be reconsidered. See `BUNDLE_B_SD_ALTERNATIVES_BACKUP.md` for the
documented escalation paths.

## Self-tests

`python -m memshield.semantic_compositor` runs sanity checks on small
synthetic frames. No SAM2 dependency.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from memshield.decoy_seed import (
    build_duplicate_object_decoy_frame,
    gaussian_blur_mask,
    shift_2d,
)


__all__ = ["compose_decoy_alpha_paste", "apply_masked_residual"]


def apply_masked_residual(
    x_base: Tensor,                 # [H, W, 3] in [0, 1]
    residual: Tensor,               # [H, W, 3] (typically clamped to [-eps_R, eps_R])
    support_mask: Tensor,           # [H, W] in [0, 1] — caller MUST detach
) -> Tensor:
    """Add a mask-supported residual to a base frame and clamp to [0, 1].

    Used in Stage 14 Bundle B sub-session 4 to apply the per-bridge-frame
    learnable residual R_k after the alpha-overlay composes the duplicate
    onto the warped clean frame.

    Formula:
        out = (x_base + support_mask * residual).clamp(0, 1)

    Caller is responsible for **detaching** `support_mask` if it was built
    from a differentiable mask source (e.g. `soften_decoy_mask` of an
    oracle decoy mask that depends on trajectory params). Detaching
    prevents the high-dim `residual` from creating a second gradient path
    into the trajectory params (anchor/delta), which would let the
    residual steer geometry instead of refining pixels — flagged as a
    blocking issue by gpt-5.4 reviewer in design consult thread
    `019dc51a-c71a-7971-bece-116a592de2f5`.

    Args:
        x_base: [H, W, 3] in [0, 1]. Output of `apply_continuation_overlay`.
        residual: [H, W, 3]. Per-bridge-frame learnable R_k. Caller is
            responsible for clamping to its ε-budget (e.g. ±8/255 via
            sign-PGD with hard clamp).
        support_mask: [H, W] in [0, 1]. Localizes where the residual can
            edit. Caller MUST pass a detached tensor.

    Returns:
        [H, W, 3] in [0, 1].
    """
    if x_base.dim() != 3 or x_base.shape[-1] != 3:
        raise ValueError(
            f"x_base must be [H, W, 3]; got {tuple(x_base.shape)}")
    if residual.shape != x_base.shape:
        raise ValueError(
            f"residual must match x_base shape {tuple(x_base.shape)}; "
            f"got {tuple(residual.shape)}")
    if support_mask.dim() != 2 or support_mask.shape != x_base.shape[:2]:
        raise ValueError(
            f"support_mask must be [H, W]={x_base.shape[:2]}; "
            f"got {tuple(support_mask.shape)}")
    # Codex pre-commit review (2026-04-25, thread
    # 019dc51a-c71a-7971-bece-116a592de2f5) flagged that the docstring's
    # detach contract should be enforced. A support_mask with grad would
    # let R steer geometry through whatever produced the mask.
    if support_mask.requires_grad:
        raise ValueError(
            "support_mask must be detached (requires_grad=False); "
            "received a tensor with requires_grad=True. Caller must "
            ".detach() the soften'd decoy mask before passing it in "
            "to prevent the residual from creating a second gradient "
            "path into trajectory params (gpt-5.4 blocking item).")
    support_3 = support_mask.unsqueeze(-1)                 # [H, W, 1]
    return (x_base + support_3 * residual).clamp(0.0, 1.0)


def compose_decoy_alpha_paste(
    x_ref: Tensor,                  # [H, W, 3] float in [0, 1]
    object_mask: Tensor,            # [H, W] float in [0, 1] (soft) or {0, 1}
    decoy_offset: Tuple[int, int],  # (dy, dx) integer translation
    feather_radius: int = 5,
    feather_sigma: float = 2.0,
) -> Tensor:
    """Silhouette alpha-feather paste of an object onto a clean reference.

    Drop-in replacement for `build_duplicate_object_decoy_frame` with no
    boundary dark-halo artifact. Same call signature.

    Output convention (identical to legacy at silhouette interior, fixed
    at silhouette exterior boundary):
        - Outside shifted silhouette: x_ref (background preserved exactly).
        - Inside shifted silhouette: shifted object pixels.
        - Feathered transition: smooth convex combination, no darkening.

    Formula:
        shifted_pixels = shift(x_ref * hard_mask, dy, dx)        # 0 outside
        shifted_hard_mask = shift(hard_mask, dy, dx)             # binary
        alpha = gaussian_blur(shifted_hard_mask)                  # [0, 1] feathered
        # Halo fix: fill shifted_pixels with x_ref outside shifted silhouette.
        fill_inside = shifted_hard_mask.unsqueeze(-1) > 0.5
        shifted_pixels_filled = where(fill_inside, shifted_pixels, x_ref)
        out = (1 - alpha).unsqueeze(-1) * x_ref + alpha.unsqueeze(-1) * shifted_pixels_filled

    Args:
        x_ref: [H, W, 3] in [0, 1]. Background frame (with original object
            visible at its true position — this compositor produces a
            DUPLICATE not a MOVE; original is left intact, in keeping with
            Stage 14's `apply_continuation_overlay` semantics).
        object_mask: [H, W] in [0, 1]. Object's silhouette at x_ref.
            Soft OK; thresholded at 0.5 for hard binarization (matches
            legacy convention).
        decoy_offset: (dy, dx) integer pixel translation. Positive dy =
            down, positive dx = right.
        feather_radius: Gaussian kernel half-width for alpha feathering.
        feather_sigma: Gaussian sigma for feathering.

    Returns:
        [H, W, 3] in [0, 1]. The composite duplicate frame.
    """
    if x_ref.dim() != 3 or x_ref.shape[-1] != 3:
        raise ValueError(
            f"x_ref must be [H, W, 3]; got {tuple(x_ref.shape)}")
    if object_mask.dim() != 2 or object_mask.shape != x_ref.shape[:2]:
        raise ValueError(
            f"object_mask must be [H, W]={x_ref.shape[:2]}; "
            f"got {tuple(object_mask.shape)}")

    hard_mask = (object_mask > 0.5).to(x_ref.dtype)              # [H, W]
    object_pixels = x_ref * hard_mask.unsqueeze(-1)              # [H, W, 3]

    dy, dx = int(decoy_offset[0]), int(decoy_offset[1])
    shifted_pixels = shift_2d(object_pixels, dy, dx, fill=0.0)   # [H, W, 3]
    shifted_hard_mask = shift_2d(hard_mask, dy, dx, fill=0.0)    # [H, W]

    # Defensive: feather_radius<=0 OR feather_sigma<=0 → no blur (avoids
    # divide-by-zero in gaussian_blur_mask's exp(-(x²)/(2·sigma²))).
    # gaussian_blur_mask itself short-circuits radius<=0 but not sigma<=0.
    if feather_radius > 0 and feather_sigma > 0:
        alpha = gaussian_blur_mask(
            shifted_hard_mask, feather_radius, feather_sigma)
    else:
        alpha = shifted_hard_mask                                # [H, W]

    # Halo fix: at pixels outside the shifted hard silhouette, the shifted
    # object pixels are 0. A convex combination `(1-α)·x_ref + α·0` darkens
    # the background where alpha > 0 due to Gaussian leakage. Fill those
    # pixels with x_ref so the convex combination resolves to x_ref there.
    fill_inside = (shifted_hard_mask > 0.5).unsqueeze(-1)        # [H, W, 1]
    shifted_pixels_filled = torch.where(
        fill_inside, shifted_pixels, x_ref)                       # [H, W, 3]

    alpha_3 = alpha.unsqueeze(-1)                                 # [H, W, 1]
    out = (1.0 - alpha_3) * x_ref + alpha_3 * shifted_pixels_filled
    return out.clamp(0.0, 1.0)


# =============================================================================
# Self-tests
# =============================================================================


def _test_shape_and_range() -> None:
    H, W = 32, 32
    x = torch.rand(H, W, 3)
    m = torch.zeros(H, W); m[10:20, 10:20] = 1.0
    out = compose_decoy_alpha_paste(x, m, (5, 5))
    assert out.shape == (H, W, 3), out.shape
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0
    print("  shape_and_range OK")


def _test_silhouette_interior_matches_object() -> None:
    """Inside the shifted silhouette (well clear of feather), the output
    pixel must equal the source object pixel."""
    H, W = 64, 64
    x = torch.zeros(H, W, 3)
    x[20:30, 20:30, 0] = 0.7   # red square = object
    x[:, :, 1] = 0.4           # green background everywhere
    m = torch.zeros(H, W); m[20:30, 20:30] = 1.0
    dy, dx = 15, 0
    out = compose_decoy_alpha_paste(
        x, m, (dy, dx), feather_radius=3, feather_sigma=1.0)
    # Pixel at (40, 25) is inside the shifted silhouette interior
    # (originally (25, 25) → shifted to (40, 25), 5 px from boundary so
    # alpha ≈ 1 even after feather).
    interior = out[40, 25]
    # Source pixel x[25, 25] = (0.7, 0.4, 0) — red foreground + green
    # background showing through. `x_ref * hard_mask` is per-pixel scalar
    # mask multiply, preserves ALL channels inside the silhouette. So
    # shifted_pixels[40, 25] = (0.7, 0.4, 0). At alpha ≈ 1, output equals
    # shifted_pixels.
    expected_interior = torch.tensor([0.7, 0.4, 0.0])
    assert torch.allclose(interior, expected_interior, atol=0.05), \
        f"interior at (40, 25) = {interior} (expected ≈ {expected_interior})"
    print("  silhouette_interior_matches_object OK")


def _test_no_dark_halo() -> None:
    """At boundary OUTSIDE the shifted silhouette but INSIDE the feather
    radius, the output must NOT be darkened relative to x_ref. This is the
    halo bug that the legacy `build_duplicate_object_decoy_frame` exhibits;
    this compositor must fix it."""
    H, W = 64, 64
    x = torch.full((H, W, 3), 0.5)  # uniform mid-grey background
    # Add a small object far from where its shifted version's halo would land.
    m = torch.zeros(H, W); m[10:14, 10:14] = 1.0
    x[10:14, 10:14] = torch.tensor([1.0, 0.0, 0.0])  # bright red object
    dy, dx = 30, 30  # shift far away — silhouette will be at [40:44, 40:44]

    new_compositor_out = compose_decoy_alpha_paste(
        x, m, (dy, dx), feather_radius=5, feather_sigma=2.0)
    legacy_out = build_duplicate_object_decoy_frame(
        x, m, (dy, dx), feather_radius=5, feather_sigma=2.0)

    # Sample a pixel just outside the shifted silhouette but inside the
    # feather radius. Shifted silhouette is [40:44, 40:44]; pixel (45, 42)
    # is 1 px outside in y. At sigma=2, alpha at that pixel is ~0.3.
    halo_pixel_new = new_compositor_out[45, 42]
    halo_pixel_legacy = legacy_out[45, 42]
    expected_clean = torch.tensor([0.5, 0.5, 0.5])

    # New compositor should preserve the background (≈ x_ref).
    new_diff_to_clean = (halo_pixel_new - expected_clean).abs().max().item()
    legacy_diff_to_clean = (halo_pixel_legacy - expected_clean).abs().max().item()

    assert new_diff_to_clean < 0.02, \
        f"new compositor at halo pixel deviates {new_diff_to_clean} " \
        f"from clean (expected < 0.02): got {halo_pixel_new}"
    # Legacy should darken (this is the bug we're documenting).
    assert legacy_diff_to_clean > new_diff_to_clean, \
        f"expected legacy to darken more than new at halo pixel; " \
        f"new diff={new_diff_to_clean}, legacy diff={legacy_diff_to_clean}"
    print(f"  no_dark_halo OK (new diff={new_diff_to_clean:.4f}, "
          f"legacy diff={legacy_diff_to_clean:.4f})")


def _test_zero_offset_preserves_x_ref_outside_object() -> None:
    """With zero offset, the duplicate is the object pasted ONTO ITS OWN
    POSITION. Outside the object silhouette, output must equal x_ref."""
    H, W = 32, 32
    torch.manual_seed(0)
    x = torch.rand(H, W, 3)
    m = torch.zeros(H, W); m[10:14, 10:14] = 1.0
    out = compose_decoy_alpha_paste(
        x, m, (0, 0), feather_radius=3, feather_sigma=1.0)
    # Pixel (0, 0) is far from the silhouette → must equal x_ref[0, 0].
    assert torch.allclose(out[0, 0], x[0, 0], atol=1e-5), \
        f"far-from-silhouette: out={out[0, 0]} vs x={x[0, 0]}"
    # Pixel (10, 10) is inside the silhouette → output equals object
    # pixel (which equals x_ref[10, 10] since hard_mask=1 there).
    assert torch.allclose(out[10, 10], x[10, 10], atol=1e-5), \
        f"silhouette-interior: out={out[10, 10]} vs x={x[10, 10]}"
    print("  zero_offset_preserves_x_ref_outside_object OK")


def _test_overlap_regime() -> None:
    """When the decoy offset is small enough that the shifted silhouette
    overlaps the original silhouette, the compositor should still produce a
    valid frame (no NaN, no out-of-range, both objects' content visible
    in their respective regions)."""
    H, W = 32, 32
    x = torch.full((H, W, 3), 0.5)
    x[10:18, 10:18] = torch.tensor([1.0, 0.0, 0.0])  # red object
    m = torch.zeros(H, W); m[10:18, 10:18] = 1.0
    # Small offset: shifted silhouette [12:20, 12:20] overlaps original.
    out = compose_decoy_alpha_paste(
        x, m, (2, 2), feather_radius=3, feather_sigma=1.0)
    # No NaN.
    assert torch.isfinite(out).all(), "non-finite values in output"
    # Range bounded.
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0
    # Original at (11, 11) is INSIDE original mask but OUTSIDE shifted
    # silhouette (which starts at 12). Should be x_ref's red.
    assert torch.allclose(out[11, 11], x[11, 11], atol=0.05), \
        f"original-region pixel: out={out[11, 11]} vs x={x[11, 11]}"
    # Shifted object at (15, 15): inside both. Output should be the
    # silhouette-extracted red (since shifted_pixels = x_ref * hard_mask,
    # at (15, 15) the source was x_ref[13, 13] which is also red object).
    assert torch.allclose(out[15, 15], torch.tensor([1.0, 0.0, 0.0]),
                          atol=0.05), \
        f"overlap-region pixel: {out[15, 15]}"
    print("  overlap_regime OK")


def _test_signature_compat_with_legacy() -> None:
    """Same call signature as build_duplicate_object_decoy_frame so Stage
    14 dispatch is a one-line swap."""
    import inspect
    new_sig = inspect.signature(compose_decoy_alpha_paste)
    legacy_sig = inspect.signature(build_duplicate_object_decoy_frame)
    assert list(new_sig.parameters.keys()) == \
        list(legacy_sig.parameters.keys()), \
        f"signature mismatch:\n  new={new_sig}\n  legacy={legacy_sig}"
    print("  signature_compat_with_legacy OK")


def _test_no_grad_through_content() -> None:
    """Stage 14 expects the duplicate frame to be content-detached (it
    uses a detached integer-rounded offset). Verify the function doesn't
    create implicit grad paths through inputs that would surprise."""
    H, W = 16, 16
    x = torch.rand(H, W, 3, requires_grad=True)
    m = torch.zeros(H, W); m[5:10, 5:10] = 1.0
    out = compose_decoy_alpha_paste(x, m, (3, 0))
    # Output is differentiable w.r.t. x_ref by design (no unwanted
    # disconnect — the content compositing is just elementwise multiply
    # and add). Stage 14 calls .detach() on the offset, not on the
    # compositor's output; the output's grad to x_ref is benign because
    # x_clean is a leaf tensor that doesn't get gradient updates.
    out.sum().backward()
    assert x.grad is not None
    print("  no_grad_through_content (grad to x_ref OK, expected) "
          "OK")


def _test_apply_masked_residual_basic() -> None:
    """Basic shape, range, and identity checks."""
    H, W = 16, 16
    x = torch.full((H, W, 3), 0.5)
    R = torch.zeros(H, W, 3)
    sup = torch.zeros(H, W); sup[6:10, 6:10] = 1.0
    out = apply_masked_residual(x, R, sup)
    assert out.shape == (H, W, 3)
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0
    # R = 0 → output = x_base.
    assert torch.allclose(out, x), "R=0 should be identity"
    # support = 0 everywhere → output = x_base regardless of R.
    R_nonzero = torch.full((H, W, 3), 0.1)
    sup_zero = torch.zeros(H, W)
    out2 = apply_masked_residual(x, R_nonzero, sup_zero)
    assert torch.allclose(out2, x), "support=0 should be identity"
    print("  apply_masked_residual_basic OK")


def _test_apply_masked_residual_localization() -> None:
    """Residual only edits pixels where support > 0; outside support,
    output equals x_base."""
    H, W = 16, 16
    x = torch.full((H, W, 3), 0.5)
    R = torch.full((H, W, 3), 0.05)        # small uniform residual
    sup = torch.zeros(H, W); sup[6:10, 6:10] = 1.0   # support only at interior
    out = apply_masked_residual(x, R, sup)
    # Outside support: out = x.
    assert torch.allclose(out[0, 0], x[0, 0]), \
        f"outside support: out={out[0, 0]} vs x={x[0, 0]}"
    # Inside support: out = (x + R).clamp = 0.5 + 0.05 = 0.55.
    assert torch.allclose(out[7, 7], torch.full((3,), 0.55), atol=1e-5), \
        f"inside support: out={out[7, 7]} (expected 0.55)"
    print("  apply_masked_residual_localization OK")


def _test_apply_masked_residual_clamps_to_unit_range() -> None:
    """Output strictly in [0, 1] even when x + support·R would overflow."""
    H, W = 8, 8
    x = torch.full((H, W, 3), 0.95)         # near upper bound
    R = torch.full((H, W, 3), 0.5)          # would push above 1
    sup = torch.ones(H, W)
    out = apply_masked_residual(x, R, sup)
    assert out.max().item() <= 1.0
    # Lower bound similarly.
    x_low = torch.full((H, W, 3), 0.05)
    R_neg = torch.full((H, W, 3), -0.5)
    out_low = apply_masked_residual(x_low, R_neg, sup)
    assert out_low.min().item() >= 0.0
    print("  apply_masked_residual_clamps_to_unit_range OK")


def _test_apply_masked_residual_grad_paths() -> None:
    """Gradient flows to residual but NOT through detached support_mask
    when caller detaches it (caller responsibility)."""
    H, W = 8, 8
    x_base = torch.full((H, W, 3), 0.5, requires_grad=False)
    R = torch.zeros(H, W, 3, requires_grad=True)
    # Build a differentiable support, then detach it explicitly (caller
    # responsibility — apply_masked_residual itself does NOT detach).
    raw_decoy = torch.zeros(H, W, requires_grad=True)
    raw_decoy.data[2:6, 2:6] = 1.0
    sup = raw_decoy.detach()                # caller's detach
    out = apply_masked_residual(x_base, R, sup)
    loss = out.sum()
    loss.backward()
    assert R.grad is not None and R.grad.abs().sum().item() > 0, \
        "R should receive non-zero gradient"
    # raw_decoy should NOT receive grad because we detached.
    assert raw_decoy.grad is None or raw_decoy.grad.abs().sum().item() == 0, \
        "detached support_mask must block grad to its source"
    print("  apply_masked_residual_grad_paths OK")


def _test_apply_masked_residual_input_validation() -> None:
    """Bad shapes raise clear errors. Non-detached support_mask raises."""
    H, W = 8, 8
    x = torch.zeros(H, W, 3)
    R = torch.zeros(H, W, 3)
    sup = torch.zeros(H, W)
    # Wrong x_base shape.
    try:
        apply_masked_residual(torch.zeros(H, W), R, sup)
        assert False, "should have raised on bad x_base"
    except ValueError:
        pass
    # Wrong residual shape.
    try:
        apply_masked_residual(x, torch.zeros(H, W), sup)
        assert False, "should have raised on bad residual"
    except ValueError:
        pass
    # Wrong support shape.
    try:
        apply_masked_residual(x, R, torch.zeros(H + 1, W))
        assert False, "should have raised on bad support"
    except ValueError:
        pass
    # support_mask with requires_grad=True raises (codex pre-commit invariant).
    sup_with_grad = torch.zeros(H, W, requires_grad=True)
    try:
        apply_masked_residual(x, R, sup_with_grad)
        assert False, "should have raised on grad-requiring support_mask"
    except ValueError as e:
        assert "detached" in str(e), f"unexpected error message: {e}"
    print("  apply_masked_residual_input_validation OK")


if __name__ == "__main__":
    print("memshield.semantic_compositor self-tests:")
    _test_shape_and_range()
    _test_silhouette_interior_matches_object()
    _test_no_dark_halo()
    _test_zero_offset_preserves_x_ref_outside_object()
    _test_overlap_regime()
    _test_signature_compat_with_legacy()
    _test_no_grad_through_content()
    _test_apply_masked_residual_basic()
    _test_apply_masked_residual_localization()
    _test_apply_masked_residual_clamps_to_unit_range()
    _test_apply_masked_residual_grad_paths()
    _test_apply_masked_residual_input_validation()
    print("memshield.semantic_compositor: all self-tests PASSED")
