"""Parity test for track_step gradient checkpointing (codex round 22, 2026-04-30).

Verifies that toggling `use_track_step_checkpointing` produces equivalent
forward logits and backward gradients on a real DAVIS clip + real SAM2.1-Tiny
predictor (eval mode, frozen weights, bf16 autocast).

Acceptance:
  - logits:  torch.allclose(out_off, out_on, atol=1e-4, rtol=1e-4) per probe fid
  - grad:    cosine(g_off, g_on) > 0.9999  AND  ||g_on-g_off||/||g_off|| < 1e-2
  - loss:    |L_on - L_off| < 1e-3 * |L_off|

Probe frames: 0 (mask-input shortcut), 1 (post-init), 7 (post-mem-window), mid, last.

Smoke-checks both __call__ and forward_with_objectness (the two live paths).

Run on Pro 6000 inside conda memshield env:
  python -m _run.parity_track_step_ckpt
"""
from __future__ import annotations

import sys
import pathlib
import torch
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from scripts.run_vadi_pilot import build_pilot_adapters
from scripts.run_vadi_v5 import load_davis_clip


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af, bf = a.flatten().double(), b.flatten().double()
    if af.norm() == 0 or bf.norm() == 0:
        return 0.0
    return (af @ bf / (af.norm() * bf.norm())).item()


def rel_l2(actual: torch.Tensor, ref: torch.Tensor) -> float:
    diff = (actual - ref).flatten().double().norm()
    nref = ref.flatten().double().norm()
    return (diff / nref).item() if nref > 0 else float("inf")


def run_one(predictor, clean_pass_factory, x_clean, prompt_mask,
            use_ckpt: bool, probes, method: str = "__call__"):
    """Build a VADIForwardFn with the requested ckpt flag, run forward+backward
    on a freshly-leafed processed tensor, return (logits_dict, loss_value, grad)."""
    from memshield.vadi_sam2_wiring import VADIForwardFn

    H, W = int(x_clean.shape[1]), int(x_clean.shape[2])
    fwd = VADIForwardFn(
        predictor=predictor,
        prompt_mask=prompt_mask,
        video_H=H, video_W=W,
        device=x_clean.device,
        autocast_dtype=torch.bfloat16,
        use_gradient_checkpointing=True,
        use_track_step_checkpointing=use_ckpt,
    )

    # Fresh leaf with grad
    proc = x_clean.detach().clone().requires_grad_(True)

    if method == "__call__":
        logits = fwd(proc, return_at=probes)  # {fid: [H, W]}
    elif method == "forward_with_objectness":
        logits, _objs = fwd.forward_with_objectness(
            proc, return_at=probes, objectness_at=probes,
        )
    else:
        raise ValueError(f"unknown method {method}")

    loss = torch.zeros((), device=x_clean.device, dtype=torch.float32)
    for fid in probes:
        loss = loss + logits[fid].float().sum()
    loss.backward()
    grad = proc.grad.detach().clone()
    out = {fid: logits[fid].detach().clone() for fid in probes}
    L = float(loss.detach().item())
    return out, L, grad


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    checkpoint = pathlib.Path("checkpoints/sam2.1_hiera_tiny.pt")
    clean_pass_factory, _, _, _, _ = build_pilot_adapters(checkpoint, device)

    # Load dog (T=51, well-known).
    x_clean, prompt_mask = load_davis_clip(pathlib.Path("data/davis"), "dog")
    x_clean = x_clean.to(device)
    print(f"dog: T={x_clean.shape[0]} H={x_clean.shape[1]} W={x_clean.shape[2]}")

    # Need a predictor instance for VADIForwardFn. Pull from the closure.
    fac_dummy = clean_pass_factory("dog", x_clean, prompt_mask)
    # The factory returned a closure; the predictor is captured inside.
    # Instead, reuse build_pilot_adapters' predictor by inspecting closure.
    # Cleanest: build predictor directly.
    from sam2.build_sam import build_sam2_video_predictor
    cfg_path = "configs/sam2.1/sam2.1_hiera_t.yaml"
    predictor = build_sam2_video_predictor(cfg_path, str(checkpoint), device=device.type)
    for p in predictor.parameters():
        p.requires_grad_(False)
    predictor.eval()

    # Probe frames
    T = int(x_clean.shape[0])
    probes = [0, 1, 7, T // 2, T - 1]
    print(f"probes = {probes}")

    print("\n=== Method: __call__ ===")
    print("Run 1: track_step checkpoint OFF")
    out_off, L_off, g_off = run_one(predictor, clean_pass_factory, x_clean,
                                    prompt_mask, use_ckpt=False,
                                    probes=probes, method="__call__")
    print(f"  loss = {L_off:.6f}, max|out|={max(o.abs().max().item() for o in out_off.values()):.4f}, "
          f"||grad||={g_off.norm().item():.6f}")
    torch.cuda.empty_cache()

    print("Run 2: track_step checkpoint ON")
    out_on, L_on, g_on = run_one(predictor, clean_pass_factory, x_clean,
                                 prompt_mask, use_ckpt=True,
                                 probes=probes, method="__call__")
    print(f"  loss = {L_on:.6f}, ||grad||={g_on.norm().item():.6f}")

    # Compare per-frame logits
    print("\n--- per-frame logit diff ---")
    fail_logits = []
    for fid in probes:
        a, b = out_off[fid], out_on[fid]
        amax = (a - b).abs().max().item()
        ok = torch.allclose(a, b, atol=1e-4, rtol=1e-4)
        print(f"  fid={fid:>3}: max|diff|={amax:.6e}  allclose(atol=1e-4)={ok}")
        if not ok:
            fail_logits.append((fid, amax))

    # Compare loss
    L_diff = abs(L_on - L_off)
    L_rel = L_diff / max(abs(L_off), 1e-9)
    print(f"\n--- loss ---  L_off={L_off:.6f}  L_on={L_on:.6f}  |Δ|={L_diff:.4e}  rel={L_rel:.4e}")
    L_ok = L_rel < 1e-3

    # Compare grad
    cos = cos_sim(g_on, g_off)
    rl2 = rel_l2(g_on, g_off)
    g_max = (g_on - g_off).abs().max().item()
    print(f"--- grad ---  cos(g_off, g_on)={cos:.6f}  ||Δg||/||g||={rl2:.4e}  max|Δg|={g_max:.4e}")
    g_ok = (cos > 0.9999) and (rl2 < 1e-2)

    print("\n=== Method: forward_with_objectness ===")
    print("Run 1: track_step checkpoint OFF")
    out_off2, L_off2, g_off2 = run_one(predictor, clean_pass_factory, x_clean,
                                       prompt_mask, use_ckpt=False,
                                       probes=probes,
                                       method="forward_with_objectness")
    torch.cuda.empty_cache()
    print("Run 2: track_step checkpoint ON")
    out_on2, L_on2, g_on2 = run_one(predictor, clean_pass_factory, x_clean,
                                    prompt_mask, use_ckpt=True,
                                    probes=probes,
                                    method="forward_with_objectness")

    fail2 = []
    for fid in probes:
        a, b = out_off2[fid], out_on2[fid]
        amax = (a - b).abs().max().item()
        ok = torch.allclose(a, b, atol=1e-4, rtol=1e-4)
        if not ok:
            fail2.append((fid, amax))
    cos2 = cos_sim(g_on2, g_off2)
    rl22 = rel_l2(g_on2, g_off2)
    print(f"  obj-method: logit fails={fail2}  cos(g)={cos2:.6f}  rel L2={rl22:.4e}")
    g_ok2 = (cos2 > 0.9999) and (rl22 < 1e-2)

    print("\n=== VERDICT ===")
    print(f"  __call__ logits OK:        {len(fail_logits) == 0}  (fails: {fail_logits})")
    print(f"  __call__ loss within 1e-3: {L_ok}")
    print(f"  __call__ grad cos>0.9999 & rel L2<1e-2: {g_ok}  (cos={cos:.6f}, rl2={rl2:.4e})")
    print(f"  forward_with_objectness logits OK: {len(fail2) == 0}")
    print(f"  forward_with_objectness grad OK:    {g_ok2}  (cos={cos2:.6f}, rl2={rl22:.4e})")
    overall = (len(fail_logits) == 0 and L_ok and g_ok and len(fail2) == 0 and g_ok2)
    print(f"\nOVERALL: {'PASS — safe to commit + deploy' if overall else 'FAIL — investigate'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
