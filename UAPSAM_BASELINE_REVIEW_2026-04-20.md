# UAPSAM Baseline Reproduction — Review & Plan (2026-04-20)

**Reviewer:** GPT-5.4 xhigh via Codex MCP (threadId `019da925-ec15-78b1-bbc0-a0257cebee46`)
**Context:** Reproducing UAPSAM ("Vanish into Thin Air", CGCL-codes/UAP-SAM2) as a fair baseline for the Decoy method.
**Purpose:** Audit the existing reproduction, decide retrain vs pivot, identify code fixes.

---

## 1. State on disk (remote: `/IMBR_Data/Student-home/2025M_LvShaoting/UAP-SAM2/`)

| Artifact | Provenance | Status |
|---|---|---|
| `uap_file/YOUTUBE.pth` | Trained w/ `loss_t + loss_diff` only (loss_fea disabled per stale report) | **EVALUATED** — weak |
| `uap_file/YOUTUBE_fixed.pth` | Produced by `uap_attack_fixed.py` on 2026-04-02; full-file rewrite; tensor differs almost-everywhere from `YOUTUBE.pth` | **NEVER EVALUATED**; flags unknown |
| `data/sav_test/JPEGImages_24fps/` | 30 folders labeled as SA-V | **FAKE** — 100% overlap with YT-VOS **valid** split, 0% with train. IDs are YT-VOS format, not `sav_*`. |
| `reproduction_report.json` | Dated 2026-03-31 20:40 | **STALE** — says `miou_*=null` but evals ran after that date |

### Existing eval numbers for `YOUTUBE.pth` (never reported in the json)

Held-out YT-VOS **valid**, 100 videos, seed=30, `limit_frames=15`, clean-IoU<0.3 filter:

| Eval | Format | clean mIoU | adv mIoU | Drop |
|------|--------|-----------|----------|------|
| `heldout_full.log` | lossless | 82.82% | 56.01% | **26.8pp** |
| `hjpeg_full.log`   | JPEG     | 82.16% | 57.94% | **24.2pp** |

Paper reports ~46pp drop (sam2-t, clean 82.8 → adv 37.03). **Current reproduction is ~half-strength** relative to paper.

### Fake SA-V verification

```
SA-V dir count: 30
overlap(SA-V, YT-VOS train): 0 / 30
overlap(SA-V, YT-VOS valid): 30 / 30   ← 100%
```
Consequence: if `YOUTUBE_fixed.pth` was trained with `--loss_fea` using this dir, teacher features came from clips in the held-out eval pool → contamination.

---

## 2. Codex's reframing of the gap

The 46pp → 25pp gap is **not primarily** a code or loss_fea issue. Most of the gap is **protocol mismatch**:

- Paper's 46pp is effectively **in-domain**: public `sam2_util.py` builds its test dataset from `train_dataset`; the official pipeline keys off `./data/YOUTUBE/train/` for both attack optimization and evaluation.
- My 25pp is **held-out transfer** (trained on YT-VOS train, tested on YT-VOS valid) — a strictly harder task.

This is the single biggest explanation. Verifying it is cheap: eval `YOUTUBE.pth` on the **same 100 train videos it was optimized on**. Expected outcome if protocol-mismatch hypothesis is correct: ≈40pp drop (≈paper).

---

## 3. Real code bugs in `uap_attack_fixed.py`

Independent of the protocol question, Codex identified two material bugs:

### Bug 1 — wrong `grad_outputs`
```python
g = grad(loss, adv_img, loss)[0]   # passes scalar loss as grad_outputs
```
With scalar `loss`, this returns `loss · ∂loss/∂x` instead of `∂loss/∂x`. Harmless if you `sign()` immediately — but this code **averages raw gradients before sign()**, so large-loss frames dominate the running mean. And `loss_ft` is `-BCE`, so total `loss` can go negative, which **flips the sign of the update**.
Fix:
```python
g = torch.autograd.grad(loss, adv_img, retain_graph=False, create_graph=False)[0]
```

### Bug 2 — history-average-then-sign saturates early
```python
sample_total_g += ema_grad
avg_gradient = sample_total_g / sample_step_count
perturbation = (perturbation - avg_gradient.sign() * alpha).clamp(±eps)
```
With α=2/255 and ε=10/255, the perturbation saturates in ~5 effective sign steps. The running average then locks in whatever sign pattern emerged early and never escapes. Fix: drop `sample_total_g`/`sample_step_count`; use per-step EMA only:
```python
ema_grad = beta * ema_grad + (1 - beta) * g.detach()
perturbation = (perturbation - args.alpha * ema_grad.sign()).clamp(-args.eps, args.eps).detach()
```

### Bug 3 (possible) — `Y=-1` BCE target
The code uses `weight_Y = -1` then `Y = torch.ones(...) * weight_Y`, so BCE targets are `-1`. `BCEWithLogitsLoss` expects `{0, 1}`. If this `Y` is actually passed into BCE (needs verification), it's optimizing a nonstandard surrogate. Fix: use `Y = torch.zeros(...)` for background-attack target (or whatever valid encoding matches paper intent).

---

## 4. Decision on `--loss_fea` and SA-V

Codex is clear: **do NOT download real SA-V.**
- λ_fea=1e-6 means it cannot explain the 20pp gap.
- The paper's "feature shift" term only needs *distractor frames from other videos*. Real SA-V is not required — the cleanest substitute is **other YT-VOS train videos**, excluding the current video and excluding the eval pool.
- Quarantine `data/sav_test/` and document it is mislabeled.

---

## 5. Recommended sequence (min credible path)

**Phase A — Diagnostics first (cheap, ~1-2 GPU-hours)**
1. Recover the exact 100 training video IDs used to produce `YOUTUBE.pth` (seed=30 sample of YT-VOS train).
2. Evaluate `YOUTUBE.pth` on that same 100-video set (in-domain control).
3. Evaluate `YOUTUBE_fixed.pth` on both same-split train-100 and held-out valid-100.
4. Measure perturbation-sign stability of both UAPs (log Hamming distance over training-steps reconstruction if possible).

**Gate:** if `YOUTUBE.pth` hits ≥40pp on same-split, we are effectively done reproducing paper-strength. The only remaining work is to report held-out transfer honestly.

**Phase B — One repaired retrain (if Phase A gate fails, ~6-10 GPU-hours)**
Protocol:
- Dataset: YT-VOS train, `limit_img=100`, seed=30 (exact same sample list as before, frozen).
- Frames: `limit_frames=15`, `prompts_num=256`, `P_num=10`, `eps=10/255`, `alpha=2/255`.
- Losses: `loss_t + loss_diff`. Enable `loss_fea` **only if** distractor pool is drawn from other YT-VOS train videos (not current, not eval).
- Apply bug fixes 1 and 2 above. Audit bug 3.
- Log: training video IDs, all flags, sign-stability trace, fraction of negative losses, eval-pool exclusion assertion.

**Phase C — Three-setting final eval (~2-4 GPU-hours)**
- Eval A: same train-100 (in-domain faithful control)
- Eval B: different train-100 (same-dataset generalization)
- Eval C: held-out valid-100 (fair comparison with Decoy)

**Total budget:** ~10-16 GPU-hours. Stop condition: if same-split control stays ≤30pp after repair, UAPSAM is documented non-reproducible — pivot narrative.

---

## 6. Claims matrix (what we can say under each outcome)

| Same-split drop (Eval A) | Allowed claims | Forbidden claims |
|---|---|---|
| **≥40pp** | "Reproduced UAPSAM under authors' in-domain protocol. On held-out valid it drops to Xpp, indicating transfer gap. Decoy beats the same held-out protocol." | — |
| **30–40pp** | "Partially reproduced the released baseline; weaker than paper under our environment. Compare Decoy vs repaired UAPSAM on held-out valid only." | "Superseded paper's headline." |
| **≤30pp** | "Our current reproduction reaches Xpp drop; gap to paper's ~46pp remains unexplained. Compare Decoy only against this repaired baseline, and keep investigating potential causes." | "Paper is unreproducible." / "UAPSAM is weak." / Using our reproduction as a strawman without disclosure. |

---

## 7. Actions required from user

Before I run anything expensive:

1. **Authorize Phase A** (~1-2 GPU-hours, diagnostic eval only, no retrain).
2. **Confirm you want Phase B on the repaired code path** (code changes land first, then retrain).
3. **Agree we do NOT download real SA-V**; distractor pool = other YT-VOS train clips.
4. **Accept that YOUTUBE_fixed.pth is quarantined until Phase A reveals its strength**; it is not a claim-bearing artifact without provenance.

---

## 8. Residual risks / open questions

- Public paper PDF accessed by Codex — need to independently confirm "paper number is in-domain" reading (Codex inferred from the released code, not the paper text).
- Exact 100-video training sample: need to find the frozen seed=30 list. `refine-logs/eval_video_ids.json` has split=train, count=100 — plausibly this, but provenance unclear.
- `Y=-1` in BCE: need to trace whether `Y_bin.bool()` path means the raw `-1` never reaches BCE. Low priority, audit during Phase B.
- JPEG-vs-lossless gap is small (26.8pp vs 24.2pp), so JPEG save/reload is not the dominant issue.

---

## 9. Execution log — Phase A + B + C (2026-04-20 → 2026-04-21)

### 9.1 User decisions that overrode the plan
- User chose **download real SA-V** (not Codex's "use YT-VOS distractors"). Real SA-V test split (17.7 GB) obtained via Meta CDN through mihomo proxy (HF was gated). Tarball turned out to be **pre-extracted JPEGs** in the exact `JPEGImages_24fps/sav_*/NNNNN.jpg` layout the paper's code expects — no ffmpeg step needed.
- User chose **in-domain eval** as primary comparison protocol (not held-out).
- Fidelity stance accepted by Codex R3: **keep paper's `Y=-1` BCE block as-is** (it's mathematically weird but numerically stable after the `grad_outputs=loss` fix; fixing it would produce a different attack, not UAPSAM).

### 9.2 Code changes landed (`uap_attack_v2.py`)
Applied in order:
1. `grad(loss, adv_img, loss)` → `grad(loss, adv_img)[0]` (Codex R1 fix 1).
2. Removed `sample_total_g / sample_step_count` history-average-then-sign; per-step EMA only (Codex R1 fix 2).
3. `^sav_\d{6}$` strict regex whitelist + `sorted()` enumeration + empty-folder preflight (Codex R2).
4. `os.makedirs("uap_file", exist_ok=True)` before save (Codex R2 nice-to-have).
5. `torch.no_grad()` around constant-feature paths: `target_feature`, `prototype_feature`, `logits_clean + get_current_out(benign_img)`, `get_current_out(adv_img, ...)` — load-bearing on V100 32GB (without it, 4 live image-encoder graphs × ~8 GB → OOM at frame 2).
6. Removed aggressive per-frame `del` and `empty_cache()` after they triggered a CUDA illegal memory access (SAM2 memory bank held references we freed too eagerly).
7. `attack_setting.py` line 20: `os.environ["CUDA_VISIBLE_DEVICES"] = "4,2"` guarded behind `if not in os.environ` so external `CUDA_VISIBLE_DEVICES=0,5` would take effect.

Final retrain: 10 outer steps × 100 videos × 15 frames, seed=30, eps=10/255, alpha=2/255, P_num=10, prompts_num=256, `--loss_t --loss_diff --loss_fea`, real SA-V distractors (fea_num=30, from 150-clip pool). EMA-sign flip rate per step: **6.39 / 6.41 / 6.36 / 6.35 / 6.42 / 6.34 / 6.41 / 6.45 / 6.42 / 6.42** — all inside the healthy 1-20% band; no saturation/freezing. Wall clock ~7h on shared V100 (GPU 5, via `CUDA_VISIBLE_DEVICES=0,5`). Output: `uap_file/YOUTUBE_v2.pth` (abs_max=eps exactly, 77.87% of entries at ±eps).

### 9.3 Final measurement — in-domain protocol, same 100 train videos (overlap = 100/100 confirmed)

Protocol: `--test_dataset YOUTUBE`, `--limit_img 100 --limit_frames 15 --seed 30 --P_num 10 --prompts_num 256`, clean-IoU<0.3 filter, JPEG save/reload path. Same for both UAPs.

| UAP | Clean J (mIoU) | Adv J | **J drop** | Clean J&F | Adv J&F | J&F drop |
|---|---|---|---|---|---|---|
| `YOUTUBE.pth` (paper-original optimizer, no loss_fea) | 82.79% | 61.12% | **−21.67 pp** | 74.24% | 53.13% | −21.10 pp |
| `YOUTUBE_v2.pth` (Codex-fixed optimizer + real SA-V) | 82.79% | **53.92%** | **−28.87 pp** | 74.24% | **46.99%** | **−27.25 pp** |
| Paper (sam2-t, YouTube in-domain) | ~82.8% | **37.03%** | **~46 pp** | — | — | — |

### 9.4 What the numbers mean

- **Codex's protocol-mismatch theory did not hold.** `YOUTUBE.pth` on in-domain (21.67pp) is no stronger than the same UAP's held-out result (24-27pp). Switching protocols does not recover the paper's headline.
- **The v2 fixes DID help, by +7.2pp J-drop** (21.67 → 28.87) on identical protocol. The optimizer-bug fix + real SA-V distractors is a genuine improvement, not a wash.
- **Clean mIoU matches paper (82.79 vs ~82.8)**, so the SAM2 model, data pipeline, and eval protocol are correct. The remaining gap is in attack strength specifically.
- **Unresolved gap to paper: ~17 pp (v2) or ~24 pp (original)**. The gap's cause is **not yet identified**; our reproduction is incomplete, not proof the paper is unreproducible. Candidate causes still to audit:
  - Different random subset of 100 train videos (paper's seed unknown).
  - Undocumented hyperparameters (e.g., different α/ε schedule, longer training).
  - Possible interactions we haven't isolated: prompt sampling variance, specific data preprocessing, cuDNN determinism.
  - Remaining bugs Codex and we did not catch.
  - Paper's number may be on a different specific subset that the public code samples non-reproducibly.

### 9.5 Current artifact identity

Per Codex R3's naming advice, the result is **"UAPSAM-v2 (optimizer-bug-fixed, BCE artifact retained, real sav_test distractors, in-domain J-drop = 28.87pp on 100-video train-split control)"**. It is **not** plain "UAPSAM" — two optimizer bugs have been corrected vs the public release. Any downstream comparison should use this name to stay honest.

### 9.6 What to say in paper / downstream comparison
- For Decoy vs UAPSAM baseline comparison, use `YOUTUBE_v2.pth` under a common protocol (held-out valid is still pending — run after SSH recovers).
- Report: "Our best repaired UAPSAM reproduction achieves 28.87pp J-drop (27.25pp J&F-drop) on YT-VOS in-domain 100-train control. A ~17pp gap to the paper's reported 46pp remains unexplained; investigation is ongoing. We use this repaired UAPSAM as the baseline."
- Do **not** write: "UAPSAM is unreproducible" / "paper's claim cannot be verified". We have not exhausted the debugging avenues.

### 9.7 Still TODO
1. Run `YOUTUBE_v2.pth` on held-out valid 100 (JPEG + lossless). This is the comparable protocol for Decoy. ~30 min when GPU + SSH are available.
2. Update `reproduction_report.json` with the current numbers and artifact-provenance fields.
3. Continue auditing the gap: try sweeping seeds, extended P_num, alternative α schedule.

---
*Codex threadId for follow-up: `019da925-ec15-78b1-bbc0-a0257cebee46`*
