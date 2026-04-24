# Post-Pilot Review (GPT-5.4 xhigh) — 2026-04-24

**Thread ID**: `019dbd43-142b-7dd1-9cb1-b7443a18c3e2`
**Repo HEAD at review time**: `10b6d6a` + uncommitted OOM ckpt fix + gate-source purity fixes.
**Pilot run**: 3 clips (dog, cows, bmx-trees) × 3 configs (K1_top, K1_random, K3_top), DAVIS 480p, ε=4/255, 100-step PGD.

**User's reframing question**: "只要攻击有效，能破坏 SAM2 分割，对数据进行保护就可以了 — 现在的结果符合这个 claim 吗?" (Is data-protection-via-attack-disruption, dropping all other VADI claims, supported by current results?)

---

## Pilot empirical state (for the record)

| Clip | K1_top | K1_random | K3_top | surrogate K3_top |
|------|--------|-----------|--------|-----|
| dog       | 0.0856 | 0.0801 | 0.4263 | 0.964 |
| cows      | 0.0534 | 0.0658 | 0.4796 | 0.995 |
| bmx-trees | 0.0639 | 0.0463 | 0.2245 | 0.987 |
| mean      | 0.0676 | 0.0641 | 0.3768 | 0.982 |

**Gate decision**: **NO-GO** (cond1 FAIL 0/3; cond2 PASS 3/3; diagnostic FAIL 0/3).

Δμ diagnostic at K3_top — |Δμ_true| dominates |Δμ_decoy| in **all 9 runs**.

---

## Reviewer verdict — one-line summary

> "You currently have **a real attack effect, a failed mechanistic story, and not yet a defensible protection paper.**"

---

## Key findings (from the review)

### 1. The narrower claim is supported only as a *pilot existence proof*, not as a *publishable protection claim*

- Effect is real at K=3 (3/3 clips). ~38% exported J-drop exists.
- But **n=3 is tiny**. Rough 95% t-interval on mean J-drop is roughly **[0.04, 0.71]** — publication-useless width.
- J-drop = 0.377 → remaining overlap ≈ J = 0.623. That's **disruption, not protection**. A human annotator or a different segmenter may still recover the mask.
- The evaluation protocol (exported uint8 → fresh SAM2) is actually one of the strongest parts — it's credible.

### 2. Two falsifications stand regardless of reframing

- **cond1 FAIL** (K1_top Δ vs K1_random mean +0.002) → "vulnerability-aware" is not purchasing anything at K=1.
- **diagnostic FAIL** (|Δμ_true| > |Δμ_decoy| uniformly) → "decoy adoption" mechanism is wrong; this is a **suppression** attack, not a decoy attack.
- Unless K=3 top clearly dominates K=3 random (UNTESTED), the paper can't be called "vulnerability-aware" at all.
- The attack should be renamed to something honest: "insert-assisted local suppression", "context-poisoning via sparse frame insertion", or "insert-mediated SAM2 suppression".

### 3. Missing baselines are mandatory even for the narrowed story

- **K3_random** — is placement at K=3 still irrelevant? Critical.
- **δ-only** (no inserts) — are inserts even necessary?
- **insert-only** (no δ) — isolate insert contribution.
- Fidelity distributions (mean/median/tails of LPIPS + SSIM on exported).
- **n ≥ 10 clips minimum**. n=3 is not a paper, it's a sanity check.

### 4. Top-venue publishability as-is — **NO**

- Prior literature already sets a high bar: DarkSAM (NeurIPS 2024), RGA on SAM (2024), DAG (ICCV 2017), Metzen universal adversarial perturbations (ICCV 2017) — these operate at multi-dataset / multi-prompt / white-and-black-box scale.
- 38% J-drop on 3 clips is not competitive on raw attack evidence.
- **Only plausible top-venue angle**: the **threat model** (publisher-side exported-uint8 poisoning for downstream-consumer protection). Lean on novelty of *threat model*, not strength of numbers.

### 5. Minimum package to get to "maybe publishable"

- 10+ clips immediately, 20-50 for a serious paper.
- K3_random, δ-only, insert-only, a placement-alternative baseline.
- Fidelity distributions + visual audit.
- Robustness to codec/re-encode/resize (threat-model critical — consumer will re-encode).
- At least one additional consumer model (e.g., SAM2Long, SAM2.1-Base) or a recovery baseline.
- A **practical protection metric**: annotation-time increase OR downstream-training degradation OR cross-model failure. Not just IoU.

---

## Claims matrix (for the reframed narrower story)

| K3_random vs K3_top | δ-only vs K3_top | What the paper can honestly claim |
|---|---|---|
| close | close | Small perturbations disrupt SAM2 on some clips, but inserts and placement are not needed. **Method novelty basically gone. Top-venue method paper: no.** |
| close | much worse | Inserts matter, placement does not. **Workshop/specialized venue**, mild contribution, unless scale + protection evidence become strong. |
| much worse | close | Placement matters, inserts do not. **Becomes vulnerability-localization story, not VADI.** Still weak unless effect scales + fidelity is strong. |
| much worse | much worse | **Best case.** Placement and inserts both matter. Original VADI story partly comes back, but mechanism must be rewritten honestly (not decoy cooperation). |
| K3_random > K3_top | any | **Scorer is wrong.** Drop "vulnerability-aware" entirely. |

**Fidelity overlay**: if LPIPS/SSIM tails are bad, you lose "stealthy protection" even if attack strength holds. Fidelity-first matters for the data-protection threat model.

---

## Recommended next step — option (d): hard-gated decisive round

Not (a) "proceed with narrowed paper now" — too weak empirically.
Not (c) "drop" — premature; one more decisive round is cheap.

**1 round on +7 clips** (total 10 clips) with exported-artifact metrics only, 4 configs:
- K3_top
- K3_random
- δ-only
- insert-only

Decision branch:
- If K3_top **clearly beats both** K3_random AND δ-only → proceed with narrowed paper (drop decoy claims, keep threat-model novelty).
- If K3_top **does NOT clearly beat both** → pivot to **(b) attack-surface analysis of SAM2** — surrogate-vs-exported gap + placement insensitivity + suppression-dominant behavior as primary content.

Estimated cost: ~7 clips × 4 configs × 5 min/config ≈ 2.3 GPU-hours on Pro 6000 GPU1. Well within budget.

---

## Narrative reframing (if proceed)

**Old story**: "VADI = vulnerability-aware decoy insertion against SAM2 video segmentation"
**New honest story**: "Publisher-side uint8-artifact poisoning for SAM2 segmentation protection via insert-assisted local suppression" (or similar).

Explicit concessions to make in the paper:
- Vulnerability scoring doesn't beat random at K=1 (possibly not at K=3 either).
- Loss function acts as suppression, not decoy adoption.
- Surrogate metric overestimates exported J-drop by ~60pp (this is actually a useful side-finding — prior MemoryShield claims of 92% J-drop likely had this artifact).

**What stays novel**:
- Threat-model framing (publisher-side data protection for exported video).
- Exported-artifact evaluation rigor.
- Concrete fidelity budgets + hard-S_feas acceptance on uint8 re-measure.

---

## Literature bar reviewer cited

- [DarkSAM (NeurIPS 2024)](https://arxiv.org/abs/2409.17874)
- [RGA on SAM](https://arxiv.org/abs/2411.02974)
- [DAG](https://openaccess.thecvf.com/content_ICCV_2017/html/Xie_Adversarial_Examples_for_ICCV_2017_paper.html)
- [Metzen universal perturbations for segmentation](https://openaccess.thecvf.com/content_ICCV_2017/html/Metzen_Universal_Adversarial_Perturbations_ICCV_2017_paper.html)
