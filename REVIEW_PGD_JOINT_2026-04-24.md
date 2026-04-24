# Research-Review: PGD joint δ + ν optimization — formulation sanity check (2026-04-24)

**Thread ID**: `019dbdaa-d34d-79e3-9331-e87d3b254784`
**Question**: Is jointly optimizing (δ, ν) in one PGD loop with shared η reasonable, given empirical additivity-collapse in early decisive results?

---

## Context

Per-video PGD in VADI jointly updates two leaf tensors:
- `δ` ∈ [T, H, W, 3] — per-frame ℓ∞-bounded perturbation (ε_0=2/255 on f0, ε_other=4/255)
- `ν` ∈ [K, H, W, 3] — per-insert perturbation, **no ℓ∞ bound**, constrained only by LPIPS ≤ 0.35 + TV hinges
- Shared η = 2/255 sign-grad
- 3-stage schedule (N_1=30 attack-only / N_2=40 fidelity regularization / N_3=30 η halved)

**Early decisive result (dog, 2026-04-24)**:

| config | δ? | ν? | exported J-drop |
|---|---|---|---|
| K3_top (both) | ✓ | ✓ | 0.449 |
| K3_delta_only_top (K=0 phantoms, δ only) | ✓ | — | 0.421 |
| K3_insert_only_top (K=3 inserts, freeze δ) | — | ✓ | 0.407 |

Joint buys only +0.03 over the better single arm.

---

## Codex verdict — one-line

> "The joint loop is **not trivially broken, but it is not a well-principled formulation either**. Right now it looks more like an **engineering convenience than a defensible joint threat model**, and your own pilot already weakens the main reason to keep it."

---

## Key findings

### 1. Biggest issue = threat-model mismatch, not optimizer mismatch

- δ is hard-ℓ∞-bounded on existing frames.
- ν is a synthesized inserted-frame channel with only soft LPIPS/TV control.
- These are **different attack classes**. Sharing one sign-PGD loop + one η gives them no common geometric meaning.
- The η = 2/255 is NOT moot: δ saturates its ε-ball in 2 steps (then just sign-flips); ν has no analogous hard budget so the same step is arbitrary. This **biases search toward the freer arm** (ν).

### 2. Formulation bakes in redundancy by design

- δ support `S_δ` is explicitly tied to insert ± 2 neighborhoods.
- Both arms thus act around the same temporal neighborhoods.
- If placement is wrong or clustered, both arms become redundant together.

### 3. K3_random > K3_top is a bigger warning than the joint/non-joint gap

Suggests vulnerability ranking is not causally aligned with attack effectiveness once inserts are present — undercuts the mechanism that justifies coupling δ and ν in the first place.

### 4. Most plausible interpretation of the additivity collapse

**(a) + (b), with some (d)**:

- **(a) Shared bottleneck**: both arms perturb the same failure pathway in SAM2's temporal update / current-frame processing → once one arm pushes into the decoy basin, the other has little left to add.
- **(b) Objective ceiling**: margin loss already near-satisfied by either arm alone → weak incentive for joint to add J-drop.
- **(d) Poor calibration**: real (ν's geometry is wrong, favorable stage-1 schedule), but not the primary explanation.

If "ν starves δ" were the main issue, joint would collapse to just one arm. Instead all three configs land in the same band → **shared bottleneck + loss saturation** dominates.

---

## The one decisive experiment: best-response composition test

1. Optimize `δ-only` → `δ*`.
2. Optimize `ν-only` → `ν*`.
3. Evaluate exported `(δ*, 0)`, `(0, ν*)`, and `(δ*, ν*)` with **no further optimization**.
4. Then start from `(δ*, ν*)` and run **alternating** updates with separate schedules η_δ ≠ η_ν.

**Interpretation key**:
- `(δ*, ν*)` marginal over best single arm + alternating adds little → **redundancy/ceiling**
- `(δ*, ν*)` or alternating jumps above current joint PGD → **simultaneous-shared-η loop IS the problem**

Also record whether **margin term is already saturated** in each single-arm optimum — if yes, strong support for the ceiling story.

---

## Method-design recommendations

Codex's ordered preference:

1. **(iii) Separate threat models** — treat δ-only and insert-only as DIFFERENT attacks with different fidelity profiles and stop trying to combine them. **Preferred**.
2. **(ii) + (i)**: if you insist on a combined attacker, use alternating / block-coordinate optimization with separate η_δ and η_ν. Keep sign-PGD for δ; **do NOT force the same update rule onto ν**.
3. **(iv)** Adam / proximal-style update for ν (sensible because of soft LPIPS/TV constraints), not for δ.
4. **(v) Mandatory either way**: explicitly report the **low marginal gain of joint over the best single arm**.

---

## Publication implication

- **As a positive-method paper**: the joint δ + ν story is **basically dead** if the decisive round confirms the pattern. Not defensible as a core contribution in its current form.
- **As an audit paper**: the story is **much stronger** — "a more complex joint attacker is not meaningfully stronger than simpler single-arm baselines, and the supposedly principled placement heuristic loses to random". Publishable if it holds across enough videos + seeds with achieved-fidelity statistics (not just hinge satisfaction).

---

## Implication for the currently-running decisive experiment

The decisive round (10 clips × 4 configs) is already collecting exactly the data needed to **quantify the additivity-collapse finding across clips**. No need to rerun. After it finishes:

- Per-clip table of (K3_top, K3_delta_only, K3_insert_only) J-drops → directly measures additivity collapse.
- If pattern holds across 10 clips with paired variance → the audit paper has its core empirical finding.

The best-response composition test (codex's proposed one-experiment-that-resolves-it) is a **followup** that could strengthen the audit paper — it's not needed before the branch decision.

---

## TODO after decisive finishes

1. Read `decisive_decision.json`. If AUDIT_PIVOT (highly likely given early signal):
   - Draft audit paper outline with three empirical pillars:
     - **Surrogate-vs-exported 60pp gap** (from pilot Round-1)
     - **Placement inversion** (K3_random > K3_top across 10 clips)
     - **Additivity collapse** (K3_joint ≈ K3_δ_only ≈ K3_ν_only across 10 clips)
   - Optionally plan the best-response composition experiment as a follow-up section.
2. If NARROWED_PROCEED: revisit whether to keep joint-PGD framing or switch to alternating / separated optimizers before the main table.
