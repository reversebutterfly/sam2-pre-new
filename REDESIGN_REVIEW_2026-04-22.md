# Insert Scheme Redesign Review (2026-04-22)

User directive: "重新考虑插帧方案，不应该考虑实现成本，不要复用已有代码实现，方案能取得好的效果才是最重要的。"

Reviewer: `gpt-5.4` xhigh via codex MCP, thread `019db2f5-ec85-7ae2-b84b-557374e69c47`.

## Round 1 — Score 8/10, verdict "almost"

Full review below. Stop-condition met (score ≥ 6, verdict contains "almost"), but the verdict is explicitly conditional on implementing the proposed hidden-state attack correctly (not as another output-logit tuning pass). Round 2 pressure-tests the biggest realization gap.

### Proposed top architecture: bilevel state-control attack on `memory_attention` K/V

Ranked attack surfaces (highest → lowest leverage):
1. `memory_attention` K/V tokens from inserted bank entries (read interface controlling future localization)
2. `maskmem_features` + `maskmem_pos_enc` (write interface feeding K/V)
3. `obj_ptr` tokens (less spatial, often secondary for relocation)
4. Image-encoder features of attacked prefix frames
5. `pred_iou` head (conditional on multimask branch, off by default)

Key design principles:
- Attack **both** read-path (cross-attention hijack) and write-path (memory write poisoning); if forced to pick one → read-path K/V hijack wins, because that is where the poisoned bank captures future queries.
- **Abandon Poisson-cloned decoy-object inserts** as the main mechanism. Use video-generator / interpolation / inpainting bases so inserts look like *plausible counterfactual frames* of the target drifting / self-occluding / transitioning, NOT a visibly pasted decoy.
- **Dynamic insertion position = finite-horizon controllability score**, not uncertainty / motion / learned RL scheduler.
- **K=4 inserts** optimal (1 early anchor + 1 middle rewrite + 2 late occupancy) — state-limited, not pixel-limited. `num_maskmem=7` only admits 6 rolling slots, so more than 4-5 cannibalizes.
- **Self-propagation loss**: force the first R=3 clean post-prefix frames to realign to the virtual poisoned state Z*, so the attack rewrites itself instead of self-healing.

### Concrete attack objective (math)

**Step 1 — clean trajectory + counterfactual**: run SAM2 clean once, get `(C_t, s_t^c, B_t^c, Q_t^c)` as model-aligned true regions + hidden states. Choose counterfactual trajectory `D_t` (real distractor track or flow-consistent displaced trajectory in free space).

**Step 2 — controllability score for position selection**:
```
S_t = Σ_{u=t+1}^{T_eval+H} γ^(u-t) × (
          ||∂ℓ_u/∂K_t||_F
          + 0.5 × ||∂ℓ_u/∂V_t||_F
          + 0.25 × ||∂ℓ_u/∂p_t||_2
       )
where ℓ_u = μ(g_u, C_u) − μ(g_u, D_u)
```
Pick top-K positions with min-gap 2, force one insert into last 2 prefix frames.

**Step 3 — bilevel optimization**:
- Outer: `(P*, Z*) = argmin_{P,Z} L_post + λ_r L_read + λ_s L_self + λ_m L_man`, each `Z_k = (K*_k, V*_k, p*_k)`
- Inner (realization): `min_{δ,z} L_post + λ_r L_read + λ_s L_self + λ_w Σ_k ||Ψ(x^ins_k(z,δ)) − Z*_k||_2^2 + λ_f L_fid`

**Step 4 — component losses**:
```
L_post = (1/|U|) Σ_u [ softplus(μ(g_u,C_u) − μ(g_u,D_u) + κ)
                      + λ_o softplus(τ_o − s_u)             # object presence
                      + λ_a |area(σ(g_u)) − area(C_u)| ]    # area preservation
                                                             # (wrong-but-present, not suppression)

L_read = (1/|U|) Σ_u [ −log A_u^ins + β log(A_u^anc + ε) ]
           A_u^ins = avg cross-attn mass from queries ∈ C_u to inserted-memory tokens
           A_u^anc = corresponding mass to conditioning frame + recent clean anchors

L_self = (1/R) Σ_{u=T_eval}^{T_eval+R-1} ||Ψ(x_u) − Z*_u||_2^2
           (R=3 clean post-prefix frames rewrite to match Z*)

L_man: manifold regularizer keeping Z* in encoder range
L_fid: LPIPS + ΔE + seam-band TV against f_prev
L_IoUflip (multimask only): softplus(max_{j≠m*} IoU_j − IoU_{m*} + κ_iou)
```

**Step 5 — PGD schedule (replaces 3-stage warmup)**:
- 40 outer steps relaxed top-K position search
- 150-200 AdamW on virtual-state inner problem
- 250-300 Adam on insert-generator latents + original-frame perturbations, full objective active from step 0
- 60-80 projected MI-FGSM in DCT + seam-band residual space, EOT over quantization/JPEG/mild-blur

### Failure modes flagged by reviewer

1. Virtual K/V state optimal under unconstrained Z* may be weakly realizable under `LPIPS ≤ 0.10` — **realization gap is the main practical risk**
2. Natural-distractor videos (cows) may need semantic identity transition, not just spatial K/V poisoning
3. Last insert too early → bank self-heals before eval window even with large insert loss
4. Without strong objectness + area constraints, optimizer collapses back into suppression
5. Long unroll is chaotic → must stabilize with latent optimization, EOT, seed averaging; else mistake optimizer noise for mechanism

### What would convince the reviewer (minimum experimental package)

- Full ablation vs CVaR baseline: output-only / write-only / read-only / combined state-control / combined + generator
- Direct memory-read instrumentation: attention mass from foreground queries to each bank slot over time
- Direct memory-write instrumentation: cosine/MSE between realized K/V and target Z*, + obj_ptr drift
- Position-study: fixed 3/7/11 vs oracle controllability vs learned scheduler (if oracle barely beats fixed, no scheduler story)
- K ablation {1..5} showing occupancy logic
- Fidelity split attacked-originals vs inserted, + insert LPIPS + human realism study
- Transfer beyond one checkpoint: SAM2Long + one other SAM2 variant
- Defense implication (write gating / attention temperature clipping / memory selection knocks attack down) — turns "stronger attack" into "architectural vulnerability"

### Claimed headline metric (reviewer's estimate)

- LPIPS ≤ 0.10 AND ≥ 70% J-drop on all non-saturated clips: **not credible** with pixel-PGD + Poisson inserts
- With generator-backed hidden-state attack: plausible on easier non-distractor clips and some distractor clips
- Robust claim: insert LPIPS ~0.12-0.14 on hard distractor cases, or K=4 + stronger generator. For 0.10 blanket claim: keep it clip-conditional.

## Round 2 — Realization gap, generator choice, honest expectations

Score after steelman: **6.5/10**, verdict **"almost, conditional"**. Loop stop condition met.

### Generator choice (realization engine)

**Do NOT use full diffusion as primary**. Recommended: **ProPainter-initialized flow-guided residual U-Net**.
- Inputs: `f_{t-1}, f_{t+1}`, forward/backward optical flow, occlusion maps, optional depth, low-dim latent `z_k`
- Outputs: soft edit mask `M_k`, displacement field `Δu_k`, RGB residual `R_k`
- Compose: `x_k^ins = Warp(x_base, Δu_k) + M_k ⊙ R_k`
- Advantages: feedforward, fully differentiable, strongly anchored to clean context, smaller realization gap than SVD/I2V
- GPU footprint: 8-12 GB single-insert latent opt over 250 steps; 24-32 GB end-to-end with 6-10 frame SAM2 horizon + attention hooks (fp16); 32-40 GB with LoRA-tuned generator

Diffusion fallback (only if generator needs richer priors): VideoLCM / AnimateLCM 4-6 steps (few-step distilled), NOT SVD 50 steps. OSV one-step exists but not safest first choice.

### Realization gap experiment (GATING TEST — highest information/GPU-hour)

**Single clip, single insert, short horizon**:
1. Choose easy non-distractor clip (e.g. `dog`), one insert after original f11
2. Optimize free virtual state `Z* = (K*, V*, p*)` injected DIRECTLY into memory bank for 200 Adam steps against `L_post + λ L_read` over next 6 clean frames
3. Freeze Z*, optimize one realizable insert `x_ins` under `LPIPS(f_prev, x_ins) ≤ 0.10` for 250 steps
4. Report **whitened**: `||Ψ(x_ins) − Z*||_2 / ||Z*||_2` (concatenated `(K, V, p)` state vector, whitened so scales comparable) AND downstream loss ratio `L_post(x_ins) / L_post(Z*)`

**Thresholds (reviewer's gating):**
- ratio ≤ 0.25 → **good**, bilevel design viable
- 0.25-0.40 → **usable**, minor scope adjustment
- 0.40-0.50 → **shaky**, re-evaluate
- \> 0.50 → **KILL the bilevel story**; downgrade to "not ready"

### obj_ptr dynamics (clarification)

- `multimask=False` does NOT freeze obj_ptr. Pointer still comes from single SAM output token → **attacked originals CAN move it** through image + memory-conditioned features.
- Split attack surface: **inserts move `maskmem_features`**, **prefix original perturbations on `f0` / frame before insert / first 2 post-insert clean frames drive pointer drift**.
- Expected cosine drift under `ε_orig = 4/255`: 10-20% on affected prefix frames. Enough to help persistence, not enough to carry attack alone.
- **Attack ceilings** (reviewer's estimate):
  - Without pointer drift at all: J-drop ceiling ≈ 0.50-0.55
  - With modest pointer drift from originals: ceiling ≈ 0.58-0.62

### Cross-attention mass measurement

- Aggregate last **2 memory-attention layers only** (closest to SAM head)
- For each frame `u`: take query tokens whose receptive fields fall in `C_u`, sum attention to memory tokens sourced from each inserted frame, avg over heads, sum over inserted slots
- Also log `A_u^cond` (conditioning frame) and `A_u^clean` (remaining recent slots)
- No trustworthy universal threshold from prior work. Practical heuristic:
  - inserted-slot mass < 0.20 → noise
  - 0.30-0.40 in last 2 layers, while beating any single clean slot → mislocalization plausible
  - \> 0.50 → strong hijack

### Chaos control (optimizer stability)

- Pilot: 3 seeds; paper: 5 seeds
- Mean-over-seeds for headline metric, MEDIAN / TRIMMED MEAN for architecture selection (prevent lucky-run bias)
- Inside optimization: 2-sample gradient averaging per step + fixed deterministic samplers + momentum + EMA/Lookahead over perturbation/latent — ~2× cost, most variance reduction

### Honest attack-strength prediction (assumptions: realization gap exists, multimask off by default, DAVIS-10 mix, LPIPS ≤ 0.10)

- **Best bet: 0.58 mean SAM2 J-drop**, bracket 0.55-0.60
- Non-distractor clips: 0.65+
- Distractor-heavy clips: drag the mean down
- **Would NOT bet 0.70** at LPIPS ≤ 0.10 with multimask off

### Steelmen against the design

Three scenarios that could kill the bilevel approach:
1. Unconstrained hostile state sits off the realizable image manifold → best LPIPS≤0.10 insert stays close to clean memory → no advantage over direct logit attack
2. Conditioning frame f0 + current-frame evidence dominate read path → poisoned bank only perturbs 1-2 frames before clean writes recover
3. `obj_ptr` carries more identity stability than assumed → spatial memory moves but pointer stays near-clean → decoder self-corrects

If ANY of these three surface in the single-insert realization test → bilevel state-control is the wrong main paper.

## Method Description (for paper/illustration)

**Final proposed scheme: Bilevel memory-state control attack on SAM2**

Given clean video `x_{0:T}` with first-frame mask `m_0`, the attacker outputs a modified video as follows:

1. **Dynamic position selection**: compute finite-horizon controllability score `S_t` at each candidate insertion time by measuring the sensitivity of the post-prefix localization loss to virtual perturbations of `memory_attention` K, V, and `obj_ptr` at time `t`. Pick top-K=4 positions with min-gap 2, forcing one insert into the last 2 prefix frames (anchor near eval boundary).

2. **Outer optimization (virtual state)**: jointly optimize positions P and virtual memory states `Z*_k = (K*_k, V*_k, p*_k)` to minimize a four-term objective: (a) post-prefix mislocalization `L_post` (wrong-but-present tracking with object-presence + area-preservation terms), (b) read-side cross-attention hijack `L_read` (foreground queries attend to inserted slots, not clean anchors), (c) self-propagation `L_self` (first R=3 post-prefix clean frames realign to Z*), (d) manifold regularizer `L_man` keeping Z* in encoder range.

3. **Inner optimization (realization)**: realize each `Z*_k` with a ProPainter-style flow-guided residual U-Net generator that outputs a soft edit mask, displacement field, and RGB residual conditioned on clean neighbors `f_{t-1}, f_{t+1}` and optical flow. Optimize generator latent `z_k` and prefix perturbations `δ` (L∞ ≤ 4/255 on originals, ≤ 8/255 inside edit region) to minimize `L_post + λ_r L_read + λ_s L_self + λ_w Σ ||Ψ(x^ins_k) − Z*_k||^2 + λ_f L_fid`.

4. **PGD schedule**: 40 outer position steps → 150-200 AdamW on virtual state → 250-300 Adam on latent+perturbation (full objective active from step 0) → 60-80 projected MI-FGSM in DCT + seam-band residual space with EOT over quantization / JPEG / mild blur.

5. **Expected result (honest bracket)**: ~0.58 mean SAM2 J-drop on DAVIS-10 at insert LPIPS ≤ 0.10 with multimask_output_for_tracking=False. If multimask flipped on, add pred-IoU candidate-flip auxiliary loss for further lift on mask-write channel.

Data flow: clean video + `m_0` → controllability score → position set → virtual state optimization → generator realization → adversarial prefix + inserts → SAM2 tracking → post-prefix J-drop measured on eval window.

## Loop status

- **Stop condition met on Round 2** (score 6.5/10, verdict "almost, conditional"). MAX_ROUNDS=4 not reached; stopped early per threshold.
- Gating next step: **single-insert realization-gap test on dog clip** (cheapest, highest-information experiment). If ratio ≤ 0.40 → proceed to full bilevel implementation. If > 0.5 → abandon and pivot.
- Difficulty: medium.
- Thread: `019db2f5-ec85-7ae2-b84b-557374e69c47`.
- Not touching top-level `REVIEW_STATE.json` (holds unrelated UAPSAM audit completion).
