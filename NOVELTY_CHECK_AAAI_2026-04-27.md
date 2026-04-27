# Novelty Check — AAAI Standards (2026-04-27)

**Reviewer:** GPT-5.4 xhigh (Codex MCP), `019dcd76-ac85-75f1-88bc-021e44ed7955`
**Trigger:** User asked novelty assessment for current claims by AAAI standards.

---

## Proposed Method (current paper, post v4.1)

**Working title:** No-Regret Adaptive Decoy Insertion Attack on SAM2 Video Segmentation

Publisher-side offline white-box attack on SAM2:
1. K=3 internal **decoy frame insertion** with semantically plausible content (duplicated object spatially shifted)
2. **Vulnerability-aware joint curriculum placement search** for W
3. **Sparse δ on bridge frames** adjacent to inserts (Stage 14, ε=4/255 with f0 protected)
4. **Anchored Stage 14 (v4.1)** — `L_keep_margin` + dense `L_keep_full` no-regression vs A0 + sparse `L_gain_suffix`
5. **No-regret adaptive wrapper** — `polish_revert`: publish `max(joint, only)` at export

---

## Per-Component Novelty (codex verdict)

| # | Component | Score | Closest prior |
|---|---|---|---|
| 1 | K=3 decoy frame insertion | **MEDIUM** | Chen et al. WACV 2021 (appending frames) |
| 2 | Joint curriculum placement search | **MEDIUM** | none direct, but reviewers may call it "obvious optimizer" |
| 3 | Sparse bridge perturbation | **MEDIUM** | UAP-SAM2 (dense δ); ours is sparse hybridization |
| 4 | Anchored Stage 14 (no-regression) | **LOW** | UAP-SAM2 multi-loss design — reads as plumbing |
| 5 | No-regret adaptive wrapper | **LOW** | candidate selection / fail-safe attack — engineering |

---

## Closest Prior Work

| Paper | Year/Venue | Overlap | Key Difference |
|---|---|---|---|
| **Chen et al., "Appending Adversarial Frames for Universal Video Attack"** | WACV 2021, arXiv 1912.04538 | Frame insertion + δ on inserts only, publisher-side, video | They APPEND (not internal insert), target classification (not VOS), don't modify originals, use generic dummy frames (not semantic decoys) |
| **UAP-SAM2 ("Vanish into Thin Air")** | NeurIPS 2025, arXiv 2510.24195 | First SAM2 attack | Dense δ on EXISTING frames, no insertion, universal across videos+prompts. mIoU 76→33.67% |
| **Li et al., "Hard Region Discovery on VOS"** | T-CSVT 2023, arXiv 2309.13857 | VOS adversarial attack | First-frame δ only, no insertion, no SAM2, gradient-based hard-region |
| **Jiang et al., "One-shot Adversarial Attack on VOS"** | ACM MM 2023 | VOS robustness analysis | Adversarial training direction, not test-time attack innovation |
| **Backdoor on Video Recognition** | CVPR 2020 | Trigger placement | Training-time backdoor, not test-time attack |
| **Stratified Adversarial Robustness with Rejection** | ICML 2023 | Accept/reject | Defense-side rejection, not attacker-side proposal selection |

**Most damaging:** Chen WACV 2021 — owns the "frame insertion + δ on inserts" primitive at high level.
**Runner-up:** UAP-SAM2 — kills any "first SAM2 attack" framing aura.

---

## Combination Novelty

The 5-component bundle IS new. But honest summary: **new combination, not new attack family**.

A reviewer can plausibly summarize as:
> "Frame-addition attack from older video work, specialized to SAM2/VOS, then strengthened with sparse local perturbation and several anti-regression heuristics."

That's enough for a paper IF threat model + mechanism + empirical insight are strong. NOT enough if the claim is "we have five novel components."

---

## AAAI Rejection Risk on Novelty

| Framing | Risk |
|---|---|
| **Current: "joint method beats only"** | **8/10** ❌ |
| Reframed: "test-time video editing attack on memory-bank VOS" + threat-model centric | **6.5/10** ⚠️ |

Why current risk is high:
- Component 1 conceptually overlaps WACV 2021
- Components 2/4/5 look incremental
- "No-regret" is overstated wording (no regret-style theorem)
- "Joint > only" framing reads as attack recipe, not insight

---

## Strongest Defensible Novelty Pitch (codex draft)

> We identify a previously unstudied threat to **memory-bank video object segmentation**: a publisher can **insert a few semantically plausible decoy frames at internal time steps** to corrupt SAM2's prompt-conditioned temporal memory, without relying on dense perturbations across the full video. Unlike prior appended-frame attacks for video classification and prior dense perturbation attacks on SAM/SAM2, our attack is a test-time video editing attack on first-frame-prompt VOS that exploits how temporal memory is built and propagated.

This is much stronger than leading with curriculum search, anchored losses, or `polish_revert`.

---

## Reframing Path (codex recommendation)

**Demote** to implementation details (not headline contributions):
- `L_keep_full`, `L_keep_margin`, `L_gain_suffix` — "stability losses we tuned to keep optimizer from regressing"
- `polish_revert` — "we report the better of joint and insert-only at export"
- joint curriculum placement search — "search procedure"

**Promote** to top contributions:
1. **Threat model**: internal semantic decoy insertion in prompt-driven VOS / SAM2 (not appended dummy frames, not dense δ on existing frames)
2. **Mechanistic finding**: SAM2's temporal memory can be hijacked by ~3 inserted decoys
3. **Sparse bridge perturbation**: works *because* of memory propagation (analytical claim, not just empirical)
4. **Empirical**: per-clip J-drop + adaptive method ≥ insert-only baseline

---

## What NOT to Claim (in paper)

- ❌ "First adversarial attack on SAM2" (UAP-SAM2 has it)
- ❌ "First adversarial attack on VOS" (T-CSVT 2023, ACM MM 2023 own it)
- ❌ "No-regret guarantee" (no regret bound theorem; use "adaptive selection" or "weak dominance" instead)
- ❌ "Joint method universally beats insert-only" (per RESEARCH_REVIEW_JOINT_VS_ONLY_2026-04-27.md, only wrapper-level claim is defensible)

---

## What CAN Be Claimed

- ✅ "First **internal-insertion** adversarial attack on prompt-driven VOS / SAM2"
- ✅ "First attack to combine semantic decoy insertion with sparse bridge perturbation, exploiting temporal memory propagation"
- ✅ "Adaptive joint attack weakly dominates insert-only at export, strict improvement on majority of clips"
- ✅ "Mechanistic insight: SAM2's prompt-conditioned memory can be hijacked by ~3 inserted decoy frames"

---

## Prioritized Action Items (post novelty check)

| # | Action | Why |
|---|---|---|
| 1 | **Read Chen WACV 2021 paper in full** — extract exact differentiators | Most damaging prior art; paper must explicitly position against it |
| 2 | **Read UAP-SAM2 (NeurIPS 2025) in full** — confirm no insertion mechanism, copy exact J-drop numbers for comparison | Required for related work + numerical comparison table |
| 3 | **Survey additional SAM/SAM2 attacks** (codex flagged): Segment(Almost)Nothing SaTML 2024, Unsegment Anything CVPR 2024, Practical Region-level Attack CVPRW 2024, Segment Shards 2024, Transferable Attacks on SAM (2410.20197), Region-guided attack on SAM (NN 2026) | None should use frame insertion (they're SAM-image), but need to confirm |
| 4 | **Reframe paper outline** around threat-model contribution + mechanistic memory-corruption finding | Drops AAAI risk from 8/10 → 6.5/10 |
| 5 | **Drop "no-regret" terminology** from claims; replace with "weakly dominates" or "adaptive selection" | "No-regret" without regret bound is reviewer red flag |
| 6 | **Add explicit ablation against insert-only-no-bridge-δ baseline** in held-out 10-clip | Differentiator vs WACV 2021 needs empirical evidence (does the bridge δ actually add over insert-only?) |
| 7 | **Run 10-clip held-out with both A0 AND v4.1 reported** (paired comparison) | Only paired comparison can support wrapper-level claim |

---

## Decision Point

**Recommendation: PROCEED WITH CAUTION** (codex 6.5/10 risk after reframe).

Conditions for going to AAAI:
- ✅ Reframe per §"Reframing Path" above
- ✅ Drop "no-regret" wording  
- ✅ Add insert-only-no-δ baseline ablation
- ✅ Pass 10-clip acceptance gates (per RESEARCH_REVIEW_JOINT_VS_ONLY_2026-04-27.md: ≥5/10 wins, mean ≥+0.05, median >0, top<40%)

If 10-clip falls short of those gates → narrow paper claim to "specific clips story" → consider another venue (e.g., a workshop, or strengthen the analysis side and target later).

---

## Searches Performed

- "SAM2 adversarial attack video object segmentation 2025"
- "video object segmentation memory bank poisoning attack"
- "decoy frame insertion adversarial attack video"
- "frame insertion attack video segmentation 2025 2026"
- "video object segmentation adversarial attack 2024 2025 perturbation"
- "no-regret adversarial attack adaptive accept reject"
- "vulnerability aware OR trigger placement adversarial attack video model"
- "anchored optimization no-regression baseline adversarial attack"

WebFetch verifications:
- arxiv.org/abs/1912.04538 (Chen WACV 2021) — confirmed appending, classification, no original modification
- arxiv.org/html/2510.24195 (UAP-SAM2) — confirmed dense UAP, no insertion
- arxiv.org/abs/2309.13857 (Li T-CSVT 2023) — confirmed first-frame attack, no insertion

---

## References

- Chen et al., *Appending Adversarial Frames for Universal Video Attack* (WACV 2021): https://arxiv.org/abs/1912.04538
- Zhou et al., *Vanish into Thin Air: Cross-prompt UAP for SAM2* (NeurIPS 2025): https://arxiv.org/abs/2510.24195
- Li et al., *Adversarial Attacks on VOS with Hard Region Discovery* (T-CSVT 2023): https://arxiv.org/abs/2309.13857
- Jiang et al., *Exploring Adversarial Robustness of VOS via One-shot Attacks* (ACM MM 2023): https://dl.acm.org/doi/10.1145/3581783.3611827
- Lu et al., *Unsegment Anything by Simulating Deformation* (CVPR 2024): https://openaccess.thecvf.com/content/CVPR2024/html/Lu_Unsegment_Anything_by_Simulating_Deformation_CVPR_2024_paper.html
- Croce & Hein, *Segment (Almost) Nothing* (SaTML 2024): https://openreview.net/forum?id=UGXBYDrlhn
- Xia et al., *Transferable Adversarial Attacks on SAM and Downstream Models* (arXiv 2410.20197): https://arxiv.org/abs/2410.20197
- Shen et al., *Practical Region-level Attack against SAMs* (CVPRW 2024): https://experts.illinois.edu/en/publications/practical-region-level-attack-against-segment-anything-models
- Liu et al., *Region-guided attack on SAM* (Neural Networks 2026): https://pubmed.ncbi.nlm.nih.gov/40921127/

Codex thread: `019dcd76-ac85-75f1-88bc-021e44ed7955` (xhigh, fresh thread for novelty check)

Sources (web search):
- [Vanish into Thin Air: Cross-prompt UAPs for SAM2 (arXiv 2510.24195)](https://arxiv.org/abs/2510.24195)
- [Appending Adversarial Frames for Universal Video Attack (arXiv 1912.04538)](https://arxiv.org/abs/1912.04538)
- [Adversarial Attacks on VOS with Hard Region Discovery (arXiv 2309.13857)](https://arxiv.org/abs/2309.13857)
- [Exploring Adversarial Robustness of VOS via One-shot Attacks (ACM MM 2023)](https://dl.acm.org/doi/10.1145/3581783.3611827)
- [Awesome-Video-Adversarial-Attack list](https://github.com/rogercmq/Awesome-Video-Adversarial-Attack)
