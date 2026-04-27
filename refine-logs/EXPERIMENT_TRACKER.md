# Experiment Tracker — v5

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| **R001** | M0 | Implement A3 hook | `BlockInsertMemoryWritesHook` patch on `SAM2VideoAdapter` (~50 LOC) | n/a (code) | self-test passes | MUST | TODO | ~2h coding |
| **R002** | M0 | Implement memory-readout extractor | Extract `memory_attention.layers[-1].cross_attention` PRE-projection V; aggregate over `T_obj(clip)` (top-32 by attention at clean c_K_clean) | n/a (code) | extractor returns non-zero d_mem on clean vs attacked | MUST | TODO | ~1h coding |
| **R003** | M0 | Implement control-frame sampler | Sample 3 random non-insert non-bridge frames per clip, fixed seed=0 | n/a (code) | unit test for determinism + non-overlap with W_attacked + bridge_frames | MUST | TODO | ~30min coding |
| **R004** | M0 | Smoke test on dog | Full v5 + A3-attacked + A3-control + d_mem trace on 1 clip | dog only | non-zero d_mem trace; memory-block changes mask vs baseline | MUST | TODO | ~1 GPU-h |
| **R005** | M1 | A0 baseline 10-clip | Stage 11-13 only (`--K 3 --placement top` no Stage 14) | 10 held-out | per-clip exported J-drop on uint8 | MUST | TODO | 3 GPU-h overnight, GPU 1 |
| **R006** | M2 | A3-baseline (full v5) on 10 | `--oracle-trajectory --oracle-traj-v4 --placement-search joint_curriculum` (no hook) | 10 held-out | per-clip J-drop, polish_applied flag | MUST | TODO | 6 GPU-h, **GATING** |
| **R007** | M2 | A3-attacked (full v5 + W_attacked block) on 10 | + `--memory-block-mode attacked` flag | 10 held-out | per-clip J-drop | MUST | TODO | 2 GPU-h (eval-only re-forward), depends R006 |
| **R008** | M2 | A3-control (full v5 + W_control block) on 10 | + `--memory-block-mode control --control-seed 0` | 10 held-out | per-clip J-drop | MUST | TODO | 2 GPU-h, depends R006 |
| **R009** | M2 | d_mem trace 3-condition × 10 | Extract M_clean / M_only / M_full at extractor layer per frame | 10 held-out | d_mem(t) trace per clip | MUST | TODO | overlap with R006-008, ~0 extra GPU-h if integrated |
| **R010** | gate | Read A3 verdict | analyze R006-R008 | n/a | strong / partial / fail tier | MUST | TODO | 0 GPU; commits to Framing A/B/C |
| **R011** | M3 | C2 RAW joint v5 (= A1-full) on 10 | full v5, NO wrapper | 10 held-out | per-clip RAW J-drop, polish_applied | MUST | TODO | shared with R006 (same forward) |
| **R012** | M3 | A1-only (skip Stage 14) on 10 | `--K 3 --placement top --use-profiled-placement <from R006 W*>` + `--no-stage14` | 10 held-out | per-clip J-drop | MUST | TODO | <30s/clip → 0.1 GPU-h total |
| **R013** | M3 | Wrapper-selected (deployment column) | postprocess from R005 + R011 | 10 held-out | max(joint, A0) per clip | MUST | TODO | 0 GPU (postprocess) |
| **R014** | M4 | A2 random K=3 + full v5 polish | `--placement random --K 3 --random-seed 0 --oracle-traj-v4` | 10 held-out | per-clip J-drop | MUST | TODO | 5 GPU-h overnight |
| **R015** | M5 | T_obj sensitivity (16 / 64) | re-extract d_mem with T_obj=16 then 64 | 10 held-out | d_mem integral per clip × 3 settings | NICE | TODO | ~1 GPU-h (re-aggregation only) |
| **R016** | M5 | Qualitative figure | per-frame visualization 3 representative clips | dog, camel, bmx-trees | LPIPS/SSIM + visual | NICE | TODO | <1 GPU-h |
| **R017** | M5 | lambda_keep_full=50 retry | `--oracle-traj-v4-lambda-keep-full 50` on previously-reverting clips | 1-3 clips from R011 reverts | per-clip J-drop, polish_applied | NICE | TODO | 2 GPU-h |

## Dependency graph

```
M0: R001 -> R002 -> R003 -> R004 (smoke)
M1: R005 (A0 baseline) — independent of M0
M2: R004 -> R006 (gates) -> {R007, R008, R009 in parallel/integrated} -> R010 (verdict)
M3: R006 reuses for R011; R005 reuses for paired comparison; R012 -> postprocess R013
M4: R005 + R014 (paired)
M5: R009 -> R015; M3 results -> R016; R011 reverts -> R017
```

## Run-launch template (copy-paste)

```bash
# M2 R006 R007 R008 R009 — A3 + d_mem on 10 held-out (Day 1 PM)
ssh lvshaoting-pro6000 'screen -dmS v5-m2-a3 bash ~/sam2-pre-new/_run/launch_v5_m2_a3.sh'

# M1 R005 — A0 baseline (Day 2 overnight)
ssh lvshaoting-pro6000 'screen -dmS v5-m1-a0 bash ~/sam2-pre-new/_run/launch_v5_m1_a0.sh'

# M3 R011 + R012 — RAW joint + A1-only (Day 2 overnight, parallel)
ssh lvshaoting-pro6000 'screen -dmS v5-m3-headline bash ~/sam2-pre-new/_run/launch_v5_m3_headline.sh'

# M4 R014 — A2 random placement (Day 3 AM)
ssh lvshaoting-pro6000 'screen -dmS v5-m4-a2 bash ~/sam2-pre-new/_run/launch_v5_m4_a2.sh'

# Heartbeat (always, Day 1-3)
ssh lvshaoting-pro6000 'screen -dmS v5-hb bash ~/sam2-pre-new/_run/heartbeat_v5.sh'
```

(Launchers to be created at M0 implementation time, modeled on `_run/launch_v4_dev4_*.sh`.)

## Pre-registered held-out 10 clips

Proposed (DAVIS-2017 val): **bear, blackswan, breakdance, cows, dance-twirl, dog, hike, horsejump-high, india, judo**.

Rationale:
- Avoids dev-4 overlap where possible (dev-4 = dog, bmx-trees, camel, libby).
- "dog" overlaps dev-4 → that's intentional — dog has v4.1 retest data to compare against (consistency check).
- 8/10 are completely held out from dev-4.
- Mix of motion patterns (camera + object motion: dance-twirl; static-camera: bear; fast object: bmx-trees-like horsejump-high).

If the user prefers a fully disjoint 10-clip held-out (no dev-4 overlap), substitute dog → boat or scooter-black. Decision deferred to user before R005 launch.

## Status legend

- TODO — not started
- IMPL — implementation in progress
- RUNNING — GPU running
- DONE — results collected
- BLOCKED — waiting (e.g. GPU contention)
- FAIL — run failed; needs retry
