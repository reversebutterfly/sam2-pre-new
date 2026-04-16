# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | Clean sanity | SAM2-tiny clean | 20 DAVIS clips | J, F, J&F, eligible count | MUST | TODO | Freeze clip list and future eval window `f10:f14` |
| R002 | M0 | Clean sanity | SAM2Long-tiny clean (`P=3, iou=0.1, unc=2`) | Same 20 clips | J, F, J&F | MUST | TODO | Separate environment |
| R003 | M0 | Pipeline check | Clean + memory reset at `f10` | 3 debug clips | J, F, J&F | MUST | TODO | Verify reset baseline before batch |
| R004 | M1 | Core comparison | Suppression Hybrid | 20 clips | J&F drop, SSIM | MUST | TODO | Same schedule as Decoy |
| R005 | M1 | Core comparison | Decoy Hybrid v5 | 20 clips | J&F drop, SSIM | MUST | TODO | Same schedule as Suppression |
| R006 | M1 | Signature extraction | Suppression signatures | Results from R004 | NegScoreRate, CollapseRate | MUST | TODO | Reuse attacked clips |
| R007 | M1 | Signature extraction | Decoy signatures | Results from R005 | PosScoreRate, DecoyHitRate, CentroidShift | MUST | TODO | Reuse attacked clips |
| R008 | M2 | Benign control | Benign insertions | 8 clips, fallback 5 | J&F drop | MUST | TODO | Same insert positions, no PGD |
| R009 | M2 | Component ablation | Suppression perturb-only | 8 clips, fallback 5 | J&F drop + signatures | MUST | TODO | Same attacked originals only |
| R010 | M2 | Component ablation | Suppression insert-only | 8 clips, fallback 5 | J&F drop + signatures | MUST | TODO | Same two inserts only |
| R011 | M2 | Causal reset | Suppression hybrid + reset | 8 clips, fallback 5 | J&F drop + signatures | MUST | TODO | Compare to clean+reset |
| R012 | M2 | Component ablation | Decoy perturb-only | 8 clips, fallback 5 | J&F drop + signatures | MUST | TODO | Same attacked originals only |
| R013 | M2 | Component ablation | Decoy insert-only | 8 clips, fallback 5 | J&F drop + signatures | MUST | TODO | Same two inserts only |
| R014 | M2 | Causal reset | Decoy hybrid + reset | 8 clips, fallback 5 | J&F drop + signatures | MUST | TODO | Compare to clean+reset |
| R015 | M3 | Transfer test | SAM2Long on Suppression attacked clips | 20 clips | J&F drop, RetentionRatio | MUST | TODO | Reuse outputs from R004 |
| R016 | M3 | Transfer test | SAM2Long on Decoy attacked clips | 20 clips | J&F drop, RetentionRatio | MUST | TODO | Reuse outputs from R005 |
| R017 | M4 | Policy sweep | SAM2Long weak gate (`iou=0.0`) | 5 clips | J&F drop, RetentionRatio | SHOULD | TODO | Hold `P=3, unc=2` fixed |
| R018 | M4 | Policy sweep | SAM2Long default gate (`iou=0.1`) | 5 clips | J&F drop, RetentionRatio | SHOULD | TODO | Reproduce default |
| R019 | M4 | Policy sweep | SAM2Long strong gate (`iou=0.2`) | 5 clips | J&F drop, RetentionRatio | SHOULD | TODO | Test confidence filtering effect |
| R020 | M4 | Qualitative | Four canonical videos | 4 clips | overlays + curves | MUST | TODO | One best Suppression, one best Decoy, one ambiguous, one failure |
