"""Synthesize per-clip profile.json files for GO/NO-GO test (codex round 5).

Reads existing v5_paper_all_merged/<clip>/<config>__ot/results.json,
converts attacked-space W back to clean-space W_clean, and writes a
minimal profile.json compatible with run_vadi_v5.py's
--use-profiled-placement loader at <out_root>/<clip>/profile.json.
"""
import json
import pathlib

OUT_ROOT = pathlib.Path("vadi_runs/v5_go_nogo_profiles")
SRC_ROOT = pathlib.Path("vadi_runs/v5_paper_all_merged")
CLIPS = [
    "bear", "blackswan", "bmx-trees", "breakdance", "camel", "cows",
    "dance-twirl", "dog", "hike", "horsejump-high", "india", "judo", "libby",
]
CONFIG = "K3_top_R8_b-dup_l-dc_o-ad_d-post_s-fs__ot"


def attacked_to_clean(W_att):
    W_att = sorted(W_att)
    return [int(w - i) for i, w in enumerate(W_att)]


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    missing = []
    for c in CLIPS:
        rj = SRC_ROOT / c / CONFIG / "results.json"
        if not rj.is_file():
            missing.append(c)
            print(f"[skip] {c}: source results.json not found at {rj}")
            continue
        raw = rj.read_text()
        raw = raw.replace("NaN", "null")
        d = json.loads(raw)
        W_att = d["W"]
        W_clean = attacked_to_clean(W_att)
        score = float(d.get("exported_j_drop") or 0.0)
        profile = {
            "clip_name": c,
            "best": {
                "subset": W_clean,
                "score": score,
                "metadata": {
                    "subset": W_clean,
                    "K": len(W_clean),
                    "source": (
                        "derived from v5_paper_all_merged for GO/NO-GO "
                        "test 2026-04-28; attacked-space W mapped back "
                        "to clean-space"
                    ),
                },
            },
        }
        cp = OUT_ROOT / c
        cp.mkdir(parents=True, exist_ok=True)
        (cp / "profile.json").write_text(json.dumps(profile, indent=2))
        print(
            "{c}: W_attacked={a} -> W_clean={n} score={s:.4f}".format(
                c=c, a=W_att, n=W_clean, s=score)
        )
    print()
    print("profiles ready in", OUT_ROOT)
    if missing:
        # Codex round 17 CRITICAL #3: a missing profile silently changes
        # the experiment because the v5 driver falls back to
        # args.placement instead of using the joint-optimal W. Hard-fail.
        import sys
        print(
            f"FATAL: {len(missing)} clip(s) missing source results.json "
            f"in {SRC_ROOT}: {missing}",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
