"""Rebuttal E4 (CPU) — sensitivity of §5.4 (collapse) and §5.5 (FPR increment) to the
lift cutoff. Re-thresholds the stored per-row pair_lift of the released hard sets
(stricter cutoffs are subsets of the released rows; the looser 1.2 set is a separate
GPU run merged in when present). Unrelated set: per-row lift re-thresholded if present.
"""
import json, os, sys
import numpy as np
R4D = os.environ.get("VGV_RESULTS", os.path.join(os.environ.get("VGV_ROOT", "."), "results"))
R5A = os.environ.get("VGV_RESULTS", os.path.join(os.environ.get("VGV_ROOT", "."), "results"))
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["medgemma", "lingshu", "internvl35", "qwen3vl",
          "medgemma27b", "lingshu32b", "qwen3vl30b", "gemma3", "qwen25vl"]   # added models included when their files exist
CUTS = [1.5, 2.0, 2.5, 3.0]

out = {}
lines = ["# E4 — lift-cutoff sensitivity (CPU re-threshold of stored pair_lift)", ""]
for m in MODELS:
    if not (os.path.exists(f"{R4D}/p3_{m}.json") and os.path.exists(f"{R5A}/t1_exp3a_{m}.json")):
        lines += [f"## {m}", "", "*hard set / unrelated set not yet available*", ""]; continue
    P = json.load(open(f"{R4D}/p3_{m}.json"))
    d1, d2 = P["d1_rows"], P["d2_rows"]
    un = json.load(open(f"{R5A}/t1_exp3a_{m}.json"))["rows"]
    ukey = "pred_yes" if "pred_yes" in un[0] else ("pred" if "pred" in un[0] else None)
    u_yes = np.array([bool(r[ukey]) if ukey else r["margin_full"] > 0 for r in un])
    u_lift = np.array([r.get("pair_lift", r.get("lift", np.nan)) for r in un], dtype=float)
    iso = np.array([r["M"] - r["M_rand"] for r in d1 if r["subset"] == "hard_pos_isolated"])
    res = {"iso_median_excess": float(np.median(iso)), "n_iso": int(len(iso)), "cuts": {}}
    lines += [f"## {m}", "", "| correlated cutoff | n corr pos | corr excess | collapse iso/corr | n hard-neg | FPR corr | FPR unrel (≤0.9) | increment |", "|---|---|---|---|---|---|---|---|"]
    fpr_un = float(u_yes.mean())
    for c in CUTS:
        cor = np.array([r["M"] - r["M_rand"] for r in d1 if r["subset"] == "hard_pos_correlated" and (r.get("pair_lift") or 0) >= c])
        neg = [r for r in d2 if (r.get("pair_lift") or 0) >= c]
        fpr_c = float(np.mean([r["pred_yes"] for r in neg])) if neg else float("nan")
        ce = float(np.median(cor)) if len(cor) else float("nan")
        ratio = float(np.median(iso) / ce) if len(cor) and ce > 0 else float("nan")
        res["cuts"][str(c)] = {"n_corr": int(len(cor)), "corr_median_excess": ce, "collapse_ratio": ratio,
                               "n_neg": len(neg), "fpr_corr": fpr_c, "fpr_unrel": fpr_un, "increment": fpr_c - fpr_un}
        lines.append(f"| ≥{c} | {len(cor)} | {ce:+.3f} | {ratio:.1f}× | {len(neg)} | {fpr_c:.3f} | {fpr_un:.3f} | {fpr_c-fpr_un:+.3f} |")
    # looser GPU set (1.2) if present -> the weak-association band row of the SAME table
    lp = f"{HERE}/p3_lift12_{m}.json"
    if os.path.exists(lp):
        L12 = json.load(open(lp)); ld1, ld2 = L12["d1_rows"], L12["d2_rows"]
        cor = np.array([r["M"] - r["M_rand"] for r in ld1 if r["subset"] == "hard_pos_correlated" and 1.2 <= (r.get("pair_lift") or 0) < 1.5])
        neg = [r for r in ld2 if 1.2 <= (r.get("pair_lift") or 0) < 1.5]
        if len(cor) and neg:
            ce = float(np.median(cor)); fpr_c = float(np.mean([r["pred_yes"] for r in neg]))
            res["band_1.2_1.5"] = {"n_corr": int(len(cor)), "corr_median_excess": ce, "n_neg": len(neg), "fpr_corr": fpr_c, "increment": fpr_c - fpr_un}
            lines.append(f"| **band 1.2–1.5** (separate run) | {len(cor)} | {ce:+.3f} | {np.median(iso)/ce if ce>0 else float('nan'):.1f}× | {len(neg)} | {fpr_c:.3f} | {fpr_un:.3f} | {fpr_c-fpr_un:+.3f} |")
    # unrelated-side cutoffs if per-row lift is stored
    if not np.all(np.isnan(u_lift)):
        lines += ["", "| unrelated cutoff | n | FPR unrel | increment vs corr(≥1.5) |", "|---|---|---|---|"]
        fpr_c15 = res["cuts"]["1.5"]["fpr_corr"]
        for uc in [0.9, 0.7, 0.5]:
            sel = u_yes[u_lift <= uc]
            if len(sel):
                lines.append(f"| ≤{uc} | {len(sel)} | {sel.mean():.3f} | {fpr_c15 - sel.mean():+.3f} |")
                res.setdefault("unrel_cuts", {})[str(uc)] = {"n": int(len(sel)), "fpr": float(sel.mean())}
    lines.append("")
    out[m] = res
json.dump(out, open(f"{HERE}/e4_lift_sensitivity.json", "w"), indent=1)
open(f"{HERE}/e4_lift_sensitivity.md", "w").write("\n".join(lines))
print("\n".join(lines))
