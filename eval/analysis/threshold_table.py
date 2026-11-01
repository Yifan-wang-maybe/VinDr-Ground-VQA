"""Rebuttal E4b (CPU) — the full threshold table for §5.4 and §5.5 (R2 Q1).
For every model with the needed files, and for every correlated-set cutoff
  band 1.2–1.5 (separate run) | ≥1.5 (paper) | ≥2.0 | ≥2.5
and unrelated-set cutoff  ≤0.9 (paper) | ≤0.7 :
  §5.4  n correlated positives, isolated excess [CI] (fixed), correlated excess [CI],
        collapse ratio, detection rate isolated → correlated
  §5.5  FPR normal (fixed; p1 negatives, paper convention), FPR unrelated, FPR correlated,
        increment corr − unrel [bootstrap CI], increment unrel − normal
Writes e4b_threshold_table.md / .json. Stricter cutoffs are subsets of the released rows.
"""
import os, json
import numpy as np
R4D = os.environ.get("VGV_RESULTS", os.path.join(os.environ.get("VGV_ROOT", "."), "results"))
R5A = os.environ.get("VGV_RESULTS", os.path.join(os.environ.get("VGV_ROOT", "."), "results"))
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["medgemma", "lingshu", "internvl35", "qwen3vl", "medgemma27b", "lingshu32b", "qwen3vl30b", "gemma3", "qwen25vl"]
NICE = {"medgemma": "MedGemma-4b", "lingshu": "Lingshu-7B", "internvl35": "InternVL3.5-8B", "qwen3vl": "Qwen3-VL-8B",
        "medgemma27b": "MedGemma-27B", "lingshu32b": "Lingshu-32B", "qwen3vl30b": "Qwen3-VL-30B", "gemma3": "Gemma-3-4b (base)", "qwen25vl": "Qwen2.5-VL-7B (base)"}
CUTS = [("band 1.2–1.5", 1.2, 1.5), ("≥1.5 (paper)", 1.5, None), ("≥2.0", 2.0, None), ("≥2.5", 2.5, None)]
UCUTS = [("≤0.9 (paper)", 0.9), ("≤0.7", 0.7)]
rng = np.random.default_rng(0); B = 10000


def bmed(v):
    v = np.asarray(v, float)
    if len(v) == 0: return (float("nan"), float("nan"), float("nan"))
    m = np.median(v[rng.integers(0, len(v), (B, len(v)))], axis=1)
    return float(np.median(v)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def bdiff(a, b):   # unpaired bootstrap of mean(b) - mean(a) for binary arrays
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0: return (float("nan"), float("nan"), float("nan"))
    sa = a[rng.integers(0, len(a), (B, len(a)))].mean(1); sb = b[rng.integers(0, len(b), (B, len(b)))].mean(1)
    d = sb - sa
    return float(b.mean() - a.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def f(x, p=2): return "nan" if x != x else f"{x:+.{p}f}"


out, long_rows, pivot54, pivot55 = {}, [], {}, {}
for m in MODELS:
    p3 = f"{R4D}/p3_{m}.json"; un = f"{R5A}/t1_exp3a_{m}.json"
    if not (os.path.exists(p3) and os.path.exists(un)):
        out[m] = "missing hard set / unrelated set"; continue
    P = json.load(open(p3)); d1, d2 = P["d1_rows"], P["d2_rows"]
    U = json.load(open(un))["rows"]
    u_lift = np.array([r.get("pair_lift", np.nan) for r in U], float); u_yes = np.array([r["margin_full"] > 0 for r in U])
    band = None
    lp = f"{HERE}/p3_lift12_{m}.json"
    if os.path.exists(lp): L = json.load(open(lp)); band = (L["d1_rows"], L["d2_rows"])
    # fixed quantities
    iso_rows = [r for r in d1 if r["subset"] == "hard_pos_isolated"]
    iso_ex = bmed([r["M"] - r["M_rand"] for r in iso_rows]); det_iso = float(np.mean([r["margin_full"] > 0 for r in iso_rows]))
    normal = None
    p1 = f"{R4D}/p1_{m}.json"
    if os.path.exists(p1):
        neg = [r for r in json.load(open(p1))["rows"] if r["gold"] == "no"]
        normal = np.array([r["pred_full"] == "yes" for r in neg])
    res = {"iso_excess": iso_ex, "det_iso": det_iso, "n_iso": len(iso_rows), "normal_fpr": float(normal.mean()) if normal is not None else None, "cuts": {}}
    for cname, lo, hi in CUTS:
        if lo == 1.2:
            if band is None: continue
            cd1, cd2 = band
            cor = [r for r in cd1 if r["subset"] == "hard_pos_correlated" and lo <= (r.get("pair_lift") or 0) < hi]
            neg_c = [r for r in cd2 if lo <= (r.get("pair_lift") or 0) < hi]
        else:
            cor = [r for r in d1 if r["subset"] == "hard_pos_correlated" and (r.get("pair_lift") or 0) >= lo]
            neg_c = [r for r in d2 if (r.get("pair_lift") or 0) >= lo]
        cor_ex = bmed([r["M"] - r["M_rand"] for r in cor]); det_cor = float(np.mean([r["margin_full"] > 0 for r in cor])) if cor else float("nan")
        ratio = iso_ex[0] / cor_ex[0] if cor and cor_ex[0] > 0 else float("nan")
        c_yes = np.array([bool(r["pred_yes"]) for r in neg_c])
        entry = {"n_corr": len(cor), "corr_excess": cor_ex, "collapse": ratio, "det_corr": det_cor, "n_neg": len(neg_c), "fpr_corr": float(c_yes.mean()) if len(c_yes) else float("nan"), "unrel": {}}
        for uname, ucut in UCUTS:
            sel = u_yes if np.all(np.isnan(u_lift)) and ucut == 0.9 else u_yes[u_lift <= ucut]
            inc = bdiff(sel, c_yes); inc_un = bdiff(normal, sel) if normal is not None else (float("nan"),) * 3
            entry["unrel"][uname] = {"n_unrel": int(len(sel)), "fpr_unrel": float(sel.mean()) if len(sel) else float("nan"), "inc_corr_minus_unrel": inc, "inc_unrel_minus_normal": inc_un}
            long_rows.append([NICE[m], cname, uname, len(cor), f(iso_ex[0]), f"{f(cor_ex[0])} [{f(cor_ex[1])}, {f(cor_ex[2])}]", "nan" if ratio != ratio else f"{ratio:.1f}×",
                              f"{det_iso:.2f} → {det_cor:.2f}", "—" if normal is None else f"{normal.mean():.3f}", f"{sel.mean():.3f}" if len(sel) else "nan", f"{c_yes.mean():.3f}" if len(c_yes) else "nan",
                              f"{f(inc[0], 3)} [{f(inc[1], 3)}, {f(inc[2], 3)}]", "—" if normal is None else f(inc_un[0], 3), len(neg_c)])
        res["cuts"][cname] = entry
        # pivot cell: correlated excess with a capped ratio (ratios of near-zero medians are not meaningful)
        if iso_ex[0] < 0.05: rtxt = "iso ≈ 0"
        elif cor_ex[0] <= 0.01: rtxt = ">100×"
        elif ratio != ratio: rtxt = "n/a"
        else: rtxt = f"{ratio:.1f}×"
        pivot54.setdefault(NICE[m], {})[cname] = f"{f(cor_ex[0])} ({rtxt})"
        pivot55.setdefault(NICE[m], {})[cname] = f(entry["unrel"]["≤0.9 (paper)"]["inc_corr_minus_unrel"][0], 3)
    out[m] = res

L = ["# E4b — §5.4 / §5.5 under alternative co-occurrence thresholds", "",
     "Stricter correlated cutoffs are subsets of the released hard set; the 1.2–1.5 band is a separately built set. Isolated positives and the normal baseline do not depend on the cutoff. CIs: bootstrap over cases (10,000). Collapse = isolated excess / correlated excess.", "",
     "## Pivot A — §5.4 correlated-positive excess (isolated excess in parentheses as a ratio; '>100×' when the correlated excess is ≤ 0.01)", "",
     "Isolated excess per model: " + "; ".join(f"{NICE[m]} {f(out[m]['iso_excess'][0])}" for m in MODELS if not isinstance(out.get(m), str)) + ".", "",
     "| model | " + " | ".join(c[0] for c in CUTS) + " |", "|---|" + "---|" * len(CUTS)]
for mm, d in pivot54.items(): L.append(f"| {mm} | " + " | ".join(d.get(c[0], "—") for c in CUTS) + " |")
L += ["", "## Pivot B — §5.5 false-positive increment, correlated − unrelated (unrelated ≤0.9)", "",
      "| model | " + " | ".join(c[0] for c in CUTS) + " |", "|---|" + "---|" * len(CUTS)]
for mm, d in pivot55.items(): L.append(f"| {mm} | " + " | ".join(d.get(c[0], "—") for c in CUTS) + " |")
L += ["", "## Full table (one row per model × correlated cutoff × unrelated cutoff)", "",
      "| model | correlated cutoff | unrelated cutoff | n corr pos | iso excess | corr excess [CI] | collapse | detection iso → corr | FPR normal | FPR unrel | FPR corr | incr corr − unrel [CI] | incr unrel − normal | n hard-neg |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in long_rows: L.append("| " + " | ".join(str(x) for x in r) + " |")
L += ["", "Models without a row: " + ", ".join(NICE[m] for m in MODELS if isinstance(out.get(m), str)) + " (hard set / unrelated set still running)."]
open(f"{HERE}/e4b_threshold_table.md", "w").write("\n".join(L))
json.dump(out, open(f"{HERE}/e4b_threshold_table.json", "w"), indent=1, default=str)
print("\n".join(L[:4 + 2 + len(pivot54) + 4 + 2 + len(pivot55)]))
