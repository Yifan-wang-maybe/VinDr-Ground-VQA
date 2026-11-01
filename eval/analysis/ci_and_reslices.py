"""Phase-1 gap closer — CIs, tests, and re-slices over cached per-case rows.
Implements experiment-report items #1 (all CIs + named tests), #1.3/#6 (the
stratified isolated-vs-correlated re-slice), #9 (per-finding table), #4-partial
(Tables 11/12 to all four models), #15 (excess by confidence decile).
Convention: median [95% CI], paired bootstrap over cases, 10,000 resamples;
proportions: Wilson 95%. Outputs: ci_tables.json + ci_tables.md here.
"""
import json, os
import numpy as np

R4D = os.environ.get("VGV_RESULTS", os.path.join(os.environ.get("VGV_ROOT", "."), "results"))
R5A = os.environ.get("VGV_RESULTS", os.path.join(os.environ.get("VGV_ROOT", "."), "results"))
OUT = os.path.dirname(os.path.abspath(__file__))
MODELS = ["medgemma", "lingshu", "qwen3vl", "internvl35"]
B = 10000
rng = np.random.RandomState(0)
out = {}


def boot_median(x, stat=np.median):
    x = np.asarray(x, float)
    idx = rng.randint(0, len(x), (B, len(x)))
    s = stat(x[idx], axis=1)
    return float(stat(x)), [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def boot_diff_unpaired(a, b, stat=np.median):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sa = stat(a[rng.randint(0, len(a), (B, len(a)))], axis=1)
    sb = stat(b[rng.randint(0, len(b), (B, len(b)))], axis=1)
    d = sa - sb
    return float(stat(a) - stat(b)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def wilson(k, n, z=1.96):
    if n == 0: return None
    p = k / n
    den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [float(c-h), float(c+h)]


def sign_test_pos(x):
    """one-sided sign test P(excess>0)>0.5 via normal approx"""
    x = np.asarray(x, float); x = x[x != 0]
    k, n = int((x > 0).sum()), len(x)
    z = (k - n/2) / np.sqrt(n/4)
    from math import erf
    p = 1 - 0.5*(1 + erf(z/np.sqrt(2)))
    return {"n": n, "k_pos": k, "z": float(z), "p_one_sided": float(p)}


# ---------- Fig 1 / Tables 4 & 11: release_mm per model ----------
fig1 = {}
for m in MODELS:
    R = json.load(open(f"{R4D}/mjsd_{m}_vindr_release_mm.json"))["rows"]
    A = np.array([r["roi_area"] or 0 for r in R])
    q1, q2 = np.quantile(A, [1/3, 2/3])
    EX = np.array([r["Mm"] - r["Mm_rand"] for r in R])
    RAWEX = np.array([r["M"] - r["M_rand"] for r in R])
    MR = np.array([r["Mm_rand"] for r in R])
    F = np.array([r["M"] - r["Mt"] for r in R])
    ent = {}
    for lab, msk in [("small", A <= q1), ("med", (A > q1) & (A <= q2)), ("large", A > q2)]:
        med, ci = boot_median(EX[msk])
        mr_med, mr_ci = boot_median(MR[msk])
        raw_med, raw_ci = boot_median(RAWEX[msk])
        f_med, f_ci = boot_median(F[msk])
        k = int((EX[msk] > 0).sum()); n = int(msk.sum())
        ent[lab] = {"excess_Mm": {"median": med, "ci": ci}, "Mm_rand": {"median": mr_med, "ci": mr_ci},
                    "excess_raw_T11": {"median": raw_med, "ci": raw_ci}, "F_T12": {"median": f_med, "ci": f_ci},
                    "pct_pos_T4": {"value": k/n, "wilson": wilson(k, n), "n": n},
                    "sign_test": sign_test_pos(EX[msk])}
    # tertile trend: paired-ish smallest vs largest (unpaired cases)
    d, ci = boot_diff_unpaired(EX[A > q2], EX[A <= q1])
    ent["trend_large_minus_small"] = {"diff": d, "ci": ci}
    fig1[m] = ent
out["fig1_table4_table11_table12"] = fig1

# ---------- Table 5: resize (paired within box) ----------
t5 = {}
for m in ["medgemma", "lingshu", "internvl35"]:
    R = json.load(open(f"{R5A}/t1_exp5_full_{m}.json"))["rows"]
    ent = {}
    for cell in ["enlarge", "shrink", "cardio"]:
        rs = [r for r in R if r["cell"] == cell]
        d = np.array([r["EX_target"] - r["EX_native"] for r in rs])
        nat, nci = boot_median([r["EX_native"] for r in rs])
        tgt, tci = boot_median([r["EX_target"] for r in rs])
        dd, dci = boot_median(d)
        ent[cell] = {"n": len(rs), "native": {"median": nat, "ci": nci},
                     "target": {"median": tgt, "ci": tci},
                     "paired_diff": {"median": dd, "ci": dci}}
    t5[m] = ent
out["table5_resize"] = t5

# ---------- Fig 2: named/other/random (paired within image) ----------
f2 = {}
for m in MODELS:
    R = json.load(open(f"{R5A}/t1_exp3b_{m}.json"))["rows"]
    ent = {}
    for k_ in ["M_roi", "M_other", "M_rand"]:
        med, ci = boot_median([r[k_] for r in R])
        ent[k_] = {"median": med, "ci": ci}
    d, ci = boot_median([r["M_roi"] - r["M_other"] for r in R])
    ent["named_minus_other"] = {"median": d, "ci": ci, "n": len(R)}
    f2[m] = ent
out["fig2_specificity"] = f2

# ---------- Fig 3: FPR groups + increments ----------
f3 = {}
for m in MODELS:
    p1 = json.load(open(f"{R4D}/p1_{m}.json"))["rows"]
    neg = [r for r in p1 if r["gold"] == "no"]
    k0, n0 = sum(1 for r in neg if r["pred_full"] == "yes"), len(neg)
    un = json.load(open(f"{R5A}/t1_exp3a_{m}.json"))["rows"]
    k1, n1 = sum(1 for r in un if r["margin_full"] > 0), len(un)
    co = json.load(open(f"{R4D}/p3_{m}.json"))["d2_rows"]
    k2, n2 = sum(1 for r in co if r["pred_yes"]), len(co)
    def inc(kA, nA, kB, nB):
        a = np.zeros(nA); a[:kA] = 1; b = np.zeros(nB); b[:kB] = 1
        sa = a[rng.randint(0, nA, (B, nA))].mean(1); sb = b[rng.randint(0, nB, (B, nB))].mean(1)
        d = sb - sa
        return float(kB/nB - kA/nA), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    i1 = inc(k0, n0, k1, n1); i2 = inc(k1, n1, k2, n2)
    f3[m] = {"normal": {"fpr": k0/n0, "wilson": wilson(k0, n0), "n": n0},
             "unrelated": {"fpr": k1/n1, "wilson": wilson(k1, n1), "n": n1},
             "correlated": {"fpr": k2/n2, "wilson": wilson(k2, n2), "n": n2},
             "increment_unrel_minus_normal": {"diff": i1[0], "ci": i1[1]},
             "increment_corr_minus_unrel": {"diff": i2[0], "ci": i2[1]}}
out["fig3_fpr"] = f3

# ---------- §5.4: isolated vs correlated — unpaired CI + stratified re-slice ----------
s54 = {}
for m in MODELS:
    R = json.load(open(f"{R4D}/p3_{m}.json"))["d1_rows"]
    iso = [r for r in R if r["subset"] == "hard_pos_isolated"]
    cor = [r for r in R if r["subset"] == "hard_pos_correlated"]
    exi = np.array([r["M"] - r["M_rand"] for r in iso]); exc = np.array([r["M"] - r["M_rand"] for r in cor])
    d, ci = boot_diff_unpaired(exi, exc)
    ent = {"n_iso": len(iso), "n_corr": len(cor),
           "iso_excess": dict(zip(["median", "ci"], boot_median(exi))),
           "corr_excess": dict(zip(["median", "ci"], boot_median(exc))),
           "unpaired_diff": {"diff": d, "ci": ci}}
    # stratified: area tertile x confidence(margin_full) tertile
    allr = iso + cor
    A = np.array([r["roi_area"] or 0 for r in allr]); C = np.array([r["margin_full"] for r in allr])
    aq = np.quantile(A, [1/3, 2/3]); cq = np.quantile(C, [1/3, 2/3])
    strata = []
    for ai, (alo, ahi) in enumerate([(-1, aq[0]), (aq[0], aq[1]), (aq[1], 1e9)]):
        for cix, (clo, chi) in enumerate([(-1e9, cq[0]), (cq[0], cq[1]), (cq[1], 1e9)]):
            si = [r for r in iso if alo < (r["roi_area"] or 0) <= ahi and clo < r["margin_full"] <= chi]
            sc = [r for r in cor if alo < (r["roi_area"] or 0) <= ahi and clo < r["margin_full"] <= chi]
            if len(si) >= 15 and len(sc) >= 15:
                strata.append({"area_bin": ai, "conf_bin": cix, "n_iso": len(si), "n_corr": len(sc),
                               "iso_med": float(np.median([r["M"] - r["M_rand"] for r in si])),
                               "corr_med": float(np.median([r["M"] - r["M_rand"] for r in sc]))})
    ent["strata"] = strata
    ent["strata_collapse_holds_in"] = sum(1 for s in strata if s["iso_med"] > s["corr_med"])
    ent["strata_total"] = len(strata)
    s54[m] = ent
out["sec54_iso_vs_corr"] = s54

# ---------- GEMeX excess CIs ----------
gx = {}
for m in ["medgemma", "qwen3vl"]:
    R = json.load(open(f"{R4D}/mjsd_{m}_gemex_mt.json"))["rows"]
    EX = np.array([r["M"] - r["M_rand"] for r in R])
    med, ci = boot_median(EX)
    MR = np.array([r["M_rand"] for r in R]); MM = np.array([r["M"] for r in R])
    ratio_s = []
    for _ in range(B):
        idx = rng.randint(0, len(R), len(R))
        mr, mm = np.mean(MR[idx]), np.mean(MM[idx])
        if mm != 0: ratio_s.append(mr / mm)
    gx[m] = {"n": len(R), "excess": {"median": med, "ci": ci},
             "fill_share_ratio_meanMrand_over_meanM": {"point": float(MR.mean()/MM.mean()) if MM.mean() != 0 else None,
                                                        "ci": [float(np.percentile(ratio_s, 2.5)), float(np.percentile(ratio_s, 97.5))]}}
out["gemex"] = gx

# ---------- Table 13: agreement strata (medgemma train run) ----------
R = json.load(open(f"{R4D}/mjsd_medgemma_vindr_train_agree.json"))["rows"]
t13 = {}
for k_ in (1, 2, 3):
    x = np.array([r["Mm"] - r["Mm_rand"] for r in R if (r.get("rad_agreement") or 0) == k_])
    med, ci = boot_median(x)
    t13[f"agree{k_}"] = {"n": len(x), "median": med, "ci": ci}
x1 = np.array([r["Mm"] - r["Mm_rand"] for r in R if (r.get("rad_agreement") or 0) == 1])
x3 = np.array([r["Mm"] - r["Mm_rand"] for r in R if (r.get("rad_agreement") or 0) == 3])
d, ci = boot_diff_unpaired(x3, x1)
t13["agree3_minus_agree1"] = {"diff": d, "ci": ci}
out["table13_agreement"] = t13

# ---------- §5.5 entailment filter ----------
ENT = {("Lung Opacity", "Consolidation"), ("Lung Opacity", "Infiltration"), ("Lung Opacity", "Atelectasis"),
       ("Lung Opacity", "Pulmonary fibrosis"), ("Infiltration", "Consolidation"), ("Consolidation", "Infiltration")}
e55 = {}
for m in MODELS:
    R = json.load(open(f"{R4D}/p3_{m}.json"))["d2_rows"]
    ka, na = sum(1 for r in R if r["pred_yes"]), len(R)
    keep = [r for r in R if (r["finding"], r["context_y"]) not in ENT]
    kf, nf = sum(1 for r in keep if r["pred_yes"]), len(keep)
    e55[m] = {"all": {"fpr": ka/na, "wilson": wilson(ka, na), "n": na},
              "filtered": {"fpr": kf/nf, "wilson": wilson(kf, nf), "n": nf}}
out["sec55_entailment"] = e55

# ---------- #9 per-finding table ----------
pft = {}
for m in MODELS:
    R = json.load(open(f"{R4D}/mjsd_{m}_vindr_release_mm.json"))["rows"]
    per = {}
    from collections import defaultdict
    byf = defaultdict(list)
    for r in R: byf[r["finding"]].append(r)
    for f, rs in sorted(byf.items()):
        ex = np.array([r["Mm"] - r["Mm_rand"] for r in rs])
        med, ci = boot_median(ex) if len(rs) >= 10 else (float(np.median(ex)), None)
        per[f] = {"n": len(rs), "area_med": float(np.median([r["roi_area"] for r in rs])),
                  "M_roi_med": float(np.median([r["Mm"] for r in rs])),
                  "M_rand_med": float(np.median([r["Mm_rand"] for r in rs])),
                  "excess": {"median": med, "ci": ci}}
    pft[m] = per
out["per_finding_table"] = pft

# ---------- #15 excess by confidence decile ----------
c15 = {}
for m in MODELS:
    R = json.load(open(f"{R4D}/mjsd_{m}_vindr_release_mm.json"))["rows"]
    CF = np.array([r["mg_full"] for r in R]); EX = np.array([r["Mm"] - r["Mm_rand"] for r in R])
    qs = np.quantile(CF, np.linspace(0, 1, 6))
    rowsq = []
    for i in range(5):
        msk = (CF >= qs[i]) & (CF <= qs[i+1]) if i == 4 else (CF >= qs[i]) & (CF < qs[i+1])
        rowsq.append({"conf_bin": i, "conf_range": [float(qs[i]), float(qs[i+1])],
                      "n": int(msk.sum()), "excess_med": float(np.median(EX[msk]))})
    c15[m] = rowsq
out["excess_by_confidence"] = c15

json.dump(out, open(f"{OUT}/ci_tables.json", "w"), indent=1)

# ---------- markdown digest ----------
L = ["# CI tables & re-slices (Phase-1 gap results)\n",
     "Convention: median [95% CI], paired bootstrap over cases, 10,000 resamples; proportions: Wilson 95%.\n"]
L.append("\n## Fig 1 / Tables 4, 11, 12 (per model x tertile)\n")
L.append("| model | tertile | Mm-excess [CI] | Mm_rand [CI] | %>0 [Wilson] | sign-test p | raw-excess (T11) | F (T12) |")
L.append("|---|---|---|---|---|---|---|---|")
for m in MODELS:
    for t in ["small", "med", "large"]:
        e = fig1[m][t]
        L.append(f"| {m} | {t} | {e['excess_Mm']['median']:+.3f} {e['excess_Mm']['ci']} | "
                 f"{e['Mm_rand']['median']:+.3f} {e['Mm_rand']['ci']} | "
                 f"{e['pct_pos_T4']['value']:.2f} {e['pct_pos_T4']['wilson']} | "
                 f"{e['sign_test']['p_one_sided']:.2e} | {e['excess_raw_T11']['median']:+.3f} | {e['F_T12']['median']:+.3f} |")
    tr = fig1[m]["trend_large_minus_small"]
    L.append(f"| {m} | large−small | {tr['diff']:+.3f} {tr['ci']} | | | | | |")
L.append("\n## §5.4 isolated vs correlated (unpaired CI + stratified)\n")
for m in MODELS:
    e = s54[m]
    L.append(f"- **{m}**: iso {e['iso_excess']['median']:+.3f} {e['iso_excess']['ci']} (n={e['n_iso']}) vs "
             f"corr {e['corr_excess']['median']:+.3f} {e['corr_excess']['ci']} (n={e['n_corr']}); "
             f"diff {e['unpaired_diff']['diff']:+.3f} {e['unpaired_diff']['ci']}; "
             f"collapse holds in {e['strata_collapse_holds_in']}/{e['strata_total']} area×confidence strata")
L.append("\n## Fig 3 FPR groups\n")
for m in MODELS:
    e = f3[m]
    L.append(f"- **{m}**: normal {e['normal']['fpr']:.3f} {e['normal']['wilson']} → unrelated {e['unrelated']['fpr']:.3f} "
             f"{e['unrelated']['wilson']} → correlated {e['correlated']['fpr']:.3f} {e['correlated']['wilson']}; "
             f"increments {e['increment_unrel_minus_normal']['diff']:+.3f} {e['increment_unrel_minus_normal']['ci']} / "
             f"{e['increment_corr_minus_unrel']['diff']:+.3f} {e['increment_corr_minus_unrel']['ci']}")
L.append("\n(Full JSON: ci_tables.json — includes Table 5 paired diffs, Fig 2 named-minus-other, GEMeX with fill-share ratio CI, Table 13 agreement, §5.5 entailment Wilson pairs, the 22x4 per-finding table, excess-by-confidence deciles.)\n")
open(f"{OUT}/ci_tables.md", "w").write("\n".join(L))
print("DONE ->", f"{OUT}/ci_tables.json", "+ ci_tables.md")
