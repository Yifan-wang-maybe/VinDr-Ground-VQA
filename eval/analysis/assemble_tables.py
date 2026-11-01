"""Assemble rebuttal tables from whatever result files exist (skips missing ones).
  T-A  base-model pairs (E1): excess by tertile [CI], M_rand, trend, balanced accuracy
  T-B  grounding fraction (E2): GF ratio-of-medians [CI] overall + by tertile, oracle GF
  T-C  scale (E3 + 30B): excess by tertile, M_rand, FPR increment where hard set exists
  T-D  lift sensitivity incl. the 1.2-1.5 band (E4) -> delegates to e4_lift_sensitivity.py output
Writes rebuttal_tables.md next to this file.
"""
import os, json, math
import numpy as np
from scipy import stats as st

R4D = os.environ.get("VGV_RESULTS", os.path.join(os.environ.get("VGV_ROOT", "."), "results"))
R5A = os.environ.get("VGV_RESULTS", os.path.join(os.environ.get("VGV_ROOT", "."), "results"))
HERE = os.path.dirname(os.path.abspath(__file__))
NICE = {"medgemma": "MedGemma-4b-it", "gemma3": "Gemma-3-4b-it (base)", "lingshu": "Lingshu-7B",
        "qwen25vl": "Qwen2.5-VL-7B (base)", "qwen3vl": "Qwen3-VL-8B", "internvl35": "InternVL3.5-8B",
        "qwen3vl30b": "Qwen3-VL-30B-A3B", "lingshu32b": "Lingshu-32B", "medgemma27b": "MedGemma-27B"}
rng = np.random.default_rng(0)


def boot_med(v, n=10000):
    v = np.asarray(v); idx = rng.integers(0, len(v), (n, len(v)))
    m = np.median(v[idx], axis=1)
    return float(np.median(v)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def tertile_table(model):
    f = f"{R4D}/mjsd_{model}_vindr_release_mm.json"
    if not os.path.exists(f): return None
    rows = [r for r in json.load(open(f))["rows"] if "Mm" in r]
    A = np.array([r["roi_area"] or 0 for r in rows]); q1, q2 = np.quantile(A, [1/3, 2/3])
    out = {"n": len(rows)}
    for lab, msk in [("small", A <= q1), ("med", (A > q1) & (A <= q2)), ("large", A > q2)]:
        sel = [rows[j] for j in np.where(msk)[0]]
        ex = np.array([r["Mm"] - r["Mm_rand"] for r in sel]); mr = np.array([r["Mm_rand"] for r in sel])
        nz = ex[ex != 0]
        p = st.binomtest(int((nz > 0).sum()), len(nz), 0.5, alternative="greater").pvalue if len(nz) else 1.0
        out[lab] = {"excess": boot_med(ex), "Mm_rand": float(np.median(mr)), "frac_pos": float((ex > 0).mean()), "p": p}
    out["trend"] = out["large"]["excess"][0] - out["small"]["excess"][0]
    return out


def balanced_acc(model):
    f = f"{R4D}/p1_{model}.json"
    if not os.path.exists(f): return None
    s = json.load(open(f))["summary"]
    return s.get("full", {}), s.get("blind", {})


def fpr_increment(model):
    p3 = f"{R4D}/p3_{model}.json"; un = f"{R5A}/t1_exp3a_{model}.json"
    if not os.path.exists(p3): return None
    d2 = json.load(open(p3))["d2_rows"]
    fpr_c = float(np.mean([r["pred_yes"] for r in d2]))
    if os.path.exists(un):
        u = json.load(open(un))["rows"]; k = "pred_yes" if "pred_yes" in u[0] else None
        fpr_u = float(np.mean([bool(r[k]) if k else r["margin_full"] > 0 for r in u]))
        return {"fpr_corr": fpr_c, "fpr_unrel": fpr_u, "increment": fpr_c - fpr_u}
    return {"fpr_corr": fpr_c}


def fmt(e): return f"{e[0]:+.2f} [{e[1]:+.2f}, {e[2]:+.2f}]"


L = ["# Rebuttal tables (auto-assembled)", ""]

# ---- T-A base pairs ----
L += ["## T-A. Medical checkpoint vs its own base (identical protocol, n=1000; balanced accuracy on the matched set, n=3000)", "",
      "| model | small | medium | large | trend L−S | M_rand (S/M/L) | balanced acc | blind acc |", "|---|---|---|---|---|---|---|---|"]
for m in ["gemma3", "medgemma", "qwen25vl", "lingshu"]:
    t = tertile_table(m)
    if t is None: L.append(f"| {NICE[m]} | *pending* | | | | | | |"); continue
    ba = balanced_acc(m)
    acc = f"{ba[0].get('balanced_acc', float('nan')):.3f}" if ba and ba[0] else "—"
    bl = f"{ba[1].get('acc', float('nan')):.3f}" if ba and ba[1] else "—"
    L.append(f"| {NICE[m]} | {fmt(t['small']['excess'])} | {fmt(t['med']['excess'])} | {fmt(t['large']['excess'])} | {t['trend']:+.2f} | "
             f"{t['small']['Mm_rand']:+.2f}/{t['med']['Mm_rand']:+.2f}/{t['large']['Mm_rand']:+.2f} | {acc} | {bl} |")
L.append("")

# ---- T-B grounding fraction ----
L += ["## T-B. Grounding fraction GF = excess / image-gain. Headline = per-case median over cases where the image moves the model (image-gain > 0.5 nats), bootstrap CI; ratio-of-medians shown for reference and flagged when the median image-gain is < 0.5 (ill-conditioned). Oracle = evidence-only reader (valid only where its image-gain is positive).", "",
      "| model | median image-gain | median excess | GF per-case (gain>0.5) [CI], n | GF tertiles S/M/L | ratio-of-medians GF | oracle GF |", "|---|---|---|---|---|---|---|"]
for m in ["medgemma", "lingshu", "internvl35", "qwen3vl", "qwen3vl30b", "lingshu32b", "medgemma27b"]:
    f = f"{HERE}/e2_{m}.json"
    if not os.path.exists(f): L.append(f"| {NICE[m]} | *pending* | | | | | |"); continue
    d = json.load(open(f)); s = d["summary"]; rows = d["rows"]
    sel = [r for r in rows if r["image_gain"] > 0.5]
    gf = np.array([r["excess"] / r["image_gain"] for r in sel])
    idx = rng.integers(0, len(gf), (10000, len(gf))); bm = np.median(gf[idx], axis=1)
    areas = np.array([r["roi_area"] or 0 for r in rows]); q1, q2 = np.quantile(areas, [1/3, 2/3])
    A = np.array([r["roi_area"] or 0 for r in sel])
    tert = [np.median(gf[A <= q1]), np.median(gf[(A > q1) & (A <= q2)]), np.median(gf[A > q2])]
    g = s["GF_ratio_of_medians"]; o = s["oracle"]
    rom = f"{g['value']:.2f} [{g['ci'][0]:.2f}, {g['ci'][1]:.2f}]" + (" ⚠ ill-conditioned" if s["median_image_gain"] < 0.5 else "")
    og = o["GF_oracle_ratio_of_medians"]["value"]
    orc = f"{og:.2f}" if 0 < o["median_oracle_gain"] and abs(og) < 5 else "n/a (OOD)"
    L.append(f"| {NICE[m]} | {s['median_image_gain']:+.2f} | {s['median_excess']:+.2f} | **{np.median(gf):.2f}** [{np.percentile(bm, 2.5):.2f}, {np.percentile(bm, 97.5):.2f}], n={len(sel)} | "
             f"{tert[0]:.2f} / {tert[1]:.2f} / {tert[2]:.2f} | {rom} | {orc} |")
L.append("")

# ---- T-C scale ----
L += ["## T-C. Scale (identical protocol, n=1000)", "",
      "| model | small | medium | large | trend | M_rand (S/M/L) | FPR corr − unrel |", "|---|---|---|---|---|---|---|"]
for m in ["qwen3vl", "qwen3vl30b", "lingshu", "lingshu32b", "medgemma", "medgemma27b"]:
    t = tertile_table(m)
    if t is None: L.append(f"| {NICE[m]} | *pending* | | | | | |"); continue
    fi = fpr_increment(m)
    inc = f"{fi['increment']:+.3f}" if fi and "increment" in fi else (f"corr {fi['fpr_corr']:.3f}" if fi else "—")
    L.append(f"| {NICE[m]} | {fmt(t['small']['excess'])} | {fmt(t['med']['excess'])} | {fmt(t['large']['excess'])} | {t['trend']:+.2f} | "
             f"{t['small']['Mm_rand']:+.2f}/{t['med']['Mm_rand']:+.2f}/{t['large']['Mm_rand']:+.2f} | {inc} |")
L.append("")

# ---- T-D lift sensitivity (per-model detail) ----
e4 = f"{HERE}/e4_lift_sensitivity.md"
if os.path.exists(e4):
    L += ["## T-D. Lift-cutoff sensitivity (per-model detail)", "", open(e4).read().split("\n", 2)[2]]
# ---- T-F control ladder (E7) + lung-field (E10) + inpainting (E11) + normal-border (E8) ----
def _b(b): return "n/a" if (b is None or b[0] is None) else f"{b[0]:+.2f} [{b[1]:+.2f}, {b[2]:+.2f}]"
L += ["## T-F. Control ladder on the same cases (excess = M_roi − M_control; median [CI]; n=1000 per model)", "",
      "| model | tertile | vs uniform random (paper) | vs finding-location-matched | vs edge/texture-matched | vs strict lung-field (E10) | median M_control: rand / loc / edge / lung |", "|---|---|---|---|---|---|---|"]
for m in ["medgemma", "lingshu", "internvl35", "qwen3vl"]:
    f7 = f"{HERE}/e7_{m}.json"; f10 = f"{HERE}/e10_{m}.json"
    if not os.path.exists(f7) and not os.path.exists(f10): L.append(f"| {NICE[m]} | *pending* | | | | | |"); continue
    s7 = json.load(open(f7))["summary"] if os.path.exists(f7) else None
    s10 = json.load(open(f10))["summary"] if os.path.exists(f10) else None
    for t in ["small", "med", "large"]:
        b7 = s7[t] if s7 else None; b10 = s10[t] if s10 else None
        L.append(f"| {NICE[m]} | {t} | {_b(b7['excess_vs_rand']) if b7 else '—'} | {_b(b7['excess_vs_flm']) if b7 else '*pending*'} | {_b(b7['excess_vs_edge']) if b7 else '*pending*'} | {_b(b10['excess_vs_lung']) if b10 else '*pending*'} | "
                 f"{(f'{b7[chr(77)+chr(95)+chr(114)+chr(97)+chr(110)+chr(100)]:+.2f}' if b7 else '—')} / {(f'{b7[chr(77)+chr(95)+chr(102)+chr(108)+chr(109)]:+.2f}' if b7 and b7['M_flm'] is not None else '—')} / {(f'{b7[chr(77)+chr(95)+chr(101)+chr(100)+chr(103)+chr(101)]:+.2f}' if b7 and b7['M_edge'] is not None else '—')} / {(f'{b10[chr(77)+chr(95)+chr(108)+chr(117)+chr(110)+chr(103)]:+.2f}' if b10 and b10['M_lung'] is not None else '—')} |")
L.append("")
L += ["### Ring vs interior at equal area (extended findings > 2% of image)", "", "| model | n | M_ring [CI] | M_interior [CI] | share ring > interior |", "|---|---|---|---|---|"]
for m in ["medgemma", "lingshu", "internvl35", "qwen3vl"]:
    f7 = f"{HERE}/e7_{m}.json"
    if not os.path.exists(f7): L.append(f"| {NICE[m]} | *pending* | | | |"); continue
    r = json.load(open(f7))["summary"]["ring_vs_interior"]
    L.append(f"| {NICE[m]} | {r['n']} | {_b(r['M_ring'])} | {_b(r['M_interior'])} | {r['frac_ring_gt_interior']:.2f} |")
L.append("")
L += ["### Normal-anatomy border masking (E8; normal radiographs, suffixed prompt): yes-rate full → locus-masked → random-masked; median margin shift locus / random; paired difference", "", "| model | n | yes-rate full / locus / random | No→Yes flips locus / random | shift locus | shift random | locus − random [CI] |", "|---|---|---|---|---|---|---|"]
for m in ["medgemma", "lingshu", "internvl35", "qwen3vl"]:
    f8 = f"{HERE}/e8_{m}.json"
    if not os.path.exists(f8): L.append(f"| {NICE[m]} | *pending* | | | | | |"); continue
    s = json.load(open(f8))["summary"]["all"]
    L.append(f"| {NICE[m]} | {s['n']} | {s['yes_rate_full']:.3f} / {s['yes_rate_locus']:.3f} / {s['yes_rate_rand']:.3f} | {s['flip_No_to_Yes_locus']:.3f} / {s['flip_No_to_Yes_rand']:.3f} | {_b(s['median_shift_locus'])} | {_b(s['median_shift_rand'])} | {_b(s['paired_locus_minus_rand'])} |")
L.append("")
L += ["### Fills (E11 inpainting added to mean / blur / black): excess by tertile", "", "| model | tertile | mean (paper, cached) | inpainting [CI] | M_rand under inpainting | F_inpaint |", "|---|---|---|---|---|---|"]
for m in ["medgemma", "lingshu", "internvl35", "qwen3vl"]:
    f11 = f"{HERE}/e11_{m}.json"
    if not os.path.exists(f11): L.append(f"| {NICE[m]} | *pending* | | | | |"); continue
    s = json.load(open(f11))["summary"]
    for t in ["small", "med", "large"]:
        b = s[t]; L.append(f"| {NICE[m]} | {t} | {b['excess_mean_cached']:+.2f} | {_b(b['excess_inpaint'])} | {b['M_inpaint_rand']:+.2f} | {b['F_inpaint']:+.2f} |")
L.append("")
# ---- T-G GEMeX text shortcut (E12): blind yes/no on GEMeX-ThinkVG closed questions, text only ----
L += ["## T-G. GEMeX-ThinkVG text shortcut (E12): text-only (blind) yes/no accuracy on the closed questions (first 399 in file order = our evaluated subset; first 1,000 for stability). Gold is 74–75% 'No'.", "",
      "| model | prompt | n | blind accuracy [Wilson] | blind balanced acc. | blind yes-rate | gold yes-rate | majority baseline |", "|---|---|---|---|---|---|---|---|"]
for m in ["medgemma", "lingshu", "internvl35", "qwen3vl"]:
    f12 = f"{HERE}/e12_{m}.json"
    if not os.path.exists(f12): L.append(f"| {NICE[m]} | *pending* | | | | | | |"); continue
    s = json.load(open(f12))["summary"]
    for key, lab in [("first399_suffix", "yes/no suffix (paper FPR convention)"), ("first399_bare", "bare question"), ("first1000_suffix", "suffix, n=1000")]:
        b = s[key]
        L.append(f"| {NICE[m]} | {lab} | {b['n']} | {b['acc']:.3f} [{b['acc_wilson'][0]:.3f}, {b['acc_wilson'][1]:.3f}] | {b['balanced_acc']:.3f} | {b['blind_yes_rate']:.2f} | {b['gold_yes_rate']:.2f} | {b['majority_baseline']:.3f} |")
L.append("")
L += ["### GEMeX image-gain on the stored rows (gold-sentence log-prob, full image − blind; n=399) vs VinDr positives", "", "| model | GEMeX median image-gain | VinDr positives median image-gain | GEMeX median M_roi / M_rand |", "|---|---|---|---|"]
GX = {"medgemma": f"{R4D}/mjsd_medgemma_gemex_mt.json", "qwen3vl": f"{R4D}/mjsd_qwen3vl_gemex_mt.json",
      "lingshu": f"{HERE}/../../gap_results/gemex_lingshu.json", "internvl35": f"{HERE}/../../gap_results/gemex_internvl35.json"}
for m in ["medgemma", "lingshu", "internvl35", "qwen3vl"]:
    try:
        g = json.load(open(GX[m]))["rows"]; v = json.load(open(f"{R4D}/mjsd_{m}_vindr_release_mm.json"))["rows"]
        gg = np.median([r["lp_full"] - r["prior"] for r in g]); vg = np.median([r["lp_full"] - r["prior"] for r in v])
        L.append(f"| {NICE[m]} | {gg:+.2f} | {vg:+.2f} | {np.median([r['M'] for r in g]):+.3f} / {np.median([r['M_rand'] for r in g]):+.3f} |")
    except Exception as e:
        L.append(f"| {NICE[m]} | n/a ({e.__class__.__name__}) | | |")
L.append("")
# ---- T-E full §5.4/§5.5 threshold table ----
e4b = f"{HERE}/e4b_threshold_table.md"
if os.path.exists(e4b):
    L += ["## T-E. §5.4 / §5.5 under alternative co-occurrence thresholds (all models)", "", open(e4b).read().split("\n", 2)[2]]

open(f"{HERE}/rebuttal_tables.md", "w").write("\n".join(L))
print("\n".join(L))
