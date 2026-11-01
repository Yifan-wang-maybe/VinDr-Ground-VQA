"""Summarise one model's benchmark run into the leaderboard quantities.
Reads $VGV_RESULTS/{p1,mjsd_*_vindr_release_mm,p3,t1_exp3a}_<model>.json and prints:
  balanced accuracy (matched set) and blind accuracy; excess by ROI-size tertile with paired-bootstrap CIs;
  median M_rand; isolated vs correlated excess and detection; FPR normal / unrelated / correlated and increments.
Usage: python summarize_model.py --model medgemma [--append results/leaderboard_mine.csv]
"""
import os, json, csv, argparse
import numpy as np

RES = os.environ.get("VGV_RESULTS", os.path.join(os.environ.get("VGV_ROOT", "."), "results", "mine"))
rng = np.random.default_rng(0)


def boot_med(v, n=10000):
    v = np.asarray(v, float); idx = rng.integers(0, len(v), (n, len(v))); m = np.median(v[idx], axis=1)
    return float(np.median(v)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--append", default=None)
    a = ap.parse_args(); m = a.model; out = {"model": m}
    p1 = f"{RES}/p1_{m}.json"
    if os.path.exists(p1):
        s = json.load(open(p1))["summary"]; out["balanced_accuracy"] = s["full"]["balanced_acc"]; out["blind_accuracy"] = s["blind"]["acc"]; out["blind_yes_rate"] = s["blind"]["yes_rate"]
        neg = [r for r in json.load(open(p1))["rows"] if r["gold"] == "no"]; out["fpr_normal"] = float(np.mean([r["pred_full"] == "yes" for r in neg]))
    occ = [r for r in json.load(open(f"{RES}/mjsd_{m}_vindr_release_mm.json"))["rows"] if "Mm" in r]
    A = np.array([r["roi_area"] or 0 for r in occ]); q1, q2 = np.quantile(A, [1/3, 2/3])
    for lab, msk in [("small", A <= q1), ("medium", (A > q1) & (A <= q2)), ("large", A > q2)]:
        sel = [occ[j] for j in np.where(msk)[0]]
        out[f"excess_{lab}"], out[f"excess_{lab}_ci_lo"], out[f"excess_{lab}_ci_hi"] = boot_med([r["Mm"] - r["Mm_rand"] for r in sel])
        out[f"M_rand_{lab}"] = float(np.median([r["Mm_rand"] for r in sel]))
    out["trend_large_minus_small"] = out["excess_large"] - out["excess_small"]
    p3 = f"{RES}/p3_{m}.json"
    if os.path.exists(p3):
        P = json.load(open(p3)); d1, d2 = P["d1_rows"], P["d2_rows"]
        iso = [r for r in d1 if r["subset"] == "hard_pos_isolated"]; cor = [r for r in d1 if r["subset"] == "hard_pos_correlated"]
        out["excess_isolated"] = float(np.median([r["M"] - r["M_rand"] for r in iso])); out["excess_correlated"] = float(np.median([r["M"] - r["M_rand"] for r in cor]))
        out["detection_isolated"] = float(np.mean([r["margin_full"] > 0 for r in iso])); out["detection_correlated"] = float(np.mean([r["margin_full"] > 0 for r in cor]))
        out["fpr_correlated"] = float(np.mean([r["pred_yes"] for r in d2]))
    un = f"{RES}/t1_exp3a_{m}.json"
    if os.path.exists(un):
        out["fpr_unrelated"] = json.load(open(un))["summary"]["FPR_unrelated"]
        if "fpr_correlated" in out: out["context_fpr_increment"] = out["fpr_correlated"] - out["fpr_unrelated"]
        if "fpr_normal" in out: out["unrelated_minus_normal"] = out["fpr_unrelated"] - out["fpr_normal"]
    for k, v in out.items(): print(f"{k:28s} {v if not isinstance(v, float) else round(v, 3)}")
    if a.append:
        new = not os.path.exists(a.append)
        with open(a.append, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out.keys()));
            if new: w.writeheader()
            w.writerow(out)


if __name__ == "__main__":
    main()
