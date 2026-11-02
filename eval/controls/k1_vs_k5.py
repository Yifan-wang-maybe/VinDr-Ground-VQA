"""E13 — does the main-table control (k = 1 placement per case) differ from a k = 5 average?
Same cases, prompt (bare) and cached mg_full / mg_coll as the release run. For each case draw five
fresh evidence-avoiding size-matched placements (seeds case*100 + j), score each, and compare
  excess_k1 = Mm - Mm_rand(cached single draw, seed = case)      [the paper's main tables]
  excess_k5 = Mm - mean_j(mg_full - mg_rand_j)                   [five-placement average]
by ROI-size tertile (medians, paired bootstrap CI of the difference) and per case.
Usage: CUDA_VISIBLE_DEVICES=0 python e13_k1_vs_k5.py --model medgemma --n 300 --out e13_medgemma.json
"""
import os, sys, json, argparse, math
import numpy as np
sys.path.insert(0, "/home/azureuser/old_cloudfiles/code/yifan.i.wang/Textok/stage4D")
from mjsd_harness import TransformersVLM, MODELS, load_vindr, collapse_roi, random_matched_boxes

R4D = "/home/azureuser/old_cloudfiles/code/yifan.i.wang/Textok/stage4D/results"
K = 5


def bmed(v, n=10000, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(v, float)
    idx = rng.integers(0, len(v), (n, len(v))); m = np.median(v[idx], axis=1)
    return [float(np.median(v)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cached = json.load(open(f"{R4D}/mjsd_{a.model}_vindr_release_mm.json"))["rows"]
    ex = load_vindr(len(cached), False, True)
    assert [e["finding"] for e in ex[:20]] == [c["finding"] for c in cached[:20]], "selection misaligned"
    adapter = TransformersVLM(MODELS[a.model])
    tk = adapter.proc.tokenizer
    yid = tk("Yes", add_special_tokens=False).input_ids[0]
    nid = tk("No", add_special_tokens=False).input_ids[0]

    def margin(img, q):
        _, d = adapter.score(img, q, "Yes"); p = d[0]
        return float(math.log(max(p[yid], 1e-12) / max(p[nid], 1e-12)))

    rows = []
    for i, (e, c) in enumerate(zip(ex[:a.n], cached[:a.n])):
        if "Mm" not in c: continue
        img = e["image"]; roi = e["boxes"]; q = e["question"]
        draws = []
        for j in range(K):
            rb = random_matched_boxes(roi, img.size, seed=i * 100 + j, avoid=e["avoid_boxes"])
            draws.append(c["mg_full"] - margin(collapse_roi(img, rb), q))
        rows.append({"finding": e["finding"], "roi_area": e["roi_area"], "Mm": c["Mm"],
                     "Mm_rand_k1": c["Mm_rand"], "Mm_rand_k5": float(np.mean(draws)), "draws": draws,
                     "excess_k1": c["Mm"] - c["Mm_rand"], "excess_k5": c["Mm"] - float(np.mean(draws))})
        if (i + 1) % 50 == 0: print(f"  {i+1}/{a.n}", flush=True)

    A = np.array([r["roi_area"] or 0 for r in rows]); q1, q2 = np.quantile(A, [1/3, 2/3])
    summ = {"model": a.model, "n": len(rows), "K": K,
            "percase_absdiff_excess_median": float(np.median([abs(r["excess_k1"] - r["excess_k5"]) for r in rows])),
            "percase_absdiff_Mrand_median": float(np.median([abs(r["Mm_rand_k1"] - r["Mm_rand_k5"]) for r in rows])),
            "between_draw_sd_median": float(np.median([np.std(r["draws"]) for r in rows])), "by_tertile": {}}
    for lab, msk in [("small", A <= q1), ("med", (A > q1) & (A <= q2)), ("large", A > q2), ("all", A > -1)]:
        sel = [rows[j] for j in np.where(msk)[0]]
        summ["by_tertile"][lab] = {"n": len(sel), "excess_k1": bmed([r["excess_k1"] for r in sel]), "excess_k5": bmed([r["excess_k5"] for r in sel]),
                                   "paired_diff_k1_minus_k5": bmed([r["excess_k1"] - r["excess_k5"] for r in sel]),
                                   "Mm_rand_k1": float(np.median([r["Mm_rand_k1"] for r in sel])), "Mm_rand_k5": float(np.median([r["Mm_rand_k5"] for r in sel]))}
    json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)
    print("E13:", json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
