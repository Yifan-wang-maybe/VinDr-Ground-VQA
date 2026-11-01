"""Rebuttal E11 — style-consistent (inpainting) fill, R5 W1.
Same design as the fills experiment (p0_fills.py): same seed-0 release cases, same ROI and the SAME
random-box geometry (seed = case index, evidence-avoiding), fresh full-image pass, suffixed prompt.
Fill = OpenCV Telea inpainting (region reconstructed from surrounding texture; no learned model),
inpaint radius 5 px. Readouts: excess_inpaint by tertile vs the cached mean-fill excess, F_inpaint.
Usage: CUDA_VISIBLE_DEVICES=4 python e11_inpaint_fill.py --model medgemma --n 250 --out e11_medgemma.json
"""
import os, sys, json, argparse, math
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0, os.environ.get("VGV_EVAL", os.path.join(os.environ.get("VGV_ROOT", "."), "eval")))
from mjsd_harness import TransformersVLM, MODELS, random_matched_boxes, load_vindr

R4D = os.environ.get("VGV_RESULTS", os.path.join(os.environ.get("VGV_ROOT", "."), "results"))
SUFFIX = " Please answer with yes or no."


def inpaint_region(img, boxes, radius=5):
    arr = np.asarray(img.convert("RGB")).copy()
    mask = np.zeros(arr.shape[:2], np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(max(x2, x1 + 1)), int(max(y2, y1 + 1))
        mask[max(0, y1):y2, max(0, x1):x2] = 1
    out = cv2.inpaint(arr, mask, radius, cv2.INPAINT_TELEA)
    return Image.fromarray(out)


def boot_med(v, n=10000, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(v, float)
    idx = rng.integers(0, len(v), (n, len(v))); m = np.median(v[idx], axis=1)
    return [float(np.median(v)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cached = json.load(open(f"{R4D}/mjsd_{a.model}_vindr_release_mm.json"))["rows"]
    ex = load_vindr(len(cached), False, True)
    assert [e["finding"] for e in ex[:20]] == [c["finding"] for c in cached[:20]], "selection misaligned"
    adapter = TransformersVLM(MODELS[a.model])
    tk = adapter.proc.tokenizer
    yid = tk("Yes", add_special_tokens=False).input_ids[0]
    nid = tk("No", add_special_tokens=False).input_ids[0]

    def score3(img, q):
        lp, d = adapter.score(img, q, "Yes"); p = d[0]
        mg = float(math.log(max(p[yid], 1e-12) / max(p[nid], 1e-12)))
        rn = float(math.log(max(p[yid], 1e-12) / max(p[yid] + p[nid], 1e-12)))
        return lp, mg, rn

    rows = []
    for i, (e, c) in enumerate(zip(ex[:a.n], cached[:a.n])):
        img = e["image"]; roi = e["boxes"]
        rnd = random_matched_boxes(roi, img.size, seed=i, avoid=e["avoid_boxes"])      # same geometry as the cached run
        q = e["question"].rstrip() + SUFFIX
        lp_f, mg_f, rn_f = score3(img, q)
        lp_r, mg_r, rn_r = score3(inpaint_region(img, roi), q)
        lp_n, mg_n, rn_n = score3(inpaint_region(img, rnd), q)
        rows.append({"finding": e["finding"], "roi_area": e["roi_area"],
                     "excess_mean_cached": (c["mg_full"] - c["mg_coll"]) - (c["mg_full"] - c["mg_rand"]),
                     "excess_inpaint": (mg_f - mg_r) - (mg_f - mg_n),
                     "M_inpaint_roi": mg_f - mg_r, "M_inpaint_rand": mg_f - mg_n,
                     "F_inpaint": (lp_f - lp_r) - (rn_f - rn_r)})
        if (i + 1) % 50 == 0: print(f"  {i+1}/{a.n}", flush=True)

    A = np.array([r["roi_area"] or 0 for r in rows]); q1, q2 = np.quantile(A, [1/3, 2/3])
    summ = {"model": a.model, "n": len(rows), "fill": "OpenCV Telea inpainting, radius 5"}
    for lab, msk in [("small", A <= q1), ("med", (A > q1) & (A <= q2)), ("large", A > q2)]:
        sel = [rows[j] for j in np.where(msk)[0]]
        summ[lab] = {"excess_inpaint": boot_med([r["excess_inpaint"] for r in sel]),
                     "excess_mean_cached": float(np.median([r["excess_mean_cached"] for r in sel])),
                     "M_inpaint_rand": float(np.median([r["M_inpaint_rand"] for r in sel])),
                     "F_inpaint": float(np.median([r["F_inpaint"] for r in sel]))}
    json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)
    print("E11:", json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
