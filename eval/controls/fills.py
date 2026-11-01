"""P0 #3 — fill sensitivity. Re-test excess under blur and black fills (mean-fill
is the paper's default), ROI and the SAME random-box geometry refilled, reusing
cached mg_full. Also collects raw lp and renormalized values per condition so
the F-shrinks-under-blur prediction is testable.
Usage: CUDA_VISIBLE_DEVICES=7 python p0_fills.py --model medgemma --n 250 --out fills_medgemma.json
"""
import os, sys, json, argparse, math
import numpy as np
sys.path.insert(0, os.environ.get("VGV_EVAL", os.path.join(os.environ.get("VGV_ROOT", "."), "eval")))
from PIL import Image, ImageFilter
from mjsd_harness import TransformersVLM, MODELS, VINDR_VQA, IMAGES, random_matched_boxes, load_vindr

R4D = os.environ.get("VGV_RESULTS", os.path.join(os.environ.get("VGV_ROOT", "."), "results"))
SUFFIX = " Please answer with yes or no."


def fill_region(img, boxes, mode):
    out = img.copy()
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(max(x2, x1+1)), int(max(y2, y1+1))
        region = out.crop((x1, y1, x2, y2))
        if mode == "black":
            patch = Image.new("RGB", region.size, (0, 0, 0))
        elif mode == "blur":
            rad = max(6, int(0.15 * max(region.size)))
            big = out.filter(ImageFilter.GaussianBlur(radius=rad)).crop((x1, y1, x2, y2))
            patch = big
        out.paste(patch, (x1, y1))
    return out


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
        lp, d = adapter.score(img, q, "Yes")
        p = d[0]
        mg = float(math.log(max(p[yid], 1e-12) / max(p[nid], 1e-12)))
        rn = float(math.log(max(p[yid], 1e-12) / max(p[yid] + p[nid], 1e-12)))
        return lp, mg, rn

    rows = []
    for i, (e, c) in enumerate(zip(ex[:a.n], cached[:a.n])):
        img = e["image"]; roi = e["boxes"]
        rnd = random_matched_boxes(roi, img.size, seed=i, avoid=e["avoid_boxes"])   # same geometry as cached run
        q = e["question"].rstrip() + SUFFIX
        lp_f, mg_f, rn_f = score3(img, q)                       # fresh full-image pass
        row = {"finding": e["finding"], "roi_area": e["roi_area"],
               "excess_mean": (c["mg_full"]-c["mg_coll"]) - (c["mg_full"]-c["mg_rand"]),
               "M_mean_raw": c["M"], "Mt_mean": c.get("Mt")}
        for mode in ("blur", "black"):
            lp_r, mg_r, rn_r = score3(fill_region(img, roi, mode), q)
            lp_n, mg_n, rn_n = score3(fill_region(img, rnd, mode), q)
            row[f"excess_{mode}"] = (mg_f - mg_r) - (mg_f - mg_n)
            row[f"M_{mode}_raw"] = lp_f - lp_r
            row[f"F_{mode}"] = (lp_f - lp_r) - (rn_f - rn_r)    # F = raw drop - renormalized drop
        rows.append(row)
        if (i+1) % 50 == 0: print(f"  {i+1}/{a.n}", flush=True)

    A = np.array([r["roi_area"] or 0 for r in rows]); q1, q2 = np.quantile(A, [1/3, 2/3])
    summ = {"model": a.model, "n": len(rows)}
    for lab, msk in [("small", A <= q1), ("med", (A > q1) & (A <= q2)), ("large", A > q2)]:
        sel = [rows[j] for j in np.where(msk)[0]]
        summ[lab] = {m: float(np.median([r[f"excess_{m}"] for r in sel])) for m in ("mean", "blur", "black")}
        summ[lab]["M_raw_blur_vs_mean"] = [float(np.median([r["M_blur_raw"] for r in sel])),
                                           float(np.median([r["M_mean_raw"] for r in sel]))]
        summ[lab]["F_blur_med"] = float(np.median([r["F_blur"] for r in sel]))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)
    print("FILLS:", json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
