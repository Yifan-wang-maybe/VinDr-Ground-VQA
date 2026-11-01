"""P2 #12 (k sensitivity) + #11 (contralateral-mirror control), one pass.
Per case (all fresh, suffixed prompt): mg_full, m_roi, 20 random draws scored
individually -> M_rand at k=5 (first five) vs k=20; plus the mirror control
(box reflected across the vertical midline; midline/bilateral findings excluded;
skipped if the mirror overlaps any annotation).
Usage: CUDA_VISIBLE_DEVICES=7 python p2_ksens_mirror.py --model medgemma --n 200 --out ksens_mirror_medgemma.json
"""
import os, sys, json, argparse, math
import numpy as np
sys.path.insert(0, os.environ.get("VGV_EVAL", os.path.join(os.environ.get("VGV_ROOT", "."), "eval")))
from mjsd_harness import TransformersVLM, MODELS, load_vindr, collapse_roi, random_matched_boxes, iou_xyxy

SUFFIX = " Please answer with yes or no."
MIDLINE = {"Cardiomegaly", "Mediastinal shift", "Aortic enlargement", "Enlarged PA"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ex = load_vindr(1000, False, True)
    adapter = TransformersVLM(MODELS[a.model])
    tk = adapter.proc.tokenizer
    yid = tk("Yes", add_special_tokens=False).input_ids[0]
    nid = tk("No", add_special_tokens=False).input_ids[0]

    def margin(img, q):
        _, d = adapter.score(img, q, "Yes")
        p = d[0]
        return float(math.log(max(p[yid], 1e-12) / max(p[nid], 1e-12)))

    rows = []
    for i, e in enumerate(ex[:a.n]):
        img = e["image"]; W, H = img.size
        roi = e["boxes"][0]
        q = e["question"].rstrip() + SUFFIX
        mg_full = margin(img, q)
        m_roi = margin(collapse_roi(img, [roi]), q)
        # 20 independent draws (seeds i*100+j), scored individually
        mr = []
        for j in range(20):
            rb = random_matched_boxes([roi], img.size, seed=i*100+j, avoid=e["avoid_boxes"])
            mr.append(margin(collapse_roi(img, rb), q))
        mr = np.array(mr)
        row = {"finding": e["finding"], "roi_area": e["roi_area"],
               "M_roi": mg_full - m_roi,
               "M_rand_k5": float(mg_full - mr[:5].mean()), "M_rand_k20": float(mg_full - mr.mean()),
               "M_rand_draw_sd": float((mg_full - mr).std())}
        # mirror control
        if e["finding"] not in MIDLINE:
            mx1, mx2 = W - roi[2], W - roi[0]
            mbox = [mx1, roi[1], mx2, roi[3]]
            if all(iou_xyxy(mbox, b) < 0.05 for b in e["avoid_boxes"]):
                row["M_mirror"] = mg_full - margin(collapse_roi(img, [mbox]), q)
        rows.append(row)
        if (i+1) % 50 == 0: print(f"  {i+1}/{a.n}", flush=True)

    d5_20 = [abs(r["M_rand_k5"] - r["M_rand_k20"]) for r in rows]
    mir = [r for r in rows if "M_mirror" in r]
    summ = {"model": a.model, "n": len(rows),
            "ksens_medabs_diff_k5_vs_k20": float(np.median(d5_20)),
            "ksens_p90": float(np.percentile(d5_20, 90)),
            "M_rand_between_draw_sd_med": float(np.median([r["M_rand_draw_sd"] for r in rows])),
            "mirror_n": len(mir),
            "excess_vs_mirror_med": float(np.median([r["M_roi"] - r["M_mirror"] for r in mir])) if mir else None,
            "excess_vs_rand_med": float(np.median([r["M_roi"] - r["M_rand_k20"] for r in rows])),
            "M_mirror_med": float(np.median([r["M_mirror"] for r in mir])) if mir else None}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)
    print("KSENS+MIRROR:", json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
