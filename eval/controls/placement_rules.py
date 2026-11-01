"""P0 #2 — random-box placement validity.
Concern: uniform random boxes land on borders/mediastinum; finding boxes sit in
lung fields -> M_rand biased low -> excess biased high. Check with stricter
placement rules, reusing cached mg_full/mg_coll from the release_mm runs (same
seed-0 case selection), so each rule costs ONE new forward per case.
Rules: body   = random box mean intensity above the image's 25th percentile
       imatch = |mean(rand) - mean(roi)| <= 25 (0-255) after up to 200 draws
Also records ROI/random centre coordinates (current + new rules) for the
distribution figure. Usage: CUDA_VISIBLE_DEVICES=6 python p0_placement.py --model medgemma --n 500 --out placement_medgemma.json
"""
import os, sys, json, argparse, math
import numpy as np
sys.path.insert(0, os.environ.get("VGV_EVAL", os.path.join(os.environ.get("VGV_ROOT", "."), "eval")))
from PIL import Image
from mjsd_harness import TransformersVLM, MODELS, VINDR_VQA, IMAGES, collapse_roi, iou_xyxy, load_vindr

R4D = os.environ.get("VGV_RESULTS", os.path.join(os.environ.get("VGV_ROOT", "."), "results"))
SUFFIX = " Please answer with yes or no."


def draw_rule(img_arr, roi_px, avoid, size, rule, rng):
    W, H = size
    x1, y1, x2, y2 = roi_px
    w, h = min(x2-x1, W-1), min(y2-y1, H-1)
    gray = img_arr.mean(axis=2)
    p25 = np.percentile(gray, 25)
    roi_mean = gray[int(y1):max(int(y2), int(y1)+1), int(x1):max(int(x2), int(x1)+1)].mean()
    best, best_gap = None, 1e9
    for _ in range(200):
        rx, ry = rng.uniform(0, W-w), rng.uniform(0, H-h)
        cand = [rx, ry, rx+w, ry+h]
        if any(iou_xyxy(cand, b) >= 0.05 for b in avoid):
            continue
        cm = gray[int(ry):int(ry+h), int(rx):int(rx+w)].mean()
        if rule == "body":
            if cm > p25: return cand, cm
            gap = p25 - cm
        else:  # imatch
            gap = abs(cm - roi_mean)
            if gap <= 25: return cand, cm
        if gap < best_gap: best, best_gap = cand, gap
    if best is None:                      # every draw overlapped an annotation: relax to ROI-avoidance only
        for _ in range(100):
            rx, ry = rng.uniform(0, W-w), rng.uniform(0, H-h)
            cand = [rx, ry, rx+w, ry+h]
            if iou_xyxy(cand, [x1, y1, x2, y2]) < 0.05:
                best = cand; break
        if best is None:
            best = [0.0, 0.0, float(w), float(h)]
    return best, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cached = json.load(open(f"{R4D}/mjsd_{a.model}_vindr_release_mm.json"))["rows"]
    ex = load_vindr(len(cached), False, True)   # same seed-0 selection as the cached run
    assert [e["finding"] for e in ex[:20]] == [c["finding"] for c in cached[:20]], "selection misaligned"
    adapter = TransformersVLM(MODELS[a.model])
    tk = adapter.proc.tokenizer
    yid = tk("Yes", add_special_tokens=False).input_ids[0]
    nid = tk("No", add_special_tokens=False).input_ids[0]

    def margin(img, q):
        _, d = adapter.score(img, q, "Yes")
        p = d[0]
        return float(math.log(max(p[yid], 1e-12) / max(p[nid], 1e-12)))

    rows = []
    rng = np.random.RandomState(123)
    from mjsd_harness import random_matched_boxes
    for i, (e, c) in enumerate(zip(ex[:a.n], cached[:a.n])):
        img = e["image"]; W, H = img.size
        arr = np.array(img, dtype=np.float32)
        roi = e["boxes"][0]
        q = e["question"].rstrip() + SUFFIX
        # everything FRESH under one prompt (cached run used a different prompt)
        mg_full = margin(img, q)
        m_roi = margin(collapse_roi(img, [roi]), q)
        cur = random_matched_boxes([roi], img.size, seed=i, avoid=e["avoid_boxes"])
        m_cur = margin(collapse_roi(img, cur), q)
        row = {"finding": e["finding"], "roi_area": e["roi_area"],
               "mg_full": mg_full, "M_roi": mg_full - m_roi,
               "M_rand_current": mg_full - m_cur,
               "roi_center": [float((roi[0]+roi[2])/2/W), float((roi[1]+roi[3])/2/H)],
               "center_current": [float((cur[0][0]+cur[0][2])/2/W), float((cur[0][1]+cur[0][3])/2/H)]}
        for rule in ("body", "imatch"):
            cand, cm = draw_rule(arr, roi, e["avoid_boxes"], img.size, rule, rng)
            m = margin(collapse_roi(img, [cand]), q)
            row[f"M_rand_{rule}"] = mg_full - m
            row[f"center_{rule}"] = [float((cand[0]+cand[2])/2/W), float((cand[1]+cand[3])/2/H)]
            row[f"intensity_ok_{rule}"] = cm is not None
        rows.append(row)
        if (i+1) % 100 == 0: print(f"  {i+1}/{a.n}", flush=True)

    A = np.array([r["roi_area"] or 0 for r in rows]); q1, q2 = np.quantile(A, [1/3, 2/3])
    summ = {"model": a.model, "n": len(rows)}
    for lab, msk in [("small", A <= q1), ("med", (A > q1) & (A <= q2)), ("large", A > q2), ("all", A > -1)]:
        sel = [rows[j] for j in np.where(msk)[0]]
        summ[lab] = {}
        for rule in ("current", "body", "imatch"):
            exv = [r["M_roi"] - r[f"M_rand_{rule}"] for r in sel]
            summ[lab][rule] = {"med_excess": float(np.median(exv)),
                               "med_M_rand": float(np.median([r[f"M_rand_{rule}"] for r in sel]))}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)
    print("PLACEMENT:", json.dumps({k: v for k, v in summ.items() if k in ("model", "all")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
