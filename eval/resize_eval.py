"""Tier-1 Exp 5 — matched-area contrast: decouple finding TYPE from box AREA by
resizing the SAME box on the SAME image concentrically. Cells: ENLARGE
(Nodule/Mass, Calcification: native -> 4.84%) and SHRINK (Pleural effusion,
Aortic enlargement -> 0.27%; Cardiomegaly reported separately - CTR border cue).
Per case, margins (first-token Yes/No log-odds, one forward each):
full; roi@native; rand@native x k; roi@target; rand@target x k. Excess at both
areas. Exclusions per the doc: resized box clear of every other annotated box
(IoU<0.05), area_achieved/target >= 0.8. Flat mean fill (matches P2).
--validate runs n_per=13 (~50 cases) and compares native-area excess to the
release-run per-finding numbers before any full run.
Usage: CUDA_VISIBLE_DEVICES=<g> python t1_exp5.py --model medgemma --n_per 13 --out results/t1_exp5_val_medgemma.json
"""
import os, sys, json, random, argparse, math
import numpy as np
sys.path.insert(0, os.environ.get("VGV_EVAL", os.path.join(os.environ.get("VGV_ROOT", "."), "eval")))
from PIL import Image
from mjsd_harness import TransformersVLM, MODELS, VINDR_VQA, IMAGES, collapse_roi, random_matched_boxes, iou_xyxy

SUFFIX = " Please answer with yes or no."
K = 5
CELLS = {"enlarge": (["Nodule/Mass", "Calcification"], 0.0484),
         "shrink": (["Pleural effusion", "Aortic enlargement"], 0.0027),
         "cardio": (["Cardiomegaly"], 0.0027)}


def rescale_box(b, target_area):
    cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
    w, h = b[2]-b[0], b[3]-b[1]
    if w <= 0 or h <= 0: return None
    s = math.sqrt(target_area / (w*h))
    w2, h2 = min(w*s, 1.0), min(h*s, 1.0)
    x0, y0 = min(max(cx-w2/2, 0), 1-w2), min(max(cy-h2/2, 0), 1-h2)
    return [x0, y0, x0+w2, y0+h2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--n_per", type=int, default=13, help="cases per finding")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    adapter = TransformersVLM(MODELS[a.model])
    tk = adapter.proc.tokenizer
    yid = tk("Yes", add_special_tokens=False).input_ids[0]
    nid = tk("No", add_special_tokens=False).input_ids[0]

    def margin(img, q):
        _, d = adapter.score(img, q, "Yes")
        p = d[0]
        return float(math.log(max(p[yid], 1e-12) / max(p[nid], 1e-12)))

    rows = [json.loads(l) for l in open(f"{VINDR_VQA}/test.jsonl")]
    all_ev = {}
    for r in rows:
        if r.get("roi_box"):
            all_ev.setdefault(r["image_id"], []).append(r["roi_box"])
    pos = [r for r in rows if r["answer_type"] == "presence" and r["answer"] == "Yes" and r["roi_box"]]
    random.Random(0).shuffle(pos)

    out_rows, excl = [], {"overlap": 0, "under_area": 0}
    for cell, (finds, tgt) in CELLS.items():
        done = {f: 0 for f in finds}
        for r in pos:
            f = r["finding"]
            if f not in finds or done[f] >= a.n_per: continue
            roi = r["roi_box"]
            rs = rescale_box(roi, tgt)
            if rs is None: continue
            others = [b for b in all_ev.get(r["image_id"], []) if b != roi]
            if any(iou_xyxy(rs, b) >= 0.05 for b in others):
                excl["overlap"] += 1; continue
            if (rs[2]-rs[0])*(rs[3]-rs[1]) / tgt < 0.8:
                excl["under_area"] += 1; continue
            img = Image.open(f"{IMAGES}/{r['image_id']}.png").convert("RGB")
            W, H = img.size
            px = lambda b: [b[0]*W, b[1]*H, b[2]*W, b[3]*H]
            avoid = [px(b) for b in all_ev.get(r["image_id"], [])]
            q = r["question"].rstrip() + SUFFIX
            m_full = margin(img, q)
            m_nat = margin(collapse_roi(img, [px(roi)]), q)
            m_tgt = margin(collapse_roi(img, [px(rs)]), q)
            rn, rt = [], []
            for k in range(K):
                rn.append(margin(collapse_roi(img, random_matched_boxes([px(roi)], img.size, seed=1000*k+len(out_rows), avoid=avoid)), q))
                rt.append(margin(collapse_roi(img, random_matched_boxes([px(rs)], img.size, seed=2000*k+len(out_rows), avoid=avoid)), q))
            out_rows.append({"cell": cell, "finding": f, "image_id": r["image_id"],
                             "n_other_evidence": len(others),
                             "area_native": (roi[2]-roi[0])*(roi[3]-roi[1]),
                             "area_target": (rs[2]-rs[0])*(rs[3]-rs[1]),
                             "EX_native": (m_full - m_nat) - float(np.mean([m_full - x for x in rn])),
                             "EX_target": (m_full - m_tgt) - float(np.mean([m_full - x for x in rt])),
                             "Mm_native": m_full - m_nat, "Mm_target": m_full - m_tgt})
            done[f] += 1
            if len(out_rows) % 10 == 0:
                print(f"  {len(out_rows)} cases", flush=True)

    summ = {"model": a.model, "n": len(out_rows), "excluded": excl}
    for cell in CELLS:
        rs = [r for r in out_rows if r["cell"] == cell]
        if rs:
            summ[cell] = {"n": len(rs),
                          "med_EX_native": float(np.median([r["EX_native"] for r in rs])),
                          "med_EX_target": float(np.median([r["EX_target"] for r in rs])),
                          "med_area_native": float(np.median([r["area_native"] for r in rs]))}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summ, "rows": out_rows}, open(a.out, "w"), indent=2)
    print("T1-EXP5 SUMMARY:", json.dumps(summ, indent=2), flush=True)


if __name__ == "__main__":
    main()
