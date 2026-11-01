"""Tier-1 Exp 3 Arm B — is the logp drop specific to the NAMED finding's region?
On hard_pos_correlated cases (finding X present+boxed, correlated Y also boxed):
same image, same question, vary ONLY the masked region:
  M_roi   = mask X's own box            (the named evidence)
  M_other = mask Y's box, rescaled concentrically to X's area (non-overlapping)
  M_rand  = mask a size-matched random box avoiding all evidence
Specific:   M_other ~ M_rand << M_roi.   Any-abnormality: M_other ~ M_roi.
Both raw M (gold-token logp) and renormalized M-tilde (first-token Yes/No renorm,
per tier-1 Exp 1) from the SAME forward. n~300/model.
Usage: CUDA_VISIBLE_DEVICES=<g> python t1_exp3b.py --model medgemma --out results/t1_exp3b_medgemma.json
"""
import os, sys, json, random, argparse, math
import numpy as np
sys.path.insert(0, os.environ.get("VGV_EVAL", os.path.join(os.environ.get("VGV_ROOT", "."), "eval")))
from PIL import Image
from mjsd_harness import TransformersVLM, MODELS, VINDR_VQA, IMAGES, collapse_roi, random_matched_boxes, iou_xyxy

SUFFIX = " Please answer with yes or no."
HS = f"{VINDR_VQA}/hardset_v1.jsonl"


def token_ids(proc):
    tk = proc.tokenizer
    return tk("Yes", add_special_tokens=False).input_ids[0], tk("No", add_special_tokens=False).input_ids[0]


def all_boxes_by_image(image_ids):
    sys.path.insert(0, os.environ.get("VGV_BUILD", os.path.join(os.environ.get("VGV_ROOT", "."), "scripts")))
    from build_vqa import load_ann, consensus
    df = load_ann(); boxed = df.dropna(subset=["x_min"])
    dims = json.load(open(f"{VINDR_VQA}/dims.json"))
    out = {}
    for iid, g in boxed[boxed.image_id.isin(image_ids)].groupby("image_id"):
        H, W = dims[iid]; per = {}
        for f, gg in g.groupby("class_name"):
            bxs = consensus(gg[["x_min", "y_min", "x_max", "y_max"]].values.tolist())
            if bxs:
                bx = max(bxs, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                per[f] = [bx[0]/W, bx[1]/H, bx[2]/W, bx[3]/H]
        out[iid] = per
    return out


def rescale_box(b, target_area):
    """Concentric resize of normalized box to target_area (aspect preserved), clipped to [0,1]."""
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
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    adapter = TransformersVLM(MODELS[a.model])
    yid, nid = token_ids(adapter.proc)

    rows = [json.loads(l) for l in open(HS)]
    cor = [r for r in rows if r["subset"] == "hard_pos_correlated" and r["roi_box"] and r["context_findings"]]
    random.Random(0).shuffle(cor)
    ab = all_boxes_by_image({r["image_id"] for r in cor[:a.n*3]})

    def sc(img, q):
        lp, dists = adapter.score(img, q, "Yes")
        p = dists[0]
        pm = math.log(max(p[yid], 1e-12) / max(p[yid] + p[nid], 1e-12))   # renormalized log p(Yes|{Yes,No})
        return lp, pm

    out_rows = []
    for r in cor:
        if len(out_rows) >= a.n: break
        per = ab.get(r["image_id"], {})
        oth = next((per[f] for f in r["context_findings"] if f in per), None)
        if oth is None: continue
        roi = r["roi_box"]; area = (roi[2]-roi[0])*(roi[3]-roi[1])
        oth2 = rescale_box(oth, area)
        if oth2 is None or iou_xyxy(oth2, roi) >= 0.05: continue
        img = Image.open(f"{IMAGES}/{r['image_id']}.png").convert("RGB")
        W, H = img.size
        px = lambda b: [b[0]*W, b[1]*H, b[2]*W, b[3]*H]
        avoid = [px(b) for b in per.values()]
        q = r["question"].rstrip() + SUFFIX
        lp_f, pm_f = sc(img, q)
        lp_r, pm_r = sc(collapse_roi(img, [px(roi)]), q)
        lp_o, pm_o = sc(collapse_roi(img, [px(oth2)]), q)
        rnd = random_matched_boxes([px(roi)], img.size, seed=len(out_rows), avoid=avoid)
        lp_n, pm_n = sc(collapse_roi(img, rnd), q)
        out_rows.append({"finding": r["finding"], "roi_area": area,
                         "M_roi": lp_f-lp_r, "M_other": lp_f-lp_o, "M_rand": lp_f-lp_n,
                         "Mt_roi": pm_f-pm_r, "Mt_other": pm_f-pm_o, "Mt_rand": pm_f-pm_n})
        if len(out_rows) % 50 == 0:
            print(f"  {len(out_rows)}/{a.n}", flush=True)

    def med(k): return float(np.median([r[k] for r in out_rows]))
    summ = {"model": a.model, "n": len(out_rows),
            **{f"med_{k}": med(k) for k in ["M_roi","M_other","M_rand","Mt_roi","Mt_other","Mt_rand"]},
            "excess_roi_vs_rand_Mt": med("Mt_roi")-med("Mt_rand"),
            "excess_other_vs_rand_Mt": med("Mt_other")-med("Mt_rand")}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summ, "rows": out_rows}, open(a.out, "w"), indent=2)
    print("T1-EXP3B SUMMARY:", json.dumps(summ, indent=2), flush=True)


if __name__ == "__main__":
    main()
