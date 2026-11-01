"""P3 — evaluate the hard-set (competition test in its proper regime).
  D1 (hard positives): M with ROI-collapse + evidence-avoiding random control,
      compared ISOLATED vs CORRELATED context. Competition => M_excess shrinks
      when context can sustain the answer.
  D2 (hard negatives): FPR on absent-but-correlated findings (prior-following),
      vs the clean-normal specificity from P1. Blind margins on a subsample
      verify the model's contextual prior is actually elevated.

Usage: CUDA_VISIBLE_DEVICES=<g> python hardset_eval.py --model medgemma --out results/p3_medgemma.json
"""
import os, sys, json, random, argparse
import numpy as np
from PIL import Image
from collections import defaultdict
from mjsd_harness import TransformersVLM, MODELS, VINDR_VQA, IMAGES, collapse_roi, random_matched_boxes

HS = os.environ.get("HARDSET_PATH", f"{VINDR_VQA}/hardset_v1.jsonl")   # override for lift-cutoff sensitivity sets
SUFFIX = " Please answer with yes or no."


def load(cap_neg=60):
    rows = [json.loads(l) for l in open(HS)]
    d1 = [r for r in rows if r["subset"].startswith("hard_pos")]
    by = defaultdict(list)
    for r in rows:
        if r["subset"] == "hard_neg":
            by[(r["finding"], r["context_y"])].append(r)
    rng = random.Random(0); d2 = []
    for k in sorted(by):
        v = by[k]; rng.shuffle(v); d2.extend(v[:cap_neg])
    return d1, d2


def avoid_map(image_ids):
    """image_id -> ALL annotated finding boxes (normalized), for the random control."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_build"))
    from build_vqa import load_ann, consensus
    df = load_ann(); boxed = df.dropna(subset=["x_min"])
    dims = json.load(open(f"{VINDR_VQA}/dims.json"))
    out = {}
    for iid, g in boxed[boxed.image_id.isin(image_ids)].groupby("image_id"):
        H, W = dims[iid]; bl = []
        for _, gg in g.groupby("class_name"):
            for bx in consensus(gg[["x_min", "y_min", "x_max", "y_max"]].values.tolist()):
                bl.append([bx[0]/W, bx[1]/H, bx[2]/W, bx[3]/H])
        out[iid] = bl
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    adapter = TransformersVLM(MODELS[a.model])
    d1, d2 = load()
    print(f"[p3] D1={len(d1)} D2={len(d2)}", flush=True)
    av = avoid_map({r["image_id"] for r in d1})

    # ---------- D1: M with paired evidence-avoiding random control ----------
    d1_rows = []
    for i, r in enumerate(d1):
        img = Image.open(f"{IMAGES}/{r['image_id']}.png").convert("RGB")
        W, H = img.size
        b = r["roi_box"]; roi = [[b[0]*W, b[1]*H, b[2]*W, b[3]*H]]
        avoid = [[x[0]*W, x[1]*H, x[2]*W, x[3]*H] for x in av.get(r["image_id"], [])]
        q = r["question"].rstrip() + SUFFIX
        ly_full, _ = adapter.score(img, q, "Yes"); ln_full, _ = adapter.score(img, q, "No")
        ly_coll, _ = adapter.score(collapse_roi(img, roi), q, "Yes")
        rnd = collapse_roi(img, random_matched_boxes(roi, img.size, seed=i, avoid=avoid))
        ly_rand, _ = adapter.score(rnd, q, "Yes")
        d1_rows.append({"subset": r["subset"], "finding": r["finding"], "roi_area": r["roi_area"],
                        "pair_lift": r.get("pair_lift"), "n_context": r["n_context"],
                        "margin_full": ly_full - ln_full,
                        "M": ly_full - ly_coll, "M_rand": ly_full - ly_rand})
        if (i + 1) % 200 == 0:
            print(f"  D1 {i+1}/{len(d1)}", flush=True)

    # ---------- D2: FPR + blind prior on a subsample ----------
    d2_rows = []
    for i, r in enumerate(d2):
        img = Image.open(f"{IMAGES}/{r['image_id']}.png").convert("RGB")
        q = r["question"].rstrip() + SUFFIX
        ly, _ = adapter.score(img, q, "Yes"); ln, _ = adapter.score(img, q, "No")
        row = {"finding": r["finding"], "context_y": r["context_y"], "pair_lift": r["pair_lift"],
               "margin_full": ly - ln, "pred_yes": bool(ly > ln)}
        if i < 800:                                    # blind prior verification subsample
            by_, _ = adapter.score(None, q, "Yes"); bn_, _ = adapter.score(None, q, "No")
            row["margin_blind"] = by_ - bn_
        d2_rows.append(row)
        if (i + 1) % 300 == 0:
            print(f"  D2 {i+1}/{len(d2)} FPR so far {np.mean([x['pred_yes'] for x in d2_rows]):.3f}", flush=True)

    # ---------- summaries ----------
    def med(x): return float(np.median(x)) if len(x) else None
    summ = {"model": a.model}
    for sub in ("hard_pos_isolated", "hard_pos_correlated"):
        rs = [r for r in d1_rows if r["subset"] == sub]
        summ[sub] = {"n": len(rs), "med_M": med([r["M"] for r in rs]),
                     "med_M_rand": med([r["M_rand"] for r in rs]),
                     "med_excess": med([r["M"] - r["M_rand"] for r in rs]),
                     "det_rate": float(np.mean([r["margin_full"] > 0 for r in rs]))}
    L = np.array([r["pair_lift"] for r in d2_rows]); P = np.array([r["pred_yes"] for r in d2_rows])
    q1, q2 = np.quantile(L, [1/3, 2/3])
    summ["hard_neg"] = {"n": len(d2_rows), "FPR": float(P.mean()),
                        "FPR_by_lift": {"low": float(P[L <= q1].mean()),
                                        "mid": float(P[(L > q1) & (L <= q2)].mean()),
                                        "high": float(P[L > q2].mean())},
                        "blind_yes_rate": float(np.mean([r["margin_blind"] > 0 for r in d2_rows if "margin_blind" in r]))}
    per_pair = defaultdict(list)
    for r in d2_rows: per_pair[(r["finding"], r["context_y"])].append(r["pred_yes"])
    summ["worst_pairs"] = sorted(({"ask": k[0], "context": k[1], "FPR": float(np.mean(v)), "n": len(v)}
                                  for k, v in per_pair.items() if len(v) >= 20),
                                 key=lambda d: -d["FPR"])[:10]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summ, "d1_rows": d1_rows, "d2_rows": d2_rows}, open(a.out, "w"), indent=2)
    print("P3 SUMMARY:", json.dumps(summ, indent=2)[:1800], flush=True)


if __name__ == "__main__":
    main()
