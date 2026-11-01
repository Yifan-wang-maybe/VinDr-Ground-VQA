"""Tier-1 Exp 3 Arm A (eval) — FPR on absent-X questions over images with an
UNRELATED finding Y (middle group of the three-group comparison). Includes the
teammates' unrelatedness verification: blind margins and Y-masked margins on
subsamples (pairs where the model's own P(Yes) is elevated blind/masked get
flagged at analysis time). Margins read from one forward via the first-token
Yes/No log-odds. Usage: CUDA_VISIBLE_DEVICES=<g> python t1_exp3a_eval.py --model medgemma --out results/t1_exp3a_medgemma.json
"""
import os, sys, json, random, argparse, math
import numpy as np
sys.path.insert(0, os.environ.get("VGV_EVAL", os.path.join(os.environ.get("VGV_ROOT", "."), "eval")))
from PIL import Image
from mjsd_harness import TransformersVLM, MODELS, VINDR_VQA, IMAGES, collapse_roi
from collections import defaultdict

SUFFIX = " Please answer with yes or no."
UNREL = f"{VINDR_VQA}/hardset_unrelated_v1.jsonl"
CAP_PER_PAIR, N_BLIND, N_YMASK = 30, 600, 600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    adapter = TransformersVLM(MODELS[a.model])
    tk = adapter.proc.tokenizer
    yid = tk("Yes", add_special_tokens=False).input_ids[0]
    nid = tk("No", add_special_tokens=False).input_ids[0]

    def margin(img, q):
        _, dists = adapter.score(img, q, "Yes")
        p = dists[0]
        return float(math.log(max(p[yid], 1e-12) / max(p[nid], 1e-12)))

    rows = [json.loads(l) for l in open(UNREL)]
    by = defaultdict(list)
    for r in rows: by[(r["finding"], r["context_y"])].append(r)
    rng = random.Random(0); sample = []
    for k in sorted(by):
        v = by[k]; rng.shuffle(v); sample.extend(v[:CAP_PER_PAIR])
    rng.shuffle(sample)
    print(f"[3a:{a.model}] {len(sample)} unrelated hard-negs", flush=True)

    # Y-box map for the masked-verification subsample
    sys.path.insert(0, os.environ.get("VGV_BUILD", os.path.join(os.environ.get("VGV_ROOT", "."), "scripts")))
    from build_vqa import load_ann, consensus
    df = load_ann(); boxed = df.dropna(subset=["x_min"])
    dims = json.load(open(f"{VINDR_VQA}/dims.json"))
    need = {r["image_id"] for r in sample[:N_YMASK]}
    ybox = {}
    for iid, g in boxed[boxed.image_id.isin(need)].groupby("image_id"):
        H, W = dims[iid]; per = {}
        for f, gg in g.groupby("class_name"):
            bxs = consensus(gg[["x_min", "y_min", "x_max", "y_max"]].values.tolist())
            if bxs:
                bx = max(bxs, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                per[f] = [bx[0]/W, bx[1]/H, bx[2]/W, bx[3]/H]
        ybox[iid] = per

    out_rows = []
    for i, r in enumerate(sample):
        img = Image.open(f"{IMAGES}/{r['image_id']}.png").convert("RGB")
        q = r["question"].rstrip() + SUFFIX
        rec = {"finding": r["finding"], "context_y": r["context_y"], "pair_lift": r["pair_lift"],
               "margin_full": margin(img, q)}
        if i < N_BLIND:
            rec["margin_blind"] = margin(None, q)
        if i < N_YMASK:
            b = ybox.get(r["image_id"], {}).get(r["context_y"])
            if b:
                W, H = img.size
                rec["margin_ymask"] = margin(collapse_roi(img, [[b[0]*W, b[1]*H, b[2]*W, b[3]*H]]), q)
        out_rows.append(rec)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(sample)} FPR so far {np.mean([x['margin_full']>0 for x in out_rows]):.3f}", flush=True)

    P = np.array([r["margin_full"] > 0 for r in out_rows])
    summ = {"model": a.model, "n": len(out_rows), "FPR_unrelated": float(P.mean()),
            "blind_yes_rate": float(np.mean([r["margin_blind"] > 0 for r in out_rows if "margin_blind" in r])),
            "ymask_yes_rate": float(np.mean([r["margin_ymask"] > 0 for r in out_rows if "margin_ymask" in r]))}
    per_x = {}
    for x in sorted({r["finding"] for r in out_rows}):
        rs = [r for r in out_rows if r["finding"] == x]
        per_x[x] = {"n": len(rs), "FPR": float(np.mean([r["margin_full"] > 0 for r in rs]))}
    summ["per_finding"] = per_x
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summ, "rows": out_rows}, open(a.out, "w"), indent=2)
    print("T1-EXP3A SUMMARY:", json.dumps({k: v for k, v in summ.items() if k != "per_finding"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
