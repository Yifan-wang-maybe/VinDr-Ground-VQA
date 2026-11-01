"""Rebuttal E8 — normal-anatomy border masking (R2 W2 complement).
On NORMAL radiographs (no finding boxes from any reader), ask the extended-finding question
(Cardiomegaly / Pleural effusion / Aortic enlargement; released matched negatives) and mask
  LOCUS  the region where that finding typically lies: a donor ROI of the same finding from another
         patient (normalized coords, donor's own size), placed on this normal image;
  RAND   an equal-size random region elsewhere on the same image (IoU<0.05 with the locus box).
Prompt = question + ' Please answer with yes or no.' (same as the paper's FPR experiments), so the
yes-rates are comparable to Fig. 3's normal group. Readouts: yes-rate under full / locus / random,
median margin shift vs full, paired bootstrap CI of (locus shift - random shift), No->Yes flip rates.
Usage: CUDA_VISIBLE_DEVICES=5 python e8_normal_border.py --model medgemma --out e8_medgemma.json
"""
import os, sys, json, argparse, math, random
import numpy as np
from PIL import Image
sys.path.insert(0, os.environ.get("VGV_EVAL", os.path.join(os.environ.get("VGV_ROOT", "."), "eval")))
sys.path.insert(0, os.environ.get("VGV_BUILD", os.path.join(os.environ.get("VGV_ROOT", "."), "scripts")))
from mjsd_harness import TransformersVLM, MODELS, VINDR_VQA, IMAGES, collapse_roi, iou_xyxy
from build_vqa import load_ann

SUFFIX = " Please answer with yes or no."
FINDINGS = ["Cardiomegaly", "Pleural effusion", "Aortic enlargement"]
N_PER = 150


def boot_med(v, n=10000, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(v)
    idx = rng.integers(0, len(v), (n, len(v))); m = np.median(v[idx], axis=1)
    return [float(np.median(v)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = load_ann()
    boxed_imgs = set(df.dropna(subset=["x_min"])["image_id"])
    normal_imgs = set(df["image_id"]) - boxed_imgs                      # no box from any reader
    rows_rel = [json.loads(l) for l in open(f"{VINDR_VQA}/test.jsonl")]
    donors = {}
    for r in rows_rel:
        if r.get("roi_box") and r.get("finding") in FINDINGS:
            donors.setdefault(r["finding"], []).append(r["roi_box"])
    prng = random.Random(8); rng = np.random.RandomState(8)
    cases = []
    for f in FINDINGS:
        neg = [r for r in rows_rel if r.get("finding") == f and str(r.get("answer", "")).lower().startswith("no")
               and r["image_id"] in normal_imgs and os.path.exists(f"{IMAGES}/{r['image_id']}.png")]
        seen = set(); uniq = []
        for r in neg:
            if r["image_id"] not in seen: seen.add(r["image_id"]); uniq.append(r)
        prng.shuffle(uniq); cases += uniq[:N_PER]
    print(f"[e8] normal images available: {len(normal_imgs)}; cases: {len(cases)}", flush=True)

    adapter = TransformersVLM(MODELS[a.model])
    tk = adapter.proc.tokenizer
    yid = tk("Yes", add_special_tokens=False).input_ids[0]
    nid = tk("No", add_special_tokens=False).input_ids[0]

    def margin(img, q):
        _, d = adapter.score(img, q, "Yes"); p = d[0]
        return float(math.log(max(p[yid], 1e-12) / max(p[nid], 1e-12)))

    out = []
    for i, r in enumerate(cases):
        img = Image.open(f"{IMAGES}/{r['image_id']}.png").convert("RGB"); W, H = img.size
        q = r["question"].rstrip() + SUFFIX
        nb = prng.choice(donors[r["finding"]])
        bw, bh = (nb[2] - nb[0]) * W, (nb[3] - nb[1]) * H
        cx, cy = (nb[0] + nb[2]) / 2 * W, (nb[1] + nb[3]) / 2 * H
        x1 = min(max(0, cx - bw / 2), W - bw); y1 = min(max(0, cy - bh / 2), H - bh)
        locus = [x1, y1, x1 + bw, y1 + bh]
        rand = None
        for _ in range(300):
            rx, ry = rng.uniform(0, max(W - bw, 1)), rng.uniform(0, max(H - bh, 1))
            cand = [rx, ry, rx + bw, ry + bh]
            if iou_xyxy(cand, locus) < 0.05: rand = cand; break
        if rand is None: continue
        mg_full = margin(img, q); mg_loc = margin(collapse_roi(img, [locus]), q); mg_rnd = margin(collapse_roi(img, [rand]), q)
        out.append({"finding": r["finding"], "image_id": r["image_id"], "area": bw * bh / (W * H),
                    "mg_full": mg_full, "mg_locus": mg_loc, "mg_rand": mg_rnd,
                    "d_locus": mg_loc - mg_full, "d_rand": mg_rnd - mg_full})
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(cases)}", flush=True)

    def block(sel):
        return {"n": len(sel),
                "yes_rate_full": float(np.mean([r["mg_full"] > 0 for r in sel])),
                "yes_rate_locus": float(np.mean([r["mg_locus"] > 0 for r in sel])),
                "yes_rate_rand": float(np.mean([r["mg_rand"] > 0 for r in sel])),
                "flip_No_to_Yes_locus": float(np.mean([(r["mg_full"] <= 0) and (r["mg_locus"] > 0) for r in sel])),
                "flip_No_to_Yes_rand": float(np.mean([(r["mg_full"] <= 0) and (r["mg_rand"] > 0) for r in sel])),
                "median_shift_locus": boot_med([r["d_locus"] for r in sel]),
                "median_shift_rand": boot_med([r["d_rand"] for r in sel]),
                "paired_locus_minus_rand": boot_med([r["d_locus"] - r["d_rand"] for r in sel])}
    summ = {"model": a.model, "prompt": "suffixed (FPR convention)", "all": block(out),
            "by_finding": {f: block([r for r in out if r["finding"] == f]) for f in FINDINGS}}
    json.dump({"summary": summ, "rows": out}, open(a.out, "w"), indent=1)
    print("E8:", json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
