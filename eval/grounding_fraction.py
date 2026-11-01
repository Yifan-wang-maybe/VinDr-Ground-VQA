"""Rebuttal E2 — grounding fraction (GF) and the ROI-only oracle reference.
Same seed-0 release selection and BARE prompt as the cached release_mm runs.
Per case:
  mg_blind   text-only margin (no image entry)      -> image_gain = mg_full(cached) - mg_blind
  oracle     everything OUTSIDE the ROI mean-filled (a reader that can only see the ROI):
             mg_oracle; mg_oblank = same image with the ROI filled too (uniform image).
             A size-matched random box avoiding the ROI lands on already-filled pixels, so
             the oracle's M_rand is 0 exactly and excess_oracle = mg_oracle - mg_oblank.
GF = excess / image_gain: 0 for a blind reader, ~1 for the oracle, measured for each model.
Usage: CUDA_VISIBLE_DEVICES=2 python e2_grounding_fraction.py --model medgemma --out e2_medgemma.json
"""
import os, sys, json, argparse, math
import numpy as np
from PIL import Image
sys.path.insert(0, os.environ.get("VGV_EVAL", os.path.join(os.environ.get("VGV_ROOT", "."), "eval")))
from mjsd_harness import TransformersVLM, MODELS, load_vindr

R4D = os.environ.get("VGV_RESULTS", os.path.join(os.environ.get("VGV_ROOT", "."), "results"))


def mean_color(img):
    return tuple(int(c) for c in np.asarray(img, dtype=np.float32).reshape(-1, 3).mean(0))


def roi_only(img, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    out = Image.new("RGB", img.size, mean_color(img))
    out.paste(img.crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1))), (x1, y1))
    return out


def boot_ratio(num, den, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    num, den = np.asarray(num), np.asarray(den)
    idx = rng.integers(0, len(num), (n, len(num)))
    r = np.median(num[idx], axis=1) / np.median(den[idx], axis=1)
    return float(np.median(num) / np.median(den)), [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
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
        _, d = adapter.score(img, q, "Yes")
        p = d[0]
        return float(math.log(max(p[yid], 1e-12) / max(p[nid], 1e-12)))

    rows = []
    for i, (e, c) in enumerate(zip(ex, cached)):
        if "Mm" not in c:
            continue
        q = e["question"]                                   # bare prompt, as in the cached run
        img = e["image"]; roi = e["boxes"][0]
        mg_blind = margin(None, q)
        orc = roi_only(img, roi)
        mg_oracle = margin(orc, q)
        mg_oblank = margin(Image.new("RGB", img.size, mean_color(img)), q)
        rows.append({"finding": e["finding"], "roi_area": e["roi_area"],
                     "mg_full": c["mg_full"], "excess": c["Mm"] - c["Mm_rand"], "Mm_rand": c["Mm_rand"],
                     "mg_blind": mg_blind, "image_gain": c["mg_full"] - mg_blind,
                     "mg_oracle": mg_oracle, "mg_oblank": mg_oblank,
                     "oracle_gain": mg_oracle - mg_blind, "oracle_excess": mg_oracle - mg_oblank})
        if (i + 1) % 100 == 0: print(f"  {i+1}/{len(cached)}", flush=True)

    exs = [r["excess"] for r in rows]; gain = [r["image_gain"] for r in rows]
    oex = [r["oracle_excess"] for r in rows]; ogain = [r["oracle_gain"] for r in rows]
    A = np.array([r["roi_area"] or 0 for r in rows]); q1, q2 = np.quantile(A, [1/3, 2/3])
    summ = {"model": a.model, "n": len(rows),
            "median_excess": float(np.median(exs)), "median_image_gain": float(np.median(gain)),
            "median_blind_margin": float(np.median([r["mg_blind"] for r in rows])),
            "GF_ratio_of_medians": dict(zip(["value", "ci"], boot_ratio(exs, gain))),
            "GF_percase_median_gain_gt_0.5": float(np.median([r["excess"] / r["image_gain"] for r in rows if r["image_gain"] > 0.5])),
            "n_gain_gt_0.5": int(sum(r["image_gain"] > 0.5 for r in rows)),
            "oracle": {"median_oracle_gain": float(np.median(ogain)), "median_oracle_excess": float(np.median(oex)),
                       "GF_oracle_ratio_of_medians": dict(zip(["value", "ci"], boot_ratio(oex, ogain)))},
            "by_tertile": {}}
    for lab, msk in [("small", A <= q1), ("med", (A > q1) & (A <= q2)), ("large", A > q2)]:
        sel = [rows[j] for j in np.where(msk)[0]]
        summ["by_tertile"][lab] = {"GF": dict(zip(["value", "ci"], boot_ratio([r["excess"] for r in sel], [r["image_gain"] for r in sel]))),
                                   "median_excess": float(np.median([r["excess"] for r in sel])),
                                   "median_image_gain": float(np.median([r["image_gain"] for r in sel]))}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)
    print("E2:", json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
