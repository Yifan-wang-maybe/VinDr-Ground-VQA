"""Rebuttal E10 — strictly-inside-lung-field random control (R4 Q2, literal ask).
Lung masks from torchxrayvision's chest PSPNet (Left Lung ∪ Right Lung) on the 1,000 release
positives of the cached bare-prompt run. Control = size-matched random box with >= 90% of its pixels
inside the lung mask (300 tries; fallback best coverage if >= 70%; else skipped), IoU < 0.05 with all
annotated boxes. One fresh forward per case; excess_vs_lung = Mm(cached) - M_lung. Also records the
ROI's own lung coverage (findings such as cardiomegaly lie mostly outside the lung fields).
Usage: CUDA_VISIBLE_DEVICES=4 python e10_lungfield_control.py --model medgemma --out e10_medgemma.json
"""
import os, sys, json, argparse, math
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, os.environ.get("VGV_EVAL", os.path.join(os.environ.get("VGV_ROOT", "."), "eval")))
from mjsd_harness import TransformersVLM, MODELS, load_vindr, collapse_roi, iou_xyxy

R4D = os.environ.get("VGV_RESULTS", os.path.join(os.environ.get("VGV_ROOT", "."), "results"))
HERE = os.path.dirname(os.path.abspath(__file__))
MASK_CACHE = f"{HERE}/lung_masks_release1000.npz"


def lung_masks(images):
    """Returns list of boolean masks (H, W) at each image's own size."""
    if os.path.exists(MASK_CACHE):
        z = np.load(MASK_CACHE, allow_pickle=True); return list(z["masks"])
    import torchxrayvision as xrv, skimage.transform
    seg = xrv.baseline_models.chestx_det.PSPNet().eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; seg = seg.to(dev)
    li, ri = seg.targets.index("Left Lung"), seg.targets.index("Right Lung")
    out = []
    with torch.no_grad():
        for k, img in enumerate(images):
            g = np.asarray(img.convert("L"), dtype=np.float32)
            x = xrv.datasets.normalize(g, 255)                       # to [-1024, 1024]
            x = skimage.transform.resize(x, (512, 512), preserve_range=True).astype(np.float32)
            pred = seg(torch.from_numpy(x)[None, None].to(dev))[0]  # (14, 512, 512)
            m = (torch.sigmoid(pred[li]) > 0.5) | (torch.sigmoid(pred[ri]) > 0.5)
            m = m.float().cpu().numpy()
            m = skimage.transform.resize(m, g.shape, order=0, preserve_range=True) > 0.5
            out.append(m)
            if (k + 1) % 100 == 0: print(f"  seg {k+1}/{len(images)}", flush=True)
    np.savez_compressed(MASK_CACHE, masks=np.array(out, dtype=object))
    return out


def coverage(mask, b):
    H, W = mask.shape
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, max(x2, x1 + 1)), min(H, max(y2, y1 + 1))
    return float(mask[y1:y2, x1:x2].mean())


def boot_med(v, n=10000, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(v, float)
    if len(v) == 0: return [None, None, None]
    idx = rng.integers(0, len(v), (n, len(v))); m = np.median(v[idx], axis=1)
    return [float(np.median(v)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cached = json.load(open(f"{R4D}/mjsd_{a.model}_vindr_release_mm.json"))["rows"]
    ex = load_vindr(len(cached), False, True)
    assert [e["finding"] for e in ex[:20]] == [c["finding"] for c in cached[:20]], "selection misaligned"
    masks = lung_masks([e["image"] for e in ex])
    adapter = TransformersVLM(MODELS[a.model])
    tk = adapter.proc.tokenizer
    yid = tk("Yes", add_special_tokens=False).input_ids[0]
    nid = tk("No", add_special_tokens=False).input_ids[0]

    def margin(img, q):
        _, d = adapter.score(img, q, "Yes"); p = d[0]
        return float(math.log(max(p[yid], 1e-12) / max(p[nid], 1e-12)))

    rng = np.random.RandomState(99)
    rows = []
    for i, (e, c, mask) in enumerate(zip(ex, cached, masks)):
        if "Mm" not in c: continue
        img = e["image"]; W, H = img.size; roi = e["boxes"][0]; w, h = roi[2] - roi[0], roi[3] - roi[1]
        avoid = e["avoid_boxes"] if e.get("avoid_boxes") else [roi]
        best, best_cov = None, -1.0
        for _ in range(300):
            rx, ry = rng.uniform(0, max(W - w, 1)), rng.uniform(0, max(H - h, 1))
            cand = [rx, ry, rx + w, ry + h]
            if any(iou_xyxy(cand, b) >= 0.05 for b in avoid): continue
            cov = coverage(mask, cand)
            if cov >= 0.9: best, best_cov = cand, cov; break
            if cov > best_cov: best, best_cov = cand, cov
        row = {"finding": e["finding"], "roi_area": e["roi_area"], "Mm": c["Mm"], "Mm_rand": c["Mm_rand"],
               "roi_lung_cov": coverage(mask, roi), "lung_frac_image": float(mask.mean())}
        if best is not None and best_cov >= 0.7:
            m = margin(collapse_roi(img, [best]), q := e["question"])
            row.update({"M_lung": c["mg_full"] - m, "excess_vs_lung": c["Mm"] - (c["mg_full"] - m), "lung_cov": best_cov,
                        "lung_center": [(best[0] + best[2]) / 2 / W, (best[1] + best[3]) / 2 / H]})
        rows.append(row)
        if (i + 1) % 100 == 0: print(f"  {i+1}/{len(cached)}", flush=True)

    A = np.array([r["roi_area"] or 0 for r in rows]); q1, q2 = np.quantile(A, [1/3, 2/3])
    ok = [r for r in rows if "M_lung" in r]
    summ = {"model": a.model, "n": len(rows), "n_placed": len(ok), "frac_strict_90": float(np.mean([r["lung_cov"] >= 0.9 for r in ok])) if ok else None,
            "median_roi_lung_cov": float(np.median([r["roi_lung_cov"] for r in rows])),
            "roi_lung_cov_by_finding": {f: float(np.median([r["roi_lung_cov"] for r in rows if r["finding"] == f])) for f in sorted({r["finding"] for r in rows})}}
    for lab, msk in [("small", A <= q1), ("med", (A > q1) & (A <= q2)), ("large", A > q2), ("all", A > -1)]:
        sel = [rows[j] for j in np.where(msk)[0]]; selok = [r for r in sel if "M_lung" in r]
        summ[lab] = {"n_placed": len(selok),
                     "excess_vs_rand_same_cases": boot_med([r["Mm"] - r["Mm_rand"] for r in selok]),
                     "excess_vs_lung": boot_med([r["excess_vs_lung"] for r in selok]),
                     "M_rand": float(np.median([r["Mm_rand"] for r in selok])) if selok else None,
                     "M_lung": float(np.median([r["M_lung"] for r in selok])) if selok else None}
    json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)
    print("E10:", json.dumps(summ, indent=1)[:1500], flush=True)


if __name__ == "__main__":
    main()
