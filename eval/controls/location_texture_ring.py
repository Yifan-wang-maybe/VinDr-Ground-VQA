"""Rebuttal E7 — control-ladder extension for R2 W2 / R5 W2 (what makes a control comparable?).
Reuses cached bare-prompt mg_full / Mm (release_mm rows); fresh forwards per case:
  FLM   finding-location-matched control: the normalized box of the SAME finding type from
        ANOTHER patient, re-centred on this image and rescaled to the ROI's area, masked
        (must avoid every annotated box on this image; up to 50 donors tried).
  EDGE  edge/texture-matched control: size-matched random box whose mean Sobel gradient
        magnitude is within 20% of the ROI's (300 tries, nearest otherwise), avoiding annotations.
  RING  extended findings only (roi_area > 2%): mask the outer ring vs the concentric interior
        at EQUAL area (interior = sides x sqrt(0.5)); tests whether evidence is boundary-localized.
excess_vs_X = Mm(cached) - M_X.  Usage: CUDA_VISIBLE_DEVICES=3 python e7_control_ladder.py --model medgemma --out e7_medgemma.json
"""
import os, sys, json, argparse, math, random
import numpy as np
from PIL import Image
sys.path.insert(0, os.environ.get("VGV_EVAL", os.path.join(os.environ.get("VGV_ROOT", "."), "eval")))
from mjsd_harness import TransformersVLM, MODELS, VINDR_VQA, IMAGES, load_vindr, collapse_roi, iou_xyxy

R4D = os.environ.get("VGV_RESULTS", os.path.join(os.environ.get("VGV_ROOT", "."), "results"))
EXTENDED_MIN_AREA = 0.02


def sobel_mag(gray):
    gx = np.zeros_like(gray); gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return np.hypot(gx, gy)


def box_mean(arr2d, b, W, H):
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, max(x2, x1 + 1)), min(H, max(y2, y1 + 1))
    return float(arr2d[y1:y2, x1:x2].mean())


def clip_box(cx, cy, w, h, W, H):
    x1 = min(max(0, cx - w / 2), W - w); y1 = min(max(0, cy - h / 2), H - h)
    return [x1, y1, x1 + w, y1 + h]


def boot_med(v, n=10000, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(v)
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
    # donor pool: normalized ROI boxes per finding from the whole release test split
    donors = {}
    for r in (json.loads(l) for l in open(f"{VINDR_VQA}/test.jsonl")):
        if r.get("roi_box") and r.get("finding"):
            donors.setdefault(r["finding"], []).append((r["image_id"], r["roi_box"]))
    adapter = TransformersVLM(MODELS[a.model])
    tk = adapter.proc.tokenizer
    yid = tk("Yes", add_special_tokens=False).input_ids[0]
    nid = tk("No", add_special_tokens=False).input_ids[0]

    def margin(img, q):
        _, d = adapter.score(img, q, "Yes"); p = d[0]
        return float(math.log(max(p[yid], 1e-12) / max(p[nid], 1e-12)))

    rng = np.random.RandomState(2026); prng = random.Random(2026)
    rows = []
    for i, (e, c) in enumerate(zip(ex, cached)):
        if "Mm" not in c: continue
        img = e["image"]; W, H = img.size; roi = e["boxes"][0]; q = e["question"]
        w, h = roi[2] - roi[0], roi[3] - roi[1]
        avoid = e["avoid_boxes"] if e.get("avoid_boxes") else [roi]
        gray = np.asarray(img.convert("L"), dtype=np.float32)
        grad = sobel_mag(gray)
        row = {"finding": e["finding"], "roi_area": e["roi_area"], "Mm": c["Mm"], "Mm_rand": c["Mm_rand"], "mg_full": c["mg_full"]}
        # --- FLM: same finding, another patient, area-matched, re-centred ---
        pool = [d for d in donors.get(e["finding"], []) if d[0] != e.get("image_id")]
        prng.shuffle(pool); flm = None
        for _, nb in pool[:50]:
            cx, cy = (nb[0] + nb[2]) / 2 * W, (nb[1] + nb[3]) / 2 * H
            cand = clip_box(cx, cy, w, h, W, H)
            if all(iou_xyxy(cand, b) < 0.05 for b in avoid): flm = cand; break
        if flm is not None:
            m = margin(collapse_roi(img, [flm]), q)
            row["M_flm"] = c["mg_full"] - m; row["excess_vs_flm"] = c["Mm"] - row["M_flm"]
            row["flm_center"] = [(flm[0] + flm[2]) / 2 / W, (flm[1] + flm[3]) / 2 / H]
        # --- EDGE-matched size-matched random box ---
        g_roi = box_mean(grad, roi, W, H); best, best_gap = None, 1e9
        for _ in range(300):
            rx, ry = rng.uniform(0, max(W - w, 1)), rng.uniform(0, max(H - h, 1))
            cand = [rx, ry, rx + w, ry + h]
            if any(iou_xyxy(cand, b) >= 0.05 for b in avoid): continue
            gap = abs(box_mean(grad, cand, W, H) - g_roi)
            if gap <= 0.2 * g_roi: best, best_gap = cand, gap; break
            if gap < best_gap: best, best_gap = cand, gap
        if best is not None:
            m = margin(collapse_roi(img, [best]), q)
            row["M_edge"] = c["mg_full"] - m; row["excess_vs_edge"] = c["Mm"] - row["M_edge"]
            row["g_roi"] = g_roi; row["g_edge"] = box_mean(grad, best, W, H); row["edge_matched"] = bool(best_gap <= 0.2 * g_roi)
        # --- RING vs INTERIOR at equal area, extended findings only ---
        if (e["roi_area"] or 0) > EXTENDED_MIN_AREA:
            s = math.sqrt(0.5); cx, cy = (roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2
            inner = [cx - w * s / 2, cy - h * s / 2, cx + w * s / 2, cy + h * s / 2]
            interior_img = collapse_roi(img, [inner])
            ring_img = collapse_roi(img, [roi])
            ib = [int(round(v)) for v in inner]
            ring_img.paste(img.crop((ib[0], ib[1], ib[2], ib[3])), (ib[0], ib[1]))
            row["M_interior"] = c["mg_full"] - margin(interior_img, q)
            row["M_ring"] = c["mg_full"] - margin(ring_img, q)
        rows.append(row)
        if (i + 1) % 100 == 0: print(f"  {i+1}/{len(cached)}", flush=True)

    A = np.array([r["roi_area"] or 0 for r in rows]); q1, q2 = np.quantile(A, [1/3, 2/3])
    summ = {"model": a.model, "n": len(rows), "n_flm": sum("M_flm" in r for r in rows), "n_edge": sum("M_edge" in r for r in rows),
            "edge_matched_frac": float(np.mean([r.get("edge_matched", False) for r in rows if "M_edge" in r])) if any("M_edge" in r for r in rows) else None}
    for lab, msk in [("small", A <= q1), ("med", (A > q1) & (A <= q2)), ("large", A > q2), ("all", A > -1)]:
        sel = [rows[j] for j in np.where(msk)[0]]
        summ[lab] = {"excess_vs_rand": boot_med([r["Mm"] - r["Mm_rand"] for r in sel]),
                     "excess_vs_flm": boot_med([r["excess_vs_flm"] for r in sel if "M_flm" in r]),
                     "excess_vs_edge": boot_med([r["excess_vs_edge"] for r in sel if "M_edge" in r]),
                     "M_rand": float(np.median([r["Mm_rand"] for r in sel])),
                     "M_flm": float(np.median([r["M_flm"] for r in sel if "M_flm" in r])) if any("M_flm" in r for r in sel) else None,
                     "M_edge": float(np.median([r["M_edge"] for r in sel if "M_edge" in r])) if any("M_edge" in r for r in sel) else None}
    extd = [r for r in rows if "M_ring" in r]
    summ["ring_vs_interior"] = {"n": len(extd),
                                "M_ring": boot_med([r["M_ring"] for r in extd]), "M_interior": boot_med([r["M_interior"] for r in extd]),
                                "frac_ring_gt_interior": float(np.mean([r["M_ring"] > r["M_interior"] for r in extd])) if extd else None,
                                "by_finding": {f: {"n": sum(r["finding"] == f for r in extd),
                                                   "M_ring": float(np.median([r["M_ring"] for r in extd if r["finding"] == f])),
                                                   "M_interior": float(np.median([r["M_interior"] for r in extd if r["finding"] == f]))}
                                               for f in sorted({r["finding"] for r in extd}) if sum(r["finding"] == f for r in extd) >= 15}}
    json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)
    print("E7:", json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
