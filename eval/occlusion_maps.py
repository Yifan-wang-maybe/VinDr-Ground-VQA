"""Rebuttal E9 v2 — occlusion-sensitivity maps with an ROI-SIZED sliding window (R3 "show situations").
Per case the mask has the evidence box's own width/height (floor: image/10), slid with half-window stride
(<= 400 positions); value = margin drop when that window is mean-filled (bare prompt). So each map asks
"where would masking an ROI-sized patch change the answer?" — the paper's intervention, moved around.
Archetypes: A grounded-small (calcification), B grounded-extended (effusion), C prior-driven (fibrosis),
D substitution: correlated positives (small queried box + large co-finding) PRE-SCREENED so MedGemma
asserts the finding on the full image (margin > 1); up to 10 screened, 3 mapped, the one whose
sensitivity sits most on the co-finding is shown. Models: MedGemma-4b, Qwen3-VL-8B.
Usage: CUDA_VISIBLE_DEVICES=3 python e9b_occlusion_maps_v2.py
"""
import os, sys, json, math
import numpy as np
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.environ.get("VGV_EVAL", os.path.join(os.environ.get("VGV_ROOT", "."), "eval")))
sys.path.insert(0, os.environ.get("VGV_EVAL", os.path.join(os.environ.get("VGV_ROOT", "."), "eval")))
from mjsd_harness import TransformersVLM, MODELS, VINDR_VQA, IMAGES, load_vindr, collapse_roi
from t1_exp3b import all_boxes_by_image

HERE = os.path.dirname(os.path.abspath(__file__))
REL_IDX = {"A grounded, small": 46, "B grounded, extended": 126, "C prior-driven": 44}
MAX_POS = 400


def windows(W, H, roi):
    w = max(roi[2] - roi[0], W / 10); h = max(roi[3] - roi[1], H / 10)
    sx, sy = w / 2, h / 2
    xs = list(np.arange(0, W - w + 1e-6, sx)); ys = list(np.arange(0, H - h + 1e-6, sy))
    if xs[-1] < W - w - 1: xs.append(W - w)
    if ys[-1] < H - h - 1: ys.append(H - h)
    while len(xs) * len(ys) > MAX_POS:          # coarsen stride if needed
        sx *= 1.25; sy *= 1.25
        xs = list(np.arange(0, W - w + 1e-6, sx)); ys = list(np.arange(0, H - h + 1e-6, sy))
    return w, h, xs, ys


def slide_map(margin, img, q, mg_full, roi):
    W, H = img.size; w, h, xs, ys = windows(W, H, roi)
    M = np.zeros((len(ys), len(xs)))
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            M[yi, xi] = mg_full - margin(collapse_roi(img, [[x, y, x + w, y + h]]), q)
    return M, (w, h, xs, ys)


def render(M, geom, W, H):
    """Accumulate window drops onto a pixel map (mean over covering windows) for display."""
    w, h, xs, ys = geom; acc = np.zeros((H, W)); cnt = np.zeros((H, W))
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            x1, y1, x2, y2 = int(x), int(y), int(min(W, x + w)), int(min(H, y + h))
            acc[y1:y2, x1:x2] += M[yi, xi]; cnt[y1:y2, x1:x2] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), 0)


def box_stat(pix, b, W, H):
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, max(x2, x1 + 1)), min(H, max(y2, y1 + 1))
    return float(pix[y1:y2, x1:x2].mean())


def main():
    ex = load_vindr(1000, False, True)
    adapters = {m: TransformersVLM(MODELS[m]) for m in ("medgemma", "qwen3vl")}
    def mk(m):
        tk = adapters[m].proc.tokenizer
        yid = tk("Yes", add_special_tokens=False).input_ids[0]; nid = tk("No", add_special_tokens=False).input_ids[0]
        def margin(img, q):
            _, d = adapters[m].score(img, q, "Yes"); p = d[0]
            return math.log(max(p[yid], 1e-12) / max(p[nid], 1e-12))
        return margin
    margins = {m: mk(m) for m in adapters}

    cases = []
    for name, i in REL_IDX.items():
        e = ex[i]
        cases.append({"name": name, "finding": e["finding"], "image": e["image"], "question": e["question"],
                      "roi": e["boxes"][0], "others": [b for b in e.get("avoid_boxes", []) if b != e["boxes"][0]], "candidate": False})
    # substitution candidates, pre-screened on MedGemma's full-image margin
    hs = [json.loads(l) for l in open(f"{VINDR_VQA}/hardset_v1.jsonl")]
    pool = [r for r in hs if r["subset"] == "hard_pos_correlated" and r["roi_area"] < 0.02 and r["n_context"] >= 1
            and os.path.exists(f"{IMAGES}/{r['image_id']}.png")][:10]
    ab = all_boxes_by_image({r["image_id"] for r in pool})
    screened = []
    for r in pool:
        img = Image.open(f"{IMAGES}/{r['image_id']}.png").convert("RGB"); W, H = img.size
        mgf = margins["medgemma"](img, r["question"])
        others = [[b[0] * W, b[1] * H, b[2] * W, b[3] * H] for f, b in ab.get(r["image_id"], {}).items() if f != r["finding"]]
        if mgf > 1 and others:
            screened.append({"name": f"D correlated context ({r['finding']}; co-findings: {', '.join(r['context_findings'][:2])})", "finding": r["finding"],
                             "image": img, "question": r["question"], "roi": [r["roi_box"][0] * W, r["roi_box"][1] * H, r["roi_box"][2] * W, r["roi_box"][3] * H],
                             "others": others, "candidate": True, "mgf": mgf})
        if len(screened) >= 3: break
    print(f"[e9b] screened substitution candidates: {len(screened)} (of {len(pool)})", flush=True)
    cases += screened

    results = []
    for c in cases:
        img = c["image"]; W, H = img.size
        rec = {"name": c["name"], "finding": c["finding"], "question": c["question"], "candidate": c["candidate"], "mg_full": {}, "pix": {}, "geom": None}
        for m in adapters:
            mg_full = margins[m](img, c["question"]); rec["mg_full"][m] = mg_full
            M, geom = slide_map(margins[m], img, c["question"], mg_full, c["roi"])
            rec["pix"][m] = render(M, geom, W, H); rec["geom"] = [geom[0], geom[1], len(geom[2]), len(geom[3])]
        pm = rec["pix"]["medgemma"]
        rec["roi_sens"] = box_stat(pm, c["roi"], W, H)
        rec["other_sens"] = max((box_stat(pm, b, W, H) for b in c["others"]), default=float("nan"))
        rec["elsewhere_max"] = float(np.percentile(pm, 99))
        results.append(rec)
        print(f"[e9b] {c['name']}: windows {rec['geom'][2]}x{rec['geom'][3]} | MedGemma sens on ROI {rec['roi_sens']:+.2f}, on co-finding {rec['other_sens']:+.2f}, 99th pct elsewhere {rec['elsewhere_max']:+.2f}", flush=True)

    subs = [r for r in results if r["candidate"]]
    pick = max(subs, key=lambda r: r["other_sens"] - r["roi_sens"]) if subs else None
    show = [r for r in results if not r["candidate"]] + ([pick] if pick else [])
    json.dump({"shown": [r["name"] for r in show], "stats": [{k: v for k, v in r.items() if k != "pix"} for r in results]},
              open(f"{HERE}/e9b_occlusion_maps_v2.json", "w"), indent=1, default=str)

    fig, axes = plt.subplots(len(show), 3, figsize=(11.5, 3.5 * len(show)))
    for r_i, rec in enumerate(show):
        c = next(cc for cc in cases if cc["name"] == rec["name"]); img = c["image"]; W, H = img.size
        base = img.copy(); dr = ImageDraw.Draw(base)
        for b in c["others"]: dr.rectangle(b, outline=(255, 220, 0), width=4)
        dr.rectangle(c["roi"], outline=(255, 60, 60), width=5)
        ax = axes[r_i, 0]; ax.imshow(base); ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(rec["name"].split(" (")[0] + f"\n{rec['finding']}", fontsize=9)
        if r_i == 0: ax.set_title("image · red = queried evidence · yellow = other findings", fontsize=9)
        for col, m, nice in [(1, "medgemma", "MedGemma-4b"), (2, "qwen3vl", "Qwen3-VL-8B")]:
            ax = axes[r_i, col]; ax.imshow(img, alpha=0.5)
            P = rec["pix"][m]; vmax = max(1.0, float(np.percentile(np.abs(P), 99.5)))
            im = ax.imshow(P, cmap="coolwarm", vmin=-vmax, vmax=vmax, alpha=0.6, interpolation="nearest")
            ax.add_patch(plt.Rectangle((c["roi"][0], c["roi"][1]), c["roi"][2] - c["roi"][0], c["roi"][3] - c["roi"][1], fill=False, edgecolor="red", lw=2))
            for b in c["others"]: ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, edgecolor="gold", lw=2))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{nice} · full margin {rec['mg_full'][m]:+.1f}", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label("margin drop, ROI-sized window", fontsize=7)
    plt.suptitle("Occlusion-sensitivity maps: an evidence-sized mean-fill window slid over the image; colour = drop in the yes/no margin when that window is masked.", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{HERE}/fig_occlusion_maps_v2.png", dpi=160); plt.savefig(f"{HERE}/fig_occlusion_maps_v2.pdf")
    print("saved fig_occlusion_maps_v2.*; shown:", [r["name"] for r in show], flush=True)


if __name__ == "__main__":
    main()
