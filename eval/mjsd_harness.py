"""M/JSD causal harness — the density-competition falsification (Phase 2).

Question: does the model USE the ROI evidence, or "see but not use" it?
For each (image, ROI bbox, question, gold answer):
  M    = logp_gold(full image) - logp_gold(ROI-collapsed image)      per-answer-token
         (how much the ROI's visual content changes the gold-answer probability)
  JSD  = Jensen-Shannon(first-answer-token dist | full ||  | ROI-collapsed)
         (whether the model's OUTPUT distribution notices the intervention)
  prior= logp_gold(no image, text-only), per-token  (language/question-prior strength)

Competition ("sees but doesn't use") predicts: on HIGH-prior samples, M ~ 0 but
JSD > 0.  Dilution predicts M ~ 0 AND JSD ~ 0.

ROI-collapse = replace the ROI bbox region with the image's mean colour
(image-space; model-agnostic v1).  v1 model = MedGemma-4b-it (transformers).

Usage (CUDA_VISIBLE_DEVICES=<gpu>):
  python mjsd_harness.py --model medgemma --dataset gemex --n 200 --out results/mjsd_medgemma_gemex.json
"""
import os, sys, json, argparse
import numpy as np
import torch
from PIL import Image

ROOT = os.environ.get("VGV_ROOT", ".")
GEMEX_ROOT = f"{ROOT}/data/gemex"
SLAKE_ROOT = f"{ROOT}/data/slake_full"
# Public layout: labels in $VGV_DATA (default <repo>/data); images in $VGV_IMAGES (default <data>/images),
# which the user regenerates from a credentialed VinDr-CXR copy with scripts/prepare_images.py.
VINDR_VQA = os.environ.get("VGV_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
IMAGES = os.environ.get("VGV_IMAGES", os.path.join(VINDR_VQA, "images"))
CANVAS = 336
MODELS = {   # Hugging Face ids by default; override any entry with VGV_MODEL_<KEY>=<local path or hub id>
    "medgemma": os.environ.get("VGV_MODEL_MEDGEMMA", "google/medgemma-4b-it"),
    "qwen3vl": os.environ.get("VGV_MODEL_QWEN3VL", "Qwen/Qwen3-VL-8B-Instruct"),
    "internvl35": os.environ.get("VGV_MODEL_INTERNVL35", "OpenGVLab/InternVL3_5-8B-HF"),
    "lingshu": os.environ.get("VGV_MODEL_LINGSHU", "lingshu-medical-mllm/Lingshu-7B"),
    "qwen3vl30b": os.environ.get("VGV_MODEL_QWEN3VL30B", "Qwen/Qwen3-VL-30B-A3B-Instruct"),
    "gemma3": os.environ.get("VGV_MODEL_GEMMA3", "google/gemma-3-4b-it"),
    "qwen25vl": os.environ.get("VGV_MODEL_QWEN25VL", "Qwen/Qwen2.5-VL-7B-Instruct"),
    "medgemma27b": os.environ.get("VGV_MODEL_MEDGEMMA27B", "google/medgemma-27b-it"),
    "lingshu32b": os.environ.get("VGV_MODEL_LINGSHU32B", "lingshu-medical-mllm/Lingshu-32B"),
}


# ---------------- data ----------------
def load_gemex(n):
    recs = [json.loads(l) for l in open(f"{GEMEX_ROOT}/thinkvg_subset.jsonl")]
    out = []
    for r in recs:
        p = os.path.join(GEMEX_ROOT, "images", r["image_path"])
        if not os.path.exists(p):
            continue
        out.append({"image": Image.open(p).convert("RGB").resize((CANVAS, CANVAS)),
                    "question": r["question"], "answer": r["answer"],
                    "boxes": r["boxes"]})
        if len(out) >= n:
            break
    return out


def load_slake(n):
    """SLAKE spatial subset (Position/Organ, en). Per-organ boxes in SOURCE-pixel
    [x,y,w,h] -> xyxy; images at source size (boxes already match). Smaller ROIs
    than GEMeX -> a sharper ROI-collapse test."""
    recs = json.load(open(f"{SLAKE_ROOT}/test.json"))
    out = []
    for r in recs:
        if r.get("q_lang") != "en" or r.get("content_type") not in ("Position", "Organ"):
            continue
        d = os.path.dirname(r["img_name"])
        det = os.path.join(SLAKE_ROOT, "imgs", d, "detection.json")
        src = os.path.join(SLAKE_ROOT, "imgs", d, "source.jpg")
        if not (os.path.exists(det) and os.path.exists(src)):
            continue
        boxes = []
        for item in json.load(open(det)):
            for _, (x, y, w, h) in item.items():
                boxes.append([x, y, x + w, y + h])
        if not boxes:
            continue
        out.append({"image": Image.open(src).convert("RGB"), "question": r["question"],
                    "answer": r["answer"], "boxes": boxes})
        if len(out) >= n:
            break
    return out


def load_vindr(n, clean=False, release=False):
    """Our v1 grounded set: presence-Yes examples with a SMALL, question-aligned ROI.
    Normalized box -> pixel coords on the loaded PNG. The clean M/JSD substrate:
    collapsing the finding ROI should drop 'Yes' confidence iff the model uses it.
    clean=True -> only Fable-5 roi_appropriate boxes (sharper substrate).
    release=True -> the frozen release TEST split (paper numbers).
    release='train' -> release TRAIN split (has per-radiologist rad_agreement)."""
    import random as _r
    # public release: data/{train,val,test}.jsonl (release=True -> test split, the paper's numbers)
    split = "train" if release == "train" else ("val" if release == "val" else "test")
    src = f"{VINDR_VQA}/{split}.jsonl"
    rows = [json.loads(l) for l in open(src)]
    all_ev = {}                               # image_id -> every annotated finding box (norm)
    for r in rows:
        if r.get("roi_box"):
            all_ev.setdefault(r["image_id"], []).append(r["roi_box"])
    pos = [r for r in rows if r["answer_type"] == "presence" and r["answer"] == "Yes" and r["roi_box"]]
    if clean:
        pos = [r for r in pos if r.get("roi_appropriate") is True]
    _r.Random(0).shuffle(pos)                 # diverse finding/prior sample (not file-order first-n)
    out = []
    for r in pos:
        p = f"{IMAGES}/{r['image_id']}.png"
        if not os.path.exists(p):
            continue
        img = Image.open(p).convert("RGB"); W, H = img.size
        b = r["roi_box"]
        out.append({"image": img, "question": r["question"], "answer": r["answer"],
                    "boxes": [[b[0]*W, b[1]*H, b[2]*W, b[3]*H]], "finding": r["finding"],
                    "roi_area": (b[2]-b[0])*(b[3]-b[1]), "rad_agreement": r.get("rad_agreement"),
                    "avoid_boxes": [[a[0]*W, a[1]*H, a[2]*W, a[3]*H] for a in all_ev.get(r["image_id"], [])]})
        if len(out) >= n:
            break
    return out


def random_matched_boxes(boxes, size, seed, avoid=None):
    """Size-matched random control boxes avoiding the true ROI (rejection sampling).
    The artifact check: collapsing a random non-evidence region should give M ~ 0.
    avoid: extra boxes to stay off (e.g. ALL annotated findings on the image, so the
    control never lands on other true evidence — multi-instance confound)."""
    import random as _r
    rng = _r.Random(seed)
    W, H = size
    keep_off = list(boxes) + list(avoid or [])
    out = []
    for x1, y1, x2, y2 in boxes:
        w, h = min(x2 - x1, W - 1), min(y2 - y1, H - 1)
        best = None
        for _ in range(120):
            rx, ry = rng.uniform(0, W - w), rng.uniform(0, H - h)
            cand = [rx, ry, rx + w, ry + h]
            if all(iou_xyxy(cand, b) < 0.05 for b in keep_off):
                best = cand; break
            best = best or cand
        out.append(best)
    return out


def iou_xyxy(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def collapse_roi(img, boxes):
    """Replace the ROI bbox region(s) with the image's mean colour (spatial-mean collapse)."""
    a = np.array(img).astype(np.float32)
    mean = a.reshape(-1, a.shape[-1]).mean(0)
    for x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((max(0, int(x1)), min(a.shape[1], int(x2))))
        y1, y2 = sorted((max(0, int(y1)), min(a.shape[0], int(y2))))
        a[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1), :] = mean
    return Image.fromarray(a.astype(np.uint8))


# ---------------- metrics ----------------
def jsd(p, q, eps=1e-12):
    p = np.asarray(p, np.float64) + eps; q = np.asarray(q, np.float64) + eps
    p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ---------------- generic transformers VLM adapter (MedGemma / Qwen3-VL) ----------------
class TransformersVLM:
    def __init__(self, path, device="cuda"):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
        self.device = device

    def _messages(self, image, question, answer=None):
        content = ([{"type": "image", "image": image}] if image is not None else []) \
            + [{"type": "text", "text": question}]
        msgs = [{"role": "user", "content": content}]
        if answer is not None:
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return msgs

    def _apply(self, msgs, gen_prompt):
        return self.proc.apply_chat_template(msgs, add_generation_prompt=gen_prompt,
                                             tokenize=True, return_dict=True, return_tensors="pt")

    @torch.no_grad()
    def score(self, image, question, answer):
        """Per-token logp of the gold answer + per-answer-position distributions (La,V).
        Uses the processor-built full conversation (answer as assistant turn) so every
        model's input_ids / masks / image tensors / position-ids stay consistent
        (naive answer-token concatenation breaks Qwen3-VL's 3D RoPE)."""
        Lp = self._apply(self._messages(image, question), True)["input_ids"].shape[1]
        enc = self._apply(self._messages(image, question, answer or "."), False)
        full_ids = enc["input_ids"]
        La = len(self.proc.tokenizer(answer or ".", add_special_tokens=False).input_ids)
        La = max(1, min(La, full_ids.shape[1] - Lp))          # answer content span [Lp, Lp+La)
        inp = {}
        for k, v in enc.items():
            if torch.is_tensor(v):
                inp[k] = v.to(self.device, self.model.dtype) if v.is_floating_point() else v.to(self.device)
            else:
                inp[k] = v
        logits = self.model(**inp).logits[0].float()
        logp = torch.log_softmax(logits, -1)
        tgt = full_ids[0, Lp:Lp + La]
        tok_lp = [logp[Lp - 1 + j, tgt[j]].item() for j in range(La)]
        dists = torch.softmax(logits[Lp - 1:Lp - 1 + La], -1).cpu().numpy()   # (La, V)
        return sum(tok_lp) / La, dists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medgemma", choices=list(MODELS))
    ap.add_argument("--dataset", default="gemex", choices=["gemex", "slake", "vindr"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clean", action="store_true", help="vindr: only roi_appropriate boxes")
    ap.add_argument("--release", nargs="?", const=True, default=False,
                    help="vindr: frozen release split; bare flag = test, '--release train' = train (has rad_agreement)")
    ap.add_argument("--with_random", action="store_true", help="also score a size-matched random-box collapse (paired control)")
    ap.add_argument("--avoid_evidence", action="store_true", help="random control avoids ALL annotated finding boxes (not just the target ROI)")
    a = ap.parse_args()

    _loader = {"gemex": load_gemex, "slake": load_slake, "vindr": load_vindr}[a.dataset]
    ex = _loader(a.n, a.clean, a.release) if a.dataset == "vindr" else _loader(a.n)
    print(f"[mjsd] {a.model} on {a.dataset}: {len(ex)} examples", flush=True)
    adapter = TransformersVLM(MODELS[a.model])
    # tier-1 Exp-1: two-token renormalized M-tilde (format-break control). Applicable
    # when the gold answer starts with Yes/No; read from the first-answer-token dist.
    _tk = adapter.proc.tokenizer
    _yid = _tk("Yes", add_special_tokens=False).input_ids[0]
    _nid = _tk("No", add_special_tokens=False).input_ids[0]
    def _renorm(d0):
        py, pn = max(float(d0[_yid]), 1e-12), max(float(d0[_nid]), 1e-12)
        return float(np.log(py / (py + pn)))
    def _margin(d0):
        # log-odds log(pYes/pNo): the NON-saturating two-token M-tilde (renorm log-prob
        # saturates at 0 for confident-Yes cases and under-reads preference dents)
        py, pn = max(float(d0[_yid]), 1e-12), max(float(d0[_nid]), 1e-12)
        return float(np.log(py / pn))

    rows = []
    for i, e in enumerate(ex):
        coll = collapse_roi(e["image"], e["boxes"])
        lp_full, fd_full = adapter.score(e["image"], e["question"], e["answer"])
        lp_coll, fd_coll = adapter.score(coll, e["question"], e["answer"])
        lp_blind, _ = adapter.score(None, e["question"], e["answer"])
        jsd_mean = float(np.mean([jsd(fd_full[j], fd_coll[j]) for j in range(fd_full.shape[0])]))
        row = {"M": lp_full - lp_coll, "JSD": jsd_mean,
               "prior": lp_blind, "lp_full": lp_full, "lp_coll": lp_coll,
               "finding": e.get("finding"), "roi_area": e.get("roi_area"),
               "rad_agreement": e.get("rad_agreement")}
        yn = str(e["answer"]).strip().lower().startswith(("yes", "no"))
        if yn:
            row["Mt"] = _renorm(fd_full[0]) - _renorm(fd_coll[0])
            row["mg_full"], row["mg_coll"] = _margin(fd_full[0]), _margin(fd_coll[0])
            row["Mm"] = row["mg_full"] - row["mg_coll"]          # margin (log-odds) M-tilde
        if a.with_random:
            avoid = e.get("avoid_boxes") if a.avoid_evidence else None
            rnd = collapse_roi(e["image"], random_matched_boxes(e["boxes"], e["image"].size, seed=i, avoid=avoid))
            lp_rnd, fd_rnd = adapter.score(rnd, e["question"], e["answer"])
            row["M_rand"] = lp_full - lp_rnd
            row["JSD_rand"] = float(np.mean([jsd(fd_full[j], fd_rnd[j]) for j in range(fd_full.shape[0])]))
            if yn:
                row["Mt_rand"] = _renorm(fd_full[0]) - _renorm(fd_rnd[0])
                row["mg_rand"] = _margin(fd_rnd[0])
                row["Mm_rand"] = row["mg_full"] - row["mg_rand"]
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(ex)}", flush=True)

    M = np.array([r["M"] for r in rows]); J = np.array([r["JSD"] for r in rows])
    P = np.array([r["prior"] for r in rows])
    hi = P >= np.median(P)                      # high-prior stratum (blind predicts gold well)
    summ = {
        "model": a.model, "dataset": a.dataset, "n": len(rows),
        "mean_M": float(M.mean()), "mean_JSD": float(J.mean()),
        **({"mean_M_rand": float(np.mean([r["M_rand"] for r in rows])),
            "median_M_rand": float(np.median([r["M_rand"] for r in rows])),
            "median_M": float(np.median(M))} if a.with_random else {}),
        "high_prior": {"mean_M": float(M[hi].mean()), "mean_JSD": float(J[hi].mean()), "n": int(hi.sum())},
        "low_prior":  {"mean_M": float(M[~hi].mean()), "mean_JSD": float(J[~hi].mean()), "n": int((~hi).sum())},
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=2)
    print("SUMMARY:", json.dumps(summ, indent=2), flush=True)
    print("\nPrediction check (competition = sees-but-doesnt-use): "
          "high-prior mean_M~0 while mean_JSD>0 ?", flush=True)


if __name__ == "__main__":
    main()
