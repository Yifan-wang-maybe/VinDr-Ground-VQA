"""P1 — baseline QA accuracy gate on the VinDr-Ground-VQA release test split.
Presence: RELATIVE Yes-vs-No scoring (cancels answer-style priors), full-image +
blind conditions on the same stratified sample -> balanced acc, sens/spec,
per-finding + per-ROI-size detection curves, blind control (~50% expected).
Open: short generation + normalized contains-match on the finding name.

Usage: CUDA_VISIBLE_DEVICES=<g> python baseline_acc.py --model medgemma --n 3000 --n_open 500 --out results/p1_medgemma.json
"""
import os, json, random, argparse
import numpy as np
import torch
from PIL import Image
from collections import defaultdict
from mjsd_harness import TransformersVLM, MODELS, VINDR_VQA, IMAGES

REL = f"{VINDR_VQA}/release"
SUFFIX = " Please answer with yes or no."


def load_presence(n):
    rows = [json.loads(l) for l in open(f"{REL}/test.jsonl")]
    pres = [r for r in rows if r["answer_type"] == "presence"]
    by = defaultdict(list)
    for r in pres:
        by[(r["answer"], r["finding"])].append(r)
    rng = random.Random(0)
    total = len(pres)
    out = []
    for k, grp in sorted(by.items()):
        rng.shuffle(grp)
        take = max(1, round(n * len(grp) / total))
        out.extend(grp[:take])
    rng.shuffle(out)
    return out[:n]


def load_open(n):
    rows = [json.loads(l) for l in open(f"{REL}/test.jsonl")]
    op = [r for r in rows if r["answer_type"] == "open"]
    random.Random(0).shuffle(op)
    return op[:n]


def finding_match(finding, text):
    t = text.lower()
    return any(p.strip().lower() in t for p in finding.replace("/", ",").split(",") if len(p.strip()) > 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--n_open", type=int, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    adapter = TransformersVLM(MODELS[a.model])

    # ---------- presence: full + blind ----------
    ex = load_presence(a.n)
    print(f"[p1] presence n={len(ex)}", flush=True)
    rows = []
    for i, r in enumerate(ex):
        img = Image.open(f"{IMAGES}/{r['image_id']}.png").convert("RGB")
        q = r["question"].rstrip() + SUFFIX
        out = {"qa_id": r["qa_id"], "finding": r["finding"], "gold": r["answer"].lower(),
               "roi_area": r.get("roi_area"), "roi_appropriate": r.get("roi_appropriate")}
        for cond, im in [("full", img), ("blind", None)]:
            ly, _ = adapter.score(im, q, "Yes")
            ln, _ = adapter.score(im, q, "No")
            out[f"pred_{cond}"] = "yes" if ly > ln else "no"
            out[f"margin_{cond}"] = ly - ln
        rows.append(out)
        if (i + 1) % 100 == 0:
            acc = np.mean([x["pred_full"] == x["gold"] for x in rows])
            print(f"  {i+1}/{len(ex)} full-acc so far {acc:.3f}", flush=True)

    # ---------- open: generation ----------
    op = load_open(a.n_open)
    print(f"[p1] open n={len(op)}", flush=True)
    open_rows = []
    for i, r in enumerate(op):
        img = Image.open(f"{IMAGES}/{r['image_id']}.png").convert("RGB")
        msgs = adapter._messages(img, r["question"])
        enc = adapter._apply(msgs, True)
        inp = {k: (v.to(adapter.device) if torch.is_tensor(v) else v) for k, v in enc.items()}
        with torch.no_grad():
            gen = adapter.model.generate(**inp, max_new_tokens=24, do_sample=False)
        txt = adapter.proc.tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        open_rows.append({"finding": r["finding"], "text": txt.strip()[:120],
                          "hit": finding_match(r["finding"], txt)})
        if (i + 1) % 100 == 0:
            print(f"  open {i+1}/{len(op)} hit-rate {np.mean([x['hit'] for x in open_rows]):.3f}", flush=True)

    # ---------- summarize ----------
    g = np.array([r["gold"] == "yes" for r in rows]); pf = np.array([r["pred_full"] == "yes" for r in rows])
    pb = np.array([r["pred_blind"] == "yes" for r in rows])
    sens = float((pf & g).sum() / g.sum()); spec = float((~pf & ~g).sum() / (~g).sum())
    summ = {"model": a.model, "n_presence": len(rows),
            "full": {"balanced_acc": (sens + spec) / 2, "sens": sens, "spec": spec,
                     "acc": float((pf == g).mean())},
            "blind": {"acc": float((pb == g).mean()),
                      "yes_rate": float(pb.mean())}}
    per_f = {}
    for f in sorted(set(r["finding"] for r in rows)):
        rs = [r for r in rows if r["finding"] == f]
        gp = [r for r in rs if r["gold"] == "yes"]; gn = [r for r in rs if r["gold"] == "no"]
        per_f[f] = {"n_pos": len(gp), "sens": float(np.mean([r["pred_full"] == "yes" for r in gp])) if gp else None,
                    "n_neg": len(gn), "spec": float(np.mean([r["pred_full"] == "no" for r in gn])) if gn else None,
                    "area_med": float(np.median([r["roi_area"] for r in gp if r["roi_area"]])) if gp else None}
    summ["per_finding"] = per_f
    pos = [r for r in rows if r["gold"] == "yes" and r["roi_area"]]
    A = np.array([r["roi_area"] for r in pos]); q1, q2 = np.quantile(A, [1/3, 2/3])
    for lab, m in [("small", A <= q1), ("med", (A > q1) & (A <= q2)), ("large", A > q2)]:
        sel = [pos[j] for j in np.where(m)[0]]
        summ[f"sens_{lab}_roi"] = float(np.mean([r["pred_full"] == "yes" for r in sel]))
    summ["open_contains_acc"] = float(np.mean([x["hit"] for x in open_rows]))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summ, "rows": rows, "open_rows": open_rows}, open(a.out, "w"), indent=2)
    print("P1 SUMMARY:", json.dumps(summ, indent=2, default=str)[:2000], flush=True)


if __name__ == "__main__":
    main()
