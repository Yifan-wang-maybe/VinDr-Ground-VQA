"""Rebuttal E12 — text shortcut on GEMeX-ThinkVG (teammate's hypothesis; R4 W2 reframing).
Text-only (blind) first-token yes/no margins for the GEMeX-ThinkVG closed yes/no questions
(annotations from the public HF parquet; no images needed). Reports blind accuracy, blind
balanced accuracy, blind yes-rate, gold yes-rate, and a question-only majority baseline, on the
first 399 items (our evaluated subset, file order) and the first 1,000. Both prompt forms.
Usage: CUDA_VISIBLE_DEVICES=0 python e12_gemex_blind.py --model medgemma --out e12_medgemma.json
"""
import os, sys, json, argparse, math
import numpy as np
sys.path.insert(0, os.environ.get("VGV_EVAL", os.path.join(os.environ.get("VGV_ROOT", "."), "eval")))
from mjsd_harness import TransformersVLM, MODELS

SUFFIX = " Please answer with yes or no."
Q = "/tmp/gemex_ann/closed_yn_first2000.jsonl"


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def block(rows, key):
    pred = np.array([r[key] > 0 for r in rows]); gold = np.array([r["gold"] == "yes" for r in rows])
    acc = float((pred == gold).mean()); tpr = float(pred[gold].mean()) if gold.any() else float("nan"); tnr = float((~pred[~gold]).mean()) if (~gold).any() else float("nan")
    return {"n": len(rows), "acc": acc, "acc_wilson": wilson(int((pred == gold).sum()), len(rows)), "balanced_acc": (tpr + tnr) / 2,
            "blind_yes_rate": float(pred.mean()), "gold_yes_rate": float(gold.mean()), "majority_baseline": float(max(gold.mean(), 1 - gold.mean())),
            "median_margin": float(np.median([r[key] for r in rows]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(Q)]
    items = [it for it in items if it.get("question_type") == "closed_ended_questions"][:a.n]   # true closed items only
    adapter = TransformersVLM(MODELS[a.model])
    tk = adapter.proc.tokenizer
    yid = tk("Yes", add_special_tokens=False).input_ids[0]
    nid = tk("No", add_special_tokens=False).input_ids[0]

    def margin(q):
        _, d = adapter.score(None, q, "Yes"); p = d[0]
        return float(math.log(max(p[yid], 1e-12) / max(p[nid], 1e-12)))

    rows = []
    for i, it in enumerate(items):
        rows.append({"question": it["question"], "gold": it["gold"], "mg_blind_bare": margin(it["question"]), "mg_blind_suffix": margin(it["question"].rstrip() + SUFFIX)})
        if (i + 1) % 200 == 0: print(f"  {i+1}/{len(items)}", flush=True)
    summ = {"model": a.model, "source": "BoKelvin/GEMeX-ThinkVG train parquet, <answer> starting Yes/No, file order",
            "first399_bare": block(rows[:399], "mg_blind_bare"), "first399_suffix": block(rows[:399], "mg_blind_suffix"),
            f"first{len(rows)}_bare": block(rows, "mg_blind_bare"), f"first{len(rows)}_suffix": block(rows, "mg_blind_suffix")}
    json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)
    print("E12:", json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
