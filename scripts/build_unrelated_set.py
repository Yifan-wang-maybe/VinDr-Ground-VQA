"""Tier-1 Exp 3 Arm A (mining, CPU) — the middle group: ask about absent X on
images carrying an UNRELATED finding Y (low co-occurrence lift). Three-group
FPR comparison then reads: normal image / abnormal-UNRELATED / abnormal-CORRELATED,
stratified within X. Unrelatedness is later verified against the model's own
blind & masked P(Yes) at eval time (teammates' rule).
Output: hardset_unrelated_v1.jsonl
"""
import os, sys, json, random, hashlib
from collections import Counter
from itertools import combinations
sys.path.insert(0, os.environ.get("VGV_BUILD", os.path.join(os.environ.get("VGV_ROOT", "."), "scripts")))
from build_vqa import load_ann, q_templates

OUT = os.environ.get("VGV_DATA", os.path.join(os.environ.get("VGV_ROOT", "."), "data"))
IMG = f"{OUT}/images"
MAX_LIFT, CAP = 0.9, 60
rng = random.Random(0)

df = load_ann(); boxed = df.dropna(subset=["x_min"])
dims = json.load(open(f"{OUT}/dims.json"))
have = {i for i in dims if os.path.exists(f"{IMG}/{i}.png")}
split_of = dict(zip(df.image_id, df.split))
pm_path = f"{OUT}/paraphrase_map.json"
pm = json.load(open(pm_path)) if os.path.exists(pm_path) else {}

fs = {iid: set(g.class_name) for iid, g in boxed.groupby("image_id")}
n_abn = len(fs)
cnt = Counter(f for s in fs.values() for f in s)
pair = Counter()
for s in fs.values():
    for x, y in combinations(sorted(s), 2): pair[(x, y)] += 1

def lift(x, y):
    c = pair.get(tuple(sorted((x, y))), 0)
    return (c / cnt[y]) / (cnt[x] / n_abn) if cnt[y] and cnt[x] else 0.0

# match the X-distribution of the CORRELATED hard-neg set so within-X strata align
corr = [json.loads(l) for l in open(f"{OUT}/hardset_v1.jsonl")]
corr_X = Counter(r["finding"] for r in corr if r["subset"] == "hard_neg")

rows, qid = [], 0
for X in sorted(corr_X):
    ys = [(lift(X, Y), Y) for Y in cnt if Y != X and cnt[Y] >= 100]
    unrel = [(l, Y) for l, Y in sorted(ys) if l <= MAX_LIFT]
    for l, Y in unrel:
        pool = [i for i in have if Y in fs.get(i, set()) and X not in fs.get(i, set())]
        rng.shuffle(pool)
        t = q_templates(X)["presence"]
        for iid in pool[:CAP]:
            qa = f"unrel_{qid:06d}"; qid += 1
            vs = pm.get(t, [t])
            q = vs[int(hashlib.md5(qa.encode()).hexdigest(), 16) % len(vs)]
            rows.append({"qa_id": qa, "image_id": iid, "split": split_of.get(iid, "train"),
                         "finding": X, "question": q, "question_template": t, "answer": "No",
                         "answer_type": "presence", "roi_box": None, "subset": "hard_neg_unrelated",
                         "context_y": Y, "pair_lift": round(l, 2),
                         "context_findings": sorted(fs.get(iid, set()))})

outp = f"{OUT}/hardset_unrelated_v1.jsonl"
with open(outp, "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} unrelated hard-negs -> {outp}")
print("X coverage:", dict(Counter(r["finding"] for r in rows)))
print("lift range:", min(r["pair_lift"] for r in rows), "-", max(r["pair_lift"] for r in rows),
      "| pairs:", len({(r['finding'], r['context_y']) for r in rows}))
