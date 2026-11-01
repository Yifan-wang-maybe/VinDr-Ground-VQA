"""P3 — build the hard-set (hard_v1): the competition test in its proper regime.
  D1 hard-positives : ask about X (present, boxed) on multi-finding images
                      (context can sustain the answer)  vs isolated-X controls.
  D2 hard-negatives : ask about X (ABSENT) on images carrying a correlated Y
                      (prior pushes Yes; truth is No). FPR vs clean normals =
                      prior-following. Pure CPU; images already on disk.
Output: vindr_ground_vqa/hardset_v1.jsonl  (labeled extension, eval-only)
"""
import os, sys, json, hashlib, random
from collections import Counter, defaultdict
from itertools import combinations
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_vqa import load_ann, consensus, q_templates

ROOT = os.environ.get("VGV_BUILD", os.path.join(os.environ.get("VGV_ROOT", "."), "scripts"))
OUT = f"{ROOT}/vindr_ground_vqa"; IMG = f"{OUT}/images"
MIN_LIFT, MIN_BOTH = float(os.environ.get("MIN_LIFT", 1.5)), 30   # env override = sensitivity sets only; default = release set
CAP_PER_PAIR, CAP_MULTI_PER_FINDING = 200, 150
rng = random.Random(0)


def main():
    df = load_ann(); boxed = df.dropna(subset=["x_min"])
    dims = json.load(open(f"{OUT}/dims.json"))
    have = {i for i in dims if os.path.exists(f"{IMG}/{i}.png")}
    split_of = dict(zip(df.image_id, df.split))
    pm_path = f"{OUT}/paraphrase_map.json"
    pm = json.load(open(pm_path)) if os.path.exists(pm_path) else {}

    fs = {iid: set(g.class_name) for iid, g in boxed.groupby("image_id")}   # full 18k stats
    n_abn = len(fs)
    cnt = Counter(f for s in fs.values() for f in s)
    pair = Counter()
    for s in fs.values():
        for a, b in combinations(sorted(s), 2): pair[(a, b)] += 1

    def lift(x, y):
        c = pair.get(tuple(sorted((x, y))), 0)
        return (c / cnt[y]) / (cnt[x] / n_abn) if cnt[y] and cnt[x] else 0.0

    def q_of(template, qa_id):
        vs = pm.get(template, [template])
        return vs[int(hashlib.md5(qa_id.encode()).hexdigest(), 16) % len(vs)]

    def box_of(iid, finding):
        g = boxed[(boxed.image_id == iid) & (boxed.class_name == finding)]
        bxs = consensus(g[["x_min", "y_min", "x_max", "y_max"]].values.tolist())
        if not bxs: return None, None
        bx = max(bxs, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))          # primary instance
        H, W = dims[iid]
        nb = [round(min(1, max(0, v)), 4) for v in (bx[0]/W, bx[1]/H, bx[2]/W, bx[3]/H)]
        return nb, int(g.rad_id.nunique())

    rows = []; qid = 0
    def emit(iid, finding, ans, subset, extra):
        nonlocal qid
        t = q_templates(finding)["presence"]; qa = f"hard_{qid:06d}"; qid += 1
        nb, rad = box_of(iid, finding) if ans == "Yes" else (None, None)
        rows.append({"qa_id": qa, "image_id": iid, "source": "vindr-cxr",
                     "split": split_of.get(iid, "train"), "finding": finding,
                     "question": q_of(t, qa), "question_template": t, "answer": ans,
                     "answer_type": "presence", "roi_box": nb,
                     "roi_area": (round((nb[2]-nb[0])*(nb[3]-nb[1]), 5) if nb else None),
                     "rad_agreement": rad, "subset": subset,
                     "context_findings": sorted(fs.get(iid, set()) - {finding}),
                     "n_context": len(fs.get(iid, set()) - {finding}), **extra})

    # ---- D1: hard-positives (correlated) vs isolated controls ----
    for X in sorted(cnt):
        iso = [i for i in have if fs.get(i) == {X}]
        multi = [i for i in have if X in fs.get(i, set()) and len(fs[i]) > 1]
        if len(iso) < 20 or len(multi) < 20:      # need both cells for the contrast
            continue
        rng.shuffle(multi)
        max_lift = lambda i: max((lift(X, y) for y in fs[i] - {X}), default=0.0)
        for iid in iso:
            emit(iid, X, "Yes", "hard_pos_isolated", {"pair_lift": None})
        for iid in multi[:CAP_MULTI_PER_FINDING]:
            emit(iid, X, "Yes", "hard_pos_correlated", {"pair_lift": round(max_lift(iid), 2)})

    # ---- D2: hard-negatives (absent X on correlated-Y images) ----
    pairs = [(lift(a, b), a, b) for (a, b), c in pair.items() if c >= MIN_BOTH]
    pairs = [(l, x, y) for l, x, y in pairs if l >= MIN_LIFT]
    for l, x, y in sorted(pairs, reverse=True):
        for X, Y in ((x, y), (y, x)):                     # both directions
            lXY = lift(X, Y)
            if lXY < MIN_LIFT: continue
            pool = [i for i in have if Y in fs.get(i, set()) and X not in fs.get(i, set())]
            rng.shuffle(pool)
            for iid in pool[:CAP_PER_PAIR]:
                emit(iid, X, "No", "hard_neg", {"pair_lift": round(lXY, 2), "context_y": Y})

    outp = os.environ.get("HARDSET_OUT", f"{OUT}/hardset_v1.jsonl")
    with open(outp, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    sub = Counter(r["subset"] for r in rows)
    print(f"[hardset] wrote {len(rows)} QA -> {outp}")
    print(f"  subsets: {dict(sub)}")
    print(f"  D1 findings covered: {len(set(r['finding'] for r in rows if r['subset'].startswith('hard_pos')))}")
    print(f"  D2 pairs: {len(set((r['finding'], r.get('context_y')) for r in rows if r['subset']=='hard_neg'))}, "
          f"lift range {min((r['pair_lift'] for r in rows if r['subset']=='hard_neg'), default=0):.1f}-"
          f"{max((r['pair_lift'] for r in rows if r['subset']=='hard_neg'), default=0):.1f}")
    yn = Counter(r["answer"] for r in rows)
    print(f"  answers: {dict(yn)} | splits: {dict(Counter(r['split'] for r in rows))}")


if __name__ == "__main__":
    main()
