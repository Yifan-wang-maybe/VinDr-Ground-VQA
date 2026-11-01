"""Finalize VinDr-Ground-VQA for release.
Fixes vs vqa_v1: (1) negatives drawn ONLY from clean normals (all radiologists agree
'No finding') -> removes the 571 Yes/No contradictions; (2) balanced per (split,finding)
-> fixes test-split skew; (3) leakage-free val carved from official train (image-level).
Enhancements (paraphrase / appropriateness / NL-location) re-applied from cache -> no API.
Outputs release/{train,val,test}.jsonl (+ prints a full stats report). CPU only.
"""
import os, sys, json, hashlib, random
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from build_vqa import load_ann, consensus, q_templates

ROOT = os.environ.get("VGV_BUILD", os.path.join(os.environ.get("VGV_ROOT", "."), "scripts"))
OUT = f"{ROOT}/vindr_ground_vqa"; IMG = f"{OUT}/images"
REL = f"{OUT}/release"
VAL_FRAC = 0.10
rng = random.Random(0)


def main():
    os.makedirs(REL, exist_ok=True)
    df = load_ann()
    dims = json.load(open(f"{OUT}/dims.json"))
    have = {i for i in dims if os.path.exists(f"{IMG}/{i}.png")}
    boxed_df = df.dropna(subset=["x_min"])
    all_boxed = set(boxed_df.image_id.unique())
    all_normal = set(df[df.class_name == "No finding"].image_id.unique())
    clean_normal = (all_normal - all_boxed) & have          # ALL rads agree normal
    abn = all_boxed & have
    split_of = dict(zip(df.image_id, df.split))

    # ---- image-level split: official test; carve VAL_FRAC of official-train images -> val ----
    def base_split(i): return split_of.get(i, "train")
    train_imgs = sorted(i for i in (abn | clean_normal) if base_split(i) == "train")
    rng.shuffle(train_imgs)
    n_val = int(len(train_imgs) * VAL_FRAC)
    val_set = set(train_imgs[:n_val])
    def rel_split(i):
        if base_split(i) == "test": return "test"
        return "val" if i in val_set else "train"

    # ---- caches for enhancement re-apply (no API) ----
    pm = json.load(open(f"{OUT}/paraphrase_map.json"))
    verd = json.load(open(f"{OUT}/box_verdicts.json"))

    rows = []; qid = 0
    # positives: every downloaded abnormal image's consensus boxes
    pos_by_split_finding = defaultdict(int)
    for iid in sorted(abn):
        H, W = dims[iid]; sp = rel_split(iid)
        g = boxed_df[boxed_df.image_id == iid]
        for finding, gg in g.groupby("class_name"):
            for bx in consensus(gg[["x_min", "y_min", "x_max", "y_max"]].values.tolist()):
                nb = [round(min(1, max(0, v)), 4) for v in (bx[0]/W, bx[1]/H, bx[2]/W, bx[3]/H)]
                t = q_templates(finding)
                for at, ans in [("presence", "Yes"), ("localization", "the highlighted region"), ("open", finding)]:
                    rows.append(_row(qid, iid, sp, finding, t[at],
                                     ans, at, nb, int(gg.rad_id.nunique()), pm, verd)); qid += 1
                pos_by_split_finding[(sp, finding)] += 1

    # negatives: for each (split,finding) draw n_pos No's from clean normals in that split (round-robin)
    cn_by_split = defaultdict(list)
    for i in sorted(clean_normal):
        cn_by_split[rel_split(i)].append(i)
    for lst in cn_by_split.values(): rng.shuffle(lst)
    ptr = defaultdict(int)
    for (sp, finding), n_pos in pos_by_split_finding.items():
        pool = cn_by_split.get(sp, [])
        if not pool: continue
        t = q_templates(finding)
        for _ in range(n_pos):
            iid = pool[ptr[sp] % len(pool)]; ptr[sp] += 1
            rows.append(_row(qid, iid, sp, finding, t["presence"], "No", "presence",
                             None, None, pm, verd)); qid += 1

    # ---- write splits ----
    by_split = defaultdict(list)
    for r in rows: by_split[r["split"]].append(r)
    for sp in ("train", "val", "test"):
        with open(f"{REL}/{sp}.jsonl", "w") as f:
            for r in by_split[sp]: f.write(json.dumps(r) + "\n")

    _report(rows, by_split, abn, clean_normal, val_set)
    json.dump({"n_qa": len(rows), "splits": {s: len(v) for s, v in by_split.items()}},
              open(f"{REL}/_manifest.json", "w"), indent=2)


def _row(qid, iid, sp, finding, template, ans, at, nb, rad, pm, verd):
    qa_id = f"vindrground_{qid:06d}"
    variants = pm.get(template, [template])
    idx = int(hashlib.md5(qa_id.encode()).hexdigest(), 16) % len(variants)
    r = {"qa_id": qa_id, "image_id": iid, "source": "vindr-cxr", "split": sp,
         "finding": finding, "question": variants[idx], "question_template": template,
         "question_variants": [v for j, v in enumerate(variants) if j != idx],
         "answer": ans, "answer_type": at, "answer_nl": None,
         "roi_box": nb, "roi_area": (round((nb[2]-nb[0])*(nb[3]-nb[1]), 5) if nb else None),
         "roi_appropriate": None, "roi_reason": None, "rad_agreement": rad,
         "enhancements": ["fable5_paraphrase"] if len(variants) > 1 else []}
    if nb:
        v = verd.get(f"{iid}|{nb}", {})
        r["roi_appropriate"] = v.get("appropriate"); r["roi_reason"] = v.get("reason")
        r["enhancements"] = sorted(set(r["enhancements"] + ["appropriateness_checked"]))
        if at == "localization" and v.get("location"):
            r["answer_nl"] = v["location"]
            r["enhancements"] = sorted(set(r["enhancements"] + ["nl_localization"]))
    return r


def _report(rows, by_split, abn, clean_normal, val_set):
    print(f"\n===== VinDr-Ground-VQA release =====")
    print(f"total QA: {len(rows)} | images: abnormal={len(abn)} clean-normal={len(clean_normal)} (val imgs={len(val_set)})")
    # contradiction check
    pos = defaultdict(set); neg = defaultdict(set)
    for r in rows:
        if r["answer"] == "Yes" and r["roi_box"]: pos[r["image_id"]].add(r["finding"])
        if r["answer"] == "No": neg[r["image_id"]].add(r["finding"])
    contra = sum(1 for i in set(pos) & set(neg) if pos[i] & neg[i])
    print(f"label contradictions (same img Yes&No same finding): {contra}  (was 571)")
    for sp in ("train", "val", "test"):
        rs = by_split[sp]
        yn = Counter(r["answer"] for r in rs if r["answer_type"] == "presence")
        at = Counter(r["answer_type"] for r in rs)
        imgs = len(set(r["image_id"] for r in rs))
        print(f"  {sp:5s}: {len(rs):6d} QA | {imgs:5d} imgs | presence Yes={yn['Yes']} No={yn['No']} | types {dict(at)}")
    areas = [r["roi_area"] for r in rows if r["roi_box"]]
    appr = [r["roi_appropriate"] for r in rows if r["roi_box"]]
    print(f"  ROI area median {np.median(areas):.4f} | boxes appropriate {sum(1 for a in appr if a is True)}/{len(appr)}")
    print(f"  findings: {len(set(r['finding'] for r in rows))}")


if __name__ == "__main__":
    main()
