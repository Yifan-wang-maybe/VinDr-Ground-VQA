"""Build grounded + balanced VinDr VQA (templated core; Fable-5 paraphrase/verify is a
separate pass). Pipeline: consensus boxes (IoU-cluster 3 radiologists) -> download+PNG
-> QA (presence/localization/open) -> balance yes/no from normal images -> jsonl.

Usage: python build_vqa.py --n_abnormal 200 --n_normal 100 --tag pilot [--longest 1024]
"""
import os, io, json, argparse, random, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dicom_to_png import download_dicom, dicom_to_png

ROOT = os.environ.get("VGV_BUILD", os.path.join(os.environ.get("VGV_ROOT", "."), "scripts"))
ANN = f"{ROOT}/vindr_cxr"
rng = random.Random(0)


def load_ann():
    tr = pd.read_csv(f"{ANN}/annotations_train.csv"); tr["split"] = "train"
    te = pd.read_csv(f"{ANN}/annotations_test.csv"); te["split"] = "test"
    return pd.concat([tr, te], ignore_index=True)


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def consensus(boxes, thr=0.3):
    """Greedy IoU-cluster rad boxes -> one mean box per instance."""
    clusters = []
    for bx in boxes:
        for cl in clusters:
            if iou(bx, cl[0]) > thr:
                cl.append(bx); break
        else:
            clusters.append([bx])
    return [np.mean(cl, axis=0).tolist() for cl in clusters]


def q_templates(finding):
    f = finding.lower()
    return {
        "presence": f"Is there {f} in this chest X-ray?",
        "localization": f"Where is the {f} located in this chest X-ray?",
        "open": "What abnormality is shown in the highlighted region of this chest X-ray?",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_abnormal", type=int, default=200)
    ap.add_argument("--n_normal", type=int, default=100)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--longest", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT = f"{ROOT}/vindr_ground_vqa"; IMG = f"{OUT}/images"
    os.makedirs(IMG, exist_ok=True)

    df = load_ann()
    boxed = df.dropna(subset=["x_min"]).copy()
    normals = df[df.class_name == "No finding"].drop_duplicates("image_id")

    abn_ids = list(boxed.image_id.unique()); rng.shuffle(abn_ids); abn_ids = abn_ids[:a.n_abnormal]
    norm_ids = list(normals.image_id.unique()); rng.shuffle(norm_ids); norm_ids = norm_ids[:a.n_normal]
    split_of = dict(zip(df.image_id, df.split))
    print(f"[build] {len(abn_ids)} abnormal + {len(norm_ids)} normal images", flush=True)

    # ---- download + convert (threaded, RESUMABLE); record original dims ----
    dims_path = f"{OUT}/dims.json"
    dims = json.load(open(dims_path)) if os.path.exists(dims_path) else {}
    def fetch(iid):
        png = f"{IMG}/{iid}.png"
        if iid in dims and os.path.exists(png):
            return iid, tuple(dims[iid])                      # already done -> skip
        try:
            p = download_dicom(iid, split_of.get(iid, "train"))
            H, W = dicom_to_png(p, png, longest=a.longest)
            os.remove(p); return iid, (H, W)
        except Exception:
            return iid, None
    todo = abn_ids + norm_ids
    n_skip = sum(1 for i in todo if i in dims and os.path.exists(f"{IMG}/{i}.png"))
    print(f"[build] {n_skip} already downloaded (resume), {len(todo)-n_skip} to fetch", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (iid, hw) in enumerate(ex.map(fetch, todo)):
            if hw: dims[iid] = list(hw)
            if (i+1) % 100 == 0:
                print(f"  fetched {i+1}/{len(todo)}", flush=True)
                json.dump(dims, open(dims_path, "w"))         # periodic checkpoint for resume
    json.dump(dims, open(dims_path, "w"))
    print(f"[build] converted {len(dims)} images total", flush=True)

    # ---- positives: consensus boxes -> QA ----
    rows = []
    qid = 0
    pos_findings = []                                   # for balancing
    for iid in abn_ids:
        if iid not in dims: continue
        H, W = dims[iid]
        sub = boxed[boxed.image_id == iid]
        for finding, g in sub.groupby("class_name"):
            insts = consensus(g[["x_min","y_min","x_max","y_max"]].values.tolist())
            for bx in insts:
                nb = [bx[0]/W, bx[1]/H, bx[2]/W, bx[3]/H]        # normalized xyxy
                nb = [min(1,max(0,v)) for v in nb]
                t = q_templates(finding)
                base = dict(image_id=iid, source="vindr-cxr", finding=finding,
                            roi_box=[round(v,4) for v in nb], split=split_of.get(iid,"train"),
                            rad_agreement=int(g.rad_id.nunique()))
                for at, q, ans in [("presence", t["presence"], "Yes"),
                                   ("localization", t["localization"], "the highlighted region"),
                                   ("open", t["open"], finding)]:
                    rows.append({"qa_id": f"{a.tag}_{qid}", "question_template": q,
                                 "question": q, "answer": ans, "answer_type": at, **base})
                    qid += 1
                pos_findings.append(finding)

    # ---- balanced negatives: presence 'No' on normal images, matched per finding ----
    from collections import Counter
    need = Counter(pos_findings)                          # match Yes count per finding
    navail = [i for i in norm_ids if i in dims]
    ni = 0
    for finding, cnt in need.items():
        for _ in range(cnt):
            if not navail: break
            iid = navail[ni % len(navail)]; ni += 1
            rows.append({"qa_id": f"{a.tag}_{qid}", "question_template": q_templates(finding)["presence"],
                         "question": q_templates(finding)["presence"], "answer": "No",
                         "answer_type": "presence", "image_id": iid, "source": "vindr-cxr",
                         "finding": finding, "roi_box": None, "split": split_of.get(iid,"train"),
                         "rad_agreement": None}); qid += 1

    os.makedirs(OUT, exist_ok=True)
    outp = f"{OUT}/vqa_{a.tag}.jsonl"
    with open(outp, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

    # ---- stats ----
    at = Counter(r["answer_type"] for r in rows)
    yn = Counter(r["answer"] for r in rows if r["answer_type"] == "presence")
    areas = [ (r["roi_box"][2]-r["roi_box"][0])*(r["roi_box"][3]-r["roi_box"][1]) for r in rows if r["roi_box"]]
    print(f"\n[build] wrote {len(rows)} QA -> {outp}")
    print(f"  answer_type: {dict(at)}")
    print(f"  presence balance: {dict(yn)}")
    print(f"  #findings covered: {len(set(r['finding'] for r in rows))}")
    if areas:
        import numpy as np
        print(f"  ROI area frac: median {np.median(areas):.3f} p10 {np.percentile(areas,10):.3f} p90 {np.percentile(areas,90):.3f}")


if __name__ == "__main__":
    main()
