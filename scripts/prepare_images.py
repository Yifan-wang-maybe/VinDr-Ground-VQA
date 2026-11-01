"""Regenerate the PNGs referenced by the VinDr-Ground-VQA labels from a credentialed VinDr-CXR copy.

For every image_id in the given label files, obtain the DICOM — either from --dicom_dir (your local
PhysioNet download, laid out as <dicom_dir>/<train|test>/<image_id>.dicom or flat) or by downloading
it from PhysioNet with the credentials in ~/.netrc — and write <out>/<image_id>.png using exactly the
paper's preprocessing (VOI-LUT, MONOCHROME1 inversion, 8-bit min-max scaling, longest side 1024 px).
Nothing is uploaded anywhere. Normalized boxes in the labels are defined on these PNGs.

Usage:
  python prepare_images.py --labels data/train.jsonl data/val.jsonl data/test.jsonl \
                           data/hardset_v1.jsonl data/hardset_unrelated_v1.jsonl --out data/images --jobs 8
  python prepare_images.py --verify data/images
"""
import os, sys, json, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dicom_to_png import download_dicom, dicom_to_png


def collect(label_files):
    ids = {}
    for f in label_files:
        for line in open(f):
            r = json.loads(line)
            ids.setdefault(r["image_id"], set()).add(r.get("split", "train"))
    return ids


def find_local(dicom_dir, iid):
    for split in ("train", "test", ""):
        p = os.path.join(dicom_dir, split, f"{iid}.dicom") if split else os.path.join(dicom_dir, f"{iid}.dicom")
        if os.path.exists(p): return p
    return None


def one(iid, splits, out_dir, dicom_dir, longest):
    out = os.path.join(out_dir, f"{iid}.png")
    if os.path.exists(out): return iid, "exists", None
    tmp = None
    try:
        p = find_local(dicom_dir, iid) if dicom_dir else None
        if p is None:
            # our 'val' split was carved from the official train split; try train first, then test
            order = ["test", "train"] if splits == {"test"} else ["train", "test"]
            last = None
            for s in order:
                try: p = download_dicom(iid, s); tmp = p; break
                except Exception as e: last = e
            if p is None: raise last
        H, W = dicom_to_png(p, out, longest=longest)
        return iid, "ok", (H, W)
    except Exception as e:
        return iid, f"fail: {str(e)[:80]}", None
    finally:
        if tmp and os.path.exists(tmp): os.remove(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--out", default="data/images")
    ap.add_argument("--dicom_dir", default=None, help="local VinDr-CXR DICOM directory (optional; else download via ~/.netrc)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--longest", type=int, default=1024)
    ap.add_argument("--verify", default=None, help="only verify an existing image directory against the labels")
    a = ap.parse_args()
    if a.verify:
        ids = collect(a.labels or [os.path.join("data", f) for f in ("train.jsonl", "val.jsonl", "test.jsonl")])
        have = {f[:-4] for f in os.listdir(a.verify) if f.endswith(".png")}
        print(f"referenced {len(ids)} images; present {len(ids & have)}; missing {len(set(ids) - have)}")
        return
    ids = collect(a.labels); os.makedirs(a.out, exist_ok=True)
    print(f"{len(ids)} images to prepare -> {a.out}", flush=True)
    dims = {}; fails = []
    with ThreadPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(one, iid, splits, a.out, a.dicom_dir, a.longest) for iid, splits in ids.items()]
        for k, fu in enumerate(as_completed(futs)):
            iid, status, hw = fu.result()
            if hw: dims[iid] = list(hw)
            if status.startswith("fail"): fails.append((iid, status))
            if (k + 1) % 200 == 0: print(f"  {k+1}/{len(futs)} done, {len(fails)} failed", flush=True)
    if dims:
        json.dump(dims, open(os.path.join(a.out, "_dims_regenerated.json"), "w"))
    print(f"done: {len(ids) - len(fails)} ok, {len(fails)} failed")
    for iid, s in fails[:20]: print("  ", iid, s)


if __name__ == "__main__":
    main()
