"""Download a VinDr-CXR DICOM (via ~/.netrc PhysioNet creds) and convert to an 8-bit
PNG with proper VOI-LUT windowing + MONOCHROME1 handling. Returns the original
(H, W) so bounding boxes (given in original DICOM pixel coords) can be normalized to
[0,1] and stored resolution-independently.
"""
import os, io, subprocess, tempfile
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image

BASE = "https://physionet.org/files/vindr-cxr/1.0.0"


def download_dicom(image_id, split):
    """wget with --auth-no-challenge (reads ~/.netrc). Returns local temp path."""
    url = f"{BASE}/{split}/{image_id}.dicom"
    fd, path = tempfile.mkstemp(suffix=".dicom", dir="/tmp")
    os.close(fd)
    r = subprocess.run(["wget", "--auth-no-challenge", "--timeout=120", "--tries=3",
                        "-q", "-O", path, url], capture_output=True)
    if r.returncode != 0 or os.path.getsize(path) < 1000:
        os.path.exists(path) and os.remove(path)
        raise RuntimeError(f"download failed {image_id}: {r.stderr.decode()[:120]}")
    return path


def dicom_to_png(dcm_path, out_png, longest=1024):
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
    H, W = arr.shape[:2]
    img = Image.fromarray(arr.astype(np.uint8)).convert("L")
    if longest and max(H, W) > longest:
        s = longest / max(H, W)
        img = img.resize((round(W * s), round(H * s)))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    img.save(out_png)
    return H, W                      # ORIGINAL dims (for box normalization)


if __name__ == "__main__":
    import sys, pandas as pd
    iid = sys.argv[1] if len(sys.argv) > 1 else \
        pd.read_csv(os.path.join(os.environ.get("VGV_VINDR_CSV", "vindr_cxr"), "annotations_train.csv")).image_id.iloc[0]
    p = download_dicom(iid, "train")
    H, W = dicom_to_png(p, f"/tmp/{iid}.png")
    os.remove(p)
    from PIL import Image as I
    print(f"{iid}: orig {H}x{W} -> png {I.open(f'/tmp/{iid}.png').size} saved /tmp/{iid}.png")
