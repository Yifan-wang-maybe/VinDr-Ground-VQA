# Regenerating the images (and, optionally, the whole dataset) from VinDr-CXR

The labels shipped in `data/` reference VinDr-CXR images by `image_id`. The pixels are **not**
included: VinDr-CXR is distributed by PhysioNet under a credentialed Data Use Agreement. You need
your own credentialed copy. Two routes:

## Route A (recommended): regenerate only the PNGs the labels need

1. Obtain credentialed access to VinDr-CXR 1.0.0 on PhysioNet and put your credentials in `~/.netrc`
   (`machine physionet.org login <user> password <pass>`), or download the DICOMs with your own tooling.
2. Run
   ```bash
   python scripts/prepare_images.py --labels data/train.jsonl data/val.jsonl data/test.jsonl \
          data/hardset_v1.jsonl data/hardset_unrelated_v1.jsonl --out data/images [--jobs 8]
   ```
   For every `image_id` it downloads the DICOM (or reads it from `--dicom_dir` if you already have
   them) and writes `data/images/<image_id>.png` with exactly our preprocessing: VOI-LUT applied,
   MONOCHROME1 inverted, min–max scaled to 8 bit, longest side 1,024 px. `data/dims.json` holds the
   original DICOM (H, W) of every image so you can map the normalized `roi_box` back to DICOM pixels.
3. Verify: `python scripts/prepare_images.py --verify data/images` checks the count (8,434 images for
   the main splits; 12,529 normals are referenced only by the hard sets) and the PNG sizes.

## Route B: rebuild the dataset itself

`build_vqa.py` (consensus boxes by greedy IoU clustering at 0.3 → presence / localization / open
questions → balanced negatives), `finalize_release.py` (clean-normal negatives, per-split balance,
leakage-free image-level val split, seed 0) and `build_hardset.py` / `build_unrelated_set.py`
(co-occurrence-controlled sets, lift thresholds 1.5 / 0.9; `MIN_LIFT=1.2` env for the weak band)
reproduce every deterministic field from the VinDr-CXR annotation CSVs. Point `VGV_VINDR_CSV` at the
directory holding `annotations_train.csv` / `annotations_test.csv`. The language-model fields
(`question` paraphrases, `answer_nl`, `roi_appropriate`, `roi_reason`) were produced by a separate API
pass (prompts in `fable5_qa.py` / `fable5_pass.py` of the original repository) and are shipped as-is;
`question_template` + `answer` + `roi_box` are the API-free core.

Only PNGs and the regenerated JSONL are written; nothing is uploaded anywhere.
