# VinDr-Ground-VQA

**A grounded, image-necessary chest-X-ray VQA benchmark for measuring whether a vision–language model
uses the radiologist-marked evidence — with the area-matched occlusion protocol from the paper.**

- **102,072 questions · 8,434 radiographs · 22 findings** (train 81,656 / val 9,652 / test 10,764)
- every positive carries a **tight radiologist consensus box** (median 1.3% of the image); every "Yes" is
  matched with an image-necessary "No" (blind accuracy of all tested models: 0.49–0.50)
- **co-occurrence-controlled sets** (isolated vs correlated positives; correlated vs unrelated vs normal
  negatives) built from association-rule lift over the full VinDr-CXR annotation release
- the **evaluation harness**: ROI-collapse with a size-matched, evidence-avoiding random control, the
  grounding fraction, the control ladder (placement, texture, lung-field, mirror, fills), and bootstrap CIs
- **reference results for nine models** (MedGemma-4b/27B, Lingshu-7B/32B, InternVL3.5-8B, Qwen3-VL-8B/30B,
  and the two base models Gemma-3-4b-it and Qwen2.5-VL-7B)

Paper: *VinDr-Ground-VQA: A Grounded …* (ML4H 2026). See `CITATION.cff`.

---

## ⚠️ What is and is not in this repository

VinDr-CXR images are distributed by PhysioNet under a credentialed Data Use Agreement, so **no pixels
are included here**. Everything we built on top of them is: questions, answers, consensus boxes,
splits, the co-occurrence sets and statistics, the code, and our results. To run anything that needs
images, obtain credentialed access to VinDr-CXR 1.0.0 and regenerate the PNGs with one command
(`scripts/REGENERATE.md`); the preprocessing is deterministic, so your PNGs match the boxes exactly.

```
VinDr-Ground-VQA/
├── data/        SCHEMA.md · cooccurrence_lift.csv · dims.json · README_DOWNLOAD.md — the label files
│                (train/val/test.jsonl, hard sets) are on Zenodo: bash scripts/fetch_data.sh <record_id>
├── scripts/     prepare_images.py (DICOM → PNG, exactly our preprocessing) · build_*.py (full regeneration)
├── eval/        mjsd_harness.py (core instrument) · accuracy_gate.py · hardset_eval.py · unrelated_eval.py ·
│                specificity_eval.py · resize_eval.py · grounding_fraction.py · occlusion_maps.py ·
│                controls/ (placement rules, fills, mirror, texture/location, lung-field, inpainting,
│                normal-border) · analysis/ (bootstrap CIs, threshold tables, summaries) · run_benchmark.sh
├── results/     leaderboard.csv · ci_tables.json · reference/ (per-model result files) · tables (.md)
└── docs/        protocol appendix · figures · GEMeX protocol note
```

## Quick start

```bash
pip install -r eval/requirements.txt
# 0. label files (Zenodo, DOI 10.5281/zenodo.XXXXXXX) -> data/   [see data/README_DOWNLOAD.md]
bash scripts/fetch_data.sh <zenodo_record_id>
# 1. images (needs PhysioNet credentials in ~/.netrc, or --dicom_dir with your local DICOMs)
python scripts/prepare_images.py --labels data/train.jsonl data/val.jsonl data/test.jsonl \
       data/hardset_v1.jsonl data/hardset_unrelated_v1.jsonl --out data/images --jobs 8
# 2. one model, all main tables (≈1–2 h on one 80 GB GPU for an 8B model)
bash eval/run_benchmark.sh medgemma 0
python eval/analysis/summarize_model.py --model medgemma --append results/leaderboard_mine.csv
```

Any model loadable with `transformers.AutoModelForImageTextToText` works: add a key to `MODELS` in
`eval/mjsd_harness.py` or set `VGV_MODEL_<KEY>` to a local path. The harness needs the first-token
distribution (`score()` returns per-token log-probabilities), so closed APIs are not supported.

## What the benchmark measures

For a positive question with evidence box R, three forwards give the yes/no log-odds margin on the
full image (`mg_full`), with R mean-filled (`mg_roi`), and with a **size-matched random region that
avoids every annotated finding** mean-filled (`mg_rand`):

- **M_roi = mg_full − mg_roi**, **M_rand = mg_full − mg_rand**, **excess = M_roi − M_rand** — the
  evidence-specific effect, reported by ROI-size tertile (small ≈ 0.3%, medium ≈ 1.7%, large ≈ 6.6% of image);
- **grounding fraction GF = excess / (mg_full − mg_blind)** over cases where the image moves the model
  (image-gain > 0.5 nats): 0 for a blind reader, 1 when all image-driven belief is removable by masking R;
- **context false-positive increment** = FPR on correlated hard negatives − FPR on unrelated-context
  negatives (the co-occurrence shortcut);
- plus balanced accuracy and blind accuracy on the matched yes/no set (ability, and the text-shortcut check).

A model is reported as (accuracy, excess/GF by tertile, context increment); models are compared on
grounding at matched accuracy. Protocol details (prompt forms, fills, placement rules, seeds, bootstrap):
`docs/protocol_appendix.md`.

## Reference results (`results/leaderboard.csv`; release test split, n = 1,000 positives per model)

| model | balanced acc. | excess small / medium / large | GF (image-dependent cases) | context-FPR increment |
|---|---|---|---|---|
| MedGemma-4b-it | 0.723 | +0.19 / +0.62 / +2.50 | 0.18 | +0.157 |
| Lingshu-7B | 0.861 | +0.12 / +0.25 / +0.88 | 0.20 | +0.157 |
| InternVL3.5-8B | 0.689 | +0.00 / +0.12 / +0.75 | 0.22 | +0.010 |
| Qwen3-VL-8B-Instruct | 0.654 | +0.00 / +0.12 / +1.75 | 0.13 | +0.041 |
| MedGemma-27B-it | — | +0.12 / +0.31 / +1.50 | 0.51 (29% image-dependent) | +0.164 |
| Lingshu-32B | — | +0.12 / +0.12 / +1.00 | 0.29 (42%) | +0.149 |
| Qwen3-VL-30B-A3B | — | +0.00 / +0.38 / +1.38 | 0.14 | +0.091 |
| Gemma-3-4b-it (base) | 0.513 | −0.12 / +0.00 / +0.62 | — | +0.120 (near-ceiling FPR) |
| Qwen2.5-VL-7B (base) | 0.533 | −0.00 / +0.12 / +0.38 | — | +0.068 |

Medians; margins are quantized in 0.125-nat steps (so .x25/.x75 values are exact; tables round half-to-even).
95% paired-bootstrap CIs for every cell are in `results/ci_tables.json`; the control ladder, fill
sensitivity, threshold sensitivity and scale tables are in `results/*.md`. Headline findings: evidence-specific
reliance is small and grows with region size; when the image moves the answer, only 13–22% of that
movement comes from the annotated evidence in 4–8B models (0–8% on small findings); correlated context
substitutes for direct evidence (3–5× collapse with detection unchanged); the medically tuned models carry
the largest co-occurrence false-positive increment at every scale tested.

## Reproducing the paper's controls

Each script in `eval/controls/` reuses the cached release run and adds one or two forwards per case:
`placement_rules.py` (body-restricted, intensity-matched), `fills.py` (blur, black), `inpaint_fill.py`,
`mirror_and_ksens.py`, `location_texture_ring.py`, `lungfield.py` (torchxrayvision lung masks),
`normal_border.py`. `eval/analysis/ci_and_reslices.py` produces every CI in the paper;
`threshold_table.py` the lift-cutoff sensitivity for §5.4–5.5.

## Language-model fields

`question` paraphrases, `answer_nl` and `roi_appropriate`/`roi_reason` were produced by a language
model — Anthropic `claude-fable-5`, August 2026 (`enhancements` marks each row). The deterministic core — `question_template`, `answer`,
`roi_box`, splits, hard sets — needs no API and is what every number above uses, except the §5.3
stratification by `roi_appropriate`. Treat `roi_appropriate` as an auxiliary flag (83.6% positive);
see `DATASHEET.md` for its provenance.

## Citation

See `CITATION.cff`. Please also cite VinDr-CXR (Nguyen et al., *Scientific Data*, 2022) and PhysioNet.

## License

Labels, derived annotations, code and results: see `LICENSE`. The images remain under the PhysioNet
Credentialed Health Data License and are not part of this release.
