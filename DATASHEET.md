# Datasheet — VinDr-Ground-VQA v1

## Motivation
Built to measure whether medical vision–language models *use* the radiologist-marked evidence when
answering, rather than text priors or co-occurring findings. Existing grounded medical VQA sets either
have organ-scale boxes (GEMeX-ThinkVG, median 23% of the image) or boxes not aligned to the question;
closed questions there are answerable from text (74% "No"; blind models reach the majority baseline).
VinDr-Ground-VQA pairs lesion-level consensus boxes with questions whose answer requires the image.

## Composition
- **Source images:** VinDr-CXR 1.0.0 (PhysioNet, credentialed): 8,434 radiographs — 99.9% of the
  box-annotated abnormal images plus a balanced subset of images unanimously labelled normal.
- **Questions:** 102,072 (train 81,656 / val 9,652 / test 10,764); per consensus box one presence
  (Yes/No), one localization and one open question; presence questions exactly Yes/No balanced per split.
- **22 findings:** Aortic enlargement, Atelectasis, Calcification, Cardiomegaly, Clavicle fracture,
  Consolidation, Emphysema, Enlarged PA, ILD, Infiltration, Lung Opacity, Lung cavity, Lung cyst,
  Mediastinal shift, Nodule/Mass, Other lesion, Pleural effusion, Pleural thickening, Pneumothorax,
  Pulmonary fibrosis, Rib fracture (+ normal negatives).
- **Boxes:** per (image, finding) one consensus box from greedy IoU clustering (threshold 0.3) of the
  radiologists' boxes; normalized to the preprocessed PNG; median area 1.3% of the image; the number of
  contributing readers is stored (`rad_agreement`, train/val).
- **Hard sets:** D1 isolated positives (675; the queried finding is the only annotated abnormality) vs
  co-occurring positives (1,350; at least one other finding annotated, any association strength — the
  maximum pairwise lift is stored per row, median 1.44, 39% ≥ 1.5; capped at 150 per finding), D2 correlated hard
  negatives (lift ≥ 1.5, ≥ 30 co-occurrences, ≤ 200 per pair), unrelated-context negatives (lift ≤ 0.9,
  ≤ 60 per pair), and a weak-association band set (1.2 ≤ lift < 1.5) for sensitivity analysis.
- **Splits:** official VinDr test images form `test`; `val` is 10% of official-train images, image-level,
  seed 0; no image appears in two splits.

## Collection and construction process
Deterministic pipeline (`scripts/build_vqa.py`, `finalize_release.py`, `build_hardset.py`,
`build_unrelated_set.py`) from the VinDr-CXR annotation CSVs; images converted with
`scripts/dicom_to_png.py` (VOI-LUT, MONOCHROME1 inversion, 8-bit, longest side 1,024). Negatives are
drawn only from images all readers marked normal, which removes Yes/No contradictions. A language-model
layer — Anthropic `claude-fable-5`, run 10–12 August 2026 (prompts in `fable5_qa.py` / `fable5_pass.py` of the build
code) — added question paraphrases and natural-language locations from text inputs only, and a
box-appropriateness flag (`roi_appropriate`, `roi_reason`) produced from image crops with the box drawn. Every row records which layers touched it (`enhancements`).

## Provenance of language-model fields and data-use note
Text fields were generated from finding names, boxes and templates — no pixels left the machine. The
`roi_appropriate` flag involved sending image crops to the Anthropic API (`claude-fable-5`); PhysioNet's guidance names
the vendor as acceptable for credentialed data and (since 2025) requires zero data retention. We
recommend treating the flag as auxiliary; a 150-crop blind human audit protocol is included in the
paper's supplement. The deterministic core reproduces every headline number without this field.

## Recommended uses
Grounding evaluation of VLMs (the occlusion protocol), analysis of text-prior and co-occurrence
shortcuts, and as a standard yes/no CXR VQA benchmark with image-necessary negatives.
Not intended for clinical decision making or for training models that will be deployed clinically
without further validation.

## Known limitations
- The occlusion test is defined on positives (a present finding has a box); negatives contribute to
  accuracy and false-positive analyses only.
- Consensus boxes inherit VinDr-CXR's annotation style; extended findings (effusion, cardiomegaly) have
  large boxes whose evidence is distributed through the box (interior ≥ ring at equal area).
- The normal subset is balanced for the benchmark, not a census of all 12,657 normal images.
- Official-test annotations are a five-reader consensus without reader identities, so `rad_agreement`
  is available for train/val only.
- Chest X-ray only; co-occurrence sets are specific to this modality's finding statistics.

## Distribution
Labels, derived annotations, statistics, code and results are released in this repository. Images are
not redistributed; users regenerate them from their own credentialed VinDr-CXR copy. Please cite both
this dataset and VinDr-CXR.

## Maintenance
Issues and corrections via the repository issue tracker. Version: v1 (release files frozen 2026-08-13;
hard sets 2026-08-16; weak-band set 2026-10-08).
