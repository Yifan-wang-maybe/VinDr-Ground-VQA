### Table R5-A. Fill sensitivity: evidence-specific excess (median) by ROI-size tertile under four fills, same cases and the same random geometry refilled

| model | fill | small | medium | large | format term F (median) |
|---|---|---|---|---|---|
| MedGemma-4b | mean pixel (paper) | +0.06 | +0.62 | +2.50 | see Table 12 |
| MedGemma-4b | Gaussian blur | +0.00 | +0.37 | +2.87 | +0.00 |
| MedGemma-4b | black | +0.25 | +0.75 | +3.63 | — |
| MedGemma-4b | inpainting (Telea) | +0.12 | +0.37 | +1.37 | +0.00 |
| Lingshu-7B | mean pixel (paper) | +0.13 | +0.25 | +0.75 | see Table 12 |
| Lingshu-7B | Gaussian blur | +0.12 | +0.25 | +0.75 | +0.00 |
| Lingshu-7B | black | +0.13 | +0.25 | +0.62 | — |
| Lingshu-7B | inpainting (Telea) | +0.00 | +0.13 | +0.50 | +0.00 |
| InternVL3.5-8B | mean pixel (paper) | +0.00 | +0.25 | +0.75 | see Table 12 |
| InternVL3.5-8B | Gaussian blur | +0.00 | +0.13 | +0.63 | -0.00 |
| InternVL3.5-8B | black | +0.00 | +0.25 | +0.63 | — |
| InternVL3.5-8B | inpainting (Telea) | +0.00 | +0.13 | +0.50 | +0.00 |
| Qwen3-VL-8B | mean pixel (paper) | +0.06 | +0.12 | +1.75 | see Table 12 |
| Qwen3-VL-8B | Gaussian blur | -0.25 | +0.25 | +1.75 | -0.00 |
| Qwen3-VL-8B | black | -0.25 | +0.00 | +1.00 | — |
| Qwen3-VL-8B | inpainting (Telea) | -0.00 | +0.00 | +1.50 | +0.00 |

Mean/blur/black: n = 250 per model (fills experiment); inpainting: n = 250 per model (E11). The ordering small < medium < large holds in every row; blur and inpainting remove the flat-patch artifact, so F → 0 and Qwen3-VL's generic masking term disappears under them. Inpainting removes less information than a flat patch, so its magnitudes are a conservative lower bound.

### Table R5-B. Control ladder: evidence-specific excess (median) by ROI-size tertile when the control region is placed by each rule

| model | control rule | small | medium | large | median cost of masking the control (M_control) |
|---|---|---|---|---|---|
| MedGemma-4b | uniform random, evidence-avoiding (paper) ¹ | +0.25 | +0.56 | +2.75 | +0.00 |
| MedGemma-4b | body-restricted (tissue only) ¹ | +0.25 | +0.56 | +3.50 | +0.00 |
| MedGemma-4b | ROI-intensity-matched ¹ | +0.25 | +0.50 | +3.25 | +0.00 |
| MedGemma-4b | texture-matched (gradient energy within 20%) ² | +0.37 | +0.62 | +2.63 | -0.00 |
| MedGemma-4b | finding-location-matched (same finding, other patient) ² | +0.25 | +0.62 | +2.31 | -0.00 |
| MedGemma-4b | strictly inside lung field (segmentation mask) ² | +0.37 | +0.75 | +2.75 | -0.13 |
| MedGemma-4b | contralateral mirror (anatomy-matched) ³ | — | — | — | +0.00 (excess vs mirror +0.37, n = 97) |
| Lingshu-7B | uniform random, evidence-avoiding (paper) ¹ | +0.13 | +0.25 | +0.75 | +0.00 |
| Lingshu-7B | body-restricted (tissue only) ¹ | +0.12 | +0.13 | +0.87 | +0.00 |
| Lingshu-7B | ROI-intensity-matched ¹ | +0.12 | +0.13 | +0.75 | +0.00 |
| Lingshu-7B | texture-matched (gradient energy within 20%) ² | +0.12 | +0.25 | +0.88 | +0.00 |
| Lingshu-7B | finding-location-matched (same finding, other patient) ² | +0.13 | +0.25 | +0.62 | +0.12 |
| Lingshu-7B | strictly inside lung field (segmentation mask) ² | +0.13 | +0.25 | +0.63 | +0.00 |
| Lingshu-7B | contralateral mirror (anatomy-matched) ³ | — | — | — | +0.00 (excess vs mirror +0.25, n = 97) |
| InternVL3.5-8B | uniform random, evidence-avoiding (paper) ¹ | +0.00 | +0.25 | +0.87 | +0.00 |
| InternVL3.5-8B | body-restricted (tissue only) ¹ | +0.00 | +0.13 | +0.87 | +0.00 |
| InternVL3.5-8B | ROI-intensity-matched ¹ | +0.00 | +0.13 | +0.87 | +0.00 |
| InternVL3.5-8B | texture-matched (gradient energy within 20%) ² | +0.12 | +0.13 | +0.75 | -0.12 |
| InternVL3.5-8B | finding-location-matched (same finding, other patient) ² | +0.00 | +0.13 | +0.50 | -0.00 |
| InternVL3.5-8B | strictly inside lung field (segmentation mask) ² | +0.00 | +0.13 | +0.63 | -0.00 |
| Qwen3-VL-8B | uniform random, evidence-avoiding (paper) ¹ | +0.00 | -0.00 | +1.75 | +0.75 |
| Qwen3-VL-8B | body-restricted (tissue only) ¹ | -0.00 | -0.25 | +1.50 | +0.75 |
| Qwen3-VL-8B | ROI-intensity-matched ¹ | -0.00 | +0.00 | +1.75 | +0.50 |
| Qwen3-VL-8B | texture-matched (gradient energy within 20%) ² | +0.00 | +0.25 | +2.25 | +0.88 |
| Qwen3-VL-8B | finding-location-matched (same finding, other patient) ² | -0.00 | +0.13 | +1.44 | +1.00 |
| Qwen3-VL-8B | strictly inside lung field (segmentation mask) ² | -0.25 | -0.00 | +1.25 | +1.25 |

¹ Placement experiment: n = 500 per model, all conditions recomputed fresh under the yes/no prompt. ² E7 / E10: n = 1,000 per model (897 / 990 / 913 placed), cached bare-prompt full-image and ROI-masked margins, one fresh control forward per case. ³ Mirror: ROI reflected across the vertical midline, midline findings excluded (n = 97 eligible per model; run for MedGemma and Lingshu). Every rung costs the three non-perturbation-sensitive models ≈ 0 and leaves excess unchanged; for Qwen3-VL every rule reproduces the same generic masking term (+0.6–1.3), so its small/medium excess stays ≈ 0 and its large-tertile excess persists.