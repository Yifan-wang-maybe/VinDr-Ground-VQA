# GEMeX comparison protocol (item #8) — drop-in facts

**Subset scored.** GEMeX-ThinkVG annotations (`BoKelvin/GEMeX-ThinkVG`, CC-BY-NC-4.0; pixels from PhysioNet-credentialed MIMIC-CXR-JPG). We keep only closed yes/no items and score the first **n = 399** whose images resolve in our MIMIC-CXR copy, through the identical two-token scoring path as VinDr. Images are letterboxed to the 336×336 evaluation canvas used throughout; ThinkVG boxes are mapped to the same canvas.

**How the box sizes were measured ("23% vs 1.3%").** For each question, the union area of its gold evidence boxes divided by image area, computed on the annotation pixel grid *before* any model-side resizing (the fraction is resize-invariant). GEMeX-ThinkVG: **median 0.23** of the image, IQR [0.10, 0.45]; 13% of questions have evidence regions covering more than half the image. VinDr-Ground-VQA, measured the same way on the consensus ROI: **median 0.013**, i.e. ≈18× smaller. (Sentence for the paper: "Per question, we compute the union area of the gold evidence boxes as a fraction of image area on the source pixel grid; GEMeX-ThinkVG has median 0.23 (IQR 0.10–0.45) versus 0.013 for VinDr-Ground-VQA.")

**Four-model excess on GEMeX** (raw gold-logp scale, M − M_rand, size-matched random boxes, n = 399, sign test one-sided):

| model | median M | median M_rand | median excess | % > 0 | sign-test p |
|---|---|---|---|---|---|
| MedGemma-4b | −0.000 | −0.012 | **+0.014** | 54.6% | 0.036 |
| Lingshu-7B | +0.071 | +0.032 | **+0.027** | 57.9% | 0.0009 |
| Qwen3-VL-8B | −0.269 | −0.196 | **−0.044** | 40.9% | 1.0 |
| InternVL3.5-8B | +0.086 | +0.104 | **−0.002** | 49.4% | 0.62 |

Reading: on organ-scale boxes the evidence-specific term is at most a trace (+0.01–0.03 nats for the two medically-tuned models, indistinguishable from zero for the other two) — one to two orders of magnitude below the VinDr excess on tight boxes (+0.875 raw / +2.5 log-odds in the large tertile). The honest wording is "at most a trace, an order of magnitude smaller," not "exactly zero for all models": with n = 399 the two medical models' tiny positives do reach nominal significance.

**Register argument.** No experiment backs the "flat patch pushes the model toward generic report prose" mechanism on GEMeX (our GEMeX pixel copy is no longer on disk, and the fills experiment ran on VinDr only). Recommend softening to conjecture: "one plausible mechanism, which we do not test here, is …". Note the VinDr fills experiment (item #3) *does* show the related fact that F ≈ 0 under blur, i.e. the format-flight term is a patch artifact, which is consistent with but does not establish the register claim on GEMeX.

**Which split / resolution caveats.** ThinkVG inherits MIMIC-CXR's official splits; our 399 items come from its released annotation file in file order (no cherry-picking), restricted to yes/no. Area fractions use the source grid, so they are unaffected by the 336×336 canvas.
