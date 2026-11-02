# One control placement (paper) vs the mean of five — same cases, bare prompt, cached full/ROI margins

For each of 300 release-test positives per model, the paper's single evidence-avoiding size-matched placement (seed = case index) is compared with the mean of five fresh placements (seeds case·100 + j). Medians with paired-bootstrap 95% CIs (10,000 resamples).

| model | tertile | n | excess, k = 1 (paper) | excess, k = 5 | paired difference k1 − k5 | median M_rand k1 / k5 |
|---|---|---|---|---|---|---|
| MedGemma-4b | small | 100 | +0.12 [-0.06, +0.44] | +0.20 [-0.03, +0.48] | +0.03 [-0.08, +0.11] | +0.00 / -0.07 |
| MedGemma-4b | medium | 100 | +0.69 [+0.38, +1.00] | +0.52 [+0.25, +0.92] | -0.00 [-0.13, +0.15] | +0.12 / +0.01 |
| MedGemma-4b | large | 100 | +2.31 [+1.62, +3.75] | +2.30 [+1.71, +3.73] | +0.02 [-0.11, +0.19] | -0.12 / -0.08 |
| MedGemma-4b | all | 300 | +0.87 [+0.63, +1.13] | +0.78 [+0.52, +1.13] | +0.01 [-0.05, +0.07] | -0.00 / -0.05 |
| MedGemma-4b | per case | 300 | median \|Δexcess\| = 0.33 nats | between-draw SD = 0.37 | | |
| Lingshu-7B | small | 100 | +0.13 [+0.12, +0.31] | +0.18 [+0.10, +0.27] | +0.00 [-0.03, +0.03] | +0.00 / +0.03 |
| Lingshu-7B | medium | 100 | +0.25 [+0.13, +0.50] | +0.29 [+0.08, +0.50] | -0.00 [-0.05, +0.03] | +0.12 / +0.05 |
| Lingshu-7B | large | 100 | +0.87 [+0.50, +1.12] | +0.83 [+0.54, +1.15] | +0.02 [-0.06, +0.08] | +0.00 / +0.05 |
| Lingshu-7B | all | 300 | +0.37 [+0.25, +0.50] | +0.35 [+0.30, +0.46] | +0.00 [-0.03, +0.03] | +0.00 / +0.05 |
| Lingshu-7B | per case | 300 | median \|Δexcess\| = 0.15 nats | between-draw SD = 0.17 | | |

Reading: a single placement is noisier per case (median |Δ| 0.33 nats for MedGemma, 0.15 for Lingshu — the between-draw SD of the control), but the tertile and overall medians are unchanged: every paired k1 − k5 difference is within ±0.03 nats with a CI that includes zero. The main tables' k = 1 protocol and a k = 5 average give the same medians; k = 5 was used only in the resize experiment, where the box area changes.