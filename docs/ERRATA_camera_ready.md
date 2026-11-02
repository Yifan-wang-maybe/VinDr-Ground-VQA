# Corrections to the submitted manuscript (for the camera-ready)

These are wording/protocol statements in the ML4H 2026 submission that the released code and data make
precise. None changes a reported number.

1. **Number of random-control placements (§4, Eq. 5–6).** The text says "We average M_rand over k = 5
   valid random placements." In the main tables (Fig. 1, Tables 4, 5, 7, 8, 9) **one** evidence-avoiding,
   size-matched placement is used per case, with seed = case index and identical across models and fill
   operators; the **five-placement average applies only to the resize experiment (Table 6)**, where the box
   area changes. Correct wording: "M_rand uses one size-matched placement per case (seed = case index);
   in the resize experiment we average five placements at each target area." The operator-sensitivity
   table (mean / blur / black / inpainting) refills the *same* single placement, so it varies only the fill.
   A direct k = 1 vs k = 5 check is reported in `results/k1_vs_k5.md`.

2. **Definition of "correlated positives" (§5.4.1, Table 2).** The D1 positive group is "the queried
   finding annotated together with at least one other finding" — any association strength (median maximum
   pairwise lift 1.44; 39% ≥ 1.5). The lift ≥ 1.5 / ≥ 30 co-occurrence rule applies to the D2 hard
   negatives only. Suggested label: "co-occurring positives"; the ≥ 1.5 subset collapses more (appendix
   threshold table).

3. **Table 8 caption.** Columns are raw median drops M (named / other / random), not excess;
   named − random reproduces Table 7.

4. **Appendix E.5.** The clearance rule's selection effect applies to the enlarge cell of Table 6 only; the
   shrink and cardiomegaly cells are unaffected.

5. **§5.2.1 / Table 4 reading.** GEMeX-ThinkVG's near-zero raw drops reflect a text prior (74% of closed
   golds are "No"; blind accuracy equals the majority baseline), not "generic perturbation"; Table 4's M is a
   gold-sentence log-probability drop and is not comparable in magnitude with Table 5.

6. **§5.5.2 wording.** With Qwen3-VL-30B's intermediate increment (+0.09), "concentrated in the medically
   trained models" → "largest in the medically trained models."

7. **Line 198 grammar; duplicate data-availability statement; one unresolved reference.**
