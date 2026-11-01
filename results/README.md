# Reference results

Per-model result files behind the paper and rebuttal tables (release test split unless noted). File naming: `<experiment>_<model>.json`.

| prefix | produced by | content |
|---|---|---|
| occlusion_release_ | eval/mjsd_harness.py --release --with_random --avoid_evidence | per-case mg_full/mg_coll/mg_rand, M, Mm, Mm_rand (n=1000 positives) |
| accuracy_gate_ | eval/accuracy_gate.py | sighted/blind yes-no accuracy on the matched set (n=3000) |
| hardset_ | eval/hardset_eval.py | D1 isolated vs correlated positives; D2 correlated hard negatives |
| unrelated_ | eval/unrelated_eval.py | unrelated-context negatives (FPR, blind, context-masked) |
| specificity_ | eval/specificity_eval.py | named vs other-finding vs random masking on correlated positives (Table 8) |
| resize_ | eval/resize_eval.py | enlarge/shrink the same box (Table 6) |
| control_placement_ / control_fills_ / control_mirror_ksens_ | eval/controls/ | placement rules; blur/black fills; mirror + k-sensitivity |
| control_ladder_ / lungfield_ / inpaint_ / normal_border_ | eval/controls/ | location/texture/ring; lung-field; inpainting; normal-anatomy border masking |
| grounding_fraction_ | eval/grounding_fraction.py | blind margins, image-gain, oracle, GF |
| gemex_blind_ | eval/gemex_blind.py | text-only yes/no on GEMeX-ThinkVG closed questions |

`ci_tables.json`: paired-bootstrap CIs for every paper table. `leaderboard.csv`: the summary table in the README. `*.md`: assembled tables (control ladder, fills, threshold sensitivity).
