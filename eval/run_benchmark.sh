#!/usr/bin/env bash
# One model, end to end, on one GPU. Produces the numbers reported in the paper's main tables.
#   bash eval/run_benchmark.sh medgemma 0
# Prerequisites: data/*.jsonl (shipped), images regenerated into $VGV_IMAGES (see scripts/REGENERATE.md),
# model weights reachable via Hugging Face (or VGV_MODEL_<KEY> pointing to a local directory).
set -euo pipefail
MODEL=${1:?model key: medgemma | lingshu | internvl35 | qwen3vl | qwen3vl30b | gemma3 | qwen25vl | medgemma27b | lingshu32b}
GPU=${2:-0}
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"
export VGV_ROOT="${VGV_ROOT:-$ROOT}"; export VGV_DATA="${VGV_DATA:-$ROOT/data}"; export VGV_IMAGES="${VGV_IMAGES:-$VGV_DATA/images}"
export VGV_RESULTS="${VGV_RESULTS:-$ROOT/results/mine}"; mkdir -p "$VGV_RESULTS"
export CUDA_VISIBLE_DEVICES=$GPU
cd "$HERE"
echo "[1/4] accuracy gate (matched yes/no set, sighted and blind; Table 3)"
python accuracy_gate.py --model "$MODEL" --n 3000 --n_open 100 --out "$VGV_RESULTS/p1_${MODEL}.json"
echo "[2/4] occlusion on release test positives (Fig. 1 / Table 5: M_roi, M_rand, excess; n=1000)"
python mjsd_harness.py --model "$MODEL" --dataset vindr --n 1000 --release --with_random --avoid_evidence --out "$VGV_RESULTS/mjsd_${MODEL}_vindr_release_mm.json"
echo "[3/4] hard sets: isolated vs correlated positives (Table 7) and correlated hard negatives (Table 9)"
python hardset_eval.py --model "$MODEL" --out "$VGV_RESULTS/p3_${MODEL}.json"
echo "[4/4] unrelated-context negatives (Table 9 middle group)"
python unrelated_eval.py --model "$MODEL" --out "$VGV_RESULTS/t1_exp3a_${MODEL}.json"
echo "done -> $VGV_RESULTS. Summarise with: python analysis/summarize_model.py --model $MODEL"
