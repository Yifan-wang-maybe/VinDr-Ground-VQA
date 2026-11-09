#!/usr/bin/env bash
# Download the VinDr-Ground-VQA label archive from Zenodo into data/ and verify checksums.
# Usage: bash scripts/fetch_data.sh [ZENODO_RECORD_ID]
set -euo pipefail
RECORD="${1:-${VGV_ZENODO_RECORD:-24073784}}"      # Zenodo record 24073784 = DOI 10.5281/zenodo.24073784 (labels v1.0)
HERE="$(cd "$(dirname "$0")" && pwd)"; DATA="$(dirname "$HERE")/data"
URL="https://zenodo.org/records/${RECORD}/files/VinDr-Ground-VQA_labels_v1.0.zip?download=1"
TMP="$(mktemp -d)"; echo "downloading ${URL}"
curl -L --fail -o "$TMP/labels.zip" "$URL"
unzip -q -o "$TMP/labels.zip" -d "$TMP/x"
SRC="$(find "$TMP/x" -maxdepth 2 -name SHA256SUMS -printf '%h\n' | head -1)"
cp "$SRC"/*.jsonl "$SRC"/SHA256SUMS "$DATA/"
cd "$DATA" && sha256sum -c SHA256SUMS --ignore-missing && echo "label files verified in $DATA"
rm -rf "$TMP"
