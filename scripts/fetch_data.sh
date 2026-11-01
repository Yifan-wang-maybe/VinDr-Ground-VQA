#!/usr/bin/env bash
# Download the VinDr-Ground-VQA label archive from Zenodo into data/ and verify checksums.
# Usage: bash scripts/fetch_data.sh [ZENODO_RECORD_ID]
set -euo pipefail
RECORD="${1:-${VGV_ZENODO_RECORD:-XXXXXXX}}"      # fill in the numeric Zenodo record id after publishing
HERE="$(cd "$(dirname "$0")" && pwd)"; DATA="$(dirname "$HERE")/data"
if [ "$RECORD" = "XXXXXXX" ]; then echo "Set the Zenodo record id: bash scripts/fetch_data.sh <record_id>"; exit 1; fi
URL="https://zenodo.org/records/${RECORD}/files/VinDr-Ground-VQA_labels_v1.0.zip?download=1"
TMP="$(mktemp -d)"; echo "downloading ${URL}"
curl -L --fail -o "$TMP/labels.zip" "$URL"
unzip -q -o "$TMP/labels.zip" -d "$TMP/x"
SRC="$(find "$TMP/x" -maxdepth 2 -name SHA256SUMS -printf '%h\n' | head -1)"
cp "$SRC"/*.jsonl "$SRC"/SHA256SUMS "$DATA/"
cd "$DATA" && sha256sum -c SHA256SUMS --ignore-missing && echo "label files verified in $DATA"
rm -rf "$TMP"
