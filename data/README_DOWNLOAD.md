# Getting the label files

The question/answer/box files are distributed through Zenodo so they carry a DOI and checksums:

**VinDr-Ground-VQA labels v1.0 — Zenodo DOI: `10.5281/zenodo.XXXXXXX`** ← fill in after the deposit is published

```bash
bash scripts/fetch_data.sh            # downloads the Zenodo archive into data/ and verifies SHA-256 sums
```

or download `VinDr-Ground-VQA_labels_v1.0.zip` from the Zenodo record and unpack it into this `data/`
directory. You should end up with:

| file | rows | SHA-256 (first 12 hex) |
|---|---|---|
| `train.jsonl` | 81,656 | see `SHA256SUMS` in the archive |
| `val.jsonl` | 9,652 | |
| `test.jsonl` | 10,764 | |
| `hardset_v1.jsonl` | 14,359 | |
| `hardset_unrelated_v1.jsonl` | 4,740 | |
| `hardset_weakband_lift1.2-1.5.jsonl` | 24,112 | |

The small files already in this folder (`SCHEMA.md`, `cooccurrence_lift.csv`, `finding_counts.json`,
`dims.json`, `manifest.json`) are identical to the copies in the Zenodo archive.

Images are **not** part of either distribution; see `scripts/REGENERATE.md`.
