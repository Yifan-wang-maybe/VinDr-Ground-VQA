# Getting the label files

The question/answer/box files are distributed through Zenodo so they carry a DOI and checksums:

**VinDr-Ground-VQA labels v1.0 — Zenodo DOI: [`10.5281/zenodo.24073784`](https://doi.org/10.5281/zenodo.24073784)**  (concept DOI for all versions: `10.5281/zenodo.24073783`)

Archive: `VinDr-Ground-VQA_labels_v1.0.zip` (5.3 MB, MD5 `ef6a6a7d49f0ac461a6b798085a903cd`), record https://zenodo.org/records/24073784

```bash
bash scripts/fetch_data.sh            # downloads the Zenodo archive into data/ and verifies SHA-256 sums
```

or download `VinDr-Ground-VQA_labels_v1.0.zip` from the Zenodo record and unpack it into this `data/`
directory. You should end up with:

| file | rows | SHA-256 (first 12 hex) |
|---|---|---|
| `train.jsonl` | 81,656 | `5a3ef1d4bf1b` |
| `val.jsonl` | 9,652 | `1ae8e0f6a415` |
| `test.jsonl` | 10,764 | `33ab6a764597` |
| `hardset_v1.jsonl` | 14,359 | `6644fff2f813` |
| `hardset_unrelated_v1.jsonl` | 4,740 | `1dbdc5542d40` |
| `hardset_weakband_lift1.2-1.5.jsonl` | 24,112 | `334f7c993c80` |

The small files already in this folder (`SCHEMA.md`, `cooccurrence_lift.csv`, `finding_counts.json`,
`dims.json`, `manifest.json`) are identical to the copies in the Zenodo archive.

Images are **not** part of either distribution; see `scripts/REGENERATE.md`.
