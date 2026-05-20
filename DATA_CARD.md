# Data Card — TEKNOFEST / MammosighTR Preprocessed Mammography Dataset

## Summary

Preprocessed PNG mammograms with image-level BI-RADS labels, derived from the
nationwide Turkish breast-cancer screening dataset released for the TEKNOFEST
2023 Artificial Intelligence in Health Competition by the Republic of Turkey
Ministry of Health. The original DICOMs are cropped to the breast region with
a YOLOX detector and exported as PNG. We additionally provide an image-level
metadata file built from the official patient-level annotation spreadsheets.

## Source and required citations

This is a **preprocessed redistribution**. Any use of this data **must cite
both**:

1. **Original source** — Koç U., Karakaş E., Sezer E. A., et al.
   *MammosighTR: Nationwide Breast Cancer Screening Mammogram Dataset with
   BI-RADS Annotations for Artificial Intelligence Applications.* Radiology:
   Artificial Intelligence, 7(6):e240841, 2025. DOI: 10.1148/ryai.240841.
   PMID: 40801802. https://pubmed.ncbi.nlm.nih.gov/40801802/
2. **This work** — Gulluk & Gevaert, *MAM-CLIP: Vision--Language Pretraining
   on Mammography Atlases for BI-RADS Classification*,
   [arXiv:2605.19359](https://arxiv.org/abs/2605.19359), 2026 (BibTeX in the
   repository README).

## License

**CC BY-NC 4.0** — non-commercial research use, with attribution (the two
citations above). Redistribution must preserve this data card and both
citations.

## Download

- Google Drive folder:
  https://drive.google.com/drive/folders/1DocoIdTt_gfU1WfbOMDeZYer2EIQLnfE?usp=sharing
- Archive: `all_pngs.zip` (in the folder above; ~11 GB)
- Image-level metadata: `metadata_image_level.json` (in the folder and in this repo)
- Integrity (SHA-256 of `all_pngs.zip`):
  `d1bff0dd996f5ed2a1c43bb82a08d94f99e2135a77afb589c7c642fdbb357ada`

Because the archive is large, Google Drive cannot virus-scan it and shows a
"can't scan this file — download anyway?" prompt; this is expected. Verify
integrity instead via the checksum after download:

```bash
shasum -a 256 all_pngs.zip   # compare against the value above
```

## Contents

```
all_pngs/<patient_id>/{LCC,LMLO,RCC,RMLO}.png
```

- **10,740** patient folders; most contain 4 views, **840** contain 3
  (one view missing in the source).
- **42,074** PNG images total.
- `patient_id` is the anonymized `HASTANO` identifier from the source.

## `metadata_image_level.json`

A JSON list; one record per image:

| Field | Description |
|---|---|
| `patient_id` | Anonymized patient id (folder name) |
| `image_filename` | `LCC.png` / `LMLO.png` / `RCC.png` / `RMLO.png` |
| `relative_path` | Path within the archive |
| `view` | `CC` or `MLO` |
| `laterality` | `left` or `right` |
| `birads` | Integer image-level BI-RADS (0,1,2,4,5) |
| `birads_label` | e.g. `"BI-RADS 4"` |
| `breast_composition` | ACR density `A`/`B`/`C`/`D` |
| `findings_locations` | Finding quadrants, translated to English |
| `patient_birads` | Original patient-level BI-RADS |

The TEKNOFEST cohort contains no BI-RADS 3 or BI-RADS 6 cases.

### Label construction (patient → image)

Patient-level BI-RADS is parsed from the source "Kategori N" field
(`Kategori N` → BI-RADS `N`). Laterality: `SAĞ` = right (RCC, RMLO),
`SOL` = left (LCC, LMLO). A quadrant cell is *empty* if NaN, `""` or `[]`.

- Patient BI-RADS == 1 → both breasts BI-RADS 1.
- Patient BI-RADS != 1:
  - findings on both sides → patient score on both breasts;
  - findings on one side → patient score on that side, BI-RADS 1 on the
    other;
  - **no findings on either side → patient score on both breasts** (2,293
    patients, mostly BI-RADS 2; this convention is intentional — BI-RADS 2/0
    findings are often not quadrant-localized).
- Both views (CC, MLO) of a breast inherit that breast's label.


### Image-level BI-RADS distribution

| BI-RADS | Count |
|--:|--:|
| 0 | 5,300 |
| 1 | 18,448 |
| 2 | 10,088 |
| 4 | 3,799 |
| 5 | 4,439 |


## Intended use & limitations

Research only (non-commercial). Labels are screening BI-RADS assessments; BI-RADS 0 indicates an incomplete assessment.
Quadrant localization is annotated at the patient level and propagated by the
rule above; it is not a per-image bounding box.
