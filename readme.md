# MAM-CLIP: Vision–Language Pretraining on Mammography Atlases for BI-RADS Classification

Official code and data release for the paper **"MAM-CLIP: Vision–Language
Pretraining on Mammography Atlases for BI-RADS Classification"**
(Halil Ibrahim Gulluk, Olivier Gevaert).

**Paper:** [arXiv:2605.19359](https://arxiv.org/abs/2605.19359)

We pretrain a vision-language model on **2,313 mammogram image–caption pairs**
extracted from two mammography atlases (CLIP-style contrastive loss + masked
language modeling, following PMC-CLIP), then fine-tune the vision encoder for
BI-RADS prediction. Pretraining on atlas captions yields large gains over an
ImageNet-pretrained baseline, especially in the low-label regime (+1% to +14%
3-class macro-F1), and 2,313 image–text pairs can be more informative than
2,000 extra labeled images.

---

## Repository structure

```
MAM-CLIP/
├── data-extract/
│   ├── extract_mam.py     # extract image–caption pairs from the Atlas of Mammography (PyMuPDF)
│   └── extract_acr.py     # extract image–caption pairs from the ACR BI-RADS Atlas (OCR / pytesseract)
├── train/
│   ├── cfg.yaml           # Hydra config (model, optimizer, data paths, logging)
│   ├── main.py            # pretraining entry point
│   ├── loaders.py         # train/valid DataLoaders
│   ├── mam_clip_load.ipynb# loading pretrained / fine-tuned weights
│   └── files/
│       ├── model.py       # VLModel + Lightning wrapper
│       ├── dataset.py     # image–text dataset + MLM masking + augmentation
│       └── nnblocks.py    # transformer fusion blocks (adapted from OpenCLIP)
├── requirements.txt
├── LICENSE                # MIT (code only)
└── DATA_CARD.md           # dataset terms, citations, download
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

### Released dataset (TEKNOFEST / MammosighTR, preprocessed)

We release **preprocessed PNG mammograms** (breast region cropped from the
original DICOMs with a YOLOX detector) together with an **image-level
metadata file** (`metadata_image_level.json`) that maps each image to its
BI-RADS label, laterality, view, breast composition and English-translated
finding locations.

- **Hugging Face (recommended):** [`gulluk/mammosightr-preprocessed`](https://huggingface.co/datasets/gulluk/mammosightr-preprocessed)
  — browse in the dataset viewer or load directly:
  ```python
  from datasets import load_dataset
  ds = load_dataset("gulluk/mammosightr-preprocessed", split="train")
  ```
- **Google Drive (raw files):** [folder](https://drive.google.com/drive/folders/1DocoIdTt_gfU1WfbOMDeZYer2EIQLnfE?usp=sharing)
  (contains `all_pngs.zip` and `metadata_image_level.json`) — see [DATA_CARD.md](DATA_CARD.md)
- **License:** CC BY-NC 4.0 (non-commercial research use).
- **Required citations — you must cite BOTH:**
  1. This paper (see [Citation](#citation)).
  2. The original dataset source: Koç et al., *MammosighTR: Nationwide
     Breast Cancer Screening Mammogram Dataset with BI-RADS Annotations for
     Artificial Intelligence Applications*, Radiology: Artificial
     Intelligence, 2025. PMID: 40801802 — https://pubmed.ncbi.nlm.nih.gov/40801802/

See [DATA_CARD.md](DATA_CARD.md) for the full data card, label-construction
rule, integrity checksum, and statistics.

The EMBED dataset is **not** redistributed here; request it from the original
authors (Jeong et al., 2023).

### Extracting the pretraining image–text pairs

The atlas books are copyrighted and are **not** redistributed. With your own
legally obtained PDFs:

```bash
python data-extract/extract_mam.py \
  --pdf_file_path /path/to/atlas_of_mammography.pdf \
  --json_file_path /path/to/atlas_mammography_data.json \
  --img_main_path  /path/to/atlas_of_mammography_images/

python data-extract/extract_acr.py \
  --pdf_file_path /path/to/birads_atlas.pdf \
  --json_file_path /path/to/birads_atlas_data.json \
  --img_main_path  /path/to/birads_atlas_images/
```

Each script writes a JSON list of `{"image": "<file>", "caption": "<text>"}`
records — the exact format consumed by the dataloader.

## Pretraining

Set the data paths and logging fields in `train/cfg.yaml`
(`data.train_json_path`, `data.valid_json_path`, `data.image_main_path`,
`trainparams.experiment_name`, `trainparams.project_name`,
`trainparams.checkpoint_dir_path`), then:

```bash
cd train
python main.py
```

Config can also be overridden from the command line (Hydra), e.g.
`python main.py vision_model.name=convnext_tiny trainparams.max_epochs=50`.
The best checkpoint is selected on `valid_loss`.

## Pretrained / fine-tuned weights

Available on Google Drive:
[model weights](https://drive.google.com/drive/folders/1faHwdyjpDB5yQX0acgBlT8A3KgsVI2MM?usp=drive_link)

| File | Description |
|---|---|
| `pretrain_convnext_tiny.ckpt`  | VLM after pretraining, ConvNeXt-Tiny vision encoder |
| `pretrain_convnext_small.ckpt` | VLM after pretraining, ConvNeXt-Small vision encoder |
| `convnext_tiny_ftune.pth`      | Vision encoder fine-tuned for classification (Tiny) |
| `convnext_small_ftune.pth`     | Vision encoder fine-tuned for classification (Small) |

See `train/mam_clip_load.ipynb` for loading examples.

## Citation

```bibtex
@misc{gulluk2026mamclip,
  title         = {MAM-CLIP: Vision--Language Pretraining on Mammography Atlases for BI-RADS Classification},
  author        = {Halil Ibrahim Gulluk and Olivier Gevaert},
  year          = {2026},
  eprint        = {2605.19359},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2605.19359}
}
```

If you use the released dataset, **also** cite the original source:

```bibtex
@article{koc2025mammosightr,
  title   = {MammosighTR: Nationwide Breast Cancer Screening Mammogram Dataset with BI-RADS Annotations for Artificial Intelligence Applications},
  author  = {Ko\c{c}, Ural and others},
  journal = {Radiology: Artificial Intelligence},
  volume  = {7},
  number  = {6},
  pages   = {e240841},
  year    = {2025},
  doi     = {10.1148/ryai.240841},
  note    = {PMID: 40801802}
}
```

## Acknowledgements

Our vision-language code builds on
[PMC-CLIP](https://github.com/WeixiongLin/PMC-CLIP) and
[OpenCLIP](https://github.com/mlfoundations/open_clip). DICOM preprocessing
follows the YOLOX breast-cropping pipeline of
[kaggle_rsna_breast_cancer](https://github.com/dangnh0611/kaggle_rsna_breast_cancer).

## License

Source code: **MIT** (see [LICENSE](LICENSE)). Released dataset: **CC BY-NC
4.0** (see [DATA_CARD.md](DATA_CARD.md)).

## Contact

Questions: gulluk@stanford.edu
