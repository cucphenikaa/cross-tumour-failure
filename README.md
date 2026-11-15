# Small lesions and label protocols: what breaks glioma-trained segmentation on other brain tumours

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24237466.svg)](https://doi.org/10.5281/zenodo.24237466)

## Archived release

The public research materials are archived on Zenodo: [10.5281/zenodo.24237466](https://doi.org/10.5281/zenodo.24237466). For reproducible citation, use the DOI associated with the specific archived release.

Supplementary material for the article *Small lesions and label protocols: what breaks glioma-trained segmentation on other brain tumours* (Bui & Nguyen, submitted to *Computerized Medical Imaging and Graphics*).

A glioma-trained nnU-Net (BraTS 2023) loses 18 Dice points on meningioma and 28 on paediatric tumours, for different reasons: missed small, edema-free meningiomas, and a different annotation protocol for paediatric non-enhancing tumour. A phenotype-routed mixture-of-experts and small-lesion synthesis do not close the gap; fine-tuning with 10–100 real meningiomas does, without forgetting glioma.

## What is released now

This repository lets readers and reviewers verify every number in the article:

| Path | Content |
| --- | --- |
| `splits/` | Case identifiers of every split: GLI patient-level 5-fold split (fold 0 used: 1,009 train / 242 validation), MEN300 test set, meningioma k-shot pool in its nested order, paediatric pool and test set |
| `results/` | Per-case metrics (Dice, HD95, volumes, spurious edema) of every model on MEN300, PED and GLI validation, and voxel-level label-confusion summaries |
| `src/eval/evaluate.py` | Metric definitions (BraTS empty-region convention), inference wrapper |
| `src/eval/compare.py` | Paired patient-level comparison: bootstrap CI, Wilcoxon signed-rank, Holm correction of the three primary endpoints |
| `src/eval/label_confusion.py` | Voxel-level label confusion and protocol-harmonised (labels 1∪2) Dice |

Models in `results/`: `E1` baseline nnU-Net ResEnc-M; `E2_moe` phenotype-routed mixture-of-experts; `E3_synthesis` small-lesion synthesis; `E7_k*` fine-tuning from E1 with k real meningiomas (`k0` = schedule-matched control, `k10a`/`k10b` = disjoint draws); `swin_unetr` public MONAI Swin UNETR (BraTS 2021).

Example — reproduce the k = 30 comparison reported in the article:

```bash
pip install -r requirements.txt
python src/eval/compare.py --a E1=results/E1/test_per_case_metrics.csv --a_gli results/E1/glival_per_case_metrics.csv \
    --b k30=results/E7_k30/test_per_case_metrics.csv --b_gli results/E7_k30/glival_per_case_metrics.csv --out out/k30_vs_E1
```

## Coming with publication

The training code (nnU-Net v2.8.1 trainers for the time-budgeted baseline, mixture-of-experts, small-lesion synthesis and k-shot fine-tuning), the data pipeline, the phenotype-descriptor and failure-modelling code, and the Kaggle notebooks will be released in this repository upon publication of the article. All hyperparameters are given in the article.

## Data

Images are from the BraTS 2023 challenge and are not redistributed; obtain them from the organisers under the BraTS data-use terms.

## License

MIT (code). Data remain subject to the BraTS data-use terms.
