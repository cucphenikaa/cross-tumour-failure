"""RQ3: voxel-level label confusion between ground truth and prediction, per cohort.

For every case: a 4x4 count matrix C[g, p] = #voxels with GT label g and predicted label p (labels 0-3:
background, 1 NCR/NC/NETC, 2 ED/SNFH, 3 ET). Cohort summary = row-normalised matrix over GT tumour labels
(where did the voxels of each true label go?), pooled over voxels and as the median of per-case shares.
Also reports a protocol-harmonised score: Dice of the non-enhancing abnormal region (labels 1 or 2), which is
insensitive to whether non-enhancing tumour is annotated as core (label 1) or as edema (label 2).

Usage:
  python label_confusion.py --cohort MEN=<gt_dir>::<pred_dir> --cohort PED=... --out <dir>
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

NAMES = ["bg", "1 NCR/NC", "2 ED", "3 ET"]


def strip_nii(name: str) -> str:
    return name[:-7] if name.endswith(".nii.gz") else name[:-4]


def nii_file(folder: Path, stem: str):
    for ext in (".nii.gz", ".nii"):
        if (folder / f"{stem}{ext}").exists():
            return folder / f"{stem}{ext}"
    return None


def case_conf(args):
    cohort, case, gt_p, pr_p = args
    g = np.asarray(nib.load(gt_p).dataobj).astype(np.int64).clip(0, 3)
    p = np.asarray(nib.load(pr_p).dataobj).astype(np.int64).clip(0, 3)
    c = np.bincount((g * 4 + p).ravel(), minlength=16).reshape(4, 4)
    row = {"cohort": cohort, "case": case}
    for i in range(4):
        for j in range(4):
            row[f"g{i}_p{j}"] = int(c[i, j])
    a_g, a_p = np.isin(g, (1, 2)), np.isin(p, (1, 2))
    s = a_g.sum() + a_p.sum()
    row["nonenh_dice"] = 1.0 if s == 0 else 2.0 * (a_g & a_p).sum() / s
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", action="append", required=True, help="NAME=<gt_dir>::<pred_dir>")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    jobs = []
    for spec in args.cohort:
        name, _, dirs = spec.partition("=")
        gt_s, _, pr_s = dirs.partition("::")
        gt_dir, pr_dir = Path(gt_s), Path(pr_s)
        for p in sorted(pr_dir.glob("*.nii*")):
            case = strip_nii(p.name)
            g = nii_file(gt_dir, case)
            if g is not None:
                jobs.append((name, case, g, p))
        print(f"[{name}] {sum(j[0] == name for j in jobs)} cases", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        df = pd.DataFrame(list(ex.map(case_conf, jobs, chunksize=2)))
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "per_case_confusion.csv", index=False)

    pd.set_option("display.width", 200)
    rows = []
    for cohort, d in df.groupby("cohort"):
        mat = np.array([[d[f"g{i}_p{j}"].sum() for j in range(4)] for i in range(4)], dtype=float)
        print(f"\n=== {cohort} (n={len(d)}): where do the voxels of each GT label go? (% of row, pooled) ===")
        tab = pd.DataFrame(100 * mat[1:] / mat[1:].sum(1, keepdims=True).clip(1), index=NAMES[1:],
                           columns=[f"pred {n}" for n in NAMES]).round(1)
        print(tab.to_string())
        for i in (1, 2, 3):
            tot = sum(d[f"g{i}_p{j}"] for j in range(4))
            has = tot > 0
            for j in range(4):
                share = (d.loc[has, f"g{i}_p{j}"] / tot[has]).median() if has.any() else np.nan
                rows.append({"cohort": cohort, "gt_label": NAMES[i], "pred_label": NAMES[j],
                             "pooled_pct": tab.iloc[i - 1, j], "median_case_pct": 100 * share,
                             "cases_with_label": int(has.sum())})
        print(f"non-enhancing region (labels 1 or 2) Dice: mean {100 * d.nonenh_dice.mean():.2f}, "
              f"median {100 * d.nonenh_dice.median():.2f}")
    pd.DataFrame(rows).to_csv(args.out / "confusion_summary.csv", index=False)
    print(f"\nSaved {args.out}/per_case_confusion.csv and confusion_summary.csv")


if __name__ == "__main__":
    main()
