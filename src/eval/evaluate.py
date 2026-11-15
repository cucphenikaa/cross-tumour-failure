"""Predict with a trained nnU-Net model on external test cohorts and compute per-case metrics.

Regions (BraTS): WT = {1,2,3}, TC = {1,3}, ET = {3}.  Labels: 1 NCR/NETC, 2 ED/SNFH, 3 ET.
Per case and region: Dice, HD95 (mm), GT / predicted volume (mL).
Empty-region convention (BraTS legacy): both empty -> Dice 1, HD95 0; one empty -> Dice 0, HD95 373.13.
Hypothesis-specific metric: spurious edema = predicted ED voxels outside GT ED (mL), reported overall and
for cases whose GT has no edema.

Usage:
  python evaluate.py --model_dir <.../nnUNetTrainer_PM__nnUNetResEncUNetMPlans__3d_fullres> \
      --checkpoint checkpoint_final.pth --cohort MEN=/kaggle/input/... --cohort PED=/kaggle/input/... \
      --out /kaggle/working/eval_baseline [--max_cases 2] [--device cpu] [--tta]
  python evaluate.py --metrics_only --cohort GLIval=<gt_dir>::<pred_dir> --out ...
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

REGIONS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}
HD_MAX = 373.13


# ------------------------------------------------------------------ locating data
def find_dir_with(root: Path, pattern: str) -> Path:
    hits = sorted({p.parent for p in root.rglob(pattern)})
    if not hits:
        raise FileNotFoundError(f"no '{pattern}' under {root}")
    return hits[0]


def strip_nii(name: str) -> str:
    """Case id from a file name ending in .nii or .nii.gz (Kaggle may decompress .nii.gz)."""
    return name[:-7] if name.endswith(".nii.gz") else name[:-4]


def nii_file(folder: Path, stem: str) -> Path | None:
    for ext in (".nii.gz", ".nii"):
        if (folder / f"{stem}{ext}").exists():
            return folder / f"{stem}{ext}"
    return None


def cohort_dirs(root: Path) -> tuple[Path, Path]:
    img = find_dir_with(root, "*_0000.nii*")
    if (img.parent / "labels").exists():
        return img, img.parent / "labels"
    for d in sorted({p.parent for p in root.rglob("*.nii*")}):
        if d != img and not any(d.glob("*_0000.nii*")):
            return img, d
    raise FileNotFoundError(f"no label folder under {root}")


# ------------------------------------------------------------------ metrics
def _surface(m: np.ndarray) -> np.ndarray:
    return m ^ ndi.binary_erosion(m, structure=np.ones((3, 3, 3), bool))


def hd95(a: np.ndarray, b: np.ndarray, spacing) -> float:
    idx = np.argwhere(a | b)
    lo, hi = np.maximum(idx.min(0) - 2, 0), idx.max(0) + 3          # crop for speed
    a, b = a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], b[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sa, sb = _surface(a), _surface(b)
    d_ab = ndi.distance_transform_edt(~sb, sampling=spacing)[sa]
    d_ba = ndi.distance_transform_edt(~sa, sampling=spacing)[sb]
    return float(np.percentile(np.concatenate([d_ab, d_ba]), 95))


def case_metrics(args: tuple[str, str, Path, Path]) -> dict:
    cohort, case, gt_path, pr_path = args
    g_img = nib.load(gt_path)
    gt = np.asarray(g_img.dataobj).astype(np.int16)
    pr = np.asarray(nib.load(pr_path).dataobj).astype(np.int16)
    assert gt.shape == pr.shape, f"{case}: shape {gt.shape} vs {pr.shape}"
    sp = tuple(float(z) for z in g_img.header.get_zooms()[:3])
    vox_ml = float(np.prod(sp)) / 1000.0
    row = {"cohort": cohort, "case": case}
    for name, labs in REGIONS.items():
        g, p = np.isin(gt, labs), np.isin(pr, labs)
        ng, np_ = int(g.sum()), int(p.sum())
        if ng == 0 and np_ == 0:
            dice, hd = 1.0, 0.0
        elif ng == 0 or np_ == 0:
            dice, hd = 0.0, HD_MAX
        else:
            dice = 2.0 * (g & p).sum() / (ng + np_)
            hd = hd95(g, p, sp)
        row.update({f"{name}_dice": dice, f"{name}_hd95": hd,
                    f"{name}_vol_gt": ng * vox_ml, f"{name}_vol_pred": np_ * vox_ml})
    ed_g, ed_p = gt == 2, pr == 2
    row["ED_vol_gt"] = ed_g.sum() * vox_ml
    row["ED_vol_pred"] = ed_p.sum() * vox_ml
    row["ED_spurious_ml"] = (ed_p & ~ed_g).sum() * vox_ml
    return row


def compute_metrics(cohort: str, gt_dir: Path, pr_dir: Path, workers: int) -> pd.DataFrame:
    preds = sorted(pr_dir.glob("*.nii*"))
    jobs = []
    for p in preds:
        case = strip_nii(p.name)
        gt = nii_file(gt_dir, case)
        if gt is not None:
            jobs.append((cohort, case, gt, p))
    print(f"  [{cohort}] {len(preds)} predictions, {len(jobs)} with ground truth", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(case_metrics, jobs, chunksize=2))
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cohort, d in df.groupby("cohort"):
        r = {"cohort": cohort, "n": len(d)}
        for name in REGIONS:
            r[f"{name} Dice"] = f"{100 * d[f'{name}_dice'].mean():.2f} ± {100 * d[f'{name}_dice'].std():.2f}"
            r[f"{name} HD95"] = f"{d[f'{name}_hd95'].median():.2f}"
        no_ed = d[d["ED_vol_gt"] == 0]
        r["mean Dice"] = f"{100 * d[[f'{n}_dice' for n in REGIONS]].mean(axis=1).mean():.2f}"
        r["spurious ED mL (all, median)"] = f"{d['ED_spurious_ml'].median():.2f}"
        r["cases w/o GT edema"] = len(no_ed)
        r["spurious ED mL (no-edema cases, median)"] = f"{no_ed['ED_spurious_ml'].median():.2f}" if len(no_ed) else "-"
        out.append(r)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ prediction
def predict(model_dir: Path, checkpoint: str, img_dir: Path, out_dir: Path, device: str, tta: bool,
            max_cases: int | None) -> None:
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    cases = sorted({strip_nii(p.name)[:-5] for p in img_dir.glob("*_0000.nii*")})
    if max_cases:
        cases = cases[:max_cases]
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cases if not (out_dir / f"{c}.nii.gz").exists()]
    print(f"  {len(cases)} cases, {len(todo)} to predict -> {out_dir}", flush=True)
    if not todo:
        return
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=tta,
                           perform_everything_on_device=(device == "cuda"),
                           device=torch.device(device), verbose=False, allow_tqdm=True)
    pred.initialize_from_trained_model_folder(str(model_dir), use_folds=(0,), checkpoint_name=checkpoint)
    inputs = [[str(nii_file(img_dir, f"{c}_{i:04d}")) for i in range(4)] for c in todo]
    outputs = [str(out_dir / c) for c in todo]
    pred.predict_from_files(inputs, outputs, save_probabilities=False, overwrite=False,
                            num_processes_preprocessing=2, num_processes_segmentation_export=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=Path)
    ap.add_argument("--checkpoint", default="checkpoint_final.pth")
    ap.add_argument("--cohort", action="append", required=True,
                    help="NAME=<dataset root>  (or NAME=<gt_dir>::<pred_dir> with --metrics_only)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tta", action="store_true", help="mirroring test-time augmentation (slow)")
    ap.add_argument("--max_cases", type=int)
    ap.add_argument("--metrics_only", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    frames = []
    for spec in args.cohort:
        name, _, root = spec.partition("=")
        if args.metrics_only:
            gt_s, _, pr_s = root.partition("::")
            gt_dir, pr_dir = Path(gt_s), Path(pr_s)
        else:
            img_dir, gt_dir = cohort_dirs(Path(root))
            pr_dir = args.out / "pred" / name
            print(f"[{name}] images {img_dir} | labels {gt_dir}", flush=True)
            predict(args.model_dir, args.checkpoint, img_dir, pr_dir, args.device, args.tta, args.max_cases)
        df = compute_metrics(name, gt_dir, pr_dir, args.workers)
        print(f"[{name}] metrics for {len(df)} cases", flush=True)
        frames.append(df)

    allm = pd.concat(frames, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    allm.to_csv(args.out / "per_case_metrics.csv", index=False)
    summ = summarise(allm)
    summ.to_csv(args.out / "summary.csv", index=False)
    (args.out / "run_info.json").write_text(json.dumps(
        {"model_dir": str(args.model_dir), "checkpoint": args.checkpoint, "tta": args.tta,
         "max_cases": args.max_cases}, indent=1))
    pd.set_option("display.width", 250)
    print(summ.T.to_string(header=False))


if __name__ == "__main__":
    main()
