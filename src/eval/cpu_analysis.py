"""CPU-only analyses for the revision (no GPU, no retraining). Reuses saved predictions.

  1. Lesion-wise metrics: re-implementation of the official BraTS 2023 evaluator (dilation 1 for MEN, 3 for PED; GT
     lesions <= 50 voxels dropped; FP not size-filtered; 374 mm penalty), plus a cross-check with the current official
     BraTS_evaluation package (panoptica) when it is installed.
  2. Protocol-harmonised evaluation: regions that do not depend on whether non-enhancing tumour is called core or
     edema - WT (1|2|3), NE (1|2), ET (3) - next to the standard TC.
  3. Intensity analysis of ground-truth label-1 voxels (z-scored within the brain per case and modality):
     label 1 predicted as edema vs as core, against GT edema and GT enhancing tumour of the same cohort, plus a
     case-grouped cross-validated AUC for separating label 1 from label 2 by the four MR intensities alone.
  4. Figures: Fig. 5 (qualitative examples), Fig. 6 (lesion-wise sensitivity and FP by size),
     Fig. S1 (intensity analysis), Fig. S2 (standard vs harmonised Dice).

Inputs are discovered under --input: test datasets (images/*_000k.nii.gz + labels/*.nii.gz) named with
"men300" / "ped" and prediction folders .../eval_<name>/pred/<COHORT>/*.nii.gz.

Usage: python cpu_analysis.py --input /kaggle/input --out /kaggle/working/cpu [--workers 4] [--max_cases N]
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

MIN_VOX, HD_MAX, HD_LW = 50, 373.13, 374.0
DIL = {"MEN": 1, "PED": 3}                   # official BraTS 2023 dilation factors per sub-challenge
STRUCT = np.ones((3, 3, 3), bool)
REGIONS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}
MODEL_NAMES = {"baseline": "E1", "phenomoe": "E2 MoE", "e3": "E3 synthesis", "smalllesion": "E3 synthesis",
               "swin": "Swin UNETR", "fs0": "k=0", "fs10": "k=10", "fs10b": "k=10b", "fs10c": "k=10c",
               "fs30": "k=30", "fs100": "k=100"}
ORDER = ["E1", "Swin UNETR", "E2 MoE", "E3 synthesis", "k=0", "k=10", "k=10b", "k=10c", "k=30", "k=100"]
MODS = ["T1", "T1c", "T2", "FLAIR"]          # nnU-Net channels _0000.._0003 (t1n, t1c, t2w, t2f)
SIZE_BINS = [0, 0.5, 2, 5, 20, 1e9]
SIZE_LABELS = ["< 0.5", "0.5–2", "2–5", "5–20", "> 20"]


# ------------------------------------------------------------------ discovery
def nii(folder: Path, stem: str) -> Path | None:
    for ext in (".nii.gz", ".nii"):
        if (folder / f"{stem}{ext}").exists():
            return folder / f"{stem}{ext}"
    return None


def stem(p: Path) -> str:
    return re.sub(r"\.nii(\.gz)?$", "", p.name)


def find_cohorts(root: Path) -> dict:
    out = {}
    for name, key in (("MEN", "men300"), ("PED", "ped-test")):
        for d in sorted(root.rglob(f"*{key}*")):
            if not d.is_dir():
                continue
            labs = sorted(d.rglob("labels"))
            imgs = sorted({p.parent for p in d.rglob("*_0000.nii*")})
            if labs and imgs:
                out[name] = {"labels": labs[0], "images": imgs[0]}
                break
    return out


def find_models(root: Path) -> dict:
    models = {}
    for pd_ in sorted(root.rglob("pred")):
        m = re.match(r"eval_(.+)$", pd_.parent.name)
        if not m or not pd_.is_dir():
            continue
        key = m.group(1).lower()
        name = MODEL_NAMES.get(key, key)
        for coh in ("MEN", "PED"):
            c = pd_ / coh
            if c.is_dir() and any(c.glob("*.nii*")):
                models.setdefault(name, {}).setdefault(coh, c)
    return dict(sorted(models.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99))


def load(p: Path) -> np.ndarray:
    return np.asarray(nib.load(p).dataobj).astype(np.uint8)


# ------------------------------------------------------------------ metrics
def dice(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = a.sum(), b.sum()
    if sa == 0 and sb == 0:
        return 1.0
    return 2.0 * np.logical_and(a, b).sum() / (sa + sb)


def hd95(a: np.ndarray, b: np.ndarray) -> float:
    if not a.any() or not b.any():
        return 0.0 if not a.any() and not b.any() else HD_MAX
    idx = np.argwhere(a | b)
    lo, hi = np.maximum(idx.min(0) - 2, 0), idx.max(0) + 3
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    a, b = a[sl], b[sl]
    sa = a & ~ndi.binary_erosion(a, STRUCT, border_value=0)
    sb = b & ~ndi.binary_erosion(b, STRUCT, border_value=0)
    da = ndi.distance_transform_edt(~sa)
    db = ndi.distance_transform_edt(~sb)
    d = np.concatenate([db[sa], da[sb]])
    return float(np.percentile(d, 95))


def lesionwise(gt: np.ndarray, pr: np.ndarray, dil: int):
    """Re-implementation of the official BraTS 2023 lesion-wise evaluator (rachitsaluja/BraTS-2023-Metrics):
    GT lesions = 26-connected components merged when their dilations (dil iterations, 3x3x3) touch; a lesion is
    matched (TP) by every predicted component that intersects its dilated mask; unmatched predicted components are
    FP (no size filter); GT lesions of <= MIN_VOX voxels are dropped from TP and FN; lesion-wise Dice / HD95 =
    sum over kept GT lesions / (kept lesions + FP), with HD95 = 374 mm for FN and FP."""
    lab_d, _ = ndi.label(ndi.binary_dilation(gt, STRUCT, iterations=dil), STRUCT)
    gl = lab_d * gt                                        # merged lesion ids restricted to GT voxels
    ids = [i for i in np.unique(gl) if i]
    sizes = ndi.sum_labels(gt, gl, ids) if ids else []
    pl, npc = ndi.label(pr, STRUCT)
    psz = ndi.sum_labels(pr, pl, range(1, npc + 1)) if npc else []
    lesions, used = [], set()
    for i, sz in zip(ids, sizes):
        hit = np.unique(pl[(lab_d == i) & (pl > 0)])      # lab_d == i is the dilation of the merged lesion
        used.update(hit.tolist())
        if sz <= MIN_VOX:
            continue
        g = gl == i
        p = np.isin(pl, hit) if hit.size else np.zeros_like(g)
        d = dice(g, p)
        lesions.append({"vol_ml": sz / 1000, "dice": d, "matched": bool(hit.size), "detected": d > 0,
                        "hd95": hd95(g, p) if p.any() else HD_LW})
    fp = [psz[j - 1] / 1000 for j in range(1, npc + 1) if j not in used]
    n = len(lesions) + len(fp)
    lw_dice = 1.0 if n == 0 else sum(l["dice"] for l in lesions) / n
    lw_hd = 0.0 if n == 0 else (sum(min(l["hd95"], HD_LW) for l in lesions) + HD_LW * len(fp)) / n
    return lesions, {"n_gt_lesions": len(lesions), "n_fp": len(fp), "fp_vol_ml": float(sum(fp)),
                     "fp_vols": ";".join(f"{v:.4f}" for v in fp), "lw_dice": lw_dice, "lw_hd95": lw_hd}


_PANOPTICA = {}


def official_job(args):
    """Cross-check with the current official BraTS evaluation package (BraTS_evaluation / panoptica):
    connected components, maximum bipartite matching on Dice > 0, no dilation, no size threshold."""
    coh, case, gt_path, model, pred_path = args
    if "ev" not in _PANOPTICA:
        from importlib.resources import files
        from panoptica import Panoptica_Evaluator
        _PANOPTICA["ev"] = Panoptica_Evaluator.load_from_config(
            str(files("brats_evaluation").joinpath("configs/config_MenPre.yaml")))
    row = {"cohort": coh, "case": case, "model": model}
    try:                                  # arrays, not paths: GT (.nii) and predictions (.nii.gz) differ in ending
        res = _PANOPTICA["ev"].evaluate(load(pred_path), load(gt_path))
    except Exception as e:                # never let the cross-check break the main analysis
        row["error"] = repr(e)[:300]
        return row
    for grp, r in res.items():
        d = r.to_dict(True)
        for k in ("n_ref_instances", "n_pred_instances", "tp", "fp", "fn", "sq_dsc", "pq_dsc"):
            row[f"{grp.upper()}_{k}"] = d.get(k)
    return row


def case_job(args):
    coh, case, gt_path, preds = args
    gt = load(gt_path)
    lrows, crows = [], []
    for model, pp in preds.items():
        pr = load(pp)
        row = {"cohort": coh, "case": case, "model": model}
        for r, labs in REGIONS.items():
            g, p = np.isin(gt, labs), np.isin(pr, labs)
            row[f"{r}_dice"] = dice(g, p)
            les, summ = lesionwise(g, p, DIL[coh])
            for k, v in summ.items():
                row[f"{r}_{k}"] = v
            for i, l in enumerate(les):
                lrows.append({"cohort": coh, "case": case, "model": model, "region": r, "lesion": i + 1, **l})
        row["NE_dice"] = dice(np.isin(gt, (1, 2)), np.isin(pr, (1, 2)))
        row["WT_vol_gt_ml"] = np.isin(gt, (1, 2, 3)).sum() / 1000
        crows.append(row)
    return crows, lrows


# ------------------------------------------------------------------ intensities
def zscore_brain(img: np.ndarray, brain: np.ndarray) -> np.ndarray:
    v = img[brain]
    return (img - v.mean()) / (v.std() + 1e-6)


def intensity_job(args):
    coh, case, gt_path, img_dir, pred_path, n_sample, seed = args
    gt, pr = load(gt_path), load(pred_path)
    imgs = [np.asarray(nib.load(nii(img_dir, f"{case}_{k:04d}")).dataobj, dtype=np.float32) for k in range(4)]
    brain = np.any([im != 0 for im in imgs], axis=0)
    Z = np.stack([zscore_brain(im, brain) for im in imgs], -1)
    groups = {"L1 → edema": (gt == 1) & (pr == 2), "L1 → core": (gt == 1) & np.isin(pr, (1, 3)),
              "L1 → background": (gt == 1) & (pr == 0), "GT edema": gt == 2, "GT enhancing": gt == 3}
    med, samp = [], []
    rng = np.random.default_rng(seed)
    for g, m in groups.items():
        n = int(m.sum())
        if n < 20:
            continue
        v = Z[m]
        med.append({"cohort": coh, "case": case, "group": g, "n_vox": n,
                    **{mod: float(np.median(v[:, i])) for i, mod in enumerate(MODS)}})
        if g in ("L1 → edema", "L1 → core", "GT edema"):
            pick = rng.choice(n, min(n, n_sample), replace=False)
            for row in v[pick]:
                samp.append((coh, case, g, *row.tolist()))
    return med, samp


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/cpu"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_cases", type=int)
    ap.add_argument("--n_sample", type=int, default=400, help="voxels per case and group for the AUC analysis")
    ap.add_argument("--skip_intensity", action="store_true")
    ap.add_argument("--official", action="store_true", help="also run the official BraTS_evaluation package")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cohorts, models = find_cohorts(args.input), find_models(args.input)
    print("cohorts:", {k: str(v["labels"]) for k, v in cohorts.items()})
    for m, d in models.items():
        print(f"model {m:13s}", {c: str(p) for c, p in d.items()})
    assert "E1" in models, "E1 predictions (eval_baseline/pred) not found"
    ped_test = sorted(args.input.rglob("ped_test_cases.txt"))
    ped69 = set(ped_test[0].read_text().split()) if ped_test else set()

    jobs = []
    for coh, d in cohorts.items():
        cases = sorted(stem(p) for p in d["labels"].glob("*.nii*"))[: args.max_cases]
        for c in cases:
            preds = {m: nii(md[coh], c) for m, md in models.items() if coh in md and nii(md[coh], c)}
            if preds:
                jobs.append((coh, c, nii(d["labels"], c), preds))
    print(f"{len(jobs)} cases x up to {len(models)} models", flush=True)
    crows, lrows = [], []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (c, l) in enumerate(ex.map(case_job, jobs, chunksize=2)):
            crows += c
            lrows += l
            if i % 50 == 0:
                print(f"  lesion-wise {i + 1}/{len(jobs)}", flush=True)
    cases = pd.DataFrame(crows)
    cases["ped69"] = cases.case.isin(ped69)
    les = pd.DataFrame(lrows)
    cases.to_csv(args.out / "casewise_lesionwise.csv", index=False)
    les.to_csv(args.out / "lesions.csv", index=False)

    if args.official:
        ojobs = [(coh, c, gp, m, pp) for coh, c, gp, pr in jobs for m, pp in pr.items()]
        orows = []
        with ProcessPoolExecutor(args.workers) as ex:
            for i, r in enumerate(ex.map(official_job, ojobs, chunksize=4)):
                orows.append(r)
                if i % 200 == 0:
                    print(f"  official evaluator {i + 1}/{len(ojobs)}", flush=True)
        pd.DataFrame(orows).to_csv(args.out / "official_panoptica.csv", index=False)
    if args.skip_intensity:
        json.dump({"MIN_VOX": MIN_VOX, "DIL": DIL}, open(args.out / "run_info_lesionwise.json", "w"))
        print("ANALYSIS DONE (lesion-wise only)")
        return
    # intensity analysis: E1 predictions, all cohorts
    ijobs = [(coh, c, gp, cohorts[coh]["images"], pr["E1"], args.n_sample, k)
             for k, (coh, c, gp, pr) in enumerate(jobs) if "E1" in pr]
    med, samp = [], []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (m, s) in enumerate(ex.map(intensity_job, ijobs, chunksize=2)):
            med += m
            samp += s
            if i % 50 == 0:
                print(f"  intensities {i + 1}/{len(ijobs)}", flush=True)
    pd.DataFrame(med).to_csv(args.out / "label1_intensity_case_medians.csv", index=False)
    pd.DataFrame(samp, columns=["cohort", "case", "group", *MODS]).to_csv(args.out / "label1_voxel_sample.csv.gz", index=False)
    json.dump({"models": {m: {c: str(p) for c, p in d.items()} for m, d in models.items()},
               "cohorts": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in cohorts.items()},
               "MIN_VOX": MIN_VOX, "DIL": DIL, "n_ped69": len(ped69)}, open(args.out / "run_info.json", "w"), indent=1)
    print("ANALYSIS DONE")


if __name__ == "__main__":
    main()
