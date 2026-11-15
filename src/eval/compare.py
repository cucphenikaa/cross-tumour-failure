"""Paired, patient-level comparison of two models from their per_case_metrics.csv files.

For every cohort x subgroup x metric:
  mean A, mean B, mean difference (B - A) with a 95% bootstrap CI (resampling patients),
  Wilcoxon signed-rank p-value, Holm-adjusted across all primary tests, and win/tie/loss counts.

Primary endpoints (pre-specified, Holm-corrected together):
  MEN  : mean Dice (WT, TC, ET averaged per case)
  PED  : WT Dice                       (label-convention-robust; see Section 6.3)
  GLI  : mean Dice                     (non-inferiority check)
Secondary (reported, uncorrected): per-region Dice, HD95, detection, spurious edema, subgroups.

Usage:
  python compare.py --a baseline=/path/eval_baseline/per_case_metrics.csv \
                    --b phenomoe=/path/eval_phenomoe/per_case_metrics.csv \
                    [--a_gli ... --b_gli ...] --out /kaggle/working/compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REG = ("WT", "TC", "ET")
N_BOOT, SEED = 10000, 0


def load(spec_main: str, spec_gli: str | None) -> tuple[str, pd.DataFrame]:
    name, _, path = spec_main.partition("=")
    df = pd.read_csv(path)
    if spec_gli:
        df = pd.concat([df, pd.read_csv(spec_gli)], ignore_index=True)
    df["mean_dice"] = df[[f"{r}_dice" for r in REG]].mean(axis=1)
    df["detected"] = (df["WT_vol_pred"] > 0).astype(float)
    return name, df


def subgroups(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Masks defined on ground truth only, so they are identical for both models."""
    no_ed = df["ED_vol_gt"] == 0
    small = df["WT_vol_gt"] <= 5.0
    return {"all": pd.Series(True, index=df.index), "no edema": no_ed, "WT <= 5 mL": small,
            "no edema & WT <= 5 mL": no_ed & small, "edema present": ~no_ed}


def paired_stats(a: np.ndarray, b: np.ndarray, higher_better: bool) -> dict:
    d = b - a
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    try:
        p = wilcoxon(b, a, zero_method="wilcox").pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    better = d > 0 if higher_better else d < 0
    worse = d < 0 if higher_better else d > 0
    return {"mean_A": a.mean(), "mean_B": b.mean(), "diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
            "p": p, "wins": int(better.sum()), "ties": int((d == 0).sum()), "losses": int(worse.sum())}


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m, running, out = len(p), 0.0, pd.Series(index=p.index, dtype=float)
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        out[i] = running
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="NAME=per_case_metrics.csv (reference model)")
    ap.add_argument("--b", required=True, help="NAME=per_case_metrics.csv (new model)")
    ap.add_argument("--a_gli", help="GLI per_case_metrics.csv of model A (optional)")
    ap.add_argument("--b_gli", help="GLI per_case_metrics.csv of model B (optional)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    na, A = load(args.a, args.a_gli)
    nb, B = load(args.b, args.b_gli)
    m = A.merge(B, on=["cohort", "case"], suffixes=("_A", "_B"))
    print(f"A = {na}, B = {nb}; paired cases per cohort: {m.groupby('cohort').size().to_dict()}")

    metrics = [("mean_dice", True), *[(f"{r}_dice", True) for r in REG], *[(f"{r}_hd95", False) for r in REG],
               ("detected", True), ("ED_spurious_ml", False)]
    primary = {("MEN", "all", "mean_dice"), ("PED", "all", "WT_dice"), ("GLIval", "all", "mean_dice")}
    rows = []
    for cohort, g in m.groupby("cohort"):
        gt = g.rename(columns={"ED_vol_gt_A": "ED_vol_gt", "WT_vol_gt_A": "WT_vol_gt"})
        for sg, mask in subgroups(gt).items():
            sub = g[mask.values]
            if len(sub) < 5:
                continue
            for met, hb in metrics:
                s = paired_stats(sub[f"{met}_A"].to_numpy(float), sub[f"{met}_B"].to_numpy(float), hb)
                rows.append({"cohort": cohort, "subgroup": sg, "metric": met, "n": len(sub),
                             "primary": (cohort, sg, met) in primary, **s})
    res = pd.DataFrame(rows)
    res["p_holm"] = np.nan
    pm = res["primary"]
    if pm.any():
        res.loc[pm, "p_holm"] = holm(res.loc[pm, "p"]).values
    args.out.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out / "paired_comparison.csv", index=False)

    def fmt(r, scale):
        return (f"{scale * r.mean_A:.2f} | {scale * r.mean_B:.2f} | {scale * r['diff']:+.2f} "
                f"[{scale * r.ci_lo:+.2f}, {scale * r.ci_hi:+.2f}] | {r.p:.3g}"
                + (f" (Holm {r.p_holm:.3g})" if r.primary else "") + f" | {r.wins}/{r.ties}/{r.losses}")

    lines = [f"# {nb} vs {na}: paired patient-level comparison", "",
             "Dice in %, HD95 in mm, detection in %, spurious edema in mL. Difference = B - A with 95% bootstrap CI;",
             "p = Wilcoxon signed-rank; W/T/L = cases where B is better / tied / worse.", "",
             f"| Cohort | Subgroup | Metric | n | {na} | {nb} | Diff [95% CI] | p | W/T/L |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    show = res[(res.subgroup == "all") | res.metric.isin(["mean_dice", "WT_dice", "detected"])]
    for _, r in show.iterrows():
        scale = 100 if r.metric.endswith("dice") or r.metric == "detected" else 1
        star = " **(primary)**" if r.primary else ""
        lines.append(f"| {r.cohort} | {r.subgroup} | {r.metric}{star} | {r.n} | " + fmt(r, scale) + " |")
    (args.out / "paired_comparison.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
