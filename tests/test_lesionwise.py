import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src" / "eval"))
from cpu_analysis import lesionwise
def vol(): return np.zeros((60, 60, 60), bool)
def cube(a, c, r): a[c[0]-r:c[0]+r, c[1]-r:c[1]+r, c[2]-r:c[2]+r] = True; return a
def run(name, gt, pr, dil=1):
    les, s = lesionwise(gt, pr, dil)
    print(f"{name:42s} GT kept={s['n_gt_lesions']} matched={[l['matched'] for l in les]} dice={[round(l['dice'],2) for l in les]} FP={s['n_fp']} lwDice={s['lw_dice']:.2f}")
    return les, s
# 1 one GT - one pred
les, s = run("1 GT - 1 pred", cube(vol(), (20,20,20), 4), cube(vol(), (21,20,20), 4)); assert s["n_gt_lesions"]==1 and les[0]["matched"] and s["n_fp"]==0
# 2 two GT - one pred bridging both (one-to-many: both matched, official behaviour)
g = cube(cube(vol(), (20,20,15), 3), (20,20,35), 3); p = vol(); p[17:23,17:23,12:38] = True
les, s = run("2 GT - 1 pred (bridging)", g, p); assert s["n_gt_lesions"]==2 and all(l["matched"] for l in les) and s["n_fp"]==0
# 3 one GT - two preds: both join the lesion's prediction, no FP
p = cube(cube(vol(), (20,20,17), 2), (20,20,23), 2)
les, s = run("1 GT - 2 preds", cube(vol(), (20,20,20), 5), p); assert s["n_gt_lesions"]==1 and s["n_fp"]==0
# 4 pred not touching any GT -> FP; GT -> FN
les, s = run("pred far from GT", cube(vol(), (15,15,15), 4), cube(vol(), (45,45,45), 4)); assert not les[0]["matched"] and s["n_fp"]==1 and s["lw_dice"]==0
# 5 GT <= 50 voxels dropped (3^3=27); pred hitting it is NOT an FP; no GT -> lw Dice over FP only
g = vol(); g[10:13,10:13,10:13] = True; p = vol(); p[10:13,10:13,10:13] = True
les, s = run("small GT (27 vox) + pred on it", g, p); assert s["n_gt_lesions"]==0 and s["n_fp"]==0
# 6 small GT dropped from denominator while big GT kept
g = cube(vol(), (40,40,40), 4); g[5:8,5:8,5:8] = True
les, s = run("big GT + small GT (missed)", g, cube(vol(), (40,40,40), 4)); assert s["n_gt_lesions"]==1 and s["n_fp"]==0
# 7 two GT lesions 2 voxels apart merge with dil=1 (dilations touch), stay separate 6 apart
g = cube(cube(vol(), (20,20,14), 3), (20,20,25), 3)   # gap of 5 voxels
_, s1 = run("2 GT gap 5, dil 1", g, g); _, s3 = run("2 GT gap 5, dil 3", g, g, dil=3)
assert s1["n_gt_lesions"]==2 and s3["n_gt_lesions"]==1
print("ALL TESTS PASSED")
