import numpy as np, pandas as pd
d = pd.read_csv("rq2/out/rq2_cases.csv")
Z = ["z_log_vol_wt","z_frac_et_wt","z_frac_ed_wt","z_et_solidity","z_log_n_lesions","z_wt_frac_near_edge","z_rel_cc","z_rel_lateral","z_t1c_tc_med","z_flair_wt_med"]
d["pid"] = d.case.str.rsplit("-", n=1).str[0]
def design(df, full):
    X = [np.ones(len(df)), (df.cohort=="MEN").astype(float), (df.cohort=="PED").astype(float)]
    if full: X += [df[z].to_numpy() for z in Z] + [df.z_log_vol_wt.to_numpy()**2]
    return np.column_stack(X)
def att(df, y="mean_dice"):
    yy = df[y].to_numpy()*100
    b0 = np.linalg.lstsq(design(df,0), yy, rcond=None)[0]; b1 = np.linalg.lstsq(design(df,1), yy, rcond=None)[0]
    return b0[1], b1[1], b0[2], b1[2], 100*(1-b1[1]/b0[1]), 100*(1-b1[2]/b0[2])
print("point", np.round(att(d),2), "n patients GLIval", d[d.cohort=="GLIval"].pid.nunique())
rng = np.random.default_rng(0)
groups = {c: [g.index.to_numpy() for _, g in d[d.cohort==c].groupby("pid")] for c in d.cohort.unique()}
for mode in ["case", "patient"]:
    res = []
    for b in range(2000):
        idx = []
        for c in d.cohort.unique():
            if mode == "case":
                ii = d.index[d.cohort==c].to_numpy(); idx.append(rng.choice(ii, len(ii)))
            else:
                gs = groups[c]; pick = rng.integers(0, len(gs), len(gs)); idx.append(np.concatenate([gs[k] for k in pick]))
        res.append(att(d.loc[np.concatenate(idx)]))
    r = np.array(res)
    for j, nm in [(4,"MEN att%"),(5,"PED att%"),(0,"MEN b0"),(2,"PED b0")]:
        print(mode, nm, np.round(np.percentile(r[:,j],[2.5,50,97.5]),1), "min/max", np.round(r[:,j].min(),1), np.round(r[:,j].max(),1))
    print(mode, "MEN att >100%:", (r[:,4]>100).mean(), " b0 closest to 0:", r[:,0].max().round(1), r[:,2].max().round(1))
