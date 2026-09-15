import glob, numpy as np, pandas as pd
OUT="/data/datasets/Ishan-Axibo/piperx_flatten_merged"
PAUSE={155,156,157,158,159,160,161,162,163,164,165,168,169,173,174,175}
ARM=[0,1,2,3,4,5,7,8,9,10,11,12]
THR=1e-4   # near-exact joint freeze (teleop held still)
WIN=30     # >=1.0s @30fps

def trim(df):
    a=np.stack(df["action"].to_numpy())[:, ARM]
    v=np.concatenate([[1.0], np.linalg.norm(np.diff(a,axis=0),axis=1)])
    low=v<THR
    keep=np.ones(len(v),bool); i=0; segs=0; removed=0
    while i<len(v):
        if low[i]:
            j=i
            while j<len(v) and low[j]: j+=1
            run=j-i
            if run>=WIN:
                # keep first 6 + last 6 frames of the hold (settle), drop the dead middle
                keep[i+6:j-6]=False; segs+=1; removed+=max(0,run-12)
            i=j
        else: i+=1
    return keep, segs, removed

print("  ep   len  segs  removed  keep%  flag")
tot=tk=0; fl_rm=0; ot_rm=0
for i in range(148,182):
    p=glob.glob(OUT+"/data/**/episode_%06d.parquet"%i,recursive=True)
    if not p: continue
    df=pd.read_parquet(p[0]); keep,segs,rm=trim(df); T=len(df)
    fl = i in PAUSE
    print("%4d %5d %5d %8d %6.1f%%  %s"%(i,T,segs,rm,100*keep.sum()/T,"<--" if fl else ""))
    tot+=T; tk+=keep.sum()
    if fl: fl_rm+=rm
    else: ot_rm+=rm
print()
print("TOTAL: %d -> %d  (removed %d, %.1f%%)"%(tot,tk,tot-tk,100*(tot-tk)/tot))
print("removed from FLAGGED 16:", fl_rm, " | from other 18:", ot_rm)
