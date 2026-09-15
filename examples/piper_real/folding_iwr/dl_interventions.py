import json, os, glob
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
TOKEN=os.environ.get("HF_TOKEN")
REPO="Ishan-Axibo/piperx_flatten_merged"
OUT="/data/datasets/Ishan-Axibo/piperx_flatten_merged"
os.makedirs(OUT, exist_ok=True)
api=HfApi(token=TOKEN)
fs=api.list_repo_files(REPO, repo_type="dataset")
IV=list(range(148,182))   # intervention episode indices
PAUSE_EPS={155,156,157,158,159,160,161,162,163,164,165,168,169,173,174,175}
# download meta + the 34 intervention parquet (defer videos to merge step)
want=[f for f in fs if f.startswith("meta/")]
for f in fs:
    if f.endswith(".parquet"):
        for i in IV:
            if f"episode_{i:06d}.parquet" in f: want.append(f)
print("downloading", len(want), "files (meta + 34 parquet)...", flush=True)
for f in want:
    hf_hub_download(REPO, f, repo_type="dataset", token=TOKEN, local_dir=OUT)
print("download done. analyzing pauses...", flush=True)

import pandas as pd
def speed(df):
    a=np.stack(df["action"].to_numpy())   # (T,14)
    arms=a[:, [0,1,2,3,4,5,7,8,9,10,11,12]]
    v=np.linalg.norm(np.diff(arms,axis=0),axis=1)   # per-step motion (rad)
    return np.concatenate([[0],v])
print(f"{'ep':>4} {'len':>5} {'pause_fr':>8} {'pause_%':>7} {'segs':>4}  flagged")
tot_in=tot_keep=0
rows={}
for i in IV:
    p=glob.glob(f"{OUT}/data/**/episode_{i:06d}.parquet", recursive=True)
    if not p: print(f"{i:>4}  MISSING"); continue
    df=pd.read_parquet(p[0]); v=speed(df); T=len(df)
    # pause = sustained low motion (<0.0015 rad/step) for >=12 frames (0.4s)
    THR=0.0015; WIN=12
    low=v<THR
    # count sustained-low frames
    segs=0; run=0; pause_fr=0; inrun=False
    for x in low:
        if x: run+=1
        else:
            if run>=WIN: segs+=1; pause_fr+=run
            run=0
    if run>=WIN: segs+=1; pause_fr+=run
    flag="<-- pause-ep" if i in PAUSE_EPS else ""
    print(f"{i:>4} {T:>5} {pause_fr:>8} {100*pause_fr/T:>6.1f}% {segs:>4}  {flag}")
    tot_in+=T; tot_keep+=(T-pause_fr); rows[i]=(T,pause_fr,segs)
print(f"\nTOTAL intervention frames: {tot_in}  ->  after pause-trim: ~{tot_keep}  (removed ~{tot_in-tot_keep})")
json.dump(rows, open(f"{OUT}/pause_analysis.json","w"))
