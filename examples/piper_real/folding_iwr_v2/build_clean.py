#!/usr/bin/env python3
"""Folding IWR v2 dataset builder.

Sources:
  base = axiboai/piper_laundry_calibrated_v2   (277 original teleop eps)
  iv1  = Ishan-Axibo/piperx_flatten_merged     (interventions #1, eps 148-181)
  iv2  = axiboai/folding_corrections_v2         (interventions #2, all 21)

Pause trim (ALL sources): near-exact joint freeze on the 12 arm joints
(||Δaction|| < 1e-4) held >=30 frames (>=1s @30fps) -> keep first/last 6
frames (settle), drop the dead middle.

Standardize all video to 224x224 h264 yuv420p @30fps crf23.

Stage once (trim+encode every unique episode), then assemble each weighting
variant by copying base once + corrections xN.

Usage:
  build_clean.py stage
  build_clean.py assemble <out_name> <mult>
"""
import json, os, glob, shutil, subprocess, sys, time
import numpy as np, pandas as pd, av, imageio_ffmpeg
from concurrent.futures import ThreadPoolExecutor

FF = imageio_ffmpeg.get_ffmpeg_exe()
FPS=30; RES=224
ARM=[0,1,2,3,4,5,7,8,9,10,11,12]          # 12 arm joints (grippers 6,13 excluded)
THR=1e-4; WIN=30; SETTLE=6
TASK="pick towel from pile, fold and stack"
CAMS=["observation.images.cam_front","observation.images.cam_left_wrist","observation.images.cam_right_wrist"]

DS="/data/datasets"
SOURCES = [
    ("base", f"{DS}/axiboai/piper_laundry_calibrated_v2", list(range(277)),   "base"),
    ("iv1",  f"{DS}/Ishan-Axibo/piperx_flatten_merged",   list(range(148,182)),"corr"),
    ("iv2",  f"{DS}/axiboai/folding_corrections_v2",       list(range(21)),    "corr"),
]
STAGE=f"{DS}/_stage_folding_iwr_v2"

def ppath(root,idx):
    g=glob.glob(root+"/data/**/episode_%06d.parquet"%idx, recursive=True)
    return g[0] if g else None
def vpath(root,cam,idx):
    g=glob.glob(root+"/videos/**/%s/episode_%06d.mp4"%(cam,idx), recursive=True)
    return g[0] if g else None

def keep_mask(df):
    a=np.stack(df["action"].to_numpy())[:,ARM]
    v=np.concatenate([[1.0], np.linalg.norm(np.diff(a,axis=0),axis=1)])
    low=v<THR; keep=np.ones(len(v),bool); drops=[]; i=0
    while i<len(v):
        if low[i]:
            j=i
            while j<len(v) and low[j]: j+=1
            if j-i>=WIN:
                a0,b0=i+SETTLE,j-SETTLE
                keep[a0:b0]=False; drops.append((a0,b0))
            i=j
        else: i+=1
    return keep,drops

def decode_count(path,sample=None):
    c=av.open(path); fr=[f for f in c.decode(video=0)]; c.close(); n=len(fr)
    if sample is None: return n
    idx=np.linspace(0,n-1,min(sample,n)).astype(int)
    return n,(np.stack([np.asarray(fr[k].to_image()) for k in idx]).astype(np.float32)/255.0)

def encode(src,dst,drops):
    os.makedirs(os.path.dirname(dst),exist_ok=True)
    vf="scale=%d:%d"%(RES,RES)
    if drops:
        cond="*".join("not(between(n\\,%d\\,%d))"%(a,b-1) for a,b in drops)
        vf="select='%s',setpts=N/FRAME_RATE/TB,%s"%(cond,vf)
    cmd=[FF,"-y","-loglevel","error","-i",src,"-vf",vf,"-r",str(FPS),
         "-c:v","libx264","-pix_fmt","yuv420p","-crf","23","-preset","veryfast","-an",dst]
    subprocess.run(cmd,check=True,capture_output=True)

def img_stat(imgs,L):
    m,s=imgs.mean((0,1,2)),imgs.std((0,1,2)); mn,mx=imgs.min((0,1,2)),imgs.max((0,1,2))
    g=lambda x:[[[float(v)]] for v in x]
    return dict(min=g(mn),max=g(mx),mean=g(m),std=g(s),count=[int(L)])
def num_stat(a):
    return dict(min=a.min(0).tolist(),max=a.max(0).tolist(),mean=a.mean(0).tolist(),
                std=a.std(0).tolist(),count=[int(a.shape[0])])

def stage_one(args):
    tag,root,idx,kind = args
    df=pd.read_parquet(ppath(root,idx)).reset_index(drop=True); T=len(df)
    keep,drops=keep_mask(df); kdf=df[keep].reset_index(drop=True); L=len(kdf)
    sd=f"{STAGE}/{tag}"; os.makedirs(sd,exist_ok=True)
    kdf.to_parquet(f"{sd}/ep{idx:06d}.parquet")
    stats={"observation.state":num_stat(np.stack(kdf["observation.state"].to_numpy())),
           "action":num_stat(np.stack(kdf["action"].to_numpy())),
           "timestamp":num_stat((np.arange(L)/FPS).reshape(-1,1))}
    for cam in CAMS:
        dst=f"{sd}/{cam}/ep{idx:06d}.mp4"
        encode(vpath(root,cam,idx),dst,drops)
        n,imgs=decode_count(dst,sample=120)
        assert n==L, f"{tag} ep{idx} {cam}: video {n} != parquet {L}"
        stats[cam]=img_stat(imgs,L)
    json.dump({"stats":stats}, open(f"{sd}/ep{idx:06d}.stats.json","w"))
    return dict(tag=tag,idx=idx,kind=kind,T=T,L=L,removed=T-L,drops=len(drops))

def cmd_stage():
    shutil.rmtree(STAGE, ignore_errors=True); os.makedirs(STAGE,exist_ok=True)
    jobs=[]
    for tag,root,eps,kind in SOURCES:
        for i in eps: jobs.append((tag,root,i,kind))
    print(f"staging {len(jobs)} unique episodes (trim+encode) ...", flush=True)
    t0=time.time(); res=[]; done=0
    with ThreadPoolExecutor(max_workers=12) as ex:
        for r in ex.map(stage_one, jobs):
            res.append(r); done+=1
            if done%40==0: print(f"  {done}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(f"{STAGE}/_report.json","w"), indent=2)
    # report per source
    print("\n===== TRIM REPORT =====")
    for tag,_,_,_ in SOURCES:
        rs=[r for r in res if r["tag"]==tag]
        Tt=sum(r["T"] for r in rs); Lt=sum(r["L"] for r in rs)
        hi=sorted(rs,key=lambda r:-(r["removed"]/max(1,r["T"])))[:3]
        print(f"{tag:5s}: {len(rs):3d} eps  {Tt:>7d} -> {Lt:>7d} frames  removed {Tt-Lt:>6d} ({100*(Tt-Lt)/Tt:.1f}%)")
        for r in hi: print(f"        worst ep{r['idx']}: {r['T']}->{r['L']} ({100*r['removed']/max(1,r['T']):.0f}%)")
    print("staged OK in %.0fs"%(time.time()-t0), flush=True)

def _load_stage(tag,idx):
    sd=f"{STAGE}/{tag}"
    df=pd.read_parquet(f"{sd}/ep{idx:06d}.parquet")
    st=json.load(open(f"{sd}/ep{idx:06d}.stats.json"))["stats"]
    return df,st

def cmd_assemble(name,mult):
    mult=int(mult)
    OUT=f"{DS}/axiboai/{name}"
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT+"/data/chunk-000",exist_ok=True); os.makedirs(OUT+"/meta",exist_ok=True)
    for cam in CAMS: os.makedirs(OUT+f"/videos/chunk-000/{cam}",exist_ok=True)
    # assembly order: all base once, then corrections repeated `mult` times
    base=[(t,i) for t,_,eps,k in SOURCES if k=="base" for i in eps]
    corr=[(t,i) for t,_,eps,k in SOURCES if k=="corr" for i in eps]
    order=list(base)
    for _ in range(mult): order+=list(corr)
    eidx=0; gidx=0; episodes=[]; stats=[]; corr_frames=0; base_frames=0
    for (tag,idx) in order:
        df,st=_load_stage(tag,idx); L=len(df)
        df=df.copy()
        df["episode_index"]=eidx; df["frame_index"]=np.arange(L)
        df["timestamp"]=np.arange(L)/FPS; df["task_index"]=0
        df["index"]=np.arange(gidx,gidx+L)
        df.to_parquet(OUT+f"/data/chunk-000/episode_{eidx:06d}.parquet")
        for cam in CAMS:
            shutil.copy(f"{STAGE}/{tag}/{cam}/ep{idx:06d}.mp4",
                        OUT+f"/videos/chunk-000/{cam}/episode_{eidx:06d}.mp4")
        stx={"episode_index":eidx,"stats":st}
        stats.append(stx); episodes.append({"episode_index":eidx,"tasks":[TASK],"length":L})
        if tag=="base": base_frames+=L
        else: corr_frames+=L
        gidx+=L; eidx+=1
    # meta
    with open(OUT+"/meta/tasks.jsonl","w") as f:
        f.write(json.dumps({"task_index":0,"task":TASK})+"\n")
    with open(OUT+"/meta/episodes.jsonl","w") as f:
        for e in episodes: f.write(json.dumps(e)+"\n")
    with open(OUT+"/meta/episodes_stats.jsonl","w") as f:
        for s in stats: f.write(json.dumps(s)+"\n")
    info=json.load(open(f"{DS}/axiboai/piper_laundry_calibrated_v2/meta/info.json"))
    info["total_episodes"]=eidx; info["total_frames"]=gidx
    info["total_videos"]=eidx*len(CAMS); info["total_chunks"]=1
    info["total_tasks"]=1; info["splits"]={"train":"0:%d"%eidx}
    for cam in CAMS:
        ft=info["features"][cam]; ft["shape"]=[RES,RES,3]
        ft["info"]["video.height"]=RES; ft["info"]["video.width"]=RES
        ft["info"]["video.codec"]="h264"; ft["info"]["video.fps"]=FPS
    json.dump(info, open(OUT+"/meta/info.json","w"), indent=4)
    tot=gidx
    print(f"ASSEMBLED {name}: {eidx} eps, {tot} frames")
    print(f"  base {base_frames} + corr {corr_frames} (x{mult}) -> corrections = {100*corr_frames/tot:.1f}% of frames")
    print(f"  OUT={OUT}")

if __name__=="__main__":
    if sys.argv[1]=="stage": cmd_stage()
    elif sys.argv[1]=="assemble": cmd_assemble(sys.argv[2], sys.argv[3])
