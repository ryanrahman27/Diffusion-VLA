import json, os, glob, shutil, subprocess, sys, time
import numpy as np, pandas as pd
import imageio_ffmpeg, av
from concurrent.futures import ThreadPoolExecutor

FF = imageio_ffmpeg.get_ffmpeg_exe()
BASE="/data/datasets/axiboai/piper_laundry_calibrated_v2"
IV="/data/datasets/Ishan-Axibo/piperx_flatten_merged"
OUT="/data/datasets/axiboai/piper_folding_iwr_v1"
FPS=30; ARM=[0,1,2,3,4,5,7,8,9,10,11,12]
THR=1e-4; WIN=30; SETTLE=6; RES=224
TASK="pick towel from pile, fold and stack"
CAMS=["observation.images.cam_front","observation.images.cam_left_wrist","observation.images.cam_right_wrist"]
IV_EPS=list(range(148,182)); OVERSAMPLE=2

def vpath(root,cam,idx): return root+"/videos/chunk-000/%s/episode_%06d.mp4"%(cam,idx)
def ppath(root,idx):
    return glob.glob(root+"/data/**/episode_%06d.parquet"%idx, recursive=True)

def enc_base(args):
    src,dst=args
    os.makedirs(os.path.dirname(dst),exist_ok=True)
    cmd=[FF,"-y","-loglevel","error","-i",src,"-vf","scale=%d:%d"%(RES,RES),"-r",str(FPS),
         "-c:v","libx264","-pix_fmt","yuv420p","-crf","23","-preset","veryfast","-an",dst]
    subprocess.run(cmd,check=True,capture_output=True); return dst

def keep_mask(df):
    a=np.stack(df["action"].to_numpy())[:,ARM]
    v=np.concatenate([[1.0],np.linalg.norm(np.diff(a,axis=0),axis=1)])
    low=v<THR; keep=np.ones(len(v),bool); i=0; drops=[]
    while i<len(v):
        if low[i]:
            j=i
            while j<len(v) and low[j]: j+=1
            if j-i>=WIN: a0,b0=i+SETTLE,j-SETTLE; keep[a0:b0]=False; drops.append((a0,b0))
            i=j
        else: i+=1
    return keep,drops

def trim_iv_video(src,dst,drops):
    os.makedirs(os.path.dirname(dst),exist_ok=True)
    vf="scale=%d:%d"%(RES,RES)
    if drops:
        cond="*".join("not(between(n\\,%d\\,%d))"%(a,b-1) for a,b in drops)
        vf="select='%s',setpts=N/FRAME_RATE/TB,%s"%(cond,vf)
    cmd=[FF,"-y","-loglevel","error","-i",src,"-vf",vf,"-r",str(FPS),
         "-c:v","libx264","-pix_fmt","yuv420p","-crf","23","-preset","veryfast","-an",dst]
    subprocess.run(cmd,check=True,capture_output=True)
    return decode_count(dst)

def decode_count(path,sample=None):
    c=av.open(path); fr=[f for f in c.decode(video=0)]; c.close(); n=len(fr)
    if sample is None: return n
    idx=np.linspace(0,n-1,min(sample,n)).astype(int)
    return n,(np.stack([np.asarray(fr[k].to_image()) for k in idx]).astype(np.float32)/255.0)

def img_stat(imgs):
    m,s=imgs.mean((0,1,2)),imgs.std((0,1,2)); mn,mx=imgs.min((0,1,2)),imgs.max((0,1,2))
    g=lambda x:[[[float(v)]] for v in x]
    return dict(min=g(mn),max=g(mx),mean=g(m),std=g(s),count=[int(imgs.shape[0])])
def num_stat(a): return dict(min=a.min(0).tolist(),max=a.max(0).tolist(),mean=a.mean(0).tolist(),std=a.std(0).tolist(),count=[int(a.shape[0])])

def make_iv(i,new_idx,gidx):
    df=pd.read_parquet(ppath(IV,i)[0]).reset_index(drop=True)
    keep,drops=keep_mask(df); kdf=df[keep].reset_index(drop=True); L=len(kdf)
    kdf["episode_index"]=new_idx; kdf["frame_index"]=np.arange(L)
    kdf["timestamp"]=np.arange(L)/FPS; kdf["task_index"]=0; kdf["index"]=np.arange(gidx,gidx+L)
    pdst=OUT+"/data/chunk-000/episode_%06d.parquet"%new_idx
    os.makedirs(os.path.dirname(pdst),exist_ok=True); kdf.to_parquet(pdst)
    stats={"observation.state":num_stat(np.stack(kdf["observation.state"].to_numpy())),
           "action":num_stat(np.stack(kdf["action"].to_numpy())),
           "timestamp":num_stat(kdf["timestamp"].to_numpy().reshape(-1,1))}
    for cam in CAMS:
        n=trim_iv_video(vpath(IV,cam,i),vpath(OUT,cam,new_idx),drops)
        assert n==L,"mismatch ep%d %s %d!=%d"%(i,cam,n,L)
        _,imgs=decode_count(vpath(OUT,cam,new_idx),sample=120); stats[cam]=img_stat(imgs)
    return L,{"episode_index":new_idx,"stats":stats},{"episode_index":new_idx,"tasks":[TASK],"length":L}

def copy_iv(src_idx,new_idx,gidx,length):
    df=pd.read_parquet(OUT+"/data/chunk-000/episode_%06d.parquet"%src_idx)
    df["episode_index"]=new_idx; df["index"]=np.arange(gidx,gidx+length)
    df.to_parquet(OUT+"/data/chunk-000/episode_%06d.parquet"%new_idx)
    for cam in CAMS: shutil.copy(vpath(OUT,cam,src_idx),vpath(OUT,cam,new_idx))

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "test"
    if mode=="test":
        os.makedirs(OUT+"/data/chunk-000",exist_ok=True)
        t=time.time(); L,st,ep=make_iv(148,9000,0)
        print("TEST ep148 -> trimmed L=%d (orig 1123) in %.1fs"%(L,time.time()-t))
        print("  frame-match asserts: PASS (h264 224x224)")
        print("  cam_front mean3ch:",[round(x[0][0],3) for x in st["stats"]["observation.images.cam_front"]["mean"]])
        print("  action count:",st["stats"]["action"]["count"])
        for cam in CAMS: os.remove(vpath(OUT,cam,9000))
        os.remove(OUT+"/data/chunk-000/episode_009000.parquet")
        print("  cleaned test artifacts")

def run_full():
    import time
    t0=time.time()
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT+"/data/chunk-000", exist_ok=True); os.makedirs(OUT+"/meta", exist_ok=True)
    for m in ["info.json","episodes.jsonl","episodes_stats.jsonl","tasks.jsonl"]:
        shutil.copy(BASE+"/meta/"+m, OUT+"/meta/"+m)
    for p in sorted(glob.glob(BASE+"/data/chunk-000/*.parquet")):
        shutil.copy(p, OUT+"/data/chunk-000/"+os.path.basename(p))
    jobs=[(vpath(BASE,cam,idx),vpath(OUT,cam,idx)) for cam in CAMS for idx in range(277)]
    print("re-encoding %d base videos -> %dx%d h264..."%(len(jobs),RES,RES), flush=True)
    done=0
    with ThreadPoolExecutor(max_workers=12) as ex:
        for _ in ex.map(enc_base, jobs):
            done+=1
            if done%150==0: print("  base %d/%d (%.0fs)"%(done,len(jobs),time.time()-t0), flush=True)
    gidx=json.load(open(BASE+"/meta/info.json"))["total_frames"]
    NB=277; new_eps=[]; new_stats=[]; first_idx={}; lengths={}
    print("trimming + appending interventions...", flush=True)
    for k,i in enumerate(IV_EPS):
        nidx=NB+k; L,st,ep=make_iv(i,nidx,gidx); gidx+=L
        first_idx[i]=nidx; lengths[i]=L; new_eps.append(ep); new_stats.append(st)
    print("  copy1 block done (%d eps). oversampling x%d..."%(len(IV_EPS),OVERSAMPLE), flush=True)
    nxt=NB+len(IV_EPS)
    for rep in range(OVERSAMPLE-1):
        for k,i in enumerate(IV_EPS):
            nidx=nxt; nxt+=1; L=lengths[i]; copy_iv(first_idx[i],nidx,gidx,L); gidx+=L
            st=json.loads(json.dumps(new_stats[k])); st["episode_index"]=nidx
            new_stats.append(st); new_eps.append({"episode_index":nidx,"tasks":[TASK],"length":L})
    total_eps=nxt; total_frames=gidx
    with open(OUT+"/meta/episodes.jsonl","a") as f:
        for ep in new_eps: f.write(json.dumps(ep)+"\n")
    with open(OUT+"/meta/episodes_stats.jsonl","a") as f:
        for st in new_stats: f.write(json.dumps(st)+"\n")
    info=json.load(open(OUT+"/meta/info.json"))
    info["total_episodes"]=total_eps; info["total_frames"]=total_frames
    info["total_videos"]=total_eps*len(CAMS); info["total_chunks"]=1
    info["splits"]={"train":"0:%d"%total_eps}
    for cam in CAMS:
        ft=info["features"][cam]; ft["shape"]=[RES,RES,3]
        ft["info"]["video.height"]=RES; ft["info"]["video.width"]=RES; ft["info"]["video.codec"]="h264"
    json.dump(info, open(OUT+"/meta/info.json","w"), indent=4)
    iv_frames=sum(lengths.values())
    print("DONE: %d eps (277 base + %d iv x%d), %d frames (%.0f%% interventions) in %.0fs"%(
        total_eps, len(IV_EPS), OVERSAMPLE, total_frames, 100*iv_frames*OVERSAMPLE/total_frames, time.time()-t0), flush=True)
    print("OUT="+OUT, flush=True)

if __name__=="__main__" and len(__import__("sys").argv)>1 and __import__("sys").argv[1]=="full":
    run_full()
