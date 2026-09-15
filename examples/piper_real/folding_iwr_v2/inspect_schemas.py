from huggingface_hub import hf_hub_download
import json, pandas as pd, numpy as np, os
TOKEN=os.environ.get("HF_TOKEN")
def info(repo, ep):
    print(f"\n########## {repo} ##########")
    ij=hf_hub_download(repo,"meta/info.json",repo_type="dataset",token=TOKEN)
    d=json.load(open(ij))
    print("episodes:",d.get("total_episodes"),"frames:",d.get("total_frames"),"fps:",d.get("fps"))
    cams=[k for k in d["features"] if "image" in k]
    print("cams/codec/shape:",{k:(d["features"][k]["shape"],d["features"][k].get("info",{}).get("video.codec")) for k in cams})
    for key in ["action","observation.state"]:
        if key in d["features"]: print(f"  {key} shape:",d["features"][key]["shape"])
    tj=hf_hub_download(repo,"meta/tasks.jsonl",repo_type="dataset",token=TOKEN)
    tasks=[json.loads(l) for l in open(tj)]
    print("  tasks:",[t.get("task") for t in tasks][:5])
    # one parquet
    pq=hf_hub_download(repo,f"data/chunk-000/episode_{ep:06d}.parquet",repo_type="dataset",token=TOKEN)
    df=pd.read_parquet(pq)
    print("  parquet cols:",list(df.columns))
    a=np.stack(df["action"].to_numpy()); s=np.stack(df["observation.state"].to_numpy())
    print(f"  ep{ep}: len={len(df)} action_dim={a.shape[1]} state_dim={s.shape[1]}")
    print("  action[0]:",np.round(a[0],3).tolist())
info("axiboai/piper_laundry_calibrated_v2",0)
info("Ishan-Axibo/piperx_flatten_merged",148)
info("axiboai/folding_corrections_v2",0)
