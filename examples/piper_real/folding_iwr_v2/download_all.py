import os, time
from huggingface_hub import snapshot_download
TOKEN=os.environ.get("HF_TOKEN")
t0=time.time()

# 1) base checkpoint (warm-start) -> params + assets
print("[1/3] base checkpoint pi05_piper_folding_v2 ...", flush=True)
snapshot_download("axiboai/pi05_piper_folding_v2", repo_type="model", token=TOKEN,
    local_dir="/data/sagar_recap/folding_iwr_v2/base_ckpt")
print("  done %.0fs"%(time.time()-t0), flush=True)

# 2) interventions1: piperx_flatten_merged eps 148-181 + meta
print("[2/3] interventions1 piperx_flatten_merged eps148-181 ...", flush=True)
pats=["meta/*"]
for i in range(148,182):
    pats.append(f"data/chunk-000/episode_{i:06d}.parquet")
    pats.append(f"videos/chunk-000/*/episode_{i:06d}.mp4")
snapshot_download("Ishan-Axibo/piperx_flatten_merged", repo_type="dataset", token=TOKEN,
    local_dir="/data/datasets/Ishan-Axibo/piperx_flatten_merged", allow_patterns=pats)
print("  done %.0fs"%(time.time()-t0), flush=True)

# 3) interventions2: folding_corrections_v2 (all 21)
print("[3/3] interventions2 folding_corrections_v2 (all) ...", flush=True)
snapshot_download("axiboai/folding_corrections_v2", repo_type="dataset", token=TOKEN,
    local_dir="/data/datasets/axiboai/folding_corrections_v2")
print("ALL DOWNLOADS DONE %.0fs"%(time.time()-t0), flush=True)
