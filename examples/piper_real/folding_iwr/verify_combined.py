import json, glob, numpy as np
OUT="/data/datasets/axiboai/piper_folding_iwr_v1"
info=json.load(open(OUT+"/meta/info.json"))
eps=[json.loads(l) for l in open(OUT+"/meta/episodes.jsonl")]
sts=[json.loads(l) for l in open(OUT+"/meta/episodes_stats.jsonl")]
pq=glob.glob(OUT+"/data/**/*.parquet",recursive=True)
mp4=glob.glob(OUT+"/videos/**/*.mp4",recursive=True)
print("info: total_eps=%d frames=%d videos=%d chunks=%d split=%s"%(
    info["total_episodes"],info["total_frames"],info["total_videos"],info["total_chunks"],info["splits"]))
print("cam_front shape/codec:", info["features"]["observation.images.cam_front"]["shape"],
      info["features"]["observation.images.cam_front"]["info"]["video.codec"])
print("counts: episodes.jsonl=%d episodes_stats=%d parquet=%d mp4=%d (expect 345/345/345/1035)"%(
    len(eps),len(sts),len(pq),len(mp4)))
print("episode_index range:", eps[0]["episode_index"], "..", eps[-1]["episode_index"], "unique:", len(set(e["episode_index"] for e in eps)))
print("stats episode_index range:", sts[0]["episode_index"], "..", sts[-1]["episode_index"], "unique:", len(set(s["episode_index"] for s in sts)))
# lerobot load
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    ds=LeRobotDataset("axiboai/piper_folding_iwr_v1", root=OUT)
    print("LeRobotDataset loaded: %d frames, %d episodes"%(ds.num_frames, ds.num_episodes))
    import torch
    for tag,fi in [("base ep0", 0), ("iv copy1 ep277", int(ds.episode_data_index["from"][277])), ("iv copy2 ep311", int(ds.episode_data_index["from"][311]))]:
        it=ds[fi]
        img=it["observation.images.cam_front"]
        print("  %s: img %s dtype=%s range[%.2f,%.2f] state=%s action=%s"%(
            tag, tuple(img.shape), img.dtype, float(img.min()), float(img.max()),
            tuple(it["observation.state"].shape), tuple(it["action"].shape)))
    print("VERIFY_OK")
except Exception as e:
    import traceback; traceback.print_exc(); print("VERIFY_FAIL")
