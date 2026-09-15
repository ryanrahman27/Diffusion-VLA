#!/usr/bin/env python3
"""Collect live training data for the folding IWR v2 A/B artifact -> one JSON blob to stdout."""
import glob, json, os, re, subprocess, shutil
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb

REPO = "/data/sagar_recap/piperx-openpi"
LOGD = "/data/sagar_recap/folding_iwr_v2/logs"
RUNS = {
    "x2": {"config": "pi05_piper_folding_iwr_v2_x2", "exp": "pi05_folding_iwr_v2_x2", "gpu": 0,
           "log": f"{LOGD}/train_x2.log", "corr_pct": 19.8, "mult": 2,
           "base": "pi05_piper_folding_v2 @20k (warm-start)"},
    "x3": {"config": "pi05_piper_folding_iwr_v2_x3", "exp": "pi05_folding_iwr_v2_x3", "gpu": 1,
           "log": f"{LOGD}/train_x3.log", "corr_pct": 27.1, "mult": 3,
           "base": "pi05_piper_folding_v2 @20k (warm-start)"},
}

def parse_wandb(wf):
    ds = datastore.DataStore(); ds.open_for_scan(wf); rows = []
    while True:
        data = ds.scan_data()
        if data is None: break
        rec = pb.Record(); rec.ParseFromString(data)
        if rec.WhichOneof("record_type") == "history":
            d = {}
            for it in rec.history.item:
                k = it.nested_key[0] if it.nested_key else it.key
                try: v = json.loads(it.value_json)
                except Exception: v = it.value_json
                if isinstance(v, (int, float)): d[k] = v
            if "loss" in d and "_step" in d: rows.append(d)
    rows.sort(key=lambda r: r["_step"]); return rows

def find_wandb_dir_for(config):
    for run in sorted(glob.glob(REPO + "/wandb/offline-run-*"), reverse=True):
        meta = os.path.join(run, "files", "wandb-metadata.json")
        if os.path.exists(meta):
            try:
                args = json.load(open(meta)).get("args", [])
                if args and args[0] == config:
                    wf = glob.glob(run + "/run-*.wandb")
                    if wf: return wf[0]
            except Exception: pass
    return None

def progress_from_log(path):
    if not os.path.exists(path): return {}
    txt = open(path, errors="ignore").read()
    m = re.findall(r"Progress on: ([\d.]+k?)it/[\d.]+kit rate:([\d.]+)s/it remaining:([\d:]+) elapsed:([\d:]+)", txt)
    if not m: return {}
    step, rate, rem, el = m[-1]
    sv = float(step[:-1])*1000 if step.endswith("k") else float(step)
    return {"step": int(sv), "rate_s_it": float(rate), "remaining": rem, "elapsed": el}

def checkpoints_for(config, exp):
    d = os.path.join(REPO, "checkpoints", config, exp); out = []
    if os.path.isdir(d):
        for s in sorted(os.listdir(d)):
            p = os.path.join(d, s)
            if s.isdigit() and os.path.isdir(p):
                try: sz = int(subprocess.run(["du","-sb",p],capture_output=True,text=True).stdout.split()[0])
                except Exception: sz = 0
                out.append({"step": int(s), "gb": round(sz/1e9, 1)})
    return out

def gpu_stats():
    try:
        r = subprocess.run(["nvidia-smi","--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().splitlines()
        g = {}
        for line in r:
            i, mu, mt, u, t, p = [x.strip() for x in line.split(",")]
            g[int(i)] = {"mem_used_gb": round(float(mu)/1024,1), "mem_total_gb": round(float(mt)/1024,1),
                         "util": int(float(u)), "temp": int(float(t)), "power_w": round(float(p))}
        return g
    except Exception: return {}

def is_running(config):
    pat = "[" + config[0] + "]" + config[1:]
    return subprocess.run(["pgrep","-f",pat], capture_output=True).returncode == 0

du = shutil.disk_usage("/data")
out = {"runs": {}, "disk": {"free_gb": round(du.free/1e9), "total_gb": round(du.total/1e9), "used_gb": round(du.used/1e9)}, "gpus": gpu_stats()}
for name, r in RUNS.items():
    wf = find_wandb_dir_for(r["config"])
    try: hist = parse_wandb(wf) if wf else []
    except Exception: hist = []
    series = [{"step": int(h["_step"]), "loss": round(h["loss"],5),
               "grad_norm": round(h.get("grad_norm",0),4), "runtime_s": round(h.get("_runtime",0))} for h in hist]
    out["runs"][name] = {"config": r["config"], "exp": r["exp"], "gpu": r["gpu"], "base": r["base"],
                         "corr_pct": r["corr_pct"], "mult": r["mult"],
                         "progress": progress_from_log(r["log"]), "checkpoints": checkpoints_for(r["config"], r["exp"]),
                         "loss_series": series, "last_loss": series[-1]["loss"] if series else None,
                         "running": is_running(r["config"])}
print(json.dumps(out))
