#!/usr/bin/env python3
"""Fix SPARC bar panel + add an acceleration 'jaggedness' figure that clearly shows sync's choppiness."""
import h5py, numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D="/home/rtx5090/Insync/sagar@axibo.com/Google Drive/VLA_data/rollouts/Jun26"
OUT="/tmp/claude-1000/-home-rtx5090-Projects-vla/61f1baa8-5f99-4a7a-9524-dcc1642661ed/scratchpad/media_compare"
METHODS=[("sync","Synchronous","stacking_v3_sync_fixed_20260626_163358.hdf5","#e8533f"),
 ("lat_noclamp","Latency (no clamp)","stacking_v3_async_fixed_20260626_163242.hdf5","#e0a93d"),
 ("lat_clamp","Latency + clamp","stacking_v3_async_clamped_fixed_20260626_163131.hdf5","#3d9be0"),
 ("rtc","RTC","stacking_v3_rtc_fixed_20260626_162926.hdf5","#4ecb8e")]
REE=[10,11,12]; GR='#0c0f14'; FG='#e6edf3'; MUT='#8a97a6'; LINE='#26384c'
keys=[m[0] for m in METHODS]
res=json.load(open(f"{OUT}/metrics.json"))
def smooth(x,k=5): return np.convolve(x,np.ones(k)/k,mode='same')

# ---- load EE accel time-series ----
data={}
for k,lab,fn,c in METHODS:
    h=h5py.File(f"{D}/{fn}",'r'); t=h['time_stamp'][:]; t=t-t[0]; dt=np.median(np.diff(t))
    ee=h['observations/eef_6d'][:].astype(float)[:,REE]; h.close()
    v=np.diff(ee,axis=0)/dt; a=np.diff(v,axis=0)/dt
    amag=smooth(np.linalg.norm(a,axis=1),3)
    data[k]=dict(t=t[1:-1][:len(amag)], a=amag, c=c, lab=lab)

# ---- FIG: metrics bars (FIXED sparc panel uses a baseline) ----
labs=[res[k]['label'].replace(" (","\n(").replace(" + ","\n+ ") for k in keys]; cols=[res[k]['color'] for k in keys]
panels=[("score","Smoothness score\n(higher = smoother)"),("sparc","SPARC\n(closer to 0 = smoother)"),
        ("pauses","Stop-start pauses\n(fewer = smoother)"),("dur","Completion time (s)\n(context)")]
fig,axs=plt.subplots(1,4,figsize=(13,3.4)); fig.patch.set_facecolor(GR)
for axx,(m,title) in zip(axs,panels):
    axx.set_facecolor(GR); vals=[res[k][m] for k in keys]
    if m=="sparc":
        base=min(vals)-0.18
        axx.bar(range(4),[v-base for v in vals],bottom=base,color=cols,width=.7)
        for i,v in enumerate(vals): axx.text(i,v,f"{v:.2f}",ha='center',va='bottom',color=FG,fontsize=8)
        axx.set_ylim(base, max(vals)+0.14); axx.axhline(0,color=LINE,lw=.6)
    else:
        b=axx.bar(range(4),vals,color=cols,width=.7)
        for bb,v in zip(b,vals): axx.text(bb.get_x()+bb.get_width()/2,v,f"{v:.3g}",ha='center',va='bottom',color=FG,fontsize=8)
        axx.set_ylim(0, max(vals)*1.22)
    axx.set_title(title,color=FG,fontsize=9); axx.set_xticks(range(4)); axx.set_xticklabels(labs,fontsize=6.5,color=MUT)
    axx.tick_params(colors=MUT,labelsize=6)
    for s in axx.spines.values(): s.set_color(LINE)
fig.suptitle("Speed-normalized smoothness — raw jerk is misleading here (faster runs show higher jerk despite being smoother)",color=FG,fontsize=10.5,y=1.04)
plt.savefig(f"{OUT}/metrics_bars.png",dpi=115,facecolor=GR,bbox_inches='tight'); print("saved metrics_bars.png (sparc fixed)")

# ---- FIG: acceleration jaggedness (2x2) — sync is visibly spiky ----
amax=np.percentile(np.concatenate([data[k]['a'] for k in keys]),99)
fig,ax=plt.subplots(2,2,figsize=(13,7),sharey=True); fig.patch.set_facecolor(GR)
for axx,k in zip(ax.flat,keys):
    d=data[k]; axx.set_facecolor(GR)
    axx.fill_between(d['t'],d['a'],color=d['c'],alpha=.30); axx.plot(d['t'],d['a'],color=d['c'],lw=.9)
    axx.set_title(f"{d['lab']}  ·  smoothness {res[k]['score']:.0f}/100",color=d['c'],fontsize=10.5)
    axx.set_xlabel("t (s)",color=MUT); axx.set_ylabel("EE accel (m/s²)",color=MUT); axx.set_ylim(0,amax)
    axx.tick_params(colors=MUT,labelsize=7)
    for s in axx.spines.values(): s.set_color(LINE)
fig.suptitle("Acceleration over time — the jaggedness the path can't show. Synchronous = spike-storm; latency+clamp = smooth.",color=FG,fontsize=11.5,y=.98)
plt.tight_layout(rect=[0,0,1,.95]); plt.savefig(f"{OUT}/accel_jagged.png",dpi=115,facecolor=GR); print("saved accel_jagged.png")
print("sync accel peak %.0f vs clamp %.0f"%(data['sync']['a'].max(), data['lat_clamp']['a'].max()))
