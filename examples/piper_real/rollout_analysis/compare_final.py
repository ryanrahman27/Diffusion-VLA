#!/usr/bin/env python3
"""Final smoothness comparison — speed-normalized metrics (SPARC + pauses), correct ranking."""
import h5py, numpy as np, io, json, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
D="/home/rtx5090/Insync/sagar@axibo.com/Google Drive/VLA_data/rollouts/Jun26"
OUT="/tmp/claude-1000/-home-rtx5090-Projects-vla/61f1baa8-5f99-4a7a-9524-dcc1642661ed/scratchpad/media_compare"
METHODS=[("sync","Synchronous","stacking_v3_sync_fixed_20260626_163358.hdf5","#e8533f"),
 ("lat_noclamp","Latency (no clamp)","stacking_v3_async_fixed_20260626_163242.hdf5","#e0a93d"),
 ("lat_clamp","Latency + clamp","stacking_v3_async_clamped_fixed_20260626_163131.hdf5","#3d9be0"),
 ("rtc","RTC","stacking_v3_rtc_fixed_20260626_162926.hdf5","#4ecb8e")]
REE=[10,11,12]; ARM=[0,1,2,3,4,5,7,8,9,10,11,12]
GR='#0c0f14'; FG='#e6edf3'; MUT='#8a97a6'; LINE='#26384c'
keys=[m[0] for m in METHODS]
def smooth(x,k=5): return np.convolve(x,np.ones(k)/k,mode='same')
def sparc(sp,fs,fc=10,amp=0.05):
    sp=np.asarray(sp,float)
    if sp.max()<=0: return 0
    Nf=int(2**np.ceil(np.log2(len(sp)))*4); M=np.abs(np.fft.rfft(sp,Nf)); M/=M.max(); f=np.fft.rfftfreq(Nf,1/fs)
    s=f<=fc; f_,M_=f[s],M[s]; k=M_>=amp
    if k.sum()<2: return 0
    f_,M_=f_[k],M_[k]; fn=f_[-1]-f_[0]; return float(-np.sum(np.sqrt((np.diff(f_)/fn)**2+np.diff(M_)**2)))
def pixacc(h,cam='cam_right_wrist',sz=(140,105)):
    raw=h[f'observations/images/{cam}'][:]
    fr=np.stack([np.asarray(Image.open(io.BytesIO(bytes(r))).convert('L').resize(sz),np.float32) for r in raw])
    return float(np.abs(fr[2:]-2*fr[1:-1]+fr[:-2]).mean())
res={}; data={}
for k,lab,fn,c in METHODS:
    h=h5py.File(f"{D}/{fn}",'r'); t=h['time_stamp'][:]; t=t-t[0]; dt=np.median(np.diff(t)); fs=1/dt
    q=h['observations/qpos'][:].astype(float); ee=h['observations/eef_6d'][:].astype(float)[:,REE]
    sp=np.linalg.norm(np.diff(ee,axis=0)/dt,axis=1); sps=smooth(sp,5)
    jerk=np.diff(np.diff(np.diff(q,axis=0)/dt,axis=0)/dt,axis=0)/dt
    pa=pixacc(h); h.close()
    low=sps<0.12*sps.max(); pauses=0; run=0
    for v in low:
        if v: run+=1
        else:
            if run>=3: pauses+=1
            run=0
    res[k]=dict(label=lab,color=c,dur=round(float(t[-1]),1),
        sparc=round(sparc(sps,fs),3), pauses=int(pauses),
        peak_jerk_raw=round(float(np.abs(jerk[:,ARM]).max()),1),  # context (speed-confounded)
        pixel_acc=round(pa,2))
    data[k]=dict(t=t,ee=ee,sp=sp,sps=sps,c=c,lab=lab,t_sp=t[1:])
# composite score from speed-normalized choppiness: SPARC (0=smooth) + pauses
def nrm(vals):
    vals=np.array(vals,float); lo,hi=vals.min(),vals.max(); return (vals-lo)/(hi-lo+1e-9)
sp_n=nrm([-res[k]['sparc'] for k in keys])    # more negative sparc -> worse(1)
pa_n=nrm([res[k]['pauses'] for k in keys])
bad=0.65*sp_n+0.35*pa_n
for i,k in enumerate(keys): res[k]['score']=round(float(100*(1-bad[i])),0)
rank=sorted(keys,key=lambda k:-res[k]['score'])
print("RANKING (smoothest->choppiest):",[(k,res[k]['score']) for k in rank])

# ---- FIG ee paths colored by SPEED ----
fig,ax=plt.subplots(2,2,figsize=(11,9.5)); fig.patch.set_facecolor(GR)
vmax=np.percentile(np.concatenate([data[k]['sp'] for k in keys]),98)
for axx,k in zip(ax.flat,keys):
    d=data[k]; ee=d['ee']; axx.set_facecolor(GR)
    pts=ee[:,:2].reshape(-1,1,2); segs=np.concatenate([pts[:-1],pts[1:]],axis=1)
    lc=LineCollection(segs,cmap='turbo',norm=plt.Normalize(0,vmax)); lc.set_array(d['sp']); lc.set_linewidth(2.2); axx.add_collection(lc)
    axx.scatter(ee[0,0],ee[0,1],c='white',s=45,zorder=5,edgecolors='k'); axx.scatter(ee[-1,0],ee[-1,1],c=d['c'],marker='*',s=200,zorder=5,edgecolors='k')
    axx.set_title(f"{d['lab']}  ·  smoothness {res[k]['score']:.0f}/100",color=d['c'],fontsize=11)
    axx.set_xlabel("ee x (m)",color=MUT); axx.set_ylabel("ee y (m)",color=MUT); axx.autoscale(); axx.set_aspect('equal',adjustable='datalim'); axx.tick_params(colors=MUT,labelsize=7)
    for s in axx.spines.values(): s.set_color(LINE)
cb=fig.colorbar(lc,ax=ax,fraction=0.025,pad=0.02); cb.set_label("EE speed (m/s)",color=FG); cb.ax.tick_params(colors=MUT)
fig.suptitle("Right end-effector path (colored by speed)  ·  ○ start  ★ end",color=FG,fontsize=13,y=.97)
plt.savefig(f"{OUT}/ee_paths.png",dpi=115,facecolor=GR,bbox_inches='tight'); print("saved ee_paths.png")

# ---- FIG metric bars: smoothness (SPARC, pauses, score) + context (jerk, time) ----
fig,axs=plt.subplots(1,4,figsize=(13,3.3)); fig.patch.set_facecolor(GR)
labs=[res[k]['label'].replace(" (","\n(").replace(" + ","\n+ ") for k in keys]; cols=[res[k]['color'] for k in keys]
panels=[("score","Smoothness score\n(higher = smoother)",False),("sparc","SPARC\n(closer to 0 = smoother)",False),
        ("pauses","Stop-start pauses\n(fewer = smoother)",False),("dur","Completion time (s)\n(context)",False)]
for axx,(m,title,_) in zip(axs,panels):
    axx.set_facecolor(GR); vals=[res[k][m] for k in keys]; b=axx.bar(range(4),vals,color=cols,width=.7)
    axx.set_title(title,color=FG,fontsize=9); axx.set_xticks(range(4)); axx.set_xticklabels(labs,fontsize=6.5,color=MUT)
    for bb,v in zip(b,vals): axx.text(bb.get_x()+bb.get_width()/2,v,f"{v:.3g}",ha='center',va='bottom' if v>=0 else 'top',color=FG,fontsize=8)
    axx.tick_params(colors=MUT,labelsize=6);
    for s in axx.spines.values(): s.set_color(LINE)
    axx.set_ylim(min(0,min(vals))*1.15 if min(vals)<0 else 0, max(vals)*1.2)
fig.suptitle("Speed-normalized smoothness (raw jerk is misleading here — faster runs show higher jerk despite being smoother)",color=FG,fontsize=10.5,y=1.04)
plt.savefig(f"{OUT}/metrics_bars.png",dpi=115,facecolor=GR,bbox_inches='tight'); print("saved metrics_bars.png")

# ---- FIG EE z over time ----
fig,axz=plt.subplots(figsize=(12,3.6)); fig.patch.set_facecolor(GR); axz.set_facecolor(GR)
for k in keys: d=data[k]; axz.plot(d['t'],d['ee'][:,2],color=d['c'],lw=1.5,label=d['lab'])
axz.set_title("EE height (z) over time — pick → place cycle (sync slowest, latency runs ~25% faster)",color=FG,fontsize=11)
axz.set_xlabel("t (s)",color=MUT); axz.set_ylabel("ee z (m)",color=MUT); axz.legend(fontsize=8); axz.tick_params(colors=MUT,labelsize=8)
for s in axz.spines.values(): s.set_color(LINE)
plt.tight_layout(); plt.savefig(f"{OUT}/ee_z.png",dpi=115,facecolor=GR); print("saved ee_z.png")

json.dump(res, open(f"{OUT}/metrics.json","w"), indent=2)
print(json.dumps(res,indent=2))
