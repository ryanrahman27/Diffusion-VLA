#!/usr/bin/env python3
import base64, json, pathlib
M=pathlib.Path("/tmp/claude-1000/-home-rtx5090-Projects-vla/61f1baa8-5f99-4a7a-9524-dcc1642661ed/scratchpad/media_compare")
def b64(fn):
    return base64.b64encode((M/fn).read_bytes()).decode()
def png(fn): return "data:image/png;base64,"+b64(fn)
res=json.load(open(M/"metrics.json"))
VID="data:video/mp4;base64,"+b64("rollouts_2x.mp4")
VID_TASK="data:video/mp4;base64,"+b64("task_demo.mp4")
IMG={k:png(k+".png") for k in ["metrics_bars","cmd_vs_achieved","cmd_jerk_overlay"]}

# order smoothest->choppiest for the leaderboard
order=sorted(res, key=lambda k:-res[k]['score'])
DESC={
 "lat_clamp":"Latency-matched + accel/vel clamps round off the speed profile.",
 "rtc":"Real-time chunking: next chunk generated conditioned on the pending one.",
 "lat_noclamp":"Latency-matched, no clamps — flows fast but jagged bursts.",
 "sync":"Blocks ~84 ms for every inference → stop-start, slowest.",
}
GH_PO="https://github.com/axiboai/piperx-openpi/blob/main"
GH_LR="https://github.com/axiboai/piperx_lerobot_setup/blob/main"

lead="".join(f'''
  <div class="lead-card" style="--c:{res[k]['color']}">
    <div class="lead-rank">#{i+1}</div>
    <div class="lead-body">
      <div class="lead-top"><span class="lead-name">{res[k]['label']}</span><span class="lead-score">{res[k]['score']:.0f}<small>/100</small></span></div>
      <div class="lead-bar"><i style="width:{res[k]['score']:.0f}%"></i></div>
      <div class="lead-desc">{DESC[k]}</div>
      <div class="lead-stats">SPARC {res[k]['sparc']} · {res[k]['pauses']} pauses · {res[k]['dur']}s</div>
    </div>
  </div>''' for i,k in enumerate(order))

rows="".join(f'''<tr><td class="m-meth" style="color:{res[k]['color']}">{res[k]['label']}</td>
  <td class="mono">{res[k]['score']:.0f}</td><td class="mono">{res[k]['sparc']}</td><td class="mono">{res[k]['pauses']}</td>
  <td class="mono">{res[k]['dur']}s</td></tr>''' for k in order)

HTML=f'''<title>Stacking v3 — inference smoothness comparison</title>
<style>
:root{{--bg:#0b0e13;--panel:#11161f;--panel2:#0e131b;--line:#1f2937;--text:#e6edf3;--mut:#8a97a6;--faint:#5b6675;
 --sync:#e8533f;--noclamp:#e0a93d;--clamp:#3d9be0;--rtc:#4ecb8e;--acc:#5ad1c4;
 --sans:ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;--mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);line-height:1.55;-webkit-font-smoothing:antialiased;
 background-image:radial-gradient(900px 480px at 85% -8%,rgba(90,209,196,.06),transparent 60%);}}
.wrap{{max-width:1080px;margin:0 auto;padding:clamp(22px,4vw,46px) clamp(15px,4vw,34px) 80px;}}
a{{color:var(--acc);text-decoration:none}} a:hover{{text-decoration:underline}}
code{{font-family:var(--mono);font-size:.85em;background:#0a0d12;border:1px solid var(--line);padding:1px 5px;border-radius:4px;color:#cdd7e2}}
.mono{{font-family:var(--mono)}} .dim{{color:var(--faint)}}
.eyebrow{{font-family:var(--mono);font-size:.72rem;letter-spacing:.28em;text-transform:uppercase;color:var(--acc);margin-bottom:12px}}
h1{{font-size:clamp(1.9rem,4.6vw,3rem);line-height:1.04;margin:0 0 12px;letter-spacing:-.02em;font-weight:800}}
.sub{{color:var(--mut);font-size:1.02rem;max-width:70ch}}
.sub b{{color:var(--text)}}
section{{margin-top:clamp(40px,6vw,70px)}}
.sec-h{{font-family:var(--mono);font-size:.74rem;letter-spacing:.24em;text-transform:uppercase;color:var(--mut);border-bottom:1px solid var(--line);padding-bottom:10px;margin-bottom:22px}}
.sec-h .n{{color:var(--acc);margin-right:10px}}
.verdict{{margin:26px 0;border-left:3px solid var(--rtc);background:linear-gradient(90deg,rgba(78,203,142,.10),transparent);padding:16px 20px;border-radius:0 8px 8px 0}}
.verdict b{{color:#fff}}
/* leaderboard */
.lead{{display:flex;flex-direction:column;gap:12px;margin-top:24px}}
.lead-card{{display:flex;gap:16px;align-items:stretch;background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--c);border-radius:12px;padding:14px 18px}}
.lead-rank{{font-family:var(--mono);font-size:1.5rem;font-weight:700;color:var(--c);align-self:center;width:42px;text-align:center}}
.lead-body{{flex:1;min-width:0}}
.lead-top{{display:flex;justify-content:space-between;align-items:baseline;gap:10px}}
.lead-name{{font-weight:700;font-size:1.05rem}}
.lead-score{{font-family:var(--mono);font-size:1.3rem;font-weight:700;color:var(--c)}} .lead-score small{{font-size:.7rem;color:var(--mut)}}
.lead-bar{{height:6px;background:#0a0d12;border-radius:4px;margin:8px 0 7px;overflow:hidden}} .lead-bar i{{display:block;height:100%;background:var(--c);border-radius:4px}}
.lead-desc{{font-size:.88rem;color:#c2ccd6}} .lead-stats{{font-family:var(--mono);font-size:.74rem;color:var(--faint);margin-top:4px}}
figure{{margin:0}} .fig{{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:10px;overflow-x:auto}}
.fig img{{display:block;width:100%;min-width:560px;max-width:100%;border-radius:6px}}
figcaption{{font-family:var(--mono);font-size:.78rem;color:var(--mut);padding:10px 6px 2px}} figcaption b{{color:var(--text)}}
.note{{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--noclamp);border-radius:10px;padding:14px 18px;color:#c2ccd6;font-size:.9rem;margin-top:16px}}
.note b{{color:var(--text)}}
table{{width:100%;border-collapse:collapse;font-size:.9rem;margin-top:6px}} .tbl{{overflow-x:auto;border:1px solid var(--line);border-radius:10px}}
th,td{{text-align:left;padding:10px 14px;border-bottom:1px solid var(--line);white-space:nowrap}}
thead th{{font-family:var(--mono);font-size:.66rem;text-transform:uppercase;letter-spacing:.08em;color:var(--faint);background:var(--panel2)}}
tbody tr:last-child td{{border-bottom:none}} .m-meth{{font-weight:600}}
.vid{{display:grid;grid-template-columns:300px 1fr;gap:20px;align-items:start}} @media(max-width:720px){{.vid{{grid-template-columns:1fr}}}}
.vid video{{width:100%;border-radius:10px;border:1px solid var(--line);background:#000}}
.badge{{display:inline-block;font-family:var(--mono);font-size:.7rem;padding:2px 8px;border-radius:20px;background:rgba(90,209,196,.14);color:var(--acc);margin-bottom:8px}}
.methods{{display:flex;flex-direction:column;gap:18px}}
.method{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px}}
.method h3{{margin:0 0 4px;font-size:1.08rem}} .method .tag{{font-family:var(--mono);font-size:.7rem;color:var(--faint)}}
.method p{{font-size:.9rem;color:#c2ccd6;margin:10px 0}}
.diagram{{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0;overflow-x:auto}}
.diagram svg{{display:block;min-width:560px;width:100%;height:auto}}
.linkrow{{display:flex;gap:10px;flex-wrap:wrap;margin-top:10px}}
.linkrow a{{font-family:var(--mono);font-size:.78rem;background:#0a0d12;border:1px solid var(--line);border-radius:100px;padding:6px 13px}}
.taskclip{{display:grid;grid-template-columns:200px 1fr;gap:20px;align-items:center;margin-top:24px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px}}
.taskclip video{{width:100%;border-radius:10px;border:1px solid var(--line);background:#000;display:block}}
.taskclip p{{margin:8px 0;font-size:.92rem;color:#c2ccd6}} .taskclip p b{{color:var(--text)}}
@media(max-width:560px){{.taskclip{{grid-template-columns:1fr}}}}
.foot{{margin-top:64px;padding-top:18px;border-top:1px solid var(--line);font-family:var(--mono);font-size:.74rem;color:var(--faint);display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}}
.dtxt{{fill:var(--text);font-family:var(--mono);font-size:12px}} .dmut{{fill:var(--mut);font-family:var(--mono);font-size:11px}} .dac{{fill:var(--acc);font-family:var(--mono);font-size:11px}}
</style>
<div class="wrap">
  <div class="eyebrow">Real-robot rollouts · pi05_piper_stacking_v3</div>
  <h1>Inference smoothness:<br>4 ways to run the same policy</h1>
  <p class="sub">Same model (<code>pi05_piper_stacking_v3</code>, no shift), same task (<b>stack red cube on blue cube</b>), four inference stacks. The policy decides <i>what</i> to do; the inference stack decides <i>how smoothly</i> it gets executed. We measure that.</p>

  <div class="taskclip">
    <figure style="margin:0"><video src="{VID_TASK}" autoplay loop muted playsinline></video></figure>
    <div>
      <div class="badge">the task</div>
      <p>Prompt: <b>&ldquo;stack red cube on blue cube&rdquo;</b>. A demonstration from the training data. All four runs below drive the <b>same trained policy</b>, <code>pi05_piper_stacking_v3</code> — only the inference stack differs.</p>
      <div class="linkrow"><a href="https://claude.ai/code/artifact/caeb362a-6976-4ffa-876b-1b8ad68556f6" target="_blank">model &amp; training dashboard ↗</a><a href="https://huggingface.co/axiboai/pi05_piper_stacking_v3" target="_blank">HF model ↗</a></div>
    </div>
  </div>

  <div class="verdict">
    <b>Verdict.</b> Smoothest → choppiest: <b style="color:var(--clamp)">Latency+clamp</b> ≳ <b style="color:var(--rtc)">RTC</b> &gt; <b style="color:var(--noclamp)">Latency (no clamp)</b> &gt; <b style="color:var(--sync)">Synchronous</b>. Synchronous blocks for every inference → stop-start motion (most pauses, worst SPARC, and slowest at 15.3 s). Compensating latency removes the stalls; clamping accel/vel rounds the motion the most; RTC stays smooth by blending chunks in-model.
  </div>

  <div class="lead">{lead}</div>

  <section>
    <div class="sec-h"><span class="n">01</span>The smoothness scores</div>
    <figure class="fig"><img src="{IMG['metrics_bars']}" alt="smoothness metrics"></figure>
    <div class="tbl" style="margin-top:16px"><table>
      <thead><tr><th>Method</th><th>Smoothness /100</th><th>SPARC →0</th><th>Stop-start pauses</th><th>Time</th></tr></thead>
      <tbody>{rows}</tbody></table></div>
    <div class="note" style="margin-top:16px"><b>What is SPARC?</b> <b>Spectral Arc Length</b> — a standard movement-smoothness metric. You take the speed profile, look at its <i>frequency spectrum</i>, and measure the length of that spectrum curve. A smooth motion has a simple spectrum (short curve → value near <b>0</b>); a jerky motion piles on extra frequencies (a longer, wigglier curve → more <b>negative</b>). Crucially it's <b>speed- and duration-independent</b>, so it fairly compares runs of different lengths and speeds. <b>Closer to 0 = smoother.</b></div>
    <div class="note" style="border-left-color:var(--noclamp)"><b>Why not just use "jerk"?</b> Raw jerk magnitude is misleading here — the latency runs finish ~25% faster (11 s vs 15 s), so they move faster and show <i>higher</i> per-step jerk despite being smoother. So we rank by speed-normalized SPARC + stop-start pauses (the same reasoning Chef Robotics use).</div>
  </section>

  <section>
    <div class="sec-h"><span class="n">02</span>Where the jaggedness actually is</div>
    <div class="note"><b>The non-obvious part:</b> in the <b>achieved</b> joint/EE motion (what the encoders record), all four look nearly identical and smooth — the robot's low-level servo controller filters the incoming commands, so you genuinely <i>can't</i> see the jaggedness there. It lives in the <b>commanded output</b> (what each inference stack sends to the arm) and in the <b>stop-start pauses</b>.</div>
    <figure class="fig" style="margin-top:16px"><img src="{IMG['cmd_jerk_overlay']}" alt="commanded jaggedness overlay">
      <figcaption><b>Commanded jaggedness over time</b> — rolling-RMS of the inference output's jerk, all four overlaid. <b>Synchronous (red) is jaggiest by far</b> (mean 674); <b>RTC (green) is smoothest</b> (124, thanks to its in-model chunk blending); the two latency runs sit in between (~300). This is the difference the servo loop then hides in the achieved motion.</figcaption></figure>
    <figure class="fig" style="margin-top:16px"><img src="{IMG['cmd_vs_achieved']}" alt="commanded vs achieved">
      <figcaption><b>Commanded vs achieved</b> (joint 8, zoomed). Synchronous's <i>commanded</i> target stutters (stop-go) and the arm lags; the controller then smooths the <i>achieved</i> motion (dashed) — which is exactly why the encoder traces look fine even though the commands, and the run, are choppier and slower.</figcaption></figure>
  </section>

  <section>
    <div class="sec-h"><span class="n">03</span>Watch the rollouts</div>
    <figure style="max-width:360px;margin:0">
      <span class="badge">▶ 2× speed · robot runs at 30 Hz</span>
      <video src="{VID}" controls playsinline preload="metadata" style="width:100%;border-radius:10px;border:1px solid var(--line);background:#000;display:block"></video>
      <figcaption>Real-robot stacking across the runs (1:55 recording, shown at 2× = 57 s, 540p). Use the scrubber to step through.</figcaption>
    </figure>
  </section>

  <section>
    <div class="sec-h"><span class="n">04</span>How each method works</div>
    <div class="methods">

      <div class="method" style="border-left:3px solid var(--sync)">
        <h3>Synchronous</h3><div class="tag">regular openpi inference · choppiest, slowest</div>
        <p>The robot runs one action chunk, then <b>blocks</b> while the policy computes the next (~84 ms round-trip). During that wait the arm holds — so motion is <b>stop · go · stop · go</b>. Every seam is a fresh prediction from a new observation, so it can also jump. Simple and correct, but the pauses dominate.</p>
        <div class="diagram">{{svg_sync}}</div>
        <div class="linkrow"><a href="{GH_PO}/scripts/serve_policy.py" target="_blank">serve_policy.py ↗</a></div>
      </div>

      <div class="method" style="border-left:3px solid var(--clamp)">
        <h3>Latency compensation (UMI &ldquo;PD1&rdquo;)</h3><div class="tag">openpi_latency_inference_smooth.py · with / without accel-vel clamps</div>
        <p>Three sensors arrive on three clocks. We <b>align the inputs</b> to one instant and <b>issue commands early</b> so the arm arrives on time — no blocking, so the motion flows and the task finishes ~25% faster. Within a chunk we interpolate to the reach-time; at the seam we crossfade (120 ms). The <b>clamp</b> variant adds accel ≤ 8 / vel ≤ 20 caps that round off the speed profile — the single smoothest run here.</p>
        <div class="diagram">{{svg_latency}}</div>
        <div class="linkrow"><a href="{GH_LR}/scripts/openpi_latency_inference_smooth.py" target="_blank">openpi_latency_inference_smooth.py ↗</a><a href="{GH_LR}/scripts/latency_matching.py" target="_blank">latency_matching.py ↗</a></div>
      </div>

      <div class="method" style="border-left:3px solid var(--rtc)">
        <h3>RTC — Real-Time Chunking</h3><div class="tag">openpi_paper_rtc_inference.py · smooth, low-amplitude</div>
        <p>Instead of predicting each chunk independently, RTC generates the <b>next</b> chunk <b>conditioned on the still-pending one</b>: during flow-matching denoising the model is pulled toward the leftover actions (hard-frozen on the part executing during latency, softly guided across an overlap, free afterward). A background thread runs inference ahead and the new chunk is swapped in seamlessly — <b>no seam, no stall</b>. A second-order command tracker (ω=15, ζ=1) further eases each retarget.</p>
        <div class="diagram">{{svg_rtc}}</div>
        <div class="linkrow"><a href="{GH_PO}/examples/piper_real/rtc_rollout/openpi_paper_rtc_inference.py" target="_blank">openpi_paper_rtc_inference.py ↗</a><a href="{GH_PO}/src/openpi/rtc" target="_blank">src/openpi/rtc/ ↗</a><a href="https://www.physicalintelligence.company/research/real_time_chunking" target="_blank">RTC paper ↗</a></div>
      </div>
    </div>
  </section>

  <section>
    <div class="sec-h"><span class="n">05</span>Repos &amp; data</div>
    <div class="linkrow">
      <a href="https://github.com/axiboai/piperx-openpi" target="_blank">piperx-openpi ↗</a>
      <a href="https://github.com/axiboai/piperx_lerobot_setup" target="_blank">piperx_lerobot_setup ↗</a>
      <a href="https://huggingface.co/axiboai/pi05_piper_stacking_v3" target="_blank">model: pi05_piper_stacking_v3 ↗</a>
      <a href="https://www.chefrobotics.ai/post/latency-aware-vision-language-action-models-vlas-with-system-identification-for-real-time-robot-control" target="_blank">Chef Robotics writeup ↗</a>
    </div>
  </section>

  <div class="foot"><span>4 rollouts · pi05_piper_stacking_v3 · 30 Hz · recorder schema openpi_rollout_v1</span><span>generated 2026-06-26</span></div>
</div>
'''

# ---------- SVG diagrams ----------
svg_sync='''<svg viewBox="0 0 720 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="synchronous timeline">
<text class="dmut" x="8" y="20">SYNCHRONOUS — robot waits for each inference</text>
<!-- exec blocks with stalls -->
<g>
<rect x="20" y="50" width="120" height="34" rx="4" fill="#e8533f" opacity="0.85"/><text class="dtxt" x="80" y="72" text-anchor="middle" fill="#0b0e13">execute</text>
<rect x="140" y="50" width="70" height="34" rx="4" fill="#3a4150"/><text class="dmut" x="175" y="71" text-anchor="middle">stall</text>
<rect x="210" y="50" width="120" height="34" rx="4" fill="#e8533f" opacity="0.85"/><text class="dtxt" x="270" y="72" text-anchor="middle" fill="#0b0e13">execute</text>
<rect x="330" y="50" width="70" height="34" rx="4" fill="#3a4150"/><text class="dmut" x="365" y="71" text-anchor="middle">stall</text>
<rect x="400" y="50" width="120" height="34" rx="4" fill="#e8533f" opacity="0.85"/><text class="dtxt" x="460" y="72" text-anchor="middle" fill="#0b0e13">execute</text>
<rect x="520" y="50" width="70" height="34" rx="4" fill="#3a4150"/><text class="dmut" x="555" y="71" text-anchor="middle">stall</text>
</g>
<text class="dmut" x="175" y="104" text-anchor="middle">~84 ms inference</text>
<text class="dmut" x="365" y="104" text-anchor="middle">arm holds</text>
<line x1="20" y1="120" x2="700" y2="120" stroke="#1f2937"/><text class="dmut" x="700" y="135" text-anchor="end">time →</text>
<text class="dac" x="20" y="135">⇒ motion = stop · go · stop · go</text>
</svg>'''

svg_latency='''<svg viewBox="0 0 720 210" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="latency compensation">
<text class="dmut" x="8" y="18">LATENCY COMPENSATION — align inputs to t_obs, issue commands early</text>
<!-- now marker -->
<line x1="430" y1="30" x2="430" y2="150" stroke="#5ad1c4" stroke-dasharray="3 3"/><text class="dac" x="430" y="166" text-anchor="middle">now</text>
<!-- camera row -->
<text class="dmut" x="8" y="52">camera</text>
<circle cx="360" cy="48" r="6" fill="#e0a93d"/><text class="dmut" x="300" y="38" >captured t−34ms</text>
<path d="M366 48 L424 48" stroke="#e0a93d" stroke-dasharray="2 2"/><text class="dmut" x="395" y="42" text-anchor="middle"></text>
<!-- proprio row -->
<text class="dmut" x="8" y="86">proprio</text>
<circle cx="420" cy="82" r="6" fill="#3d9be0"/><text class="dmut" x="320" y="100">sampled ~now (−0.3ms)</text>
<!-- t_obs align -->
<line x1="360" y1="40" x2="360" y2="150" stroke="#8a97a6" stroke-dasharray="2 4"/><text class="dmut" x="360" y="128" text-anchor="middle">t_obs</text>
<text class="dmut" x="365" y="86">→ interpolate proprio back to t_obs</text>
<!-- command row -->
<text class="dmut" x="8" y="120">command</text>
<circle cx="430" cy="116" r="6" fill="#4ecb8e"/><path d="M436 116 L600 116" stroke="#4ecb8e"/><polygon points="600,112 608,116 600,120" fill="#4ecb8e"/>
<text class="dmut" x="520" y="110" text-anchor="middle">issued now, targets t+176ms</text>
<text class="dac" x="608" y="120">arrives on time</text>
<line x1="20" y1="150" x2="700" y2="150" stroke="#1f2937"/>
<!-- clocks -->
<text class="dtxt" x="20" y="186">34 ms camera transport</text><text class="dtxt" x="250" y="186">0.3 ms proprio</text><text class="dtxt" x="430" y="186">176 ms / arm execution</text>
<text class="dmut" x="20" y="202">measured on this rig (system identification)</text>
</svg>'''

svg_rtc='''<svg viewBox="0 0 720 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="real-time chunking">
<text class="dmut" x="8" y="18">REAL-TIME CHUNKING — generate the next chunk conditioned on the pending one</text>
<!-- chunk A executing -->
<text class="dmut" x="8" y="52">chunk A</text>
<rect x="80" y="38" width="180" height="24" rx="3" fill="#4ecb8e" opacity="0.5"/><text class="dmut" x="170" y="55" text-anchor="middle" fill="#0b0e13">executed</text>
<rect x="260" y="38" width="120" height="24" rx="3" fill="#4ecb8e" opacity="0.85"/><text class="dtxt" x="320" y="55" text-anchor="middle" fill="#0b0e13">pending (leftover)</text>
<!-- chunk B generated, overlapping -->
<text class="dmut" x="8" y="98">chunk B</text>
<rect x="260" y="84" width="120" height="24" rx="3" fill="#3a4150"/><text class="dmut" x="320" y="101" text-anchor="middle">guided by A</text>
<rect x="380" y="84" width="200" height="24" rx="3" fill="#4ecb8e" opacity="0.85"/><text class="dtxt" x="480" y="101" text-anchor="middle" fill="#0b0e13">new actions (free)</text>
<!-- overlap bracket -->
<line x1="260" y1="70" x2="260" y2="118" stroke="#5ad1c4" stroke-dasharray="3 3"/><line x1="380" y1="70" x2="380" y2="118" stroke="#5ad1c4" stroke-dasharray="3 3"/>
<text class="dac" x="320" y="132" text-anchor="middle">overlap: blended in-model (no seam)</text>
<!-- prefix mask -->
<text class="dmut" x="8" y="162">prefix mask</text>
<rect x="80" y="150" width="100" height="16" fill="#e8533f" opacity="0.8"/><text class="dmut" x="130" y="180" text-anchor="middle">frozen</text>
<rect x="180" y="150" width="200" height="16" fill="#e0a93d" opacity="0.7"/><text class="dmut" x="280" y="180" text-anchor="middle">soft-guided</text>
<rect x="380" y="150" width="200" height="16" fill="#3d9be0" opacity="0.5"/><text class="dmut" x="480" y="180" text-anchor="middle">free</text>
<text class="dmut" x="600" y="100">inference runs</text><text class="dmut" x="600" y="114">ahead on a thread</text>
</svg>'''

HTML=HTML.replace("{svg_sync}",svg_sync).replace("{svg_latency}",svg_latency).replace("{svg_rtc}",svg_rtc)
out=M.parent/"dashboard_compare.html"; out.write_text(HTML); print("wrote",out,round(len(HTML)/1e6,2),"MB")
