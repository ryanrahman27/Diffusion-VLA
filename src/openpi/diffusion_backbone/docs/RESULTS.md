# Results log — Diffusion-Backbone VLA study

Running lab notebook. Append raw numbers + observations as experiments finish;
these get transcribed into `paper.tex` tables. Newest entries at the bottom.

Reference (openpi published π0.5 @ 30k, LIBERO): Spatial 98.8 / Object 98.2 /
Goal 98.0 / Long (libero_10) 92.4 / avg 96.85.

Variants: A = stock π0.5 prefix-LM (Gemma) baseline · D = Qwen2.5-7B (AR) ·
C = Dream-7B (diffusion) · B = D-backbone + expanded attention topology.
Identifying contrast: C − D.

---

## Design decision (2026-08) — standalone action expert + A′

Architecture map found that pi0.5's flow expert is **fused** into the backbone's
self-attention (matched layer count / head geometry / RoPE / adaRMS), so it
cannot be kept identical while the backbone changes. Resolution: a **standalone
DiT-style flow expert** (`action_expert.py`) shared identically across
Aprime/B/C/D; only a small input adapter (backbone width → expert width) differs
(~2–4M of ~117M params — verified). C−D stays clean.

New variant **A′** (Gemma backbone + standalone expert) = architecture-matched
baseline; ladder A′→D→C isolates decoupling cost then backbone prior. Model A
(stock pi0.5, fused) kept as reference. `action_expert.py` shape-tested (correct
[B,H,32] for both 2048/3584 widths, backprop OK, adaLN-Zero → zero output at init).

Run matrix now 5 variants × 3 seeds (A already done).

---

## Stage 1 — Model A baseline reproduction (LIBERO)

**Training.** 3 seeds, fork of `pi05_libero` (full FT from `pi05_base`),
batch_size=32, num_workers=16, 30k steps (final ckpt step 29999), save/keep 10k.
Loss curves healthy (loss 0.075→0.017, grad_norm stable ~0.07); param_norm
climbs mildly (LR left at pi05's 5e-5 — flagged, eval is the judge).

**Eval.** LIBERO, 4 suites × 10 tasks × 50 trials (500 rollouts/suite),
no-Docker path (policy server + py3.8 LIBERO client, MUJOCO_GL=egl).

### Success rates (%)

Full run: 4 suites x 10 tasks x 50 trials = 500 rollouts/suite, 3 seeds.

| Seed | Spatial | Object | Goal | Long (10) | Avg |
|------|--------:|-------:|-----:|----------:|----:|
| 0    | 97.0 | 99.0 | 96.8 | 93.8 | 96.65 |
| 1    | 98.0 | 97.8 | 96.8 | 94.0 | 96.65 |
| 2    | 98.2 | 98.2 | 97.6 | 91.4 | 96.35 |
| **mean ± sd** | **97.7 ± 0.6** | **98.3 ± 0.6** | **97.1 ± 0.5** | **93.1 ± 1.4** | **96.6 ± 0.2** |
| published | 98.8 | 98.2 | 98.0 | 92.4 | 96.85 |

**Verdict: Model A REPRODUCES published pi0.5.** Every suite within ~1%; avg
96.6 vs 96.85 (Δ −0.3); Long-horizon slightly exceeds published (93.1 vs 92.4).
Achieved with batch 32 (vs published 256), validating the batch-size choice.
Harness validated -> proceed to Stage 2 (Dream-7B backbone swap).

Notes:
- Smoke test (seed 0, libero_spatial, 5 trials/task = 50 eps): 96% (48/50) —
  matched the full-run number; eval pipeline validated (no-Docker path,
  MUJOCO_GL=egl). Harmless EGL/libGLU teardown traceback at exit — cosmetic.
- LR left at pi05's 5e-5 (tuned for batch 256); the mild param_norm climb did
  NOT hurt reproduction, so no LR change needed for the baseline.

---

## Stage 2a — Model A' (Gemma backbone + standalone expert)

Same Gemma backbone as A, but action tokens run through the BACKBONE and a
decoupled StandaloneActionExpert produces the vector field (vs A's fused expert).
Trained PyTorch (train_pytorch, torchrun 4-GPU DDP, global batch 32, 30k steps,
from converted pi05_base_pytorch). Eval via cached sample_actions (prefix KV-cache
validated: 94% vs 92% recompute on the spatial smoke, ~1.7x faster).

### Success rates (%)

| Seed | Spatial | Object | Goal | Long (10) | Avg |
|------|--------:|-------:|-----:|----------:|----:|
| 0 | 95.4 | 92.8 | 94.2 | 82.4 | 91.2 |
| 1 | 97.2 | 90.0 | 87.2 | 74.6 | 87.3 |
| 2 | 97.4 | 95.0 | 90.2 | 81.8 | 91.1 |
| **mean ± sd** | **96.7 ± 1.1** | **92.6 ± 2.5** | **90.5 ± 3.5** | **79.6 ± 4.3** | **89.9 ± 2.3** |
| A (fused, ref) | 97.7 | 98.3 | 97.1 | 93.1 | 96.6 |
| **A' − A** | **−1.0** | **−5.7** | **−6.6** | **−13.5** | **−6.7** |

**Finding: real decoupling cost, growing with task difficulty.** Spatial ~tied
(−1.0), object/goal ~−6, Long-horizon −13.5. The fused expert's deep per-layer
interaction with the backbone matters, most for long/multi-stage tasks.

Implications:
- **C−D stays clean:** C and D both use the IDENTICAL standalone expert, so the
  decoupling cost is common to both and cancels in the C−D contrast.
- A' (not A) is the correct architecture-matched baseline for C/D.
- A' is weakest on **Long-horizon** — exactly the regime where the diffusion
  prior is hypothesized to help most. So **C vs D on Long** is the key watch.
- Caveat: the gap conflates "decoupled" with the capacity of THIS standalone
  expert (~117M, 6 DiT layers) vs the fused Gemma-300M. Could strengthen the
  expert to close it; for C−D what matters is it stays identical across A'/B/C/D.

---

## Stage 2b — Model D (Qwen2.5-7B AR backbone), single seed

Strengthened LoRA recipe (unfrozen SigLIP + LoRA r=64) after the first frozen-vision
r=16 run eval'd ~57% everywhere. Trained PyTorch (torchrun DDP, global batch 32,
30k steps), exp db_D_seed0_v2. **Single seed (n=1)** — user chose to skip seeds 1/2;
reported honestly as n=1, error bars pending (NOT fabricated as 3-seed).

### Success rate (%), seed 0

| Suite | D (Qwen AR) | A′ (Gemma, 3-seed) | A (fused, 3-seed) |
|-------|--:|--:|--:|
| Spatial | 94.2 | 96.7 | 97.7 |
| Object  | 82.0 | 92.6 | 98.3 |
| Goal    | 71.4 | 90.5 | 97.1 |
| Long(10)| 57.4 | 79.6 | 93.1 |
| **Avg** | **76.3** | 89.9 | 96.6 |

Notes:
- Competitive on spatial (94.2), drops on harder suites (goal 71, long 57). Lower
  than A′ overall (76.3 vs 89.9) — likely LoRA-on-frozen-Qwen vs A′'s full-FT, but
  a reasonable regime (not the broken 57%-everywhere).
- **C uses the IDENTICAL recipe** (Dream-7B, LoRA r=64, unfrozen SigLIP, single seed),
  so C−D is a fair matched-seed comparison. Hypothesis: diffusion prior helps most
  where D is weakest (goal, long-horizon).
- D/C are n=1 preliminary; A/A′ are 3-seed. Add D/C seeds later for error bars.

---

## Stage 2c — Model C (Dream-7B diffusion backbone), single seed — HEADLINE

Identical recipe to D (Dream-7B, LoRA r=64, unfrozen SigLIP, standalone expert),
same shared norm stats, single seed. Trained PyTorch (torchrun DDP, global batch
32, 30k steps, exp `db_C_seed0_v2`), final loss ~0.020 (matches D/A′). Dream
integration required one fix: its `AutoModel` is a MaskedLM-head model
(`MaskedLMOutput`, no `last_hidden_state`), so `Pi0DreamModel._qwen_forward`
requests `output_hidden_states=True` and takes `hidden_states[-1]` (commit 8c5d8ec).

### Success rates (%), seed 0 — the C−D contrast

| Suite | **C (Dream, diff.)** | D (Qwen, AR) | **C − D** | A′ (Gemma, 3-seed) |
|-------|--:|--:|--:|--:|
| Spatial | 93.0 | 94.2 | −1.2 | 96.7 |
| Object  | 93.2 | 82.0 | **+11.2** | 92.6 |
| Goal    | 73.2 | 71.4 | +1.8 | 90.5 |
| Long(10)| 69.4 | 57.4 | **+12.0** | 79.6 |
| **Avg** | **82.2** | 76.3 | **+5.9** | 89.9 |

**HEADLINE: the diffusion prior wins, and most where AR is weakest.** C−D = +5.9
avg, concentrated on Long (+12.0) and Object (+11.2); Spatial a ceiling tie
(−1.2), Goal small positive (+1.8). Matched scale/init/expert/vision/recipe/seed
→ attributable to the pretraining objective. This is the pre-registered *positive*
outcome. C sits below A′/A in absolute terms (decoupling + LoRA-vs-full-FT cost),
but that cost is common to C and D and cancels in C−D.

Per-task rates (each suite, seed 0):
- Spatial: 0.98 0.94 1.0 1.0 0.76 0.92 1.0 0.82 1.0 0.88 → 0.930
- Object:  0.96 0.84 0.96 0.96 0.96 0.90 1.0 0.94 0.98 0.82 → 0.932
- Goal:    0.38 0.88 0.98 0.38 0.90 0.48 0.76 1.0 0.98 0.58 → 0.732
- Long:    0.80 0.96 0.68 0.90 0.72 0.90 0.66 0.92 0.28 0.12 → 0.694

**EVAL-HARNESS FAULT (resolved) — read before trusting any raw "Total success
rate" line.** The first C eval sweep produced garbage cumulative totals (spatial
"0.0", goal "0.184") from two independent failures: (1) an overlapping second
eval client tee'd into the same `eval_logs/c_seed0_*.log` (spatial showed 575
episodes / 12 task-rate lines instead of 500/10 — the two trailing 0.0 "tasks"
were the second client); (2) the single long-lived policy server **degraded
mid-suite** — goal ran last and tasks 4–10 all returned exactly 0.0 (dead-server
signature; `opening handshake failed` in the server log). The *per-task* rates
survived intact, so spatial (0.930) and object (0.932, clean 500/10) are read
from those. Goal + Long were re-run with a **fresh server per suite** and all
stale clients killed first — clean, no zero-tails → 0.732 / 0.694 above. Takeaway
for future evals: restart the server per suite, `pkill -f "libero/main.py"` before
launching, and always sanity-check per-task rates for a zero-tail, never trust the
cumulative total alone.

---

## Stage 3 — Representation probing (C vs D), seed 0

Frozen action-token features (mean-pooled, tau=1, conditioning-only) from the
trained C/D checkpoints, 4000 IDENTICAL frames (shuffle=False), capacity-matched
linear probes over 5 splits. Pipeline: `scripts/extract_probe_features.py` +
`openpi.diffusion_backbone.probing`.

### Linear-probe predictivity

| Target | C (Dream) | D (Qwen) | C − D |
|--------|--:|--:|--:|
| action_next (R²)   | 0.874 | 0.869 | +0.006 |
| action_chunk (R²)  | 0.889 | 0.887 | +0.002 |
| state (R²)         | 0.964 | 0.954 | +0.009 |
| gripper (acc/AUROC)| 0.969 (0.997) | 0.969 (0.997) | +0.001 |

**FINDING: rollout–probe DISSOCIATION.** C and D are tied at ceiling on every
low-level target (|C−D| ≤ 0.01), yet C rolls out +5.9 avg / +12.0 Long. So the
diffusion prior's behavioral win is NOT explained by better linear decodability of
low-level action/state/gripper — that info is equally present in both backbones.
Caveats written into paper §9.2: (1) ceiling (both 0.87–0.96, little headroom);
(2) targets are low-level, single-frame, in-distribution — the Long-horizon gap is
a closed-loop/compounding phenomenon static probing can't see; abstract targets
(phase, spatial relations) deferred (need sim labels); (3) MLP nonlinear probe did
NOT converge (max-iter, variance like 0.271±0.271) — inconclusive, omitted.
Interpretation: advantage acts via higher-level/temporal structure or closed-loop
robustness, not linear decodability of immediate control. Rules out the simplest
mechanistic explanation.
