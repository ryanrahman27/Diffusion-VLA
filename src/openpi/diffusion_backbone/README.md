# Diffusion-Trained Backbones for Flow-Matching VLA Policies

Controlled study: does a **diffusion-trained** bidirectional backbone (Dream-7B)
produce better action-token hidden states than a **matched autoregressive**
backbone (Qwen2.5-7B) when both are paired with the *same* flow-matching action
expert? The backbone is the only controlled variable; the action expert, action
space, flow objective, LIBERO dataset, and control stack are held fixed.

The full write-up is in [`docs/proposal.tex`](docs/proposal.tex).

## Model variants

| Variant | Backbone | Role |
|---|---|---|
| **A** | stock pi0.5 prefix-LM (Gemma) | honest baseline (deployed model) |
| **D** | Qwen2.5-7B (AR) | matched control — Dream's base model |
| **C** | Dream-7B (diffusion, init from Qwen2.5-7B) | main proposed model |
| **B** | Model D's backbone + expanded attention topology | topology ablation |
| C-large | DiffusionGemma 26B MoE | optional scale confirmation |

Contrasts: **`C − D`** = diffusion prior (the identifying claim), `B − D` =
attention-topology effect, `C − A` = confounded upper bound.

## Layout

```
diffusion_backbone/
├── constants.py        locked params: models, seeds, k, LIBERO suites, action dim
├── backbones/          shared backbone interface + per-variant adapters (A/B/C/D)
│   ├── base.py         BackboneAdapter contract (enforces the controlled comparison)
│   ├── qwen_backbone.py    Model D
│   ├── dream_backbone.py   Model C (delegates to hidden_read for the k-step read)
│   └── topology.py         Model B
├── hidden_read.py      k-step diffusion hidden-state extraction (proposal Sec. 5.2)
├── model.py            wires a swappable backbone onto the shared flow expert
├── train_config.py     TrainConfig per (variant, seed); forks pi05_libero
├── probing/            representation probes + capacity matching (Sec. 9.2)
├── eval/               LIBERO sweep + statistics (Sec. 9.1)
└── docs/proposal.tex   the research proposal
```

Backbone integration follows openpi's `transformers_replace` pattern: patched HF
modeling code goes under
`src/openpi/models_pytorch/transformers_replace/models/{qwen2,dream}/`.
Model weights are pulled from the HuggingFace hub at runtime and cached — **never
committed** to this repo.

## Stages (from the proposal's training plan)

1. **Baseline reproduction** — reproduce the pi0.5 LIBERO baseline (Model A) with
   the existing `pi05_libero` config to lock the harness. *Start here.*
2. **Backbone swap** — Model C (Dream-7B).
3. **Topology ablation** — Model B.
4. **Matched control** — Model D (Qwen2.5-7B); compute `C − D`.
5. **LoRA vs full FT** — LoRA is the default (preserves the C-vs-D pretraining
   difference); full FT is a robustness check.
6. **LIBERO evaluation** — full sweep, seeds, CIs.
7. **Representation probing** — probes against simulator ground truth.

## Run matrix

Primary: 4 variants × 3 seeds = **12 training runs** + 12 eval sweeps + probing.
MVP: A/C/D on LIBERO-Long, 3 seeds = **9 runs** (still gives the `C − D` contrast).

Compute (measured): ~1.17 s/step full-FT on 4×A100 → ~670 A100-h, ~7 days.
On 4×H100/H200 at ~1–2 steps/s → ~3 days. H200's 141 GB matters only for C-large.
