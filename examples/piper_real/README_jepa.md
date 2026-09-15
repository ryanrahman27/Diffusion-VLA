# π₀.₅ + multi-horizon JEPA (PiperX)

Extension under `src/openpi/jepa/`. **SigLIP is unchanged.** Learnable query tokens cross-attend over patch-level SigLIP latents at horizons `t+2`, `t+5`, `t+10`, producing **12 prefix tokens** (4 queries × 3 horizons). Concat+project of those tokens **modulates the action expert via adaRMS** (added to the flow-matching timestep cond).

## Train

```bash
uv run python scripts/compute_norm_stats.py --config-name pi05_piperx_flatten
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run python scripts/train_pi05_jepa.py pi05_piperx_jepa \
  --exp-name=jepa_v1 --overwrite
```

## Knobs (`Pi0JepaConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `jepa_horizon_steps` | `(2, 5, 10)` | Future frame offsets (dataset frames) |
| `jepa_queries_per_horizon` | `4` | Cross-attn queries per horizon |
| `jepa_loss_weight` | `0.1` | Cosine JEPA loss scale |
| `num_future_prefix_tokens` | `12` | Property: horizons × queries |

## Inference

Same RGB / state / prompt as π₀.₅. JEPA prefix tokens and adaRMS cond are computed **online** from current frames only.
