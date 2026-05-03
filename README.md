# Frozen Speech-to-LLM Adapter Stress Test

Counterfactual training and evaluation harness for frozen-frozen
speech-to-LLM adapters on the StressTest sentence-stress benchmark.
Code, configurations, evaluation logs, and per-seed result JSONs.

## Repo layout

```
src/                       adapter, losses, probes, data modules
scripts/                   stage{0..7} train + eval + smoke_test
configs/locked_cells.yaml  append-only per-stage configuration
results/                   per-seed summary.json, grad_norm logs, raw training logs
```

## Locked stack

- WavLM-Large layer 16 frozen (`microsoft/wavlm-large`)
- 84M-param adapter: Conv1d(1024→4096, k=4, s=4) → MLP-2 → RMSNorm → +modality token
- Qwen3-8B-Instruct frozen (`Qwen/Qwen3-8B`, `enable_thinking=False`)
- AdamW lr=5e-5, 400 steps, warmup 300 (R1.8) / 500 (R0), cosine to 1e-6
- Pinned: `transformers==4.51.3`, `torch>=2.5,<2.7`, `peft>=0.13`

## Reproduce

```bash
pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES=<gpu>           # ~24 GB VRAM sufficient
export HF_HOME=/path/to/hf_cache             # large-disk cache

# Smoke test (no model loads; verifies cohort-sanity numbers)
python scripts/smoke_test.py

# R0 BLSP baseline (5 seeds, ~30 min/seed on A100)
for s in 1234 1235 1236 1237 1238; do
  python scripts/stage2_train.py --seed $s --max_steps 600
done

# R1.8 cohort (5 seeds, ~37 min/seed)
for s in 1234 1235 1236 1237 1238; do
  python scripts/stage3_train.py --seed $s \
    --lambda_kl 1.0 --lambda_cf 5.0 --lambda_artifact 1.0 \
    --lambda_cond 0.5 --lambda_nce 0.0 --max_steps 400 \
    --aug aggressive_stage3_config
done

# Stage 4 controls
bash scripts/stage4_run_controls.sh
bash scripts/stage4_run_control_b.sh
python scripts/stage4_shortcut_probes_cohort.py
python scripts/stage4_finalize.py

# Stage 6 LoRA
python scripts/stage3_train.py --seed 1234 --lora_rank 4 --lora_alpha 16 \
  --frozen_adapter_ckpt outputs/stage3p8/A_R1p8_seed1234.pt --max_steps 400
python scripts/stage3_train.py --seed 1234 --lora_rank 8 --lora_alpha 32 \
  --frozen_adapter_ckpt outputs/stage3p8/A_R1p8_seed1234.pt --max_steps 400

# Stage 7 styled-teacher pilot
python scripts/stage3_train.py --seed 1234 --styled_teacher \
  --lambda_kl 1.0 --lambda_cf 5.0 --max_steps 400
```

Total compute: ~250 GPU-h on a single A100-class device.

## Data access

This repo contains derived data (`cf_pairs_*.jsonl`, `proj_P.pt`,
per-seed eval logs) but **no raw audio**. Audio is accessed via
HuggingFace Hub:

- `slprl/StressPresso` (CC-BY-NC-4.0)
- `slprl/Stress-17K-raw` (CC-BY-NC-4.0)
- `ylacombe/expresso` (CC-BY-NC-4.0)
- `openslr/librispeech_asr`, config `clean` (CC-BY-4.0)

Adapter checkpoints (~6.1 GB total across cohorts) are released on
HuggingFace:

- Collection: [`nur-dev/beyond-transcript-alignment`](https://huggingface.co/collections/nur-dev/beyond-transcript-alignment-69f7af0198a059e1d6b9bd5f)

Per-repo:

| Repo | Contents | Size |
|---|---|---|
| [`datasets/nur-dev/stress17k-counterfactual-pairs`](https://huggingface.co/datasets/nur-dev/stress17k-counterfactual-pairs) | cf_pairs jsonl + desc_only_baseline + proj_P | 0.10 GB |
| [`nur-dev/frozen-stress-r0-blsp`](https://huggingface.co/nur-dev/frozen-stress-r0-blsp) | R0 BLSP cohort, 5 seeds | 1.68 GB |
| [`nur-dev/frozen-stress-r1p8-counterfactual`](https://huggingface.co/nur-dev/frozen-stress-r1p8-counterfactual) | R1.8 counterfactual cohort, 5 seeds | 1.78 GB |
| [`nur-dev/frozen-stress-stage4-controla`](https://huggingface.co/nur-dev/frozen-stress-stage4-controla) | Stage 4 Control A (text-only adapter), 3 seeds | 1.07 GB |
| [`nur-dev/frozen-stress-stage4-controlb`](https://huggingface.co/nur-dev/frozen-stress-stage4-controlb) | Stage 4 Control B (shuffled audio), 3 seeds | 1.07 GB |
| [`nur-dev/frozen-stress-lora-r4`](https://huggingface.co/nur-dev/frozen-stress-lora-r4) | Stage 6 LoRA rank 4 (Qwen3-8B q_proj/v_proj) | 0.01 GB |
| [`nur-dev/frozen-stress-lora-r8`](https://huggingface.co/nur-dev/frozen-stress-lora-r8) | Stage 6 LoRA rank 8 | 0.02 GB |
| [`nur-dev/frozen-stress-styled-teacher`](https://huggingface.co/nur-dev/frozen-stress-styled-teacher) | Stage 7 styled-teacher bridging adapter | 0.36 GB |

## Configuration

Experimental thresholds, λ schedules, and decision rules are pinned in
[`configs/locked_cells.yaml`](configs/locked_cells.yaml) using an
append-only per-stage structure.

## License

- Code: BSD-3-Clause (see [`LICENSE`](LICENSE))
- Adapter checkpoints (HF): CC-BY-NC-4.0 (inherited from training datasets)
- Derived data (`cf_pairs_*.jsonl`): CC-BY-NC-4.0 (inherited from Stress-17K-raw)

Adapter weights were trained on CC-BY-NC-4.0 datasets and inherit the
non-commercial restriction. Use is permitted for academic research,
ablation studies, reproducibility checks, and pedagogy.
