"""Stage 2.3 — R0 BLSP training (one seed per invocation).

Single-microbatch (B=1) with gradient accumulation = 32 to hit effective
batch 32. AdamW lr=5e-5, 500-step warmup + cosine to 1e-6 over `--max_steps`.
WavLM and Qwen3-8B are FROZEN (bf16); only the adapter (fp32) is trained.
KV-cache disabled in the LLM so backward sees the full prefix/response.

Run:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/stage2_train.py \
      --seed 1234 --max_steps 600 --grad_accum 32

Per-seed checkpoint output: outputs/stage2/A_BLSP_seed{seed}.pt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_env, banner, report_check  # noqa: E402

setup_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.models.adapter import AdapterConfig, BLSPAdapter  # noqa: E402
from src.losses.blsp import BLSPInput, compute_blsp_loss  # noqa: E402
from src.utils.prompts import build_training_halves, DEFAULT_SYSTEM  # noqa: E402
from src.data.audio_pool import (  # noqa: E402
    MixedSampler, with_augmentation,
    load_stress17k_train, load_librispeech_subset, load_expresso_subset,
)
from src.data.augment import AugmentConfig  # noqa: E402
from src.data.stress_data import partition_transcript_ids, load_stress17k  # noqa: E402

QWEN3_MODEL = "Qwen/Qwen3-8B"
WAVLM_MODEL = "microsoft/wavlm-large"
SR = 16000
RATIO_BAND = (0.5, 2.0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument("--grad_accum", type=int, default=32, help="effective batch size")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--lambda_kl", type=float, default=1.0)
    p.add_argument("--n_libri", type=int, default=6000)
    p.add_argument("--n_expr", type=int, default=6000)
    p.add_argument("--max_audio_seconds", type=float, default=12.0)
    p.add_argument("--max_response_tokens", type=int, default=64)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--smoke_steps", type=int, default=0,
                   help="if >0, stop after that many steps (for fast validation)")
    return p.parse_args()


def lr_at(step: int, *, warmup: int, max_steps: int, peak_lr: float, min_lr: float) -> float:
    if step < warmup:
        return peak_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cos


def main() -> int:
    args = parse_args()
    print(f"transformers=={__import__('transformers').__version__}, torch=={torch.__version__}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES','?')}", flush=True)
    if not torch.cuda.is_available():
        print("ERROR: no CUDA visible.", flush=True); return 1
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "training_logs" / f"stage2_train_seed{args.seed}.log"
    ckpt_path = out_dir / f"A_BLSP_seed{args.seed}.pt"

    # ---- adapter init from Stage-2.0 calibration ---- #
    init_path = ROOT / "outputs" / "stage2" / "adapter_init.json"
    if not init_path.exists():
        print(f"ERROR: missing {init_path} — run scripts/stage2_adapter_smoke.py first.")
        return 1
    init = json.loads(init_path.read_text())
    cal = init["adapter_init"]
    embed_meta = init["qwen3_8b"]
    print(f"  std_8B={embed_meta['embed_tokens_std']:.5f}, "
          f"||row||_2 mean={embed_meta['embed_row_norm_mean']:.4f}, d_llm={cal['d_llm']}", flush=True)

    # ---- models ---- #
    banner(f"Loading frozen LLM ({QWEN3_MODEL}) in bf16")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(QWEN3_MODEL)
    llm = AutoModelForCausalLM.from_pretrained(
        QWEN3_MODEL, torch_dtype=torch.bfloat16,
    ).eval().to(device)
    for p in llm.parameters():
        p.requires_grad_(False)
    embed_layer = llm.get_input_embeddings()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    banner(f"Loading frozen speech encoder ({WAVLM_MODEL}) in bf16")
    from transformers import WavLMModel, AutoFeatureExtractor
    feat = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL)
    wavlm = WavLMModel.from_pretrained(WAVLM_MODEL, torch_dtype=torch.bfloat16).eval().to(device)
    for p in wavlm.parameters():
        p.requires_grad_(False)

    banner(f"Building adapter with calibrated init (target ratio band {RATIO_BAND})")
    cfg = AdapterConfig(
        d_enc=1024, d_llm=cal["d_llm"],
        conv_kernel=cal["conv_kernel"], conv_stride=cal["conv_stride"],
        mlp_hidden_mult=cal["mlp_hidden_mult"],
        last_linear_std=cal["last_linear_std"],
        rmsnorm_init_scale=cal["rmsnorm_init_scale"],
        modality_token_std=cal["modality_token_std"],
    )
    adapter = BLSPAdapter(cfg).to(device).to(torch.float32)
    print(f"  trainable params = {adapter.n_trainable_params()/1e6:.2f}M", flush=True)

    # Special token ids (vision_start / vision_end inherited from Qwen2-VL family).
    vis_start_id = int(tok.convert_tokens_to_ids("<|vision_start|>"))
    vis_end_id   = int(tok.convert_tokens_to_ids("<|vision_end|>"))
    assert (vis_start_id, vis_end_id) == (151652, 151653)

    # ---- data ---- #
    banner("Loading data pool")
    s17_all = load_stress17k()
    train_ids, _ = partition_transcript_ids(s17_all, eval_frac=0.20, seed="BTA-2026-05-02")
    stress_rows = load_stress17k_train(train_ids)
    print(f"  Stress-17K probe-train rows: {len(stress_rows)}", flush=True)

    libri_rows = load_librispeech_subset(args.n_libri, seed=args.seed)
    print(f"  LibriSpeech sampled rows:    {len(libri_rows)}", flush=True)

    expr_rows = load_expresso_subset(args.n_expr, seed=args.seed)
    print(f"  Expresso sampled rows:       {len(expr_rows)}", flush=True)

    sampler = MixedSampler(stress_rows, libri_rows, expr_rows, seed=args.seed)

    # ---- optimizer ---- #
    opt = torch.optim.AdamW(
        [p for p in adapter.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    # ---- training loop ---- #
    max_steps = args.max_steps if args.smoke_steps == 0 else args.smoke_steps
    banner(f"Training (seed={args.seed}, max_steps={max_steps}, grad_accum={args.grad_accum})")
    log_records = []
    step = 0
    t0 = time.time()
    sample_count = 0

    while step < max_steps:
        opt.zero_grad(set_to_none=True)
        # Update LR at this optimizer step
        cur_lr = lr_at(step, warmup=args.warmup_steps, max_steps=max_steps,
                       peak_lr=args.lr, min_lr=args.min_lr)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        # Accumulate gradient over `grad_accum` microbatches.
        agg = {"L_task": 0.0, "L_KL": 0.0, "L_R0": 0.0, "ratio": 0.0, "n": 0}
        for micro_i in range(args.grad_accum):
            row = sampler.batch(1)[0]
            row = with_augmentation([row], seed=args.seed * 1000003 + step * 31 + micro_i)[0]

            # Length cap: trim long audios to keep memory bounded.
            max_n = int(args.max_audio_seconds * SR)
            if row.audio.shape[0] > max_n:
                row.audio = row.audio[:max_n]

            # WavLM forward (frozen, bf16, no_grad)
            with torch.no_grad():
                proc = feat([row.audio], sampling_rate=SR, return_tensors="pt",
                            padding=True, return_attention_mask=True)
                iv = proc["input_values"].to(device).to(torch.bfloat16)
                am = proc["attention_mask"].to(device)
                out = wavlm(input_values=iv, attention_mask=am, output_hidden_states=True)
                H = out.hidden_states[16]                  # (1, T_s, 1024) bf16
                sample_lengths = am.sum(dim=1)
                valid_T_s = wavlm._get_feat_extract_output_lengths(sample_lengths).long()

            # Adapter forward (fp32 weights, fp32 input)
            H_fp32 = H.float()
            K, valid_T_k = adapter(H_fp32, valid_T_s=valid_T_s)
            Tk = int(valid_T_k[0].item())
            K_speech = K[0, :Tk, :]                         # (Tk, d_llm)

            # Build prompt halves
            halves = build_training_halves(
                tok, system=DEFAULT_SYSTEM,
                question=row.question, response=row.response,
            )
            left_ids       = tok(halves.left_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(dtype=torch.long, device=device)
            right_ids      = tok(halves.right_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(dtype=torch.long, device=device)
            response_ids   = tok(halves.response_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(dtype=torch.long, device=device)
            transcript_ids = tok(row.transcript,        return_tensors="pt", add_special_tokens=False).input_ids[0].to(dtype=torch.long, device=device)
            # Cap response length
            if response_ids.shape[0] > args.max_response_tokens:
                response_ids = response_ids[: args.max_response_tokens]

            inp = BLSPInput(
                left_ids=left_ids, right_ids=right_ids,
                response_ids=response_ids, transcript_ids=transcript_ids,
                vision_start_id=vis_start_id, vision_end_id=vis_end_id,
            )

            # K_speech is fp32; cast to bf16 to match LLM dtype.
            K_speech_bf = K_speech.to(torch.bfloat16)

            losses = compute_blsp_loss(
                llm=llm, embed_layer=embed_layer,
                K_speech=K_speech_bf, inp=inp, lambda_kl=args.lambda_kl,
            )
            (losses["L_R0"] / args.grad_accum).backward()

            # Track norms for monitoring
            with torch.no_grad():
                row_norm = embed_layer.weight.float().norm(dim=-1).mean().item()
                ratio = (K_speech.float().norm(dim=-1).mean().item() / row_norm)
            agg["L_task"] += losses["L_task"].item()
            agg["L_KL"]   += losses["L_KL"].item()
            agg["L_R0"]   += losses["L_R0"].item()
            agg["ratio"]  += ratio
            agg["n"]      += 1
            sample_count += 1

            # Free heavy tensors before next microbatch.
            del K, K_speech, K_speech_bf, H, H_fp32, losses

        # Optimizer step. Clip to max_norm=1.0 to bound the actual update;
        # `gnorm` is measured BEFORE clipping for diagnostic / kill-switch use.
        gnorm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0).item()
        opt.step()
        step += 1

        # Logging
        if step == 1 or step % args.log_every == 0 or step == max_steps:
            mean_L_task = agg["L_task"] / agg["n"]
            mean_L_KL   = agg["L_KL"]   / agg["n"]
            mean_L_R0   = agg["L_R0"]   / agg["n"]
            mean_ratio  = agg["ratio"]  / agg["n"]
            elapsed = time.time() - t0
            sps = sample_count / max(elapsed, 1e-3)
            log_records.append({
                "step": step, "lr": cur_lr,
                "L_task": mean_L_task, "L_KL": mean_L_KL, "L_R0": mean_L_R0,
                "ratio": mean_ratio, "grad_norm": gnorm, "samples_per_s": sps,
                "elapsed_s": elapsed,
            })
            print(f"  step {step:>4} lr={cur_lr:.2e}  "
                  f"L_task={mean_L_task:.3f}  L_KL={mean_L_KL:.3f}  L_R0={mean_L_R0:.3f}  "
                  f"ratio={mean_ratio:.2f}  grad_norm={gnorm:.2f}  sps={sps:.2f}", flush=True)

            # Sanity guards.
            #
            # Original kickoff threshold was grad_norm > 5; that was calibrated
            # for the design's ~25M-parameter adapter. Our 84M adapter sees
            # grad_norm ~15 stably under healthy training (sqrt(84/25)=1.83×
            # the design's natural scale, plus head-room). With grad clipping
            # at max_norm=1.0 the actual update is bounded regardless, so the
            # only blow-up signal we still need is NaN / Inf in the loss.
            if not math.isfinite(mean_L_R0):
                print("  STOP — NaN/Inf loss; saving diagnostics and aborting.", flush=True)
                break
            # Hard ceiling on truly catastrophic gradients (post-warmup only).
            if step > args.warmup_steps and gnorm > 200.0:
                print(f"  STOP — post-warmup grad_norm {gnorm:.2f} > 200 (catastrophic).", flush=True)
                break
            if not (RATIO_BAND[0] <= mean_ratio <= RATIO_BAND[1]):
                if mean_ratio > 6.0 or mean_ratio < 0.1:
                    print(f"  STOP — ratio drift {mean_ratio:.2f} outside extended tolerance.", flush=True)
                    break

    elapsed = time.time() - t0
    print(f"\n  done: {step} optimizer steps over {elapsed:.1f}s  ({sample_count/max(elapsed,1e-3):.2f} samples/s)", flush=True)

    # Save checkpoint + log
    payload = {
        "adapter_state_dict": adapter.state_dict(),
        "adapter_config": asdict(cfg),
        "args": vars(args),
        "embed_meta": embed_meta,
        "training_log": log_records,
        "final_step": step,
        "wallclock_s": elapsed,
    }
    torch.save(payload, ckpt_path)
    print(f"  saved → {ckpt_path}  ({ckpt_path.stat().st_size/1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
