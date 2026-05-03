"""Stage 3.0.5 — pre-training shortcut-probe sanity on raw H.

Run BEFORE training any R1 adapter. Three probes (voice_id, tts_vs_real,
domain) on `pool(WavLM_L16(augmented(x)))` — i.e., features from frozen
WavLM-L16 after the Stage 3.1 augmentation pipeline. No adapter in the
loop. If any probe scores ≥ 70 % on held-out, the augmentation pipeline
is NOT anonymizing the target signal. STOP and revise augmentation
before training (kickoff §3.0.5: cheap ~1 h, saves ~50 h of wasted
adapter training).

Sample composition (n=1000 default, balanced):
  250 Stress-17K-raw probe-train (nova / echo)
  250 LibriSpeech-clean-100 (real, many speakers)
  250 Expresso-read (real, ex01..ex04)
  250 StressPresso (real, held-out from training but allowed for probe-fit)

Run:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/stage3_pretrain_shortcut_probes.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_env, banner  # noqa: E402

setup_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.augment import AugmentConfig, augment_one, aggressive_stage3_config  # noqa: E402
from src.data.stress_data import load_stresspresso_test  # noqa: E402
from src.probes.shortcut_probes import run_three_shortcut_probes  # noqa: E402


WAVLM_MODEL = "microsoft/wavlm-large"
SR = 16000

CACHE = Path("/raid/nurgaly/datasets/Beyond_Transcript_Alignment/cache")
STRESS_POOL_PATH = CACHE / "stress17k_probe_train_pool.pkl"
LIBRI_POOL_PATH  = CACHE / "librispeech_pool_n6000.pkl"
EXPR_POOL_PATH   = CACHE / "expresso_pool_n6000.pkl"

OUT_DIR = ROOT / "outputs" / "stage3"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_per_source", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_audio_seconds", type=float, default=8.0)
    p.add_argument("--no_augment", action="store_true",
                   help="Sanity baseline: probe raw WavLM features without augmentation. "
                        "These should typically be HIGH (no anonymization yet).")
    p.add_argument("--aggressive", action="store_true",
                   help="Use aggressive_stage3_config(): apply_prob=1.0, tighter "
                        "bandlimit floor, lower SNR, RMS normalization. "
                        "Stage-3.0.5 revision after default fail.")
    p.add_argument("--out_suffix", type=str, default="",
                   help="Append to output JSON name (e.g. '_aggressive').")
    return p.parse_args()


def _load_pickle(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _trim(arr: np.ndarray, max_n: int) -> np.ndarray:
    return arr[:max_n] if arr.shape[0] > max_n else arr


def collect_sample_set(n_per_source: int, *, seed: int, max_audio_seconds: float
                       ) -> list[dict]:
    """Build a balanced pool of ~4*n_per_source items across the four sources.

    Each entry: {audio: np.float32 16k mono, source: str, voice: str}.
    """
    rng = random.Random(seed)
    samples: list[dict] = []
    max_n = int(max_audio_seconds * SR)

    # Stress-17K probe-train (TTS, nova/echo)
    print(f"  loading {STRESS_POOL_PATH.name}…", flush=True)
    stress = _load_pickle(STRESS_POOL_PATH)
    rng.shuffle(stress)
    for r in stress[:n_per_source]:
        samples.append({
            "audio":  _trim(np.asarray(r["audio_array"], dtype=np.float32), max_n),
            "source": "stress17k",
            "voice":  r["voice_name"] or "unk",
        })
    print(f"    +{min(n_per_source, len(stress))} stress17k", flush=True)

    # LibriSpeech (real)
    print(f"  loading {LIBRI_POOL_PATH.name}…", flush=True)
    libri = _load_pickle(LIBRI_POOL_PATH)
    rng.shuffle(libri)
    for r in libri[:n_per_source]:
        samples.append({
            "audio":  _trim(np.asarray(r["audio"], dtype=np.float32), max_n),
            "source": "librispeech",
            "voice":  str(r["meta"].get("speaker_id", "unk")),
        })
    print(f"    +{min(n_per_source, len(libri))} librispeech", flush=True)

    # Expresso (real, 4 speakers)
    print(f"  loading {EXPR_POOL_PATH.name}…", flush=True)
    expr = _load_pickle(EXPR_POOL_PATH)
    rng.shuffle(expr)
    for r in expr[:n_per_source]:
        samples.append({
            "audio":  _trim(np.asarray(r["audio"], dtype=np.float32), max_n),
            "source": "expresso",
            "voice":  str(r["meta"].get("speaker_id", "unk")),
        })
    print(f"    +{min(n_per_source, len(expr))} expresso", flush=True)

    # StressPresso (real, n=202 total; just take all if n_per_source > 202)
    print(f"  loading StressPresso (n=202)…", flush=True)
    sp = load_stresspresso_test()
    rng.shuffle(sp)
    take = min(n_per_source, len(sp))
    import soxr
    for it in sp[:take]:
        arr = np.asarray(it.audio_array, dtype=np.float32)
        if it.audio_sr != SR:
            arr = soxr.resample(arr, it.audio_sr, SR).astype(np.float32)
        samples.append({
            "audio":  _trim(arr, max_n),
            "source": "stresspresso",
            "voice":  it.speaker_id or "unk",
        })
    print(f"    +{take} stresspresso", flush=True)

    rng.shuffle(samples)
    return samples


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("ERROR: no CUDA visible.", flush=True); return 1
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    banner("Stage 3.0.5 — pre-training shortcut probes on raw H")
    print(f"  augmentation: {'OFF (no_augment baseline)' if args.no_augment else 'ON (Stage 3.1 pipeline)'}",
          flush=True)
    print(f"  n_per_source: {args.n_per_source}", flush=True)

    samples = collect_sample_set(
        args.n_per_source, seed=args.seed,
        max_audio_seconds=args.max_audio_seconds,
    )
    print(f"  total samples: {len(samples)}", flush=True)

    # ---- Load WavLM, augment + forward ----
    banner(f"Loading frozen WavLM ({WAVLM_MODEL}) in bf16")
    from transformers import WavLMModel, AutoFeatureExtractor
    feat_ex = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL)
    wavlm = WavLMModel.from_pretrained(WAVLM_MODEL, torch_dtype=torch.bfloat16).eval().to(device)
    for p in wavlm.parameters():
        p.requires_grad_(False)

    cfg = AugmentConfig()
    if args.no_augment:
        cfg = AugmentConfig(apply_prob=0.0)
    elif args.aggressive:
        cfg = aggressive_stage3_config()
    print(f"  augment cfg: apply_prob={cfg.apply_prob}, bandlimit_choices={cfg.bandlimit_choices}, "
          f"snr_db_choices={cfg.snr_db_choices}, rms_norm_target={cfg.rms_norm_target}, "
          f"use_ffmpeg_codecs={cfg.use_ffmpeg_codecs}", flush=True)

    banner("Forward all samples through WavLM-L16 (frozen)")
    feats = []
    sources = []
    voices = []
    t0 = time.time()
    for i, s in enumerate(samples):
        rng_pair = random.Random(args.seed * 1003 + i)
        x_aug = augment_one(s["audio"], cfg=cfg, rng=rng_pair)
        with torch.no_grad():
            proc = feat_ex([x_aug], sampling_rate=SR, return_tensors="pt",
                           padding=True, return_attention_mask=True)
            iv = proc["input_values"].to(device).to(torch.bfloat16)
            am = proc["attention_mask"].to(device)
            out = wavlm(input_values=iv, attention_mask=am, output_hidden_states=True)
            H = out.hidden_states[16]               # (1, T_s, 1024)
            sample_lengths = am.sum(dim=1)
            valid_T_s = wavlm._get_feat_extract_output_lengths(sample_lengths).long()
            T = int(valid_T_s[0].item())
            pooled = H[0, :T, :].float().mean(dim=0).cpu().numpy()  # (1024,)
        feats.append(pooled)
        sources.append(s["source"])
        voices.append(s["voice"])
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(samples)} done in {elapsed:.1f}s ({(i+1)/elapsed:.1f} samples/s)",
                  flush=True)
    elapsed = time.time() - t0
    print(f"  forward complete: {elapsed:.1f}s  ({len(samples)/elapsed:.2f} samples/s)", flush=True)

    feats = np.asarray(feats)
    print(f"  features shape: {feats.shape}, dtype: {feats.dtype}", flush=True)

    # ---- Run three probes ----
    banner("Linear probes (sklearn LogisticRegression)")
    results = run_three_shortcut_probes(
        features=feats, source=sources, voice_or_speaker=voices, seed=args.seed,
    )
    for name, r in results.items():
        flag = "  ⚠ FAIL-70pct" if r.fail_70pct else "  PASS"
        if np.isnan(r.eval_acc):
            print(f"  {name:<14}  (skipped — insufficient data)", flush=True)
        else:
            print(f"  {name:<14}  n_classes={r.n_classes}  chance={r.chance:.4f}  "
                  f"train_acc={r.train_acc:.4f}  eval_acc={r.eval_acc:.4f}{flag}",
                  flush=True)

    # ---- Decision ----
    any_fail = any(r.fail_70pct for r in results.values() if not np.isnan(r.eval_acc))
    print("", flush=True)
    if any_fail:
        print("=== STAGE 3.0.5 DECISION: AUGMENTATION INSUFFICIENT ===", flush=True)
        print("  At least one shortcut probe ≥ 70 % on raw H after augmentation.", flush=True)
        print("  STOP — revise augmentation before training any R1 cell.", flush=True)
    else:
        print("=== STAGE 3.0.5 DECISION: PASS ===", flush=True)
        print("  All probes < 70 %. Augmentation pipeline anonymizes the targeted shortcuts.", flush=True)
        print("  Proceed to Stage 3.2 (L_R1 implementation).", flush=True)

    payload = {
        "args":     vars(args),
        "n_total":  len(samples),
        "n_per_source": args.n_per_source,
        "feat_dim": int(feats.shape[1]),
        "results":  {k: asdict(v) for k, v in results.items()},
        "decision": "PASS" if not any_fail else "FAIL",
    }
    out_path = OUT_DIR / f"pretrain_shortcut_probes{args.out_suffix}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved → {out_path}", flush=True)
    return 0 if not any_fail else 3


if __name__ == "__main__":
    sys.exit(main())
