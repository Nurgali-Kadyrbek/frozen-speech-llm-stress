"""Stage 4 — shortcut probes on R1.8 cohort A_R1p8_seed{1234..1238}.

Per-seed tts_vs_real (4-class) and domain (binary) on pool(A_R1.8(H)),
balanced sample 200 per source × 4 sources = 800 audios per seed.

Compare to Stage 3.0.5 raw H aggressive_aug baselines:
  tts_vs_real(raw H, aggressive_aug):  0.89
  domain(raw H, aggressive_aug):       0.99

PASS condition (per kickoff §4.2):
  Per-seed tts_vs_real(A_R1.8) < 0.79 (= 0.89 − 0.10) AND
  per-seed domain(A_R1.8) < 0.89 (= 0.99 − 0.10)
  on ≥ 4/5 seeds.

Run:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/stage4_shortcut_probes_cohort.py
"""
from __future__ import annotations

import json
import pickle
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_env, banner  # noqa: E402

setup_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.models.adapter import AdapterConfig, BLSPAdapter  # noqa: E402
from src.data.augment import AugmentConfig, augment_one, aggressive_stage3_config  # noqa: E402
from src.data.stress_data import load_stresspresso_test  # noqa: E402
from src.probes.shortcut_probes import fit_linear_probe  # noqa: E402

WAVLM_MODEL = "microsoft/wavlm-large"
SR = 16000

CACHE = Path("/raid/nurgaly/datasets/Beyond_Transcript_Alignment/cache")
STRESS_POOL_PATH = CACHE / "stress17k_probe_train_pool.pkl"
LIBRI_POOL_PATH  = CACHE / "librispeech_pool_n6000.pkl"
EXPR_POOL_PATH   = CACHE / "expresso_pool_n6000.pkl"

R1P8_DIR = ROOT / "outputs" / "stage3p8"
OUT_JSON = ROOT / "outputs" / "stage4" / "cohort_shortcut_probes.json"

# Stage 3.0.5 aggressive_aug baselines.
RAW_H_TTS_VS_REAL_BASELINE = 0.89
RAW_H_DOMAIN_BASELINE      = 0.99


def _load_pickle(p: Path) -> list[dict]:
    with open(p, "rb") as f:
        return pickle.load(f)


def collect_balanced_samples(*, n_per_source: int = 200, seed: int = 0,
                              max_audio_seconds: float = 8.0) -> list[dict]:
    rng = random.Random(seed)
    samples: list[dict] = []
    max_n = int(max_audio_seconds * SR)

    stress = _load_pickle(STRESS_POOL_PATH)
    rng.shuffle(stress)
    for r in stress[:n_per_source]:
        arr = np.asarray(r["audio_array"], dtype=np.float32)[:max_n]
        samples.append({"audio": arr, "source": "stress17k", "voice": r["voice_name"] or "unk"})

    libri = _load_pickle(LIBRI_POOL_PATH)
    rng.shuffle(libri)
    for r in libri[:n_per_source]:
        arr = np.asarray(r["audio"], dtype=np.float32)[:max_n]
        samples.append({"audio": arr, "source": "librispeech",
                        "voice": str(r["meta"].get("speaker_id", "unk"))})

    expr = _load_pickle(EXPR_POOL_PATH)
    rng.shuffle(expr)
    for r in expr[:n_per_source]:
        arr = np.asarray(r["audio"], dtype=np.float32)[:max_n]
        samples.append({"audio": arr, "source": "expresso",
                        "voice": str(r["meta"].get("speaker_id", "unk"))})

    sp = load_stresspresso_test()
    rng.shuffle(sp)
    take = min(n_per_source, len(sp))
    import soxr
    for it in sp[:take]:
        arr = np.asarray(it.audio_array, dtype=np.float32)
        if it.audio_sr != SR:
            arr = soxr.resample(arr, it.audio_sr, SR).astype(np.float32)
        arr = arr[:max_n]
        samples.append({"audio": arr, "source": "stresspresso", "voice": it.speaker_id or "unk"})

    rng.shuffle(samples)
    return samples


@torch.no_grad()
def adapter_pool_features(*, adapter, wavlm, feat_ex, samples, device,
                          aug_cfg: AugmentConfig, seed: int) -> tuple[np.ndarray, list[str]]:
    feats = []
    sources = []
    for i, s in enumerate(samples):
        x_aug = augment_one(s["audio"], cfg=aug_cfg, rng=random.Random(seed * 1009 + i))
        proc = feat_ex([x_aug], sampling_rate=SR, return_tensors="pt",
                       padding=True, return_attention_mask=True)
        iv = proc["input_values"].to(device).to(torch.bfloat16)
        am = proc["attention_mask"].to(device)
        wav_out = wavlm(input_values=iv, attention_mask=am, output_hidden_states=True)
        H = wav_out.hidden_states[16].float()
        sample_lengths = am.sum(dim=1)
        valid_T_s = wavlm._get_feat_extract_output_lengths(sample_lengths).long()
        K, vTk = adapter(H, valid_T_s=valid_T_s)
        Tk = int(vTk[0].item())
        feats.append(K[0, :Tk, :].mean(dim=0).cpu().float().numpy())
        sources.append(s["source"])
    return np.stack(feats), sources


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: no CUDA visible.", flush=True); return 1
    device = "cuda"

    banner("Loading WavLM-Large (frozen, bf16)")
    from transformers import WavLMModel, AutoFeatureExtractor
    feat_ex = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL)
    wavlm = WavLMModel.from_pretrained(WAVLM_MODEL, torch_dtype=torch.bfloat16).eval().to(device)
    for p in wavlm.parameters():
        p.requires_grad_(False)

    banner("Collecting balanced 800-sample evaluation set")
    samples = collect_balanced_samples(n_per_source=200, seed=0)
    print(f"  {len(samples)} samples", flush=True)

    aug_cfg = aggressive_stage3_config()

    seeds = [1234, 1235, 1236, 1237, 1238]
    by_seed = {}
    for s in seeds:
        ckpt_path = R1P8_DIR / f"A_R1p8_seed{s}.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: missing {ckpt_path} — skipping", flush=True)
            continue
        banner(f"Seed {s}: forward through A_R1.8 + run shortcut probes")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = AdapterConfig(**ckpt["adapter_config"])
        adapter = BLSPAdapter(cfg).to(device).to(torch.float32)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        adapter.eval()

        t0 = time.time()
        feats, sources = adapter_pool_features(
            adapter=adapter, wavlm=wavlm, feat_ex=feat_ex,
            samples=samples, device=device, aug_cfg=aug_cfg, seed=s,
        )
        print(f"  forward done in {time.time()-t0:.1f}s; feats {feats.shape}", flush=True)

        # tts_vs_real (4-class)
        src_classes = ["stress17k", "stresspresso", "librispeech", "expresso"]
        src_to_int  = {c: i for i, c in enumerate(src_classes)}
        y_4   = np.asarray([src_to_int[s] for s in sources], dtype=np.int64)
        res_tts = fit_linear_probe(feats, y_4, name="tts_vs_real", seed=0)

        # domain (binary)
        is_synth = np.asarray([s == "stress17k" for s in sources], dtype=np.int64)
        y_2  = np.where(is_synth == 1, 0, 1).astype(np.int64)
        res_dom = fit_linear_probe(feats, y_2, name="domain", seed=0)

        d_tts = res_tts.eval_acc - RAW_H_TTS_VS_REAL_BASELINE
        d_dom = res_dom.eval_acc - RAW_H_DOMAIN_BASELINE
        per_seed_pass = bool(
            res_tts.eval_acc < (RAW_H_TTS_VS_REAL_BASELINE - 0.10)
            and res_dom.eval_acc < (RAW_H_DOMAIN_BASELINE - 0.10)
        )
        print(f"  tts_vs_real eval_acc = {res_tts.eval_acc:.4f}  "
              f"(raw H baseline {RAW_H_TTS_VS_REAL_BASELINE}, drop {d_tts:+.4f})", flush=True)
        print(f"  domain      eval_acc = {res_dom.eval_acc:.4f}  "
              f"(raw H baseline {RAW_H_DOMAIN_BASELINE}, drop {d_dom:+.4f})", flush=True)
        print(f"  per-seed PASS: {per_seed_pass}", flush=True)

        by_seed[s] = {
            "tts_vs_real_eval_acc": res_tts.eval_acc,
            "tts_vs_real_train_acc": res_tts.train_acc,
            "tts_vs_real_drop_vs_raw_H_baseline":  d_tts,
            "domain_eval_acc":      res_dom.eval_acc,
            "domain_train_acc":     res_dom.train_acc,
            "domain_drop_vs_raw_H_baseline":       d_dom,
            "raw_H_tts_baseline":   RAW_H_TTS_VS_REAL_BASELINE,
            "raw_H_domain_baseline": RAW_H_DOMAIN_BASELINE,
            "per_seed_PASS":         per_seed_pass,
        }

        del adapter
        torch.cuda.empty_cache()

    n_pass = sum(1 for v in by_seed.values() if v["per_seed_PASS"])
    cohort_pass = n_pass >= 4

    banner("Cohort summary")
    tts_vals = [v["tts_vs_real_eval_acc"] for v in by_seed.values()]
    dom_vals = [v["domain_eval_acc"]      for v in by_seed.values()]
    if tts_vals:
        print(f"  tts_vs_real cohort mean = {np.mean(tts_vals):.4f} (σ={np.std(tts_vals, ddof=1):.4f})", flush=True)
    if dom_vals:
        print(f"  domain      cohort mean = {np.mean(dom_vals):.4f} (σ={np.std(dom_vals, ddof=1):.4f})", flush=True)
    print(f"  per-seed PASS in {n_pass}/{len(by_seed)} seeds", flush=True)
    print(f"  cohort PASS (≥4/5): {cohort_pass}", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "by_seed": by_seed,
        "cohort_summary": {
            "n_pass":        n_pass,
            "n_seeds":       len(by_seed),
            "cohort_PASS":   cohort_pass,
            "tts_vs_real_cohort_mean": float(np.mean(tts_vals)) if tts_vals else None,
            "tts_vs_real_cohort_sigma": float(np.std(tts_vals, ddof=1)) if len(tts_vals) > 1 else None,
            "domain_cohort_mean":     float(np.mean(dom_vals)) if dom_vals else None,
            "domain_cohort_sigma":    float(np.std(dom_vals, ddof=1)) if len(dom_vals) > 1 else None,
        },
        "raw_H_baselines": {
            "tts_vs_real": RAW_H_TTS_VS_REAL_BASELINE,
            "domain":      RAW_H_DOMAIN_BASELINE,
        },
    }, indent=2))
    print(f"\nsaved → {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
