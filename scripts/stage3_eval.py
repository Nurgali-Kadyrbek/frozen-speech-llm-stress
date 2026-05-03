"""Stage 3.4 — per-seed evaluation of an A_R1 checkpoint.

Computes (in addition to the Stage-2 components):
  (A)  Probe-G(A_R1)       — adapter, 20 paraphrases × 202 StressPresso items
  (B)  Probe-K(A_R1)       — linear + MLP-2 on pool(A_R1(H))
  (C)  Cascade-T baseline  — Whisper ASR transcript → Qwen3-8B Probe-G
  (D)  K_T  baseline       — embed(true transcript) → Qwen3-8B Probe-G
  (E)  Probe-G-oracle re-confirm — Stage 1a's exact prompt on Qwen3-8B

  C.1  Within-domain Probe-K stratified by {Stress-17K-nova,
       Stress-17K-echo, StressPresso}
  C.2  Cross-domain transfer Probe-K — train on Stress-17K probe-train
       (TTS only); eval on StressPresso (real only)
  C.3  Domain-stratified Probe-G on StressPresso — per speaker_id
       and R0-vs-R1 directional consistency
  C.4  Domain-separability red flag — binary linear probe on
       pool(A_R1(H)) vs raw pool(WavLM_L16(H))

  T-only — C_phi_t_only diagnostic on Stress-17K probe-eval
           (kickoff §3.2: > 0.30 = lexical contamination).

Run:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/stage3_eval.py \
      --seed 1234 --checkpoint outputs/stage3/A_R1_seed1234.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_env, banner  # noqa: E402

setup_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.models.adapter import AdapterConfig, BLSPAdapter  # noqa: E402
from src.losses.r1 import CPhi, restricted_t_features  # noqa: E402
from src.data.stress_data import (  # noqa: E402
    load_stresspresso_test, load_stress17k, partition_transcript_ids,
)
from src.utils.prompts import AUDIO_PLACEHOLDER, DEFAULT_SYSTEM  # noqa: E402
from src.probes.probe_g import (  # noqa: E402
    SYSTEM_PROBE_G, NEUTRAL_PARAPHRASES, EXPLICIT_PARAPHRASES,
    bootstrap_by_cluster,
)
from src.probes.probe_k import (  # noqa: E402
    ProbeK, FitConfig, fit_probe, predict, within_transcript_argmax,
    accuracy as acc_fn,
)
from src.probes.shortcut_probes import fit_linear_probe  # noqa: E402

# Re-use Stage-2 helpers verbatim so behavior matches.
from scripts.stage2_eval import (  # noqa: E402
    chat_text_with_audio_marker, long_tensor, score_two_candidates,
    run_probe_g_variant, adapter_audio_embeddings, text_embeddings_from_string,
    whisper_asr_transcripts, adapter_pool_for_probe_k,
)


# ---------- Stage 4 Control A — text-only forward path ---------- #

def _pad_text_to_multiple_of_4(K_T_pertoken: torch.Tensor, proj_P_W: torch.Tensor,
                                proj_P_b: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Apply P projection and pad to target_T = max(4, ceil(T_text/4)*4)."""
    H_text = K_T_pertoken @ proj_P_W + proj_P_b           # (T_text, 1024)
    T_text = H_text.shape[0]
    target_T = max(4, ((T_text + 3) // 4) * 4)
    if target_T > T_text:
        pad_len = target_T - T_text
        pad = torch.zeros(pad_len, H_text.shape[1],
                           dtype=H_text.dtype, device=H_text.device)
        H_text = torch.cat([H_text, pad], dim=0)
    return H_text, target_T


@torch.no_grad()
def adapter_audio_embeddings_text_only(adapter, embed_layer, tokenizer,
                                       proj_P_W, proj_P_b, items, device) -> list[torch.Tensor]:
    """Like adapter_audio_embeddings but uses P(embed_tokens(transcript))
    instead of WavLM(audio). For Stage 4 Control A eval."""
    out = []
    for it in items:
        ids = tokenizer(it.transcription, return_tensors="pt",
                        add_special_tokens=False).input_ids[0].to(
                            dtype=torch.long, device=device)
        K_T_pertoken = embed_layer(ids).float()        # (T_text, 4096)
        H_text, target_T = _pad_text_to_multiple_of_4(K_T_pertoken, proj_P_W, proj_P_b)
        H_fp32 = H_text.unsqueeze(0)                    # (1, target_T, 1024)
        valid_T_s = torch.tensor([target_T], dtype=torch.long, device=device)
        K, vTk = adapter(H_fp32, valid_T_s=valid_T_s)
        Tk = int(vTk[0].item())
        out.append(K[0, :Tk, :].to(torch.bfloat16))
    return out


@torch.no_grad()
def adapter_pool_for_probe_k_text_only(adapter, embed_layer, tokenizer,
                                       proj_P_W, proj_P_b, items, device) -> torch.Tensor:
    """Mean-pool A_textK(P(K_T)) per item; shape (N, 1, d_llm)."""
    pool_list = []
    for it in items:
        ids = tokenizer(it.transcription, return_tensors="pt",
                        add_special_tokens=False).input_ids[0].to(
                            dtype=torch.long, device=device)
        K_T_pertoken = embed_layer(ids).float()
        H_text, target_T = _pad_text_to_multiple_of_4(K_T_pertoken, proj_P_W, proj_P_b)
        H_fp32 = H_text.unsqueeze(0)
        valid_T_s = torch.tensor([target_T], dtype=torch.long, device=device)
        K, vTk = adapter(H_fp32, valid_T_s=valid_T_s)
        Tk = int(vTk[0].item())
        pool_list.append(K[0, :Tk, :].mean(dim=0).cpu().float())
    return torch.stack(pool_list).unsqueeze(1)

QWEN3_MODEL = "Qwen/Qwen3-8B"
WAVLM_MODEL = "microsoft/wavlm-large"
SR = 16000
N_MAX_CLASSES = 13


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--skip_cascade", action="store_true",
                   help="skip Whisper-ASR cascade-T (saves ~5 min)")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--bootstrap_iter", type=int, default=1000)
    p.add_argument("--control_mode", type=str, default="none",
                   choices=("none", "text_only"),
                   help="text_only = Stage 4 Control A: feed P(embed_tokens(transcript)) "
                        "to adapter instead of WavLM(audio). Used for A_textK eval.")
    p.add_argument("--proj_P_path", type=str, default=None,
                   help="Path to outputs/stage4/proj_P.pt for control_mode=text_only.")
    p.add_argument("--lora_path", type=str, default=None,
                   help="Stage 6: directory containing PEFT LoRA weights to load on top of Qwen3-8B.")
    return p.parse_args()


# ---------- Branch-D diagnostics (C.1 / C.2 / C.4) ---------- #

def split_s17_eval_by_voice(s17_eval) -> dict[str, list]:
    """Stratify Stress-17K probe-eval by TTS voice."""
    nova = [it for it in s17_eval if it.voice_name == "nova"]
    echo = [it for it in s17_eval if it.voice_name == "echo"]
    return {"nova": nova, "echo": echo}


def probe_k_eval_subset(probe, pool: torch.Tensor, n_words: torch.Tensor,
                        y: torch.Tensor, device: str) -> float:
    """Run linear probe on a subset of pool/n_words and compare to y."""
    p = predict(probe, pool, n_words, device)
    return float(acc_fn(p, y))


def diagnostic_c1_within_domain(
    *, probe_linear, pool_eval, y_eval, n_words_eval, voice_eval,
    pool_sp, y_sp, n_words_sp, device, chance: float,
) -> dict:
    """C.1 — within-domain Probe-K stratified by {nova, echo, real}.

    PASS condition: each subset accuracy within ≤ 8 pp of cross-subset mean
    AND each above (chance + 0.10) absolute floor.
    """
    nova_idx = np.array([i for i, v in enumerate(voice_eval) if v == "nova"])
    echo_idx = np.array([i for i, v in enumerate(voice_eval) if v == "echo"])

    subsets = {}
    if len(nova_idx) > 0:
        subsets["nova"] = (
            pool_eval[nova_idx], y_eval[nova_idx], n_words_eval[nova_idx],
        )
    if len(echo_idx) > 0:
        subsets["echo"] = (
            pool_eval[echo_idx], y_eval[echo_idx], n_words_eval[echo_idx],
        )
    subsets["stresspresso"] = (pool_sp, y_sp, n_words_sp)

    accs = {}
    for k, (P, Y, NW) in subsets.items():
        accs[k] = probe_k_eval_subset(probe_linear, P, NW, Y, device)

    accs_arr = np.array(list(accs.values()))
    mean = float(accs_arr.mean())
    spread = float(accs_arr.max() - accs_arr.min())
    above_floor = bool((accs_arr >= chance + 0.10).all())
    within_8pp = bool((np.abs(accs_arr - mean) <= 0.08).all())

    return {
        "by_subset": accs,
        "n_per_subset": {k: int(len(v[1])) for k, v in subsets.items()},
        "cross_subset_mean": mean,
        "max_spread": spread,
        "above_chance_plus_10pp_floor": above_floor,
        "within_8pp_of_cross_mean": within_8pp,
        "chance": chance,
        "PASS": bool(within_8pp and above_floor),
    }


def diagnostic_c2_cross_domain_transfer(
    *, probe_linear,
    pool_eval, y_eval, n_words_eval,           # in-domain (Stress-17K-eval, TTS only)
    pool_sp,   y_sp,   n_words_sp,             # cross-domain (StressPresso, real)
    device, chance: float,
    desc_only_floor: float | None = None,
) -> dict:
    """C.2 — train ONLY on Stress-17K (TTS); eval ONLY on StressPresso.

    Stage 3.6 / Option C update: the original ratio-only criterion is
    augmented with quantitative floors against the description-only
    probe baseline (`desc_only_floor`, default loaded from
    outputs/stage3p6/desc_only_baseline.json).

    PASS conditions (ALL three required):
      (i)   cross_domain_acc > desc_only_floor + 0.05
      (ii)  cross_domain_acc / in_domain_acc ≥ 0.80
      (iii) in_domain_acc > desc_only_floor + 0.10

    The floor enforces "the adapter is doing more than what the description
    text alone could provide" — the meaningful test for shortcut exploitation
    via the word-substitution L_NCE-cond term (kickoff Option C update 2).
    """
    in_acc = probe_k_eval_subset(probe_linear, pool_eval, n_words_eval, y_eval, device)
    cross_acc = probe_k_eval_subset(probe_linear, pool_sp, n_words_sp, y_sp, device)

    # Bootstrap CI on cross-domain accuracy.
    rng = np.random.default_rng(0)
    n = pool_sp.shape[0]
    p_sp_pred = predict(probe_linear, pool_sp, n_words_sp, device).cpu().numpy()
    correct = (p_sp_pred == y_sp.numpy()).astype(np.float64)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        boots.append(correct[idx].mean())
    boots = np.asarray(boots)
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))
    above_chance = bool(ci_lo > chance)
    ratio = float(cross_acc / max(in_acc, 1e-9))

    if desc_only_floor is None:
        desc_only_floor = float("nan")
    cond_i   = bool(cross_acc  > desc_only_floor + 0.05)
    cond_ii  = bool(ratio       >= 0.80)
    cond_iii = bool(in_acc      > desc_only_floor + 0.10)
    return {
        "in_domain_accuracy":      in_acc,
        "cross_domain_accuracy":   cross_acc,
        "cross_domain_ci_lo":      ci_lo,
        "cross_domain_ci_hi":      ci_hi,
        "ratio_cross_over_in":     ratio,
        "ci_above_chance":         above_chance,
        "desc_only_floor":         desc_only_floor,
        "cond_i_cross_above_floor_plus_005":  cond_i,
        "cond_ii_ratio_above_080":            cond_ii,
        "cond_iii_in_above_floor_plus_010":   cond_iii,
        "PASS":                    bool(cond_i and cond_ii and cond_iii),
        "chance":                  chance,
    }


def diagnostic_c3_domain_stratified_probe_g(
    *, adapter_rows: list[dict], sp_items, r0_summary: dict | None,
) -> dict:
    """C.3 — Probe-G accuracy_neutral stratified by speaker_id; check R0 vs
    R1 directional consistency across strata.

    PASS condition: A_R1 vs R0 uplift sign is positive in ≥ 75 % of strata
    (with ≥ 5 items per stratum). If R0 summary missing, fall back to
    direction = sign(A_R1 - 0.50) and require positive in ≥ 75 %.
    """
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for r in adapter_rows:
        # neutral subset only
        if r.get("is_explicit"):
            continue
        i = r["item"]
        sp = sp_items[i].speaker_id or "unknown"
        by_speaker[sp].append(r)

    strata = []
    for sp, rows in by_speaker.items():
        if len(rows) < 5:
            continue
        accs = np.array([r["correct"] for r in rows], dtype=np.float64)
        strata.append({
            "speaker_id": sp,
            "n": len(rows),
            "accuracy_neutral_R1": float(accs.mean()),
        })

    if r0_summary is not None:
        r0_acc_n = r0_summary.get("accuracy_neutral", float("nan"))
        for s in strata:
            s["uplift_vs_R0"] = s["accuracy_neutral_R1"] - r0_acc_n
            s["uplift_positive"] = bool(s["uplift_vs_R0"] > 0.0)
    else:
        for s in strata:
            s["uplift_vs_R0"] = s["accuracy_neutral_R1"] - 0.50
            s["uplift_positive"] = bool(s["uplift_vs_R0"] > 0.0)

    if strata:
        n_pos = sum(1 for s in strata if s["uplift_positive"])
        frac_pos = n_pos / len(strata)
    else:
        n_pos = 0
        frac_pos = float("nan")
    return {
        "n_strata": len(strata),
        "fraction_uplift_positive": frac_pos,
        "strata": strata,
        "r0_baseline_accuracy_neutral": (r0_summary or {}).get("accuracy_neutral"),
        "PASS": bool(strata and frac_pos >= 0.75),
    }


def diagnostic_b3_style_probe(
    *, adapter, wavlm, feat, device, n_per_speaker: int = 200,
) -> dict:
    """Stage 3.6 / B3 case-discrimination diagnostic.

    Linear probe on pool(A(H)) → Expresso style (5-class default / happy /
    sad / whisper / sarcastic). Holds out speakers (train on 3, eval on 1).

    Decision (kickoff B3):
      Near chance (≤ 0.30): adapter is too coarse (case ii — explains why
        L_artifact never fires; K_a vs K_a'' divergence is uniformly small).
      Significantly above chance: adapter encodes style (case i — L_artifact
        SHOULD fire, but δ may need re-tuning).
    """
    import pickle
    import random as _random
    from collections import defaultdict
    from src.data.audio_pool import SimpleSample

    CACHE = Path("/raid/nurgaly/datasets/Beyond_Transcript_Alignment/cache")
    EXPR_POOL_PATH = CACHE / "expresso_pool_n6000.pkl"
    with open(EXPR_POOL_PATH, "rb") as f:
        rows = [SimpleSample(**r) for r in pickle.load(f)]

    # Bucket by (speaker, style); keep up to n_per_speaker per speaker.
    by_sp: dict[str, list] = defaultdict(list)
    for r in rows:
        sp = r.meta.get("speaker_id", "")
        st = r.meta.get("style", "")
        if not sp or not st:
            continue
        by_sp[sp].append((r, st))

    speakers = sorted(by_sp.keys())
    if len(speakers) < 2:
        return {"error": "need >= 2 speakers", "PASS": False}

    style_targets = ["default", "happy", "sad", "whisper", "enunciated"]
    style_to_int = {s: i for i, s in enumerate(style_targets)}

    rng = _random.Random(0)
    holdout_sp = speakers[-1]
    train_speakers = speakers[:-1]
    train_data, eval_data = [], []
    for sp in train_speakers:
        local = [(r, s) for r, s in by_sp[sp] if s in style_to_int]
        rng.shuffle(local)
        train_data.extend(local[:n_per_speaker])
    local_eval = [(r, s) for r, s in by_sp[holdout_sp] if s in style_to_int]
    rng.shuffle(local_eval)
    eval_data = local_eval[:n_per_speaker]

    # Forward all through frozen WavLM + adapter, mean-pool.
    @torch.no_grad()
    def _pool_features(rows_with_style):
        feats, ys = [], []
        max_n = 8 * SR
        for r, st in rows_with_style:
            x = r.audio[:max_n]
            proc = feat([x], sampling_rate=SR, return_tensors="pt",
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
            ys.append(style_to_int[st])
        return np.stack(feats), np.asarray(ys, dtype=np.int64)

    print(f"  forwarding {len(train_data)} train + {len(eval_data)} eval audios…", flush=True)
    Xtr, ytr = _pool_features(train_data)
    Xev, yev = _pool_features(eval_data)

    from src.probes.shortcut_probes import fit_linear_probe
    res = fit_linear_probe(
        np.concatenate([Xtr, Xev], axis=0),
        np.concatenate([ytr, yev], axis=0),
        name="style", seed=0, eval_frac=len(yev) / max(1, len(ytr) + len(yev)),
    )
    eval_acc = res.eval_acc
    chance = 1.0 / len(style_targets)
    return {
        "n_styles":        len(style_targets),
        "train_speakers":  list(train_speakers),
        "holdout_speaker": holdout_sp,
        "n_train":         int(Xtr.shape[0]),
        "n_eval":          int(Xev.shape[0]),
        "chance":          chance,
        "eval_acc":        eval_acc,
        "case":            ("ii_too_coarse" if eval_acc <= 0.30 else "i_encodes_style"),
        "interpretation":  ("L_artifact silent likely because adapter doesn't encode style"
                            if eval_acc <= 0.30 else
                            "adapter encodes style; L_artifact silence may be δ mis-set"),
    }


def diagnostic_c4_domain_probe_on_adapter(
    *, adapter, wavlm, feat, device, n_per_source: int = 200,
) -> dict:
    """C.4 — binary domain probe on pool(A_R1(H)). Red flag if higher than
    raw-H domain probe (Stage 3.0.5 measured raw at 0.99-1.00).
    """
    import pickle
    import random
    import soxr

    CACHE = Path("/raid/nurgaly/datasets/Beyond_Transcript_Alignment/cache")
    STRESS_POOL_PATH = CACHE / "stress17k_probe_train_pool.pkl"
    LIBRI_POOL_PATH  = CACHE / "librispeech_pool_n6000.pkl"
    EXPR_POOL_PATH   = CACHE / "expresso_pool_n6000.pkl"

    rng = random.Random(0)
    samples: list[tuple[np.ndarray, str]] = []   # (audio, source)

    with open(STRESS_POOL_PATH, "rb") as f:
        stress = pickle.load(f)
    rng.shuffle(stress)
    for r in stress[:n_per_source]:
        samples.append((np.asarray(r["audio_array"], dtype=np.float32), "stress17k"))

    with open(LIBRI_POOL_PATH, "rb") as f:
        libri = pickle.load(f)
    rng.shuffle(libri)
    for r in libri[:n_per_source]:
        samples.append((np.asarray(r["audio"], dtype=np.float32), "librispeech"))

    with open(EXPR_POOL_PATH, "rb") as f:
        expr = pickle.load(f)
    rng.shuffle(expr)
    for r in expr[:n_per_source]:
        samples.append((np.asarray(r["audio"], dtype=np.float32), "expresso"))

    sp_items = load_stresspresso_test()
    rng.shuffle(sp_items)
    for it in sp_items[:n_per_source]:
        arr = np.asarray(it.audio_array, dtype=np.float32)
        if it.audio_sr != SR:
            arr = soxr.resample(arr, it.audio_sr, SR).astype(np.float32)
        samples.append((arr, "stresspresso"))

    rng.shuffle(samples)

    # Forward each through frozen WavLM → adapter → mean pool over audio tokens.
    feats_list = []
    sources = []
    max_n = 8 * SR
    for x, src in samples:
        x = x[:max_n]
        with torch.no_grad():
            proc = feat([x], sampling_rate=SR, return_tensors="pt",
                        padding=True, return_attention_mask=True)
            iv = proc["input_values"].to(device).to(torch.bfloat16)
            am = proc["attention_mask"].to(device)
            wav_out = wavlm(input_values=iv, attention_mask=am, output_hidden_states=True)
            H = wav_out.hidden_states[16].float()
            sample_lengths = am.sum(dim=1)
            valid_T_s = wavlm._get_feat_extract_output_lengths(sample_lengths).long()
            K, vTk = adapter(H, valid_T_s=valid_T_s)
            Tk = int(vTk[0].item())
            pooled = K[0, :Tk, :].mean(dim=0).cpu().float().numpy()
        feats_list.append(pooled)
        sources.append(src)

    feats_arr = np.stack(feats_list)
    is_synth = np.array([s == "stress17k" for s in sources], dtype=np.int64)
    y = is_synth.astype(np.int64)
    res = fit_linear_probe(feats_arr, y, name="domain_A_R1", seed=0)
    return {
        "n_samples": len(samples),
        "domain_acc_A_R1":   res.eval_acc,
        "domain_train_acc":  res.train_acc,
        "n_per_source_used": n_per_source,
        "RED_FLAG_higher_than_raw_H": bool(res.eval_acc > 0.99),
    }


# ---------- T-only check ---------- #

def t_only_eval(c_phi_t_only, s17_eval, device) -> dict:
    """Run C_phi_t_only on probe-eval transcripts; report accuracy + PASS flag."""
    correct = 0
    total = 0
    for it in s17_eval:
        rT_np = restricted_t_features(it.transcription, n_words=it.n_words)
        rT = torch.from_numpy(rT_np).to(device=device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = c_phi_t_only(rT)
            pred = int(logits.argmax(dim=-1).item())
        if pred == it.stress_index:
            correct += 1
        total += 1
    acc = correct / max(1, total)
    return {
        "n_eval": total, "n_correct": correct, "accuracy": acc,
        "lexical_contamination_flag": bool(acc > 0.30),
    }


# ---------- Main ---------- #

def main() -> int:
    args = parse_args()
    print(f"transformers=={__import__('transformers').__version__}, torch=={torch.__version__}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES','?')}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("ERROR: GPU required."); return 1

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / "stage3_eval"
    seed_dir = out_dir / f"seed{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load A_R1 + C_phi heads ---- #
    banner(f"Loading R1 checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = AdapterConfig(**ckpt["adapter_config"])
    adapter = BLSPAdapter(cfg).to(device).to(torch.float32)
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    adapter.eval()
    d_llm = cfg.d_llm
    d_T = 8194
    c_phi_full = CPhi(d_in=d_llm + d_T, hidden=256, n_classes=N_MAX_CLASSES).to(device).to(torch.float32)
    c_phi_full.load_state_dict(ckpt["c_phi_full_state_dict"])
    c_phi_full.eval()
    c_phi_t_only = CPhi(d_in=d_T, hidden=256, n_classes=N_MAX_CLASSES).to(device).to(torch.float32)
    c_phi_t_only.load_state_dict(ckpt["c_phi_t_only_state_dict"])
    c_phi_t_only.eval()
    print(f"  loaded; adapter trainable = {adapter.n_trainable_params()/1e6:.2f}M, "
          f"C_phi_full + C_phi_t_only = {(sum(p.numel() for p in c_phi_full.parameters()) + sum(p.numel() for p in c_phi_t_only.parameters()))/1e6:.2f}M", flush=True)

    # ---- Load Qwen3-8B + WavLM ---- #
    banner(f"Loading Qwen3-8B (bf16) and WavLM-Large (bf16)")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN3_MODEL)
    llm = AutoModelForCausalLM.from_pretrained(QWEN3_MODEL, torch_dtype=torch.bfloat16).eval().to(device)
    if args.lora_path:
        from peft import PeftModel
        lora_dir = Path(args.lora_path)
        if not lora_dir.exists():
            print(f"ERROR: missing LoRA dir {lora_dir}"); return 1
        print(f"  attaching LoRA from {lora_dir}", flush=True)
        llm = PeftModel.from_pretrained(llm, str(lora_dir)).eval().to(device)
        n_lora = sum(p.numel() for p in llm.parameters() if p.requires_grad)
        print(f"  PEFT model attached; LoRA params reported trainable: {n_lora/1e6:.2f}M "
              f"(eval mode — gradients off downstream)", flush=True)
    embed_layer = llm.get_input_embeddings()

    from transformers import WavLMModel, AutoFeatureExtractor
    wavlm_feat = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL)
    wavlm = WavLMModel.from_pretrained(WAVLM_MODEL, torch_dtype=torch.bfloat16).eval().to(device)

    # Stage 4 Control A: load proj_P for text-only forward path.
    proj_P_W = None
    proj_P_b = None
    if args.control_mode == "text_only":
        proj_P_path = Path(args.proj_P_path) if args.proj_P_path else (
            ROOT / "outputs" / "stage4" / "proj_P.pt")
        if not proj_P_path.exists():
            print(f"ERROR: missing {proj_P_path}", flush=True); return 1
        P_payload = torch.load(proj_P_path, map_location=device, weights_only=False)
        proj_P_W = P_payload["weight"].to(device=device, dtype=torch.float32)
        proj_P_b = P_payload["bias"].to(device=device, dtype=torch.float32)
        print(f"  control_mode=text_only: loaded proj_P from {proj_P_path.name}", flush=True)

    vis_start_id = int(tok.convert_tokens_to_ids("<|vision_start|>"))
    vis_end_id   = int(tok.convert_tokens_to_ids("<|vision_end|>"))

    sp_items = load_stresspresso_test()
    print(f"  StressPresso loaded: {len(sp_items)} items", flush=True)

    # ---- (A) Probe-G(A_R1) ---- #
    banner("(A) Probe-G(A_R1) — adapter, 20 paraphrases × 202 items")
    t0 = time.time()
    if args.control_mode == "text_only":
        K_speech_list = adapter_audio_embeddings_text_only(
            adapter, embed_layer, tok, proj_P_W, proj_P_b, sp_items, device)
        print(f"  [text_only] K_speech via P(K_T) computed in {time.time()-t0:.1f}s", flush=True)
    else:
        K_speech_list = adapter_audio_embeddings(adapter, wavlm, wavlm_feat, sp_items, device)
        print(f"  K_speech computed in {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    res_adapter = run_probe_g_variant(
        name="adapter", llm=llm, embed_layer=embed_layer, tok=tok,
        items=sp_items, device=device, audio_embeddings=K_speech_list,
        paraphrases_neutral=NEUTRAL_PARAPHRASES,
        paraphrases_explicit=EXPLICIT_PARAPHRASES,
        vis_start_id=vis_start_id, vis_end_id=vis_end_id,
    )
    print(f"  done in {time.time()-t0:.1f}s", flush=True)
    sN = res_adapter["summary"]
    print(f"    accuracy total={sN['accuracy']:.4f}  neutral={sN['accuracy_neutral']:.4f}  explicit={sN['accuracy_explicit']:.4f}", flush=True)

    # ---- (D) K_T baseline ---- #
    banner("(D) K_T baseline — embed(true transcript) in audio slot")
    t0 = time.time()
    K_T_list = text_embeddings_from_string(tok, embed_layer, [it.transcription for it in sp_items], device)
    res_kt = run_probe_g_variant(
        name="K_T", llm=llm, embed_layer=embed_layer, tok=tok,
        items=sp_items, device=device, audio_embeddings=K_T_list,
        paraphrases_neutral=NEUTRAL_PARAPHRASES,
        paraphrases_explicit=EXPLICIT_PARAPHRASES,
        vis_start_id=vis_start_id, vis_end_id=vis_end_id,
    )
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    # ---- (D2) K_T_styled baseline (Stage 7) ---- #
    banner("(D2) K_T_styled baseline — embed(transcript + ' [stress on word: <word>]') in audio slot")
    t0 = time.time()
    styled_texts = [
        f"{it.transcription} [stress on word: {it.stressed_word}]"
        for it in sp_items
    ]
    K_Tst_list = text_embeddings_from_string(tok, embed_layer, styled_texts, device)
    res_kt_styled = run_probe_g_variant(
        name="K_T_styled", llm=llm, embed_layer=embed_layer, tok=tok,
        items=sp_items, device=device, audio_embeddings=K_Tst_list,
        paraphrases_neutral=NEUTRAL_PARAPHRASES,
        paraphrases_explicit=EXPLICIT_PARAPHRASES,
        vis_start_id=vis_start_id, vis_end_id=vis_end_id,
    )
    print(f"  done in {time.time()-t0:.1f}s; "
          f"acc total={res_kt_styled['summary']['accuracy']:.4f} "
          f"neutral={res_kt_styled['summary']['accuracy_neutral']:.4f} "
          f"explicit={res_kt_styled['summary']['accuracy_explicit']:.4f}", flush=True)

    # ---- (C) Cascade-T ---- #
    res_cascade = None
    res_cascade_tl = None
    if not args.skip_cascade:
        banner("(C) Cascade-T baseline — Whisper-ASR transcript")
        t0 = time.time()
        predicted_transcripts = whisper_asr_transcripts(sp_items, device)
        K_pred_list = text_embeddings_from_string(tok, embed_layer, predicted_transcripts, device)
        res_cascade = run_probe_g_variant(
            name="cascade_T", llm=llm, embed_layer=embed_layer, tok=tok,
            items=sp_items, device=device, audio_embeddings=K_pred_list,
            paraphrases_neutral=NEUTRAL_PARAPHRASES,
            paraphrases_explicit=EXPLICIT_PARAPHRASES,
            vis_start_id=vis_start_id, vis_end_id=vis_end_id,
        )
        print(f"  done in {time.time()-t0:.1f}s", flush=True)

        # (C2) Cascade-T+L baseline (Stage 7) — Whisper transcript + " [stress on word: X]"
        banner("(C2) Cascade-T+L baseline — Whisper transcript + stress label")
        t0 = time.time()
        styled_pred_texts = [
            f"{tr} [stress on word: {it.stressed_word}]"
            for tr, it in zip(predicted_transcripts, sp_items)
        ]
        K_predL_list = text_embeddings_from_string(tok, embed_layer, styled_pred_texts, device)
        res_cascade_tl = run_probe_g_variant(
            name="cascade_T_plus_L", llm=llm, embed_layer=embed_layer, tok=tok,
            items=sp_items, device=device, audio_embeddings=K_predL_list,
            paraphrases_neutral=NEUTRAL_PARAPHRASES,
            paraphrases_explicit=EXPLICIT_PARAPHRASES,
            vis_start_id=vis_start_id, vis_end_id=vis_end_id,
        )
        print(f"  done in {time.time()-t0:.1f}s; "
              f"acc total={res_cascade_tl['summary']['accuracy']:.4f} "
              f"neutral={res_cascade_tl['summary']['accuracy_neutral']:.4f} "
              f"explicit={res_cascade_tl['summary']['accuracy_explicit']:.4f}", flush=True)

    # ---- (E) Oracle re-confirm ---- #
    banner("(E) Probe-G-oracle re-confirm")
    from src.data.stress_data import _wrap_word
    SYS_ORACLE = (
        "You are a careful reader. Use the speaker's word emphasis to choose the "
        "correct interpretation."
    )
    USER_ORACLE = (
        "In the transcript, the speaker emphasizes the word '{word}'. Use that "
        "emphasis to interpret the meaning.\nTranscript: {marked}\n"
        "Which interpretation is correct?\nA) {opt_a}\nB) {opt_b}"
    )
    cand_a_ids = long_tensor(tok, " A", device)
    cand_b_ids = long_tensor(tok, " B", device)
    rows_oracle = []
    t0 = time.time()
    with torch.no_grad():
        for it in sp_items:
            marked = _wrap_word(it.transcription, it.stress_index)
            user_text = USER_ORACLE.format(
                word=it.stressed_word, marked=marked,
                opt_a=it.options[0], opt_b=it.options[1],
            )
            left = tok.apply_chat_template(
                [{"role": "system", "content": SYS_ORACLE},
                 {"role": "user",   "content": user_text}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ) + "Answer:"
            ids = long_tensor(tok, left, device).unsqueeze(0)
            scores = []
            for cand in (cand_a_ids, cand_b_ids):
                full_ids = torch.cat([ids[0], cand]).unsqueeze(0)
                embeds = embed_layer(full_ids)
                attn = torch.ones_like(full_ids)
                logits = llm(inputs_embeds=embeds, attention_mask=attn).logits[0]
                T_c = cand.shape[0]
                cand_start = full_ids.shape[1] - T_c
                pred = logits[cand_start - 1: cand_start - 1 + T_c].float()
                log_probs = F.log_softmax(pred, dim=-1)
                scores.append(float(log_probs[torch.arange(T_c, device=device), cand].mean().item()))
            s_A, s_B = scores
            pred_label = 0 if s_A > s_B else 1
            margin = (s_A - s_B) if it.label == 0 else (s_B - s_A)
            rows_oracle.append({
                "item": len(rows_oracle), "transcription_id": it.transcription_id,
                "label": it.label, "pred": pred_label,
                "correct": int(pred_label == it.label),
                "score_A": s_A, "score_B": s_B, "signed_margin": margin,
            })
    accs = np.array([r["correct"] for r in rows_oracle], dtype=np.float64)
    margs = np.array([r["signed_margin"] for r in rows_oracle], dtype=np.float64)
    clusters = [r["transcription_id"] for r in rows_oracle]
    acc_mean, acc_lo, acc_hi = bootstrap_by_cluster(accs, clusters, n_iter=args.bootstrap_iter)
    marg_mean, marg_lo, marg_hi = bootstrap_by_cluster(margs, clusters, n_iter=args.bootstrap_iter)
    res_oracle = {
        "summary": {
            "accuracy": acc_mean, "accuracy_ci_lo": acc_lo, "accuracy_ci_hi": acc_hi,
            "signed_margin": marg_mean, "signed_margin_ci_lo": marg_lo, "signed_margin_ci_hi": marg_hi,
            "n_correct": int(accs.sum()), "n": len(accs),
        },
        "rows": rows_oracle,
    }
    print(f"  oracle accuracy = {acc_mean:.4f}  expected ~0.7871 ± 2pp from Stage 1a", flush=True)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    # ---- (B) Probe-K(A_R1) ---- #
    banner("(B) Probe-K(A_R1) — linear + MLP-2 probe on pool(A_R1(H))")
    t0 = time.time()
    s17_all = load_stress17k()
    train_ids, eval_ids = partition_transcript_ids(s17_all, eval_frac=0.20, seed="BTA-2026-05-02")
    s17_train = [it for it in s17_all if it.transcription_id in train_ids]
    s17_eval  = [it for it in s17_all if it.transcription_id in eval_ids]
    print(f"  Stress-17K train rows: {len(s17_train)}  eval rows: {len(s17_eval)}", flush=True)

    if args.control_mode == "text_only":
        print(f"  [text_only] computing pool(A_textK(P(K_T))) on Stress-17K train...", flush=True)
        pool_train = adapter_pool_for_probe_k_text_only(
            adapter, embed_layer, tok, proj_P_W, proj_P_b, s17_train, device)
        print(f"  [text_only] computing pool(A_textK(P(K_T))) on Stress-17K eval...", flush=True)
        pool_eval  = adapter_pool_for_probe_k_text_only(
            adapter, embed_layer, tok, proj_P_W, proj_P_b, s17_eval, device)
        print(f"  [text_only] computing pool(A_textK(P(K_T))) on StressPresso...", flush=True)
        pool_sp    = adapter_pool_for_probe_k_text_only(
            adapter, embed_layer, tok, proj_P_W, proj_P_b, sp_items, device)
    else:
        print(f"  computing pool(A_R1(H)) on Stress-17K train...", flush=True)
        pool_train = adapter_pool_for_probe_k(adapter, wavlm, wavlm_feat, s17_train, device)
        print(f"  computing pool(A_R1(H)) on Stress-17K eval...", flush=True)
        pool_eval  = adapter_pool_for_probe_k(adapter, wavlm, wavlm_feat, s17_eval, device)
        print(f"  computing pool(A_R1(H)) on StressPresso...", flush=True)
        pool_sp    = adapter_pool_for_probe_k(adapter, wavlm, wavlm_feat, sp_items, device)
    print(f"  pool extraction done in {time.time()-t0:.1f}s", flush=True)

    n_words_train  = torch.tensor([it.n_words      for it in s17_train], dtype=torch.long)
    y_train        = torch.tensor([it.stress_index for it in s17_train], dtype=torch.long)
    n_words_eval   = torch.tensor([it.n_words      for it in s17_eval],  dtype=torch.long)
    y_eval         = torch.tensor([it.stress_index for it in s17_eval],  dtype=torch.long)
    n_words_sp     = torch.tensor([len(it.transcription.split()) for it in sp_items], dtype=torch.long)
    y_sp           = torch.tensor([it.stress_index for it in sp_items],  dtype=torch.long)
    eval_tids      = [it.transcription_id for it in s17_eval]
    sp_tids        = [it.transcription_id for it in sp_items]
    voice_eval     = [it.voice_name for it in s17_eval]

    fit_cfg = FitConfig(epochs=80, lr=1e-3, batch_size=256, weight_decay=1e-4, seed=args.seed)
    probe_results = {}
    probe_linear_obj = None
    for head in ("linear", "mlp2"):
        probe = ProbeK(d_in=pool_train.shape[-1], n_classes=N_MAX_CLASSES,
                       cell_mode="single", n_layers_used=1, head=head)
        info = fit_probe(probe, pool_train, y_train, n_words_train,
                         pool_eval,  y_eval,  n_words_eval,
                         fit_cfg, device)
        eval_cands_lookup = defaultdict(set)
        for it in s17_eval:
            eval_cands_lookup[it.transcription_id].add(int(it.stress_index))
        eval_cands_lookup = {k: sorted(v) for k, v in eval_cands_lookup.items()}
        sp_cands_lookup = defaultdict(set)
        for it in sp_items:
            sp_cands_lookup[it.transcription_id].add(int(it.stress_index))
        sp_cands_lookup = {k: sorted(v) for k, v in sp_cands_lookup.items()}

        p_eval_full = predict(probe, pool_eval, n_words_eval, device)
        eval_acc_full = acc_fn(p_eval_full, y_eval)
        p_eval_within = within_transcript_argmax(probe, pool_eval, eval_tids, eval_cands_lookup, device)
        eval_acc_within = acc_fn(p_eval_within, y_eval)
        p_sp_full = predict(probe, pool_sp, n_words_sp, device)
        sp_acc_full = acc_fn(p_sp_full, y_sp)
        p_sp_within = within_transcript_argmax(probe, pool_sp, sp_tids, sp_cands_lookup, device)
        sp_acc_within = acc_fn(p_sp_within, y_sp)
        rob_fp16 = acc_fn(predict(probe, pool_eval, n_words_eval, device, dtype=torch.float16), y_eval)
        rob_s01  = acc_fn(predict(probe, pool_eval, n_words_eval, device, noise_sigma=0.1, noise_seed=0), y_eval)

        probe_results[head] = {
            "fit_best_eval_acc": info["best"]["eval_acc"],
            "fit_final_eval_acc": info["final_eval_acc"],
            "eval_acc_full": eval_acc_full,
            "eval_acc_within": eval_acc_within,
            "sp_acc_full": sp_acc_full,
            "sp_acc_within": sp_acc_within,
            "robust_fp16": rob_fp16,
            "robust_sigma_01": rob_s01,
        }
        print(f"  Probe-K {head}: eval_full={eval_acc_full:.3f} within={eval_acc_within:.3f} "
              f"sp_full={sp_acc_full:.3f} fp16={rob_fp16:.3f} σ0.1={rob_s01:.3f}", flush=True)
        if head == "linear":
            probe_linear_obj = probe

    # ---- C.1 — within-domain Probe-K stratified ---- #
    banner("C.1 — within-domain Probe-K stratified by {nova, echo, stresspresso}")
    chance_k = 1.0 / N_MAX_CLASSES
    c1 = diagnostic_c1_within_domain(
        probe_linear=probe_linear_obj,
        pool_eval=pool_eval, y_eval=y_eval, n_words_eval=n_words_eval,
        voice_eval=voice_eval,
        pool_sp=pool_sp, y_sp=y_sp, n_words_sp=n_words_sp,
        device=device, chance=chance_k,
    )
    print(f"  by_subset: {c1['by_subset']}", flush=True)
    print(f"  spread={c1['max_spread']:.4f}  PASS={c1['PASS']}", flush=True)

    # ---- C.2 — cross-domain transfer ---- #
    banner("C.2 — cross-domain transfer Probe-K (TTS → real)")
    desc_only_path = ROOT / "outputs" / "stage3p6" / "desc_only_baseline.json"
    desc_only_floor = float("nan")
    if desc_only_path.exists():
        desc_only_payload = json.loads(desc_only_path.read_text())
        # Stage 3.6: use the active template's desc-only acc, not the initial one.
        desc_only_floor = float(desc_only_payload.get("active_desc_only_floor",
                                  desc_only_payload["desc_only_eval_acc"]["word_sub"]))
        print(f"  desc-only floor (word-sub) = {desc_only_floor:.4f} "
              f"(loaded from {desc_only_path.name})", flush=True)
    else:
        print(f"  WARNING: {desc_only_path.name} missing — C.2 reverts to legacy ratio-only PASS", flush=True)
    c2 = diagnostic_c2_cross_domain_transfer(
        probe_linear=probe_linear_obj,
        pool_eval=pool_eval, y_eval=y_eval, n_words_eval=n_words_eval,
        pool_sp=pool_sp,   y_sp=y_sp,   n_words_sp=n_words_sp,
        device=device, chance=chance_k,
        desc_only_floor=desc_only_floor,
    )
    print(f"  in-domain={c2['in_domain_accuracy']:.4f}  "
          f"cross-domain={c2['cross_domain_accuracy']:.4f}  "
          f"ratio={c2['ratio_cross_over_in']:.3f}  "
          f"CI=({c2['cross_domain_ci_lo']:.3f}, {c2['cross_domain_ci_hi']:.3f})", flush=True)
    print(f"  cond_i (cross > {desc_only_floor:.3f}+0.05={desc_only_floor+0.05:.3f}): {c2['cond_i_cross_above_floor_plus_005']}", flush=True)
    print(f"  cond_ii (ratio >= 0.80): {c2['cond_ii_ratio_above_080']}", flush=True)
    print(f"  cond_iii (in > {desc_only_floor:.3f}+0.10={desc_only_floor+0.10:.3f}): {c2['cond_iii_in_above_floor_plus_010']}", flush=True)
    print(f"  PASS={c2['PASS']}", flush=True)

    # ---- C.3 — domain-stratified Probe-G + R0 comparison ---- #
    banner("C.3 — domain-stratified Probe-G on StressPresso (by speaker_id)")
    r0_summary_path = ROOT / "outputs" / "stage2_eval" / f"seed{args.seed}" / "summary.json"
    r0_summary = None
    if r0_summary_path.exists():
        r0_summary = json.loads(r0_summary_path.read_text()).get("adapter")
        print(f"  R0 baseline loaded from {r0_summary_path}", flush=True)
    else:
        print(f"  WARNING: R0 baseline missing — direction taken vs 0.50", flush=True)
    c3 = diagnostic_c3_domain_stratified_probe_g(
        adapter_rows=res_adapter["rows"], sp_items=sp_items, r0_summary=r0_summary,
    )
    print(f"  n_strata={c3['n_strata']}  fraction_uplift_positive={c3['fraction_uplift_positive']}  PASS={c3['PASS']}", flush=True)

    # ---- C.4 — domain probe on A_R1(H) red flag ---- #
    banner("C.4 — domain probe on pool(A_R1(H)) (red flag check)")
    t0 = time.time()
    c4 = diagnostic_c4_domain_probe_on_adapter(
        adapter=adapter, wavlm=wavlm, feat=wavlm_feat, device=device, n_per_source=200,
    )
    print(f"  domain_acc(A_R1) = {c4['domain_acc_A_R1']:.4f}  "
          f"raw H baseline ≈ 0.99-1.00  "
          f"RED_FLAG={c4['RED_FLAG_higher_than_raw_H']}  ({time.time()-t0:.1f}s)", flush=True)

    # ---- B3 case-discrimination — Expresso style probe ---- #
    banner("B3 — Expresso style probe on pool(A_R1(H)) (case i vs case ii)")
    t0 = time.time()
    b3_style = diagnostic_b3_style_probe(
        adapter=adapter, wavlm=wavlm, feat=wavlm_feat, device=device, n_per_speaker=120,
    )
    if "error" in b3_style:
        print(f"  skipped: {b3_style['error']}", flush=True)
    else:
        print(f"  style_eval_acc = {b3_style['eval_acc']:.4f}  "
              f"chance = {b3_style['chance']:.4f}  "
              f"case = {b3_style['case']}  ({time.time()-t0:.1f}s)", flush=True)

    # ---- T-only — C_phi_t_only on probe-eval ---- #
    banner("T-only diagnostic — C_phi_t_only(restricted_T) → Φ on probe-eval")
    t_only = t_only_eval(c_phi_t_only, s17_eval, device)
    print(f"  T-only accuracy = {t_only['accuracy']:.4f}  "
          f"({t_only['n_correct']}/{t_only['n_eval']}); chance ≈ {1/N_MAX_CLASSES:.3f}; "
          f"contamination_flag (>0.30) = {t_only['lexical_contamination_flag']}", flush=True)

    # ---- Save ---- #
    banner("Saving evaluation results")
    payload = {
        "seed":       args.seed,
        "checkpoint": args.checkpoint,
        "adapter":   res_adapter["summary"],
        "K_T":       res_kt["summary"],
        "cascade_T": (res_cascade["summary"] if res_cascade is not None else None),
        "oracle_reconfirm": res_oracle["summary"],
        "probe_k":   probe_results,
        "C1_within_domain":         c1,
        "C2_cross_domain_transfer": c2,
        "C3_domain_stratified_probe_g": c3,
        "C4_domain_probe_on_adapter":   c4,
        "B3_style_probe":               b3_style,
        "t_only_diagnostic":            t_only,
        "K_T_styled":                   res_kt_styled["summary"],
        "cascade_T_plus_L":             (res_cascade_tl["summary"] if res_cascade_tl is not None else None),
    }
    (seed_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str))
    rows_payload = {
        "adapter":   res_adapter["rows"],
        "K_T":       res_kt["rows"],
        "cascade_T": (res_cascade["rows"] if res_cascade is not None else None),
        "oracle_reconfirm": res_oracle["rows"],
    }
    (seed_dir / "rows.json").write_text(json.dumps(rows_payload, indent=2, default=str))
    print(f"  saved → {seed_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
