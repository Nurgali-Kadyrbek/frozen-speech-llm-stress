"""Stage 3.3 — R1 structural counterfactual training (one seed per invocation).

Per-step batch composition (target effective batch = 32, microbatch=1):
  16 cf-pair members  (8 Stress-17K cf pairs, 2-AFC prompt format, L_BLSP+L_cf+L_cond+L_NCE)
   7 individual stress (Stress-17K probe-train, full-text response, L_BLSP only)
   5 LibriSpeech       (ASR question, L_BLSP only)
   4 expresso pair members  (2 pairs, role-split with asymmetric L_artifact gradient)

Roles enumerated per microbatch let role-specific loss assembly happen
without branching the LLM forward.

Per-component grad-norm diagnostic runs every `--diag_every` optimizer
steps on a single representative cf-pair-member microbatch.

Run:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/stage3_train.py \
      --seed 1234 --max_steps 400 --grad_accum 32 --warmup_steps 500
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_env, banner  # noqa: E402

setup_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.models.adapter import AdapterConfig, BLSPAdapter  # noqa: E402
from src.losses.blsp import BLSPInput, compute_blsp_loss  # noqa: E402
from src.losses.r1 import (  # noqa: E402
    CPhi, restricted_t_features, make_phi_description_embeds,
    build_phi_word_embed_cache, make_phi_word_embed,
    compute_l_cf_from_logits, compute_l_artifact, compute_l_cond_pred,
    compute_l_nce_cond, ArtifactCacheEntry, ArtifactPairCache,
    per_component_grad_norm,
)
from src.utils.prompts import (  # noqa: E402
    build_training_halves, DEFAULT_SYSTEM, AUDIO_PLACEHOLDER,
)
from src.data.audio_pool import (  # noqa: E402
    SimpleSample, load_librispeech_subset, load_expresso_subset,
    STRESS_QUESTION, ASR_QUESTION,
)
from src.data.augment import (  # noqa: E402
    AugmentConfig, augment_one, aggressive_stage3_config,
)
from src.data.cf_pairs import (  # noqa: E402
    CfPair, ArtifactPair, read_cf_pairs_jsonl, read_artifact_pairs_jsonl,
)


QWEN3_MODEL = "Qwen/Qwen3-8B"
WAVLM_MODEL = "microsoft/wavlm-large"
SR = 16000
RATIO_BAND = (0.5, 2.0)

CACHE = Path("/raid/nurgaly/datasets/Beyond_Transcript_Alignment/cache")
STRESS_POOL_PATH = CACHE / "stress17k_probe_train_pool.pkl"

CF_TRAIN_JSONL    = ROOT / "outputs" / "stage3" / "cf_pairs_train.jsonl"
CF_ARTIFACT_JSONL = ROOT / "outputs" / "stage3" / "cf_pairs_artifact.jsonl"

# Single-token IDs for ' A' and ' B' in Qwen3-8B tokenizer (verified empirically).
TOK_ID_SP_A = 362
TOK_ID_SP_B = 425

# Φ vocabulary: stress index ∈ [0, 12]. Stage-1 yaml: n_max_classes=13.
N_MAX_CLASSES = 13


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--grad_accum", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--lambda_kl", type=float, default=1.0)
    p.add_argument("--lambda_cf", type=float, default=1.0)
    p.add_argument("--lambda_artifact", type=float, default=1.0)
    p.add_argument("--lambda_cond", type=float, default=0.5)
    p.add_argument("--lambda_nce", type=float, default=0.5)
    p.add_argument("--delta_artifact", type=float, default=0.10)
    p.add_argument("--tau_nce", type=float, default=0.07)
    p.add_argument("--n_libri", type=int, default=6000)
    p.add_argument("--n_expr", type=int, default=6000)
    p.add_argument("--max_audio_seconds", type=float, default=10.0)
    p.add_argument("--max_response_tokens", type=int, default=64)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--diag_every", type=int, default=25,
                   help="Per-component grad-norm diagnostic cadence.")
    p.add_argument("--use_aggressive_aug", action="store_true",
                   help="Use aggressive_stage3_config() (post-Stage 3.0.5 strengthened pipeline).")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--ckpt_name", type=str, default=None,
                   help="Override checkpoint filename (default A_BLSP_seed{S}.pt or A_R1_seed{S}.pt).")
    p.add_argument("--smoke_steps", type=int, default=0,
                   help="if >0, stop after that many steps (for fast validation)")
    # Stage 4 controls
    p.add_argument("--control_mode", type=str, default="none",
                   choices=("none", "text_only"),
                   help="text_only = Stage 4 Control A: replace audio path with P(embed_tokens(transcript))")
    p.add_argument("--cf_pairs_path", type=str, default=None,
                   help="Override cf_pairs jsonl. Used by Stage 4 Control B with shuffled pairs.")
    p.add_argument("--proj_P_path", type=str, default=None,
                   help="Path to outputs/stage4/proj_P.pt for control_mode=text_only.")
    # Stage 6 LoRA on LLM (R1.8 adapter frozen)
    p.add_argument("--lora_rank", type=int, default=0,
                   help="Stage 6: if > 0, wrap Qwen3-8B with LoRA on q_proj/v_proj (all layers). "
                        "Adapter is loaded frozen from --frozen_adapter_ckpt; only LoRA matrices train.")
    p.add_argument("--lora_alpha", type=float, default=32.0,
                   help="LoRA α (default 32, alpha = 4×rank at rank 8)")
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--frozen_adapter_ckpt", type=str, default=None,
                   help="Stage 6: load A_R1.8 adapter from this path and freeze (no grad). "
                        "Required when --lora_rank > 0.")
    # Stage 7 styled-teacher
    p.add_argument("--styled_teacher", action="store_true",
                   help="Stage 7: L_BLSP teacher text becomes "
                        "transcript + ' [stress on word: <word>]' for stress rows.")
    return p.parse_args()


def lr_at(step: int, *, warmup: int, max_steps: int, peak_lr: float, min_lr: float) -> float:
    if step < warmup:
        return peak_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cos


# ---------- Sample types ---------- #

class _Sample:
    """Lightweight per-microbatch sample bundle.

    Stage 3.6 / B1: q_pos and q_neg carry the precomputed word-substitution
    description embeddings for L_NCE-cond. Both are (d_llm,) fp32 tensors
    pulled from a per-(transcription_id, phi_index) cache built at training
    script init (build_phi_word_embed_cache). For non-cf roles they remain
    None.
    """
    __slots__ = ("audio", "role", "transcript", "question", "response",
                 "options", "label", "phi_a", "phi_a_prime", "n_words",
                 "pair_id", "aug_seed", "q_pos", "q_neg",
                 "transcription_id", "word_at_phi")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


# ---------- Sampling ---------- #

def build_microbatch_plan(grad_accum: int) -> list[str]:
    """Deterministic role plan per accumulation cycle.

    For grad_accum=32:
      16 cf-pair members  (8 pairs, paired in adjacent microbatches)
       7 stress_individual
       5 librispeech
       4 expresso_artifact (2 pairs, paired in adjacent microbatches)
    """
    plan: list[str] = []
    # 8 cf pairs: a, a', a, a', ...
    for _ in range(8):
        plan.extend(["cf_a", "cf_a_prime"])
    # stress individual
    plan.extend(["stress_individual"] * 7)
    # libri
    plan.extend(["librispeech"] * 5)
    # 2 expresso artifact pairs
    for _ in range(2):
        plan.extend(["expresso_a", "expresso_a_prime"])
    if grad_accum != len(plan):
        # Pad with stress_individual or trim
        while len(plan) < grad_accum:
            plan.append("stress_individual")
        if len(plan) > grad_accum:
            plan = plan[:grad_accum]
    return plan


class StageThreeSampler:
    """Pulls rows from the four pools by role.

    For paired roles (cf_a / cf_a_prime / expresso_a / expresso_a_prime),
    a sentinel `pair_id` is generated so the artifact cache and the cf-pair
    augmentation seed agree.
    """

    def __init__(
        self, *,
        stress_pool: list[dict],
        libri_pool: list[SimpleSample],
        expr_pool:  list[SimpleSample],
        cf_pairs:   list[CfPair],
        art_pairs:  list[ArtifactPair],
        seed:       int,
        phi_word_embed_cache: dict | None = None,
    ):
        self.stress_by_audio_id = {r["audio_id"]: r for r in stress_pool}
        self.stress_pool = stress_pool
        self.libri_pool  = libri_pool
        self.expr_pool   = expr_pool
        self.cf_pairs    = cf_pairs
        self.art_pairs   = art_pairs
        self.rng         = random.Random(seed)
        self._pair_counter = 0
        # Stage 3.6: lookup table {(transcription_id, phi) → embedding}.
        self.phi_word_embed_cache = phi_word_embed_cache or {}

    def _next_pair_id(self) -> int:
        self._pair_counter += 1
        return self._pair_counter

    def draw_cf_pair(self) -> tuple[_Sample, _Sample, int]:
        cp = self.rng.choice(self.cf_pairs)
        ra = self.stress_by_audio_id.get(cp.audio_id_a)
        rb = self.stress_by_audio_id.get(cp.audio_id_a_prime)
        if ra is None or rb is None:
            return self.draw_cf_pair()
        pair_id = self._next_pair_id()
        aug_seed = self.rng.getrandbits(63)
        # Stage 3.6 / B1: look up word-substitution description embeds for
        # L_NCE-cond. q_pos[member] = correct-phi embed; q_neg[member] = partner-phi embed.
        q_a       = self.phi_word_embed_cache.get((cp.transcription_id, int(cp.stress_index_a)))
        q_a_prime = self.phi_word_embed_cache.get((cp.transcription_id, int(cp.stress_index_a_prime)))
        # Stage 7 styled teacher: extract the actual word at each stress index.
        words_a = ra.get("words") or []
        word_a = words_a[cp.stress_index_a] if 0 <= cp.stress_index_a < len(words_a) else ""
        word_a_prime = words_a[cp.stress_index_a_prime] if 0 <= cp.stress_index_a_prime < len(words_a) else ""
        s_a = _Sample(
            audio=np.asarray(ra["audio_array"], dtype=np.float32),
            role="cf_a",
            transcript=cp.transcription,
            question=STRESS_QUESTION,
            response=" A" if cp.label_a == 0 else " B",
            options=cp.options,
            label=cp.label_a,
            phi_a=cp.stress_index_a,
            phi_a_prime=cp.stress_index_a_prime,
            n_words=cp.n_words,
            pair_id=pair_id,
            aug_seed=aug_seed,
            q_pos=q_a, q_neg=q_a_prime,
            transcription_id=cp.transcription_id,
            word_at_phi=word_a,
        )
        s_b = _Sample(
            audio=np.asarray(rb["audio_array"], dtype=np.float32),
            role="cf_a_prime",
            transcript=cp.transcription,
            question=STRESS_QUESTION,
            response=" A" if cp.label_a_prime == 0 else " B",
            options=cp.options,
            label=cp.label_a_prime,
            phi_a=cp.stress_index_a_prime,
            phi_a_prime=cp.stress_index_a,
            n_words=cp.n_words,
            pair_id=pair_id,
            aug_seed=aug_seed,
            q_pos=q_a_prime, q_neg=q_a,    # roles flip for partner
            transcription_id=cp.transcription_id,
            word_at_phi=word_a_prime,
        )
        return s_a, s_b, pair_id

    def draw_stress_individual(self) -> _Sample:
        r = self.rng.choice(self.stress_pool)
        words = r.get("words") or []
        si = r["stress_index"]
        word = words[si] if 0 <= si < len(words) else ""
        return _Sample(
            audio=np.asarray(r["audio_array"], dtype=np.float32),
            role="stress_individual",
            transcript=r["transcription"],
            question=STRESS_QUESTION,
            response=r["options"][r["label"]],
            options=tuple(r["options"]),
            label=r["label"],
            phi_a=r["stress_index"],
            phi_a_prime=None,
            n_words=r["n_words"],
            pair_id=None,
            aug_seed=self.rng.getrandbits(63),
            word_at_phi=word,
        )

    def draw_libri(self) -> _Sample:
        r = self.rng.choice(self.libri_pool)
        return _Sample(
            audio=r.audio,
            role="librispeech",
            transcript=r.transcript,
            question=ASR_QUESTION,
            response=r.transcript,
            options=None,
            label=None,
            phi_a=None,
            phi_a_prime=None,
            n_words=None,
            pair_id=None,
            aug_seed=self.rng.getrandbits(63),
        )

    def draw_artifact_pair(self) -> tuple[_Sample, _Sample, int]:
        ap = self.rng.choice(self.art_pairs)
        ra = self.expr_pool[ap.pool_idx_a]
        rb = self.expr_pool[ap.pool_idx_a_prime]
        pair_id = self._next_pair_id()
        aug_seed = self.rng.getrandbits(63)
        # Canonicalize the transcript across the pair so tokenized response
        # lengths match (otherwise L_artifact is silently zeroed by the
        # response_ids.shape mismatch guard).
        canonical_transcript = ra.transcript
        s_a = _Sample(
            audio=ra.audio, role="expresso_a",
            transcript=canonical_transcript, question=ASR_QUESTION,
            response=canonical_transcript, options=None, label=None,
            phi_a=None, phi_a_prime=None, n_words=None,
            pair_id=pair_id, aug_seed=aug_seed,
        )
        s_b = _Sample(
            audio=rb.audio, role="expresso_a_prime",
            transcript=canonical_transcript, question=ASR_QUESTION,
            response=canonical_transcript, options=None, label=None,
            phi_a=None, phi_a_prime=None, n_words=None,
            pair_id=pair_id, aug_seed=aug_seed,
        )
        return s_a, s_b, pair_id

    def cycle(self, plan: list[str]) -> list[_Sample]:
        """Realize one accumulation cycle's role plan into _Samples."""
        out: list[_Sample] = []
        i = 0
        while i < len(plan):
            r = plan[i]
            if r == "cf_a":
                a, b, _ = self.draw_cf_pair()
                out.append(a); out.append(b)
                assert plan[i+1] == "cf_a_prime"
                i += 2
            elif r == "stress_individual":
                out.append(self.draw_stress_individual())
                i += 1
            elif r == "librispeech":
                out.append(self.draw_libri())
                i += 1
            elif r == "expresso_a":
                a, b, _ = self.draw_artifact_pair()
                out.append(a); out.append(b)
                assert plan[i+1] == "expresso_a_prime"
                i += 2
            else:
                raise ValueError(f"unknown role: {r}")
        return out


# ---------- Per-microbatch processing ---------- #

def build_2afc_question(options: tuple[str, str]) -> str:
    return f"{STRESS_QUESTION}\nA) {options[0]}\nB) {options[1]}"


def encode_for_llm(
    *, tok, embed_layer, llm, K_speech_bf,
    left_text: str, right_text: str, response_text: str,
    vis_start_id: int, vis_end_id: int, max_response_tokens: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one Qwen3-8B forward and return (response_position_logits,
    response_ids, log_p_response).

    response_position_logits: (T_resp, vocab) — logits at positions that
    predict the response tokens. Used by both L_BLSP (full-vocab CE) and
    L_cf (restricted softmax over {A, B}).
    """
    left_ids = tok(left_text, return_tensors="pt", add_special_tokens=False
                   ).input_ids[0].to(dtype=torch.long, device=device)
    right_ids = tok(right_text, return_tensors="pt", add_special_tokens=False
                    ).input_ids[0].to(dtype=torch.long, device=device)
    response_ids = tok(response_text, return_tensors="pt", add_special_tokens=False
                       ).input_ids[0].to(dtype=torch.long, device=device)
    if response_ids.shape[0] > max_response_tokens:
        response_ids = response_ids[:max_response_tokens]
    T_resp = response_ids.shape[0]

    with torch.no_grad():
        E_left  = embed_layer(left_ids)
        E_right = embed_layer(right_ids)
        E_resp  = embed_layer(response_ids)
        vs = embed_layer(torch.tensor([vis_start_id], device=device))
        ve = embed_layer(torch.tensor([vis_end_id],   device=device))
    speech_prefix = torch.cat([E_left, vs, K_speech_bf, ve, E_right], dim=0)
    full = torch.cat([speech_prefix, E_resp], dim=0).unsqueeze(0)
    attn = torch.ones(1, full.shape[1], dtype=torch.long, device=device)
    out = llm(inputs_embeds=full, attention_mask=attn)
    logits = out.logits[0]
    T_total = logits.shape[0]
    start = T_total - T_resp - 1
    end   = T_total - 1
    response_pos_logits = logits[start:end].float()
    log_p = F.log_softmax(response_pos_logits, dim=-1)
    return response_pos_logits, response_ids, log_p


# ---------- Main ---------- #

def _stage6_finalize(elapsed, step, sample_count,
                      artifact_trigger_count, artifact_check_count,
                      main_clip_fires_count, main_clip_check_count,
                      grad_norm_log, args, ckpt_path) -> int:
    artifact_trigger_rate = artifact_trigger_count / max(1, artifact_check_count)
    main_clip_rate = main_clip_fires_count / max(1, main_clip_check_count)
    print(f"  artifact trigger rate: {artifact_trigger_count}/{artifact_check_count} "
          f"= {artifact_trigger_rate:.2%}", flush=True)
    print(f"  main-step clip rate:   {main_clip_fires_count}/{main_clip_check_count} "
          f"= {main_clip_rate:.2%}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    print(f"transformers=={__import__('transformers').__version__}, torch=={torch.__version__}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES','?')}", flush=True)
    if not torch.cuda.is_available():
        print("ERROR: no CUDA visible.", flush=True); return 1
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / "stage3"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "training_logs" / f"stage3_train_seed{args.seed}.log"
    ckpt_name = args.ckpt_name or f"A_R1_seed{args.seed}.pt"
    ckpt_path = out_dir / ckpt_name

    # ---- adapter init from Stage 2 calibration cache ---- #
    init_path = ROOT / "outputs" / "stage2" / "adapter_init.json"
    if not init_path.exists():
        print(f"ERROR: missing {init_path} — Stage 2 must run first.")
        return 1
    init = json.loads(init_path.read_text())
    cal = init["adapter_init"]
    embed_meta = init["qwen3_8b"]
    print(f"  std_8B={embed_meta['embed_tokens_std']:.5f} (cached, kickoff R2)", flush=True)

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

    # Stage 6 LoRA: wrap with PEFT BEFORE getting embed layer reference,
    # because PEFT may rewire submodules. Embedding layer remains frozen.
    if args.lora_rank > 0:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            init_lora_weights="gaussian",
            task_type=TaskType.CAUSAL_LM,
        )
        llm = get_peft_model(llm, lora_config)
        # Make sure all non-LoRA params are still frozen.
        for n, p in llm.named_parameters():
            if "lora_" not in n:
                p.requires_grad_(False)
        n_lora = sum(p.numel() for p in llm.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in llm.parameters())
        print(f"  Stage 6 LoRA active: rank={args.lora_rank}, "
              f"alpha={args.lora_alpha}, dropout={args.lora_dropout}", flush=True)
        print(f"  trainable LoRA params: {n_lora/1e6:.2f}M ({n_lora/n_total*100:.3f}% of LLM)",
              flush=True)
        # Cast LoRA matrices to fp32 for stable training (forward stays bf16
        # via PEFT's autocast); follow standard LoRA convention.
        for n, p in llm.named_parameters():
            if "lora_" in n:
                p.data = p.data.to(torch.float32)
    embed_layer = llm.get_input_embeddings()
    d_llm = embed_layer.weight.shape[1]
    print(f"  loaded in {time.time()-t0:.1f}s, d_llm={d_llm}", flush=True)

    banner(f"Loading frozen speech encoder ({WAVLM_MODEL}) in bf16")
    from transformers import WavLMModel, AutoFeatureExtractor
    feat = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL)
    wavlm = WavLMModel.from_pretrained(WAVLM_MODEL, torch_dtype=torch.bfloat16).eval().to(device)
    for p in wavlm.parameters():
        p.requires_grad_(False)

    banner(f"Building adapter with calibrated init")
    cfg = AdapterConfig(
        d_enc=1024, d_llm=cal["d_llm"],
        conv_kernel=cal["conv_kernel"], conv_stride=cal["conv_stride"],
        mlp_hidden_mult=cal["mlp_hidden_mult"],
        last_linear_std=cal["last_linear_std"],
        rmsnorm_init_scale=cal["rmsnorm_init_scale"],
        modality_token_std=cal["modality_token_std"],
    )
    adapter = BLSPAdapter(cfg).to(device).to(torch.float32)
    print(f"  adapter trainable params = {adapter.n_trainable_params()/1e6:.2f}M", flush=True)

    # Stage 6: load frozen adapter + frozen C_phi heads from R1.8 cohort.
    frozen_c_phi_full_state = None
    frozen_c_phi_t_only_state = None
    if args.frozen_adapter_ckpt:
        ckpt_path_in = Path(args.frozen_adapter_ckpt)
        if not ckpt_path_in.exists():
            print(f"ERROR: missing {ckpt_path_in}"); return 1
        print(f"  Loading frozen adapter from {ckpt_path_in}", flush=True)
        loaded = torch.load(ckpt_path_in, map_location=device, weights_only=False)
        adapter.load_state_dict(loaded["adapter_state_dict"])
        for p in adapter.parameters():
            p.requires_grad_(False)
        adapter.eval()
        print(f"    adapter frozen ({sum(1 for p in adapter.parameters() if p.requires_grad)} trainable params)",
              flush=True)
        frozen_c_phi_full_state   = loaded.get("c_phi_full_state_dict")
        frozen_c_phi_t_only_state = loaded.get("c_phi_t_only_state_dict")

    banner(f"Building C_phi heads (full + T-only)")
    d_T = 8194
    c_phi_full = CPhi(d_in=d_llm + d_T, hidden=256, n_classes=N_MAX_CLASSES).to(device).to(torch.float32)
    c_phi_t_only = CPhi(d_in=d_T, hidden=256, n_classes=N_MAX_CLASSES).to(device).to(torch.float32)
    if frozen_c_phi_full_state is not None:
        c_phi_full.load_state_dict(frozen_c_phi_full_state)
        for p in c_phi_full.parameters():
            p.requires_grad_(False)
        c_phi_full.eval()
        print(f"  C_phi_full loaded + frozen", flush=True)
    if frozen_c_phi_t_only_state is not None:
        c_phi_t_only.load_state_dict(frozen_c_phi_t_only_state)
        for p in c_phi_t_only.parameters():
            p.requires_grad_(False)
        c_phi_t_only.eval()
        print(f"  C_phi_t_only loaded + frozen", flush=True)
    n_full = sum(p.numel() for p in c_phi_full.parameters() if p.requires_grad)
    n_tonly = sum(p.numel() for p in c_phi_t_only.parameters() if p.requires_grad)
    print(f"  C_phi_full trainable params  = {n_full/1e6:.2f}M", flush=True)
    print(f"  C_phi_t_only trainable params = {n_tonly/1e6:.2f}M", flush=True)

    vis_start_id = int(tok.convert_tokens_to_ids("<|vision_start|>"))
    vis_end_id   = int(tok.convert_tokens_to_ids("<|vision_end|>"))
    assert (vis_start_id, vis_end_id) == (151652, 151653)

    # Stage 3.6 / B1: build per-(transcription_id, phi) word-substitution
    # description embeds. Replaces the abstract-template `phi_embeds` cache
    # used in Stage 3 (which had ≥0.99 cos sim across phis → dead L_NCE).
    print("Building B1 phi-word embed cache (will be populated below after data load)…", flush=True)

    # ---- data ---- #
    banner("Loading data pools")
    if not STRESS_POOL_PATH.exists():
        print(f"ERROR: missing {STRESS_POOL_PATH} — run scripts/stage3_build_cf_pairs.py first.")
        return 1
    with open(STRESS_POOL_PATH, "rb") as f:
        stress_pool = pickle.load(f)
    print(f"  Stress-17K probe-train pool: {len(stress_pool)} rows", flush=True)

    libri_rows = load_librispeech_subset(args.n_libri, seed=args.seed)
    print(f"  LibriSpeech sampled: {len(libri_rows)} rows", flush=True)
    expr_rows = load_expresso_subset(args.n_expr, seed=args.seed)
    print(f"  Expresso sampled:    {len(expr_rows)} rows", flush=True)

    cf_pairs_path = Path(args.cf_pairs_path) if args.cf_pairs_path else CF_TRAIN_JSONL
    cf_pairs = read_cf_pairs_jsonl(cf_pairs_path)
    art_pairs = read_artifact_pairs_jsonl(CF_ARTIFACT_JSONL)
    print(f"  cf_pairs_train:    {len(cf_pairs)}  (from {cf_pairs_path.name})", flush=True)
    print(f"  cf_pairs_artifact: {len(art_pairs)}", flush=True)

    # Stage 4 Control A: load proj_P for text-only forward path.
    proj_P_W = None
    proj_P_b = None
    if args.control_mode == "text_only":
        proj_P_path = Path(args.proj_P_path) if args.proj_P_path else (
            ROOT / "outputs" / "stage4" / "proj_P.pt")
        if not proj_P_path.exists():
            print(f"ERROR: missing {proj_P_path} — run scripts/stage4_fit_proj_P.py first.")
            return 1
        P_payload = torch.load(proj_P_path, map_location=device, weights_only=False)
        proj_P_W = P_payload["weight"].to(device=device, dtype=torch.float32)   # (4096, 1024)
        proj_P_b = P_payload["bias"].to(device=device, dtype=torch.float32)     # (1024,)
        print(f"  control_mode=text_only: loaded proj_P from {proj_P_path.name} "
              f"(W{tuple(proj_P_W.shape)}, b{tuple(proj_P_b.shape)})", flush=True)
        print(f"  proj_P fit MSE={P_payload.get('train_mse'):.4f}, "
              f"cos_sim={P_payload.get('train_cos_sim'):.4f}", flush=True)

    # Stage 3.6 / B1: build per-(transcription_id, phi) embed cache from cf pairs.
    stress_by_audio_id = {r["audio_id"]: r for r in stress_pool}
    phi_word_embed_cache = build_phi_word_embed_cache(
        cf_pairs=cf_pairs, stress_by_audio_id=stress_by_audio_id,
        embed_layer=embed_layer, tokenizer=tok, device=torch.device(device),
    )
    print(f"  phi_word_embed_cache: {len(phi_word_embed_cache)} entries", flush=True)
    # Stage 3.6 / B1 cos-sim gate (kickoff Option C update 1):
    # The L_NCE-relevant cos sim is between (q_pos, q_neg) within a SINGLE
    # cf-pair (same transcript, different phi). Sample cf pairs directly.
    if len(cf_pairs) >= 100:
        rng_pair = random.Random(args.seed)
        sampled_pairs = rng_pair.sample(list(cf_pairs), min(500, len(cf_pairs)))
        pair_sims = []
        for cp in sampled_pairs:
            q_pos = phi_word_embed_cache.get((cp.transcription_id, int(cp.stress_index_a)))
            q_neg = phi_word_embed_cache.get((cp.transcription_id, int(cp.stress_index_a_prime)))
            if q_pos is None or q_neg is None:
                continue
            qp = F.normalize(q_pos, dim=-1)
            qn = F.normalize(q_neg, dim=-1)
            pair_sims.append(float((qp * qn).sum().item()))
        if pair_sims:
            ps = np.asarray(pair_sims)
            print(f"  paired (q_pos, q_neg) cos-sim (n={len(ps)} cf pairs):", flush=True)
            print(f"    mean={ps.mean():.3f}  median={np.median(ps):.3f}  "
                  f"min={ps.min():.3f}  max={ps.max():.3f}  "
                  f"frac>0.95={(ps > 0.95).mean():.2%}  "
                  f"frac>0.85={(ps > 0.85).mean():.2%}", flush=True)
            print(f"  (gate: mean ≤ 0.85)", flush=True)
            if ps.mean() > 0.85:
                print("  ❌ COS-SIM GATE FAIL — q_pos/q_neg degenerate within cf pairs.", flush=True)
                return 1
            print(f"  ✓ cos-sim gate PASS (mean {ps.mean():.3f} ≤ 0.85)", flush=True)

    sampler = StageThreeSampler(
        stress_pool=stress_pool, libri_pool=libri_rows,
        expr_pool=expr_rows, cf_pairs=cf_pairs, art_pairs=art_pairs,
        seed=args.seed, phi_word_embed_cache=phi_word_embed_cache,
    )
    plan = build_microbatch_plan(args.grad_accum)
    print(f"  microbatch plan ({len(plan)} per cycle): "
          f"cf={plan.count('cf_a')+plan.count('cf_a_prime')}, "
          f"stress_ind={plan.count('stress_individual')}, "
          f"libri={plan.count('librispeech')}, "
          f"expresso={plan.count('expresso_a')+plan.count('expresso_a_prime')}", flush=True)

    # ---- augmentation ---- #
    aug_cfg = aggressive_stage3_config() if args.use_aggressive_aug else AugmentConfig()
    print(f"  aug: apply_prob={aug_cfg.apply_prob}, codec_choices={aug_cfg.codec_choices}, "
          f"rms_norm={aug_cfg.rms_norm_target}", flush=True)

    # ---- optimizer ---- #
    if args.lora_rank > 0:
        # Stage 6: only LoRA matrices train. Adapter + C_phi heads frozen.
        lora_params = [p for n, p in llm.named_parameters() if p.requires_grad and "lora_" in n]
        if not lora_params:
            print("ERROR: lora_rank > 0 but no LoRA params found", flush=True); return 1
        opt = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=args.weight_decay)
        print(f"  optimizer: AdamW on {sum(p.numel() for p in lora_params)/1e6:.2f}M LoRA params, "
              f"lr={args.lr}, weight_decay={args.weight_decay}", flush=True)
    else:
        opt = torch.optim.AdamW(
            [
                {"params": [p for p in adapter.parameters() if p.requires_grad]},
                {"params": [p for p in c_phi_full.parameters() if p.requires_grad]},
                {"params": [p for p in c_phi_t_only.parameters() if p.requires_grad]},
            ],
            lr=args.lr, weight_decay=args.weight_decay,
        )

    # ---- training loop ---- #
    max_steps = args.max_steps if args.smoke_steps == 0 else args.smoke_steps
    banner(f"Training (seed={args.seed}, max_steps={max_steps}, grad_accum={args.grad_accum})")
    log_records = []
    step = 0
    t0 = time.time()
    sample_count = 0
    artifact_cache = ArtifactPairCache()

    # B2 — per-component grad-norm diagnostic accumulator.
    grad_norm_log = []
    artifact_trigger_count = 0
    artifact_check_count = 0
    main_clip_fires_count = 0       # Stage 3.7: clip-rate over main training steps
    main_clip_check_count = 0

    def _one_microbatch(s: _Sample, step: int) -> dict:
        """Run forward+loss assembly for one role; backward done outside.

        Returns a dict {component_name: scalar_tensor (on CUDA)} representing
        loss components; caller backwards their sum.
        """
        if args.control_mode == "text_only":
            # Stage 4 Control A: replace audio path with P(embed_tokens(transcript)).
            # Skip WavLM and augmentation. Pad H_text to multiple of 4 ≥ 4
            # (the conv kernel requires ≥ 4 frames). Set valid_T_s to the
            # padded length so the adapter's conv produces the full output
            # over the padded sequence; mean-pool will see padded positions
            # but the bias is small for our short-transcript edge cases.
            with torch.no_grad():
                ids = tok(s.transcript, return_tensors="pt",
                          add_special_tokens=False).input_ids[0].to(
                              dtype=torch.long, device=device)
                K_T_pertoken = embed_layer(ids).float()        # (T_text, 4096)
                H_text = K_T_pertoken @ proj_P_W + proj_P_b    # (T_text, 1024)
                T_text = H_text.shape[0]
                target_T = max(4, ((T_text + 3) // 4) * 4)     # round up to multiple of 4, ≥ 4
                if target_T > T_text:
                    pad_len = target_T - T_text
                    pad = torch.zeros(pad_len, H_text.shape[1],
                                       dtype=H_text.dtype, device=H_text.device)
                    H_text = torch.cat([H_text, pad], dim=0)
            H_fp32 = H_text.unsqueeze(0)                       # (1, target_T, 1024)
            valid_T_s = torch.tensor([target_T], dtype=torch.long, device=device)
        else:
            # Trim audio
            max_n = int(args.max_audio_seconds * SR)
            if s.audio.shape[0] > max_n:
                s.audio = s.audio[:max_n]
            # Augment with the role-specific seed (paired roles share aug_seed
            # so cf-pair / artifact-pair members get identical augmentation).
            rng = random.Random(s.aug_seed)
            x_aug = augment_one(s.audio, cfg=aug_cfg, rng=rng)

            # WavLM forward (frozen, no grad).
            with torch.no_grad():
                proc = feat([x_aug], sampling_rate=SR, return_tensors="pt",
                            padding=True, return_attention_mask=True)
                iv = proc["input_values"].to(device).to(torch.bfloat16)
                am = proc["attention_mask"].to(device)
                wav_out = wavlm(input_values=iv, attention_mask=am, output_hidden_states=True)
                H = wav_out.hidden_states[16]
                sample_lengths = am.sum(dim=1)
                valid_T_s = wavlm._get_feat_extract_output_lengths(sample_lengths).long()
            H_fp32 = H.float()

        # Adapter forward (fp32 with grad).
        K, valid_T_k = adapter(H_fp32, valid_T_s=valid_T_s)
        Tk = int(valid_T_k[0].item())
        K_speech = K[0, :Tk, :]                     # (Tk, d_llm), fp32
        K_speech_bf = K_speech.to(torch.bfloat16)

        # Build prompt halves per role.
        if s.role in ("cf_a", "cf_a_prime"):
            question_text = build_2afc_question(s.options)
        else:
            question_text = s.question
        halves = build_training_halves(
            tok, system=DEFAULT_SYSTEM,
            question=question_text, response=s.response,
        )

        # LLM forward.
        response_pos_logits, response_ids, log_p_response = encode_for_llm(
            tok=tok, embed_layer=embed_layer, llm=llm, K_speech_bf=K_speech_bf,
            left_text=halves.left_text, right_text=halves.right_text,
            response_text=halves.response_text,
            vis_start_id=vis_start_id, vis_end_id=vis_end_id,
            max_response_tokens=args.max_response_tokens, device=device,
        )

        # Diagnostic ratio (no grad path).
        with torch.no_grad():
            row_norm = embed_layer.weight.float().norm(dim=-1).mean().item()
            ratio = (K_speech.float().norm(dim=-1).mean().item() / row_norm)

        comps: dict[str, torch.Tensor] = {}

        # ---- L_BLSP for everyone except expresso pair members ---- #
        T_resp = response_ids.shape[0]
        if s.role != "expresso_a" and s.role != "expresso_a_prime":
            # Speech-branch L_task on full-vocab softmax.
            L_task = -log_p_response[
                torch.arange(T_resp, device=device), response_ids
            ].mean()
            # Text-branch teacher (no grad).
            # Stage 7: when styled_teacher is on AND this row has a stress
            # index (cf or stress_individual), append " [stress on word: X]"
            # to the transcript before tokenization.
            if (args.styled_teacher and s.word_at_phi
                    and s.role in ("cf_a", "cf_a_prime", "stress_individual")):
                styled_text = f"{s.transcript} [stress on word: {s.word_at_phi}]"
            else:
                styled_text = s.transcript
            with torch.no_grad():
                left_ids = tok(halves.left_text, return_tensors="pt",
                                add_special_tokens=False).input_ids[0
                                ].to(dtype=torch.long, device=device)
                right_ids = tok(halves.right_text, return_tensors="pt",
                                 add_special_tokens=False).input_ids[0
                                 ].to(dtype=torch.long, device=device)
                tr_ids = tok(styled_text, return_tensors="pt",
                              add_special_tokens=False).input_ids[0
                              ].to(dtype=torch.long, device=device)
                vs = embed_layer(torch.tensor([vis_start_id], device=device))
                ve = embed_layer(torch.tensor([vis_end_id],   device=device))
                K_text = embed_layer(tr_ids)
                E_left  = embed_layer(left_ids)
                E_right = embed_layer(right_ids)
                E_resp  = embed_layer(response_ids)
                text_full = torch.cat([E_left, vs, K_text, ve, E_right, E_resp], dim=0).unsqueeze(0)
                attn_t = torch.ones(1, text_full.shape[1], dtype=torch.long, device=device)
                out_t = llm(inputs_embeds=text_full, attention_mask=attn_t)
                T_total_t = out_t.logits.shape[1]
                start = T_total_t - T_resp - 1
                logits_t_resp = out_t.logits[0, start:start+T_resp].float()
                log_p_text = F.log_softmax(logits_t_resp, dim=-1)
                p_text = log_p_text.exp()
            L_KL = (p_text * (log_p_text - log_p_response)).sum(dim=-1).mean()
            comps["L_task"] = L_task
            comps["L_KL"]   = L_KL

        # ---- L_cf on cf-pair members (single-token margin) ---- #
        if s.role in ("cf_a", "cf_a_prime") and T_resp >= 1:
            # response is " A" or " B", first token id is the answer.
            correct_id = int(response_ids[0].item())
            wrong_id = TOK_ID_SP_B if correct_id == TOK_ID_SP_A else TOK_ID_SP_A
            L_cf = compute_l_cf_from_logits(
                response_position_logits=response_pos_logits[0],
                correct_letter_id=correct_id, wrong_letter_id=wrong_id,
            )
            comps["L_cf"] = L_cf

        # ---- L_cond_pred + T-only diagnostic on cf-pair members ---- #
        if s.role in ("cf_a", "cf_a_prime"):
            pooled_A_H = K_speech.mean(dim=0)            # (d_llm,) fp32
            rT_np = restricted_t_features(s.transcript, n_words=int(s.n_words or 0))
            rT = torch.from_numpy(rT_np).to(device=device, dtype=torch.float32)
            cd = compute_l_cond_pred(
                c_phi_full=c_phi_full, c_phi_t_only=c_phi_t_only,
                pooled_A_H=pooled_A_H, restricted_T=rT,
                phi_label=int(s.phi_a), n_max_classes=N_MAX_CLASSES,
            )
            comps["L_cond_pred"]  = cd["L_cond_pred"]
            comps["L_t_only_aux"] = cd["L_t_only_aux"]

        # ---- L_NCE_cond on cf-pair members (B1: word-substitution descriptions) ---- #
        if (s.role in ("cf_a", "cf_a_prime") and s.phi_a_prime is not None
                and s.q_pos is not None and s.q_neg is not None):
            pooled_A_H = K_speech.mean(dim=0) if "L_cond_pred" not in comps else pooled_A_H
            q_pos = s.q_pos.detach().to(device=device, dtype=torch.float32)
            q_neg = s.q_neg.detach().to(device=device, dtype=torch.float32)
            L_NCE = compute_l_nce_cond(
                pooled_A_H=pooled_A_H, q_pos=q_pos, q_neg=q_neg, tau=args.tau_nce,
            )
            comps["L_NCE_cond"] = L_NCE

        # ---- L_artifact on expresso pair members (asymmetric, role-2 grad) ---- #
        if s.role == "expresso_a":
            # Cache detached log_p for partner consumption.
            artifact_cache.put(int(s.pair_id), ArtifactCacheEntry(
                detached_log_p=log_p_response.detach(),
                response_ids=response_ids.detach(),
            ))
            # No backward contribution from this microbatch.
        elif s.role == "expresso_a_prime":
            entry = artifact_cache.consume(int(s.pair_id))
            if entry is not None:
                # Match response lengths.
                if entry.response_ids.shape[0] != response_ids.shape[0]:
                    L_art = torch.tensor(0.0, device=device, requires_grad=False)
                else:
                    L_art = compute_l_artifact(
                        detached_log_p_a=entry.detached_log_p,
                        log_p_a_prime=log_p_response,
                        delta_artifact=args.delta_artifact,
                    )
                comps["L_artifact"] = L_art

        comps["_ratio"] = torch.tensor(ratio, device=device, requires_grad=False)
        return comps

    while step < max_steps:
        opt.zero_grad(set_to_none=True)
        cur_lr = lr_at(step, warmup=args.warmup_steps, max_steps=max_steps,
                       peak_lr=args.lr, min_lr=args.min_lr)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        cycle = sampler.cycle(plan)
        agg = {
            "L_task": 0.0, "L_KL": 0.0, "L_cf": 0.0, "L_artifact": 0.0,
            "L_cond_pred": 0.0, "L_t_only_aux": 0.0, "L_NCE_cond": 0.0,
            "ratio": 0.0, "n_total": 0,
            "n_cf": 0, "n_artifact": 0, "n_blsp": 0, "n_cond": 0, "n_nce": 0,
        }
        n_cf_pred_correct = 0
        n_cf_t_only_correct = 0
        n_cf_seen = 0

        for s in cycle:
            comps = _one_microbatch(s, step)
            # Build accumulation gradient: compose roles-relevant pieces.
            losses = []
            if "L_task" in comps and "L_KL" in comps:
                L_blsp = comps["L_task"] + args.lambda_kl * comps["L_KL"]
                losses.append(L_blsp)
                agg["L_task"] += float(comps["L_task"].item())
                agg["L_KL"]   += float(comps["L_KL"].item())
                agg["n_blsp"] += 1
            if "L_cf" in comps:
                losses.append(args.lambda_cf * comps["L_cf"])
                agg["L_cf"] += float(comps["L_cf"].item())
                agg["n_cf"] += 1
            if "L_cond_pred" in comps:
                losses.append(args.lambda_cond * comps["L_cond_pred"])
                losses.append(comps["L_t_only_aux"])     # ungated; trains C_phi_t_only
                agg["L_cond_pred"] += float(comps["L_cond_pred"].item())
                agg["L_t_only_aux"] += float(comps["L_t_only_aux"].item())
                agg["n_cond"] += 1
            if "L_NCE_cond" in comps:
                losses.append(args.lambda_nce * comps["L_NCE_cond"])
                agg["L_NCE_cond"] += float(comps["L_NCE_cond"].item())
                agg["n_nce"] += 1
            if "L_artifact" in comps:
                la = float(comps["L_artifact"].item())
                losses.append(args.lambda_artifact * comps["L_artifact"])
                agg["L_artifact"] += la
                agg["n_artifact"] += 1
                artifact_check_count += 1
                if la > 0.0:
                    artifact_trigger_count += 1
            agg["ratio"] += float(comps["_ratio"].item())
            agg["n_total"] += 1

            if losses:
                total = sum(losses)
                (total / args.grad_accum).backward()
            sample_count += 1

        # B2 — per-component grad-norm diagnostic + Stage 3.7 pre/post-clip
        # contribution + clip-rate logging. Runs BEFORE the main clipping/
        # optimizer step on a single representative cf-pair-member microbatch
        # so we can decompose gradients by loss term without disturbing the
        # accumulated gradient state. We restore state below.
        if (step + 1) % args.diag_every == 0 and step + 1 < max_steps - 5:
            saved_grads = {id(p): (p.grad.detach().clone() if p.grad is not None else None)
                            for p in (list(adapter.parameters())
                                      + list(c_phi_full.parameters())
                                      + list(c_phi_t_only.parameters()))}
            try:
                diag_a, diag_b, _ = sampler.draw_cf_pair()
                diag_comps = _one_microbatch(diag_a, step)
                pc = {}
                per_term_flat = {}     # flat gradient vectors per component
                # Use the actual trainable params (adapter for Stage 3.x; LoRA
                # matrices for Stage 6 LoRA mode). If empty (shouldn't happen
                # but guard) skip the diagnostic.
                if args.lora_rank > 0:
                    adapter_params = [p for n, p in llm.named_parameters()
                                       if p.requires_grad and "lora_" in n]
                else:
                    adapter_params = [p for p in adapter.parameters() if p.requires_grad]
                if not adapter_params:
                    print(f"  [grad-norm @ {step+1}] no trainable params; skipping diagnostic", flush=True)
                    continue
                lam = {
                    "L_task":      1.0,
                    "L_KL":        args.lambda_kl,
                    "L_cf":        args.lambda_cf,
                    "L_cond_pred": args.lambda_cond,
                    "L_NCE_cond":  args.lambda_nce,
                }
                for cname in ("L_task", "L_KL", "L_cf", "L_cond_pred", "L_NCE_cond"):
                    L = diag_comps.get(cname)
                    if L is None or not torch.is_tensor(L) or not L.requires_grad or L.dim() != 0:
                        continue
                    try:
                        grads = torch.autograd.grad(
                            L, adapter_params, retain_graph=True, allow_unused=True,
                        )
                        flat = torch.cat([
                            g.float().flatten() if g is not None else torch.zeros_like(p).float().flatten()
                            for g, p in zip(grads, adapter_params)
                        ])
                        per_term_flat[cname] = flat
                        pc[cname] = float(flat.norm().item())
                    except RuntimeError:
                        pc[cname] = float("nan")
                if "L_task" in pc and "L_KL" in pc:
                    pc["L_BLSP"] = pc["L_task"] + args.lambda_kl * pc["L_KL"]

                # Stage 3.7 — pre-clip cosine contribution.
                # Total update direction = Σ λ_i · grad_i (matches the
                # actual main-step gradient that the optimizer sees, modulo
                # the diagnostic-vs-cycle averaging — the direction matches
                # in expectation per-cf-pair-member microbatch).
                contrib_pre = {}
                contrib_post = {}
                lam_used = {}
                clip_fires = False
                clip_factor = 1.0
                if per_term_flat:
                    total = torch.zeros_like(next(iter(per_term_flat.values())))
                    for cname, g in per_term_flat.items():
                        total = total + lam.get(cname, 1.0) * g
                    total_norm = float(total.norm().item())
                    clip_fires = bool(total_norm > 1.0)
                    clip_factor = (1.0 / total_norm) if clip_fires else 1.0
                    total_post = total * clip_factor
                    for cname, g in per_term_flat.items():
                        # Cosine is scale-invariant ⇒ pre-clip and post-clip
                        # cosines are identical by construction. Log both for
                        # the kickoff spec; their equality is a sanity check.
                        denom = float((g.norm() * total.norm()).item()) + 1e-12
                        cos_pre = float((g @ total).item()) / denom
                        denom_p = float((g.norm() * total_post.norm()).item()) + 1e-12
                        cos_post = float((g @ total_post).item()) / denom_p
                        contrib_pre[cname]  = cos_pre
                        contrib_post[cname] = cos_post
                        lam_used[cname]     = lam.get(cname, 1.0)

                rec = {
                    "step": step + 1,
                    "grad_norms":          pc,
                    "lambdas":             lam_used,
                    "cosine_pre_clip":     contrib_pre,
                    "cosine_post_clip":    contrib_post,
                    "total_grad_norm":     float(total.norm().item()) if per_term_flat else None,
                    "clip_fires":          clip_fires,
                    "clip_factor":         clip_factor,
                }
                grad_norm_log.append(rec)
                norms_str = "  ".join(f"{k}={v:.2f}" for k, v in pc.items())
                print(f"  [grad-norm @ {step+1}] {norms_str}", flush=True)
                cos_str = "  ".join(f"{k}={v:.3f}" for k, v in contrib_pre.items())
                print(f"  [cos-contrib @ {step+1}] {cos_str}  "
                      f"clip_fires={clip_fires}  total_norm={rec['total_grad_norm']:.2f}", flush=True)
            finally:
                # Restore accumulated gradients + free graph.
                for p in (list(adapter.parameters())
                           + list(c_phi_full.parameters())
                           + list(c_phi_t_only.parameters())):
                    saved = saved_grads.get(id(p))
                    p.grad = saved.clone() if saved is not None else None

        # Stage 3.7 — main-step clip-rate tracking (NOT the diagnostic
        # microbatch; this is the actual training-step clip rate).
        # Capture the pre-clip total grad norm as it was JUST after the
        # accumulation cycle finished, by computing it via clip_grad_norm_
        # call below — clip_grad_norm_ returns the PRE-clip norm.

        # Optimizer step. clip_grad_norm_ returns the PRE-clip total norm,
        # so we can track the rate at which clipping actually fires on the
        # main accumulated gradient (Stage 3.7 / Codex new diagnostic).
        if args.lora_rank > 0:
            clip_params = [p for n, p in llm.named_parameters() if p.requires_grad and "lora_" in n]
        else:
            clip_params = (list(adapter.parameters())
                            + list(c_phi_full.parameters())
                            + list(c_phi_t_only.parameters()))
        gnorm = torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0).item()
        main_clip_check_count += 1
        if gnorm > 1.0:
            main_clip_fires_count += 1
        opt.step()
        artifact_cache.clear()
        step += 1

        # ---- Logging ---- #
        if step == 1 or step % args.log_every == 0 or step == max_steps:
            n_total = max(1, agg["n_total"])
            mean_ratio  = agg["ratio"]  / n_total
            mean_L_task = agg["L_task"] / max(1, agg["n_blsp"])
            mean_L_KL   = agg["L_KL"]   / max(1, agg["n_blsp"])
            mean_L_cf   = agg["L_cf"]   / max(1, agg["n_cf"])
            mean_L_art  = agg["L_artifact"] / max(1, agg["n_artifact"])
            mean_L_cond = agg["L_cond_pred"] / max(1, agg["n_cond"])
            mean_L_t    = agg["L_t_only_aux"] / max(1, agg["n_cond"])
            mean_L_nce  = agg["L_NCE_cond"]   / max(1, agg["n_nce"])
            elapsed = time.time() - t0
            sps = sample_count / max(elapsed, 1e-3)
            rec = {
                "step": step, "lr": cur_lr,
                "L_task": mean_L_task, "L_KL": mean_L_KL,
                "L_cf": mean_L_cf, "L_artifact": mean_L_art,
                "L_cond": mean_L_cond, "L_t_only": mean_L_t,
                "L_NCE": mean_L_nce,
                "ratio": mean_ratio, "grad_norm": gnorm,
                "samples_per_s": sps, "elapsed_s": elapsed,
            }
            log_records.append(rec)
            print(f"  step {step:>4} lr={cur_lr:.2e}  "
                  f"L_task={mean_L_task:.3f}  L_KL={mean_L_KL:.3f}  "
                  f"L_cf={mean_L_cf:.3f}  L_art={mean_L_art:.3f}  "
                  f"L_cond={mean_L_cond:.3f}  L_t={mean_L_t:.3f}  "
                  f"L_NCE={mean_L_nce:.3f}  "
                  f"ratio={mean_ratio:.2f}  gnorm={gnorm:.2f}  sps={sps:.2f}",
                  flush=True)

            # Sanity guards.
            if any(not math.isfinite(v) for v in (mean_L_task, mean_L_KL,
                    mean_L_cf, mean_L_art, mean_L_cond, mean_L_nce)):
                print("  STOP — NaN/Inf in a component; aborting.", flush=True)
                break
            if step > args.warmup_steps and gnorm > 200.0:
                print(f"  STOP — post-warmup grad_norm {gnorm:.2f} > 200 (catastrophic).", flush=True)
                break
            if not (RATIO_BAND[0] <= mean_ratio <= RATIO_BAND[1]):
                if mean_ratio > 6.0 or mean_ratio < 0.1:
                    print(f"  STOP — ratio drift {mean_ratio:.2f} outside extended tolerance.", flush=True)
                    break

    elapsed = time.time() - t0
    print(f"\n  done: {step} optimizer steps over {elapsed:.1f}s  "
          f"({sample_count/max(elapsed,1e-3):.2f} samples/s)", flush=True)

    # Save checkpoint + log
    artifact_trigger_rate = (artifact_trigger_count / max(1, artifact_check_count))
    main_clip_rate = main_clip_fires_count / max(1, main_clip_check_count)
    if args.lora_rank > 0:
        # Stage 6: save LoRA weights via PEFT (separate dir);
        # adapter state already on disk under --frozen_adapter_ckpt.
        ckpt_lora_dir = ckpt_path.with_suffix("")        # strip .pt
        ckpt_lora_dir.mkdir(parents=True, exist_ok=True)
        llm.save_pretrained(str(ckpt_lora_dir))
        meta_path = ckpt_lora_dir.parent / (ckpt_lora_dir.name + "_meta.pt")
        torch.save({
            "adapter_config": asdict(cfg),
            "args": vars(args),
            "embed_meta": embed_meta,
            "training_log": log_records,
            "grad_norm_log": grad_norm_log,
            "artifact_trigger_rate": artifact_trigger_rate,
            "artifact_trigger_count": artifact_trigger_count,
            "artifact_check_count": artifact_check_count,
            "main_clip_fires_count": main_clip_fires_count,
            "main_clip_check_count": main_clip_check_count,
            "main_clip_rate":         main_clip_rate,
            "final_step": step,
            "wallclock_s": elapsed,
            "frozen_adapter_ckpt":  args.frozen_adapter_ckpt,
        }, meta_path)
        print(f"  saved LoRA weights → {ckpt_lora_dir}", flush=True)
        print(f"  saved meta → {meta_path}", flush=True)
        # Skip the regular checkpoint save below.
        ckpt_path = ckpt_lora_dir   # for the final-print summary
        return _stage6_finalize(elapsed, step, sample_count,
                                  artifact_trigger_count, artifact_check_count,
                                  main_clip_fires_count, main_clip_check_count,
                                  grad_norm_log, args, ckpt_path)
    payload = {
        "adapter_state_dict": adapter.state_dict(),
        "c_phi_full_state_dict":   c_phi_full.state_dict(),
        "c_phi_t_only_state_dict": c_phi_t_only.state_dict(),
        "adapter_config": asdict(cfg),
        "args": vars(args),
        "embed_meta": embed_meta,
        "training_log": log_records,
        "grad_norm_log": grad_norm_log,
        "artifact_trigger_rate": artifact_trigger_rate,
        "artifact_trigger_count": artifact_trigger_count,
        "artifact_check_count": artifact_check_count,
        "main_clip_fires_count": main_clip_fires_count,
        "main_clip_check_count": main_clip_check_count,
        "main_clip_rate":         main_clip_rate,
        "final_step": step,
        "wallclock_s": elapsed,
    }
    torch.save(payload, ckpt_path)
    # Also dump grad_norm + clip log as JSON for off-line inspection.
    grad_clip_json_path = ckpt_path.with_suffix("").parent / f"grad_norm_clip_log_seed{args.seed}.json"
    grad_clip_json_path.write_text(json.dumps({
        "lambdas": {"lambda_kl": args.lambda_kl, "lambda_cf": args.lambda_cf,
                    "lambda_artifact": args.lambda_artifact,
                    "lambda_cond": args.lambda_cond, "lambda_nce": args.lambda_nce},
        "main_clip_fires_count": main_clip_fires_count,
        "main_clip_check_count": main_clip_check_count,
        "main_clip_rate":        main_clip_rate,
        "artifact_trigger_count": artifact_trigger_count,
        "artifact_check_count":   artifact_check_count,
        "artifact_trigger_rate":  artifact_trigger_rate,
        "diag_log":               grad_norm_log,
    }, indent=2, default=str))
    print(f"  saved → {grad_clip_json_path}", flush=True)
    print(f"  saved → {ckpt_path}  ({ckpt_path.stat().st_size/1e6:.1f} MB)", flush=True)
    print(f"  artifact trigger rate: {artifact_trigger_count}/{artifact_check_count} "
          f"= {artifact_trigger_rate:.2%}", flush=True)
    print(f"  main-step clip rate:   {main_clip_fires_count}/{main_clip_check_count} "
          f"= {main_clip_rate:.2%}", flush=True)
    if grad_norm_log:
        # Summary: mean per-component norm and ratio to L_BLSP.
        mean_norms = {}
        for k in ("L_task", "L_KL", "L_cf", "L_cond_pred", "L_NCE_cond", "L_BLSP"):
            vals = [r.get(k) for r in grad_norm_log if r.get(k) is not None and not math.isnan(r.get(k))]
            if vals:
                mean_norms[k] = float(sum(vals) / len(vals))
        print(f"  grad-norm summary (mean across diag steps):", flush=True)
        for k, v in mean_norms.items():
            print(f"    ‖∇{k}‖ = {v:.2f}", flush=True)
        if "L_BLSP" in mean_norms:
            for k, v in mean_norms.items():
                if k == "L_BLSP" or k in ("L_task", "L_KL"):
                    continue
                ratio = mean_norms["L_BLSP"] / max(v, 1e-9)
                print(f"    ‖∇L_BLSP‖ / ‖∇{k}‖ = {ratio:.2f}x", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
