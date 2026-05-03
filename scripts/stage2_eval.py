"""Stage 2.4 — per-seed evaluation of an A_BLSP checkpoint.

Computes:
  (A) Probe-G(A_BLSP)     — adapter, 20 paraphrases × 202 StressPresso items
  (B) Probe-K(A_BLSP)     — linear + MLP-2 on pool(A_BLSP(H))
  (C) Cascade-T baseline  — Whisper ASR transcript → Qwen3-8B Probe-G
  (D) K_T  baseline       — embed(true transcript) → Qwen3-8B Probe-G
  (E) Probe-G-oracle re-confirm — Stage 1a's exact prompt on Qwen3-8B

Run:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/stage2_eval.py \
      --seed 1234 --checkpoint outputs/stage2/A_BLSP_seed1234.pt
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

from src.utils.env import setup_env, banner  # noqa: E402

setup_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.models.adapter import AdapterConfig, BLSPAdapter  # noqa: E402
from src.data.stress_data import (  # noqa: E402
    load_stresspresso_test, load_stress17k, partition_transcript_ids,
)
from src.utils.prompts import AUDIO_PLACEHOLDER, DEFAULT_SYSTEM  # noqa: E402
from src.probes.probe_g import (  # noqa: E402
    SYSTEM_PROBE_G, NEUTRAL_PARAPHRASES, EXPLICIT_PARAPHRASES,
    bootstrap_by_cluster,
)
from src.probes.probe_k import (  # noqa: E402
    ProbeK, FitConfig, fit_probe, predict, masked_softmax_logits,
    within_transcript_argmax, accuracy,
)

QWEN3_MODEL = "Qwen/Qwen3-8B"
WAVLM_MODEL = "microsoft/wavlm-large"
WHISPER_MODEL = "openai/whisper-large-v3"
SR = 16000
N_MAX_CLASSES = 13   # Stage 1bc lock


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--skip_cascade", action="store_true",
                   help="skip Whisper-ASR cascade-T (saves ~5 min)")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--bootstrap_iter", type=int, default=1000)
    return p.parse_args()


def chat_text_with_audio_marker(tok, *, system: str, user_with_marker: str) -> tuple[str, str]:
    full = tok.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_with_marker},
        ],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    parts = full.split(AUDIO_PLACEHOLDER, 1)
    if len(parts) != 2:
        raise ValueError("audio placeholder lost")
    return parts[0], parts[1] + "Answer:"


def long_tensor(tok, text: str, device: str) -> torch.Tensor:
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return ids.to(dtype=torch.long, device=device)


@torch.no_grad()
def score_two_candidates(llm, embed_layer, *,
                         left_ids, audio_embed, right_ids,
                         vis_start_id, vis_end_id, cand_a_ids, cand_b_ids,
                         device) -> tuple[float, float]:
    """Return (mean log-prob ' A', mean log-prob ' B') for one item × one prompt.

    `audio_embed` is the contents of the audio slot (may be K_speech or text
    embeddings). The vision_start / vision_end bookends wrap it.
    """
    vs = embed_layer(torch.tensor([vis_start_id], device=device))
    ve = embed_layer(torch.tensor([vis_end_id],   device=device))
    base = torch.cat([
        embed_layer(left_ids),
        vs, audio_embed, ve,
        embed_layer(right_ids),
    ], dim=0)

    out = []
    for cand_ids in (cand_a_ids, cand_b_ids):
        full = torch.cat([base, embed_layer(cand_ids)], dim=0).unsqueeze(0)
        T_total = full.shape[1]
        attn = torch.ones(1, T_total, dtype=torch.long, device=device)
        logits = llm(inputs_embeds=full, attention_mask=attn).logits[0]
        T_c = cand_ids.shape[0]
        cand_start = T_total - T_c
        pred = logits[cand_start - 1: cand_start - 1 + T_c].float()
        log_probs = F.log_softmax(pred, dim=-1)
        out.append(float(log_probs[torch.arange(T_c, device=device), cand_ids].mean().item()))
    return out[0], out[1]


def run_probe_g_variant(
    *, name: str, llm, embed_layer, tok, items, device,
    audio_embeddings: list[torch.Tensor],   # one per item
    paraphrases_neutral, paraphrases_explicit,
    vis_start_id: int, vis_end_id: int,
) -> dict:
    """Generic Probe-G runner. `audio_embeddings[i]` is what goes in the audio
    slot for item i. For neutral/explicit-cue conditions we vary the question
    text but keep the slot contents identical.
    """
    rows = []
    cand_a_ids = long_tensor(tok, " A", device)
    cand_b_ids = long_tensor(tok, " B", device)
    for i, it in enumerate(items):
        audio_embed = audio_embeddings[i]
        for is_explicit, paraphrases in [
            (False, paraphrases_neutral),
            (True,  paraphrases_explicit),
        ]:
            for p_idx, paraphrase in enumerate(paraphrases):
                q = paraphrase.format(word=it.stressed_word) if is_explicit else paraphrase
                user = f"{AUDIO_PLACEHOLDER}\n{q}\nA) {it.options[0]}\nB) {it.options[1]}"
                left_text, right_text = chat_text_with_audio_marker(
                    tok, system=SYSTEM_PROBE_G, user_with_marker=user,
                )
                left_ids  = long_tensor(tok, left_text,  device)
                right_ids = long_tensor(tok, right_text, device)
                s_A, s_B = score_two_candidates(
                    llm, embed_layer,
                    left_ids=left_ids, audio_embed=audio_embed, right_ids=right_ids,
                    vis_start_id=vis_start_id, vis_end_id=vis_end_id,
                    cand_a_ids=cand_a_ids, cand_b_ids=cand_b_ids, device=device,
                )
                pred = 0 if s_A > s_B else 1
                margin = (s_A - s_B) if it.label == 0 else (s_B - s_A)
                rows.append({
                    "item": i, "transcription_id": it.transcription_id,
                    "is_explicit": is_explicit, "p_idx": p_idx,
                    "label": it.label, "pred": pred,
                    "correct": int(pred == it.label),
                    "score_A": s_A, "score_B": s_B, "signed_margin": margin,
                })
    accs    = np.array([r["correct"] for r in rows], dtype=np.float64)
    margs   = np.array([r["signed_margin"] for r in rows], dtype=np.float64)
    cl_all  = [r["transcription_id"] for r in rows]
    is_e    = np.array([r["is_explicit"] for r in rows])

    acc_mean, acc_lo, acc_hi             = bootstrap_by_cluster(accs, cl_all)
    marg_mean, marg_lo, marg_hi          = bootstrap_by_cluster(margs, cl_all)
    accN_mean, accN_lo, accN_hi          = bootstrap_by_cluster(accs[~is_e], [c for c, e in zip(cl_all, is_e) if not e])
    accE_mean, accE_lo, accE_hi          = bootstrap_by_cluster(accs[is_e],  [c for c, e in zip(cl_all, is_e) if e])
    margN_mean, margN_lo, margN_hi       = bootstrap_by_cluster(margs[~is_e], [c for c, e in zip(cl_all, is_e) if not e])
    margE_mean, margE_lo, margE_hi       = bootstrap_by_cluster(margs[is_e],  [c for c, e in zip(cl_all, is_e) if e])

    summary = {
        "name": name,
        "n_rows": len(rows),
        "accuracy": acc_mean, "accuracy_ci_lo": acc_lo, "accuracy_ci_hi": acc_hi,
        "signed_margin": marg_mean, "signed_margin_ci_lo": marg_lo, "signed_margin_ci_hi": marg_hi,
        "accuracy_neutral":  accN_mean,  "accuracy_neutral_ci_lo":  accN_lo,  "accuracy_neutral_ci_hi":  accN_hi,
        "accuracy_explicit": accE_mean,  "accuracy_explicit_ci_lo": accE_lo,  "accuracy_explicit_ci_hi": accE_hi,
        "signed_margin_neutral":  margN_mean,  "signed_margin_neutral_ci_lo":  margN_lo,  "signed_margin_neutral_ci_hi":  margN_hi,
        "signed_margin_explicit": margE_mean,  "signed_margin_explicit_ci_lo": margE_lo,  "signed_margin_explicit_ci_hi": margE_hi,
    }
    return {"summary": summary, "rows": rows}


@torch.no_grad()
def adapter_audio_embeddings(adapter, wavlm, feat, items, device) -> list[torch.Tensor]:
    """Run WavLM-L16 + adapter on each item's audio. Returns K_speech bf16 list."""
    out = []
    for it in items:
        wav = np.asarray(it.audio_array, dtype=np.float32)
        proc = feat([wav], sampling_rate=SR, return_tensors="pt",
                    padding=True, return_attention_mask=True)
        iv = proc["input_values"].to(device).to(torch.bfloat16)
        am = proc["attention_mask"].to(device)
        wavlm_out = wavlm(input_values=iv, attention_mask=am, output_hidden_states=True)
        H = wavlm_out.hidden_states[16].float()
        sample_lengths = am.sum(dim=1)
        valid_T_s = wavlm._get_feat_extract_output_lengths(sample_lengths).long()
        K, vTk = adapter(H, valid_T_s=valid_T_s)
        Tk = int(vTk[0].item())
        out.append(K[0, :Tk, :].to(torch.bfloat16))
    return out


@torch.no_grad()
def text_embeddings_from_string(tok, embed_layer, texts: list[str], device) -> list[torch.Tensor]:
    out = []
    for t in texts:
        ids = long_tensor(tok, t, device)
        out.append(embed_layer(ids))
    return out


@torch.no_grad()
def whisper_asr_transcripts(items, device) -> list[str]:
    """Whisper-large-v3 ASR on each StressPresso audio."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL, torch_dtype=torch.bfloat16).eval().to(device)
    out = []
    import soxr
    for it in items:
        wav = np.asarray(it.audio_array, dtype=np.float32)
        if it.audio_sr != SR:
            wav = soxr.resample(wav, it.audio_sr, SR).astype(np.float32)
        feats = proc(wav, sampling_rate=SR, return_tensors="pt").input_features.to(device).to(torch.bfloat16)
        gen_ids = model.generate(feats, max_new_tokens=128, num_beams=1, do_sample=False)
        text = proc.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        out.append(text)
    del model, proc
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def adapter_pool_for_probe_k(adapter, wavlm, feat, items, device) -> torch.Tensor:
    """Mean-pool A(H(audio)) per item over the valid audio-token range.
    Returns shape (N, 1, d_llm)."""
    pool_list = []
    for it in items:
        wav = np.asarray(it.audio_array, dtype=np.float32)
        proc = feat([wav], sampling_rate=SR, return_tensors="pt",
                    padding=True, return_attention_mask=True)
        iv = proc["input_values"].to(device).to(torch.bfloat16)
        am = proc["attention_mask"].to(device)
        wavlm_out = wavlm(input_values=iv, attention_mask=am, output_hidden_states=True)
        H = wavlm_out.hidden_states[16].float()
        sample_lengths = am.sum(dim=1)
        valid_T_s = wavlm._get_feat_extract_output_lengths(sample_lengths).long()
        K, vTk = adapter(H, valid_T_s=valid_T_s)
        Tk = int(vTk[0].item())
        pool_list.append(K[0, :Tk, :].mean(dim=0).cpu().float())
    return torch.stack(pool_list).unsqueeze(1)


def main() -> int:
    args = parse_args()
    print(f"transformers=={__import__('transformers').__version__}, torch=={torch.__version__}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES','?')}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("ERROR: GPU 6 required."); return 1

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / "stage2_eval"
    seed_dir = out_dir / f"seed{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load checkpoint ---- #
    banner(f"Loading adapter checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = AdapterConfig(**ckpt["adapter_config"])
    adapter = BLSPAdapter(cfg).to(device).to(torch.float32)
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    adapter.eval()
    print(f"  adapter loaded; trainable params = {adapter.n_trainable_params()/1e6:.2f}M", flush=True)

    # ---- Load Qwen3-8B + WavLM ---- #
    banner(f"Loading Qwen3-8B (bf16) and WavLM-Large (bf16)")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN3_MODEL)
    llm = AutoModelForCausalLM.from_pretrained(QWEN3_MODEL, torch_dtype=torch.bfloat16).eval().to(device)
    embed_layer = llm.get_input_embeddings()

    from transformers import WavLMModel, AutoFeatureExtractor
    wavlm_feat = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL)
    wavlm = WavLMModel.from_pretrained(WAVLM_MODEL, torch_dtype=torch.bfloat16).eval().to(device)

    vis_start_id = int(tok.convert_tokens_to_ids("<|vision_start|>"))
    vis_end_id   = int(tok.convert_tokens_to_ids("<|vision_end|>"))

    # ---- Load StressPresso ---- #
    banner("Loading StressPresso (n=202)")
    sp_items = load_stresspresso_test()
    print(f"  loaded {len(sp_items)} items", flush=True)

    # ---- Adapter Probe-G ---- #
    banner("(A) Probe-G(A_BLSP) — adapter, 20 paraphrases × 202 items")
    t0 = time.time()
    K_speech_list = adapter_audio_embeddings(adapter, wavlm, wavlm_feat, sp_items, device)
    print(f"  computed K_speech for 202 items in {time.time()-t0:.1f}s", flush=True)
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
    print(f"    signed margin total={sN['signed_margin']:+.4f}  neutral={sN['signed_margin_neutral']:+.4f}  explicit={sN['signed_margin_explicit']:+.4f}", flush=True)

    # ---- K_T baseline ---- #
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
    sK = res_kt["summary"]
    print(f"    accuracy total={sK['accuracy']:.4f}  neutral={sK['accuracy_neutral']:.4f}  explicit={sK['accuracy_explicit']:.4f}", flush=True)
    print(f"    signed margin total={sK['signed_margin']:+.4f}  neutral={sK['signed_margin_neutral']:+.4f}  explicit={sK['signed_margin_explicit']:+.4f}", flush=True)

    # ---- Cascade-T (Whisper ASR transcript) ---- #
    res_cascade = None
    if not args.skip_cascade:
        banner("(C) Cascade-T baseline — Whisper-ASR transcript")
        t0 = time.time()
        # Free WavLM and Qwen3 GPU mem to fit Whisper too — actually all three should fit
        # in the 53 GB free; but Whisper-large-v3 is 1.5B in bf16 ~3 GB.
        predicted_transcripts = whisper_asr_transcripts(sp_items, device)
        print(f"  ASR done for 202 items in {time.time()-t0:.1f}s", flush=True)
        K_pred_list = text_embeddings_from_string(tok, embed_layer, predicted_transcripts, device)
        t0 = time.time()
        res_cascade = run_probe_g_variant(
            name="cascade_T", llm=llm, embed_layer=embed_layer, tok=tok,
            items=sp_items, device=device, audio_embeddings=K_pred_list,
            paraphrases_neutral=NEUTRAL_PARAPHRASES,
            paraphrases_explicit=EXPLICIT_PARAPHRASES,
            vis_start_id=vis_start_id, vis_end_id=vis_end_id,
        )
        print(f"  Probe-G done in {time.time()-t0:.1f}s", flush=True)
        sC = res_cascade["summary"]
        print(f"    accuracy total={sC['accuracy']:.4f}  neutral={sC['accuracy_neutral']:.4f}  explicit={sC['accuracy_explicit']:.4f}", flush=True)
        print(f"    signed margin total={sC['signed_margin']:+.4f}  neutral={sC['signed_margin_neutral']:+.4f}  explicit={sC['signed_margin_explicit']:+.4f}", flush=True)

    # ---- Probe-G-oracle re-confirm (Stage 1a's exact prompt) ---- #
    banner("(E) Probe-G-oracle re-confirm (Stage 1a's [[X]] markup prompt)")
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
    t0 = time.time()
    rows_oracle = []
    cand_a_ids = long_tensor(tok, " A", device)
    cand_b_ids = long_tensor(tok, " B", device)
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
        # No audio slot for oracle — compute scores via standard inputs_embeds path.
        ids = long_tensor(tok, left, device).unsqueeze(0)
        # Forward both candidates
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
    print(f"  oracle accuracy = {acc_mean:.4f}  ({int(accs.sum())} / {len(accs)})  CI ({acc_lo:.4f}, {acc_hi:.4f})", flush=True)
    print(f"  oracle margin   = {marg_mean:+.4f}  CI ({marg_lo:+.4f}, {marg_hi:+.4f})", flush=True)
    print(f"  expected: 0.787 ± 2pp from Stage 1a; observed-Stage1a delta = {abs(acc_mean - 0.7871)*100:.2f}pp", flush=True)
    res_oracle = {
        "summary": {
            "accuracy": acc_mean, "accuracy_ci_lo": acc_lo, "accuracy_ci_hi": acc_hi,
            "signed_margin": marg_mean, "signed_margin_ci_lo": marg_lo, "signed_margin_ci_hi": marg_hi,
            "n_correct": int(accs.sum()), "n": len(accs),
        },
        "rows": rows_oracle,
    }
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    # ---- (B) Probe-K(A_BLSP) ---- #
    banner("(B) Probe-K(A_BLSP) — linear + MLP-2 probe on pool(A_BLSP(H))")
    t0 = time.time()
    s17_all = load_stress17k()
    train_ids, eval_ids = partition_transcript_ids(s17_all, eval_frac=0.20, seed="BTA-2026-05-02")
    s17_train = [it for it in s17_all if it.transcription_id in train_ids]
    s17_eval  = [it for it in s17_all if it.transcription_id in eval_ids]
    print(f"  Stress-17K train rows: {len(s17_train)}  eval rows: {len(s17_eval)}", flush=True)

    # Compute pool(A(H)) for train, eval, and StressPresso
    print(f"  computing pool(A(H)) on Stress-17K train...", flush=True)
    pool_train = adapter_pool_for_probe_k(adapter, wavlm, wavlm_feat, s17_train, device)
    print(f"  computing pool(A(H)) on Stress-17K eval...", flush=True)
    pool_eval  = adapter_pool_for_probe_k(adapter, wavlm, wavlm_feat, s17_eval, device)
    print(f"  computing pool(A(H)) on StressPresso...", flush=True)
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

    fit_cfg = FitConfig(epochs=80, lr=1e-3, batch_size=256, weight_decay=1e-4, seed=args.seed)
    probe_results = {}
    for head in ("linear", "mlp2"):
        probe = ProbeK(d_in=pool_train.shape[-1], n_classes=N_MAX_CLASSES,
                       cell_mode="single", n_layers_used=1, head=head)
        info = fit_probe(probe, pool_train, y_train, n_words_train,
                         pool_eval,  y_eval,  n_words_eval,
                         fit_cfg, device)
        # Eval predictions
        from src.probes.probe_k import predict, accuracy as acc_fn, within_transcript_argmax
        # Helper: build candidate lookup for within-transcript axis on eval and sp.
        from collections import defaultdict
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
    }
    (seed_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    rows_payload = {
        "adapter":   res_adapter["rows"],
        "K_T":       res_kt["rows"],
        "cascade_T": (res_cascade["rows"] if res_cascade is not None else None),
        "oracle_reconfirm": res_oracle["rows"],
    }
    (seed_dir / "rows.json").write_text(json.dumps(rows_payload, indent=2))
    print(f"  saved → {seed_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
