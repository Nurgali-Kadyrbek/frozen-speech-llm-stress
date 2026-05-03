"""Stage 0.5 — `inputs_embeds` sanity-check suite (PRE_IMPLEMENTATION_DESIGN §9 Stage 0.5).

Four BLOCKING checks. All must pass before any adapter training:
  (1) Round-trip identity:        input_ids vs inputs_embeds → identical logits.
  (2) Direction-sensitivity:      perturbing K_T at scale ||row|| changes the score.
  (3) Strong-text-injection:      embed("the answer is A") makes the scorer prefer "A".
  (4) ChatML bookend / whitespace: <|vision_start|>=151652, <|vision_end|>=151653; round-trip.

Run on GPU 6:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/stage0p5_inputs_embeds_sanity.py
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_env, banner, report_check  # noqa: E402

setup_env()

import torch  # noqa: E402

QWEN3_MODEL = "Qwen/Qwen3-1.7B"

# Tolerances (design § Stage 0.5)
ROUND_TRIP_MAX_ABS = 1e-3       # fp32 should be ~0; allow tiny FP-reordering noise
DIR_SENSITIVITY_MIN_NATS = 0.05 # perturbing K_T by ||row|| must shift mean log-prob ≥ this
STRONG_INJECT_MIN_MARGIN = 1.0  # log P(" A") - log P(" B") nats; needs to be a clear preference
EXPECTED_BOOKEND_IDS = (151652, 151653)


def build_scoring_pieces(tok, device):
    """Return tokenized pieces of: left | [audio_slot] | right.

    The 'audio slot' is where K_T (the prefix the adapter would emit) goes.
    Layout is a closed ChatML user turn followed by an opened assistant turn,
    matching PRE_IMPLEMENTATION_DESIGN §3.3:
        left  = <|im_start|>user\\n
        K_T   = continuous tensor (text-embedded for these checks)
        right = <|im_end|>\\n<|im_start|>assistant\\nThe answer is
    Candidates ' A' / ' B' are scored as the next-token continuation.
    """
    left_text  = "<|im_start|>user\n"
    right_text = "<|im_end|>\n<|im_start|>assistant\nThe answer is"
    cand_texts = [" A", " B"]

    def to_ids(s):
        return tok(s, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)

    return {
        "left":   to_ids(left_text),
        "right":  to_ids(right_text),
        "cands":  [to_ids(c) for c in cand_texts],
        "cand_texts": cand_texts,
    }


def score_candidate(model, embed_layer, audio_embed, pieces, cand_idx):
    """Mean log-prob of candidate tokens given full inputs_embeds sequence.

    Sequence: [left, audio_embed, right, candidate_tokens]. Score = mean log_softmax
    at candidate positions (per design §3.5: 'Use mean log-prob per token, not sum').
    audio_embed: (T_audio, d_llm) — a continuous tensor (the would-be adapter output).
    """
    left_ids  = pieces["left"]
    right_ids = pieces["right"]
    cand_ids  = pieces["cands"][cand_idx]

    # Embed text pieces
    left_emb  = embed_layer(left_ids)        # (T_left,  d)
    right_emb = embed_layer(right_ids)       # (T_right, d)
    cand_emb  = embed_layer(cand_ids)        # (T_c,     d)

    full_emb = torch.cat([left_emb, audio_embed, right_emb, cand_emb], dim=0).unsqueeze(0)
    T_total = full_emb.shape[1]

    with torch.no_grad():
        out = model(inputs_embeds=full_emb,
                    attention_mask=torch.ones(1, T_total, dtype=torch.long, device=full_emb.device))

    # logits at position t predict token t+1.
    # Candidate occupies positions [T_total - T_c, T_total).
    # So to predict token at position p, read logits[p-1].
    T_c = cand_ids.shape[0]
    cand_start = T_total - T_c
    pred_logits = out.logits[0, cand_start - 1 : cand_start - 1 + T_c]  # (T_c, vocab)
    log_probs = torch.log_softmax(pred_logits.float(), dim=-1)
    return log_probs[torch.arange(T_c, device=log_probs.device), cand_ids].mean().item()


def check_1_round_trip(model, embed_layer, tok, device, fails):
    banner("Stage 0.5 / Check 1 — Round-trip identity (input_ids vs inputs_embeds)")
    msgs = [{"role": "user", "content": "What is 2 + 2?"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    attn = torch.ones_like(ids)

    embeds = embed_layer(ids)
    with torch.no_grad():
        out_ids = model(input_ids=ids, attention_mask=attn).logits
        out_emb = model(inputs_embeds=embeds, attention_mask=attn).logits

    max_abs = (out_ids - out_emb).abs().max().item()
    rms = (out_ids - out_emb).pow(2).mean().sqrt().item()
    print(f"  T_total={ids.shape[1]}, max|Δlogits|={max_abs:.3e}, rms={rms:.3e}", flush=True)
    report_check(f"max abs logit diff < {ROUND_TRIP_MAX_ABS:.0e}",
                 max_abs < ROUND_TRIP_MAX_ABS,
                 f"got {max_abs:.3e}", fails)


def check_2_direction_sensitivity(model, embed_layer, tok, device, fails):
    banner("Stage 0.5 / Check 2 — Direction-sensitivity")
    pieces = build_scoring_pieces(tok, device)

    # K_T = embedding of a transcript-like sentence (used as adapter-slot replacement).
    K_text = "neutral context with little information about the answer"
    k_ids = tok(K_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    K_T = embed_layer(k_ids)  # (T_audio, d_llm)

    # Margin = log P(" A") - log P(" B")
    s_A_clean = score_candidate(model, embed_layer, K_T, pieces, 0)
    s_B_clean = score_candidate(model, embed_layer, K_T, pieces, 1)
    margin_clean = s_A_clean - s_B_clean

    # Perturb at the scale of the embedding manifold:
    # token rows of embed_tokens have ||row||_2 ≈ std × sqrt(d_llm). A random
    # tensor with the same per-element std produces vectors with the same per-row ||·||.
    embed_std = embed_layer.weight.std().item()
    row_norm  = embed_layer.weight.norm(dim=-1).mean().item()
    torch.manual_seed(0)
    delta = torch.randn_like(K_T) * embed_std
    delta_norm = delta.norm(dim=-1).mean().item()
    print(f"  ||row||_2 mean = {row_norm:.3f},  ||delta||_2 mean = {delta_norm:.3f}", flush=True)

    K_perturbed = K_T + delta
    s_A_pert = score_candidate(model, embed_layer, K_perturbed, pieces, 0)
    s_B_pert = score_candidate(model, embed_layer, K_perturbed, pieces, 1)
    margin_pert = s_A_pert - s_B_pert

    shift_A = abs(s_A_pert - s_A_clean)
    shift_B = abs(s_B_pert - s_B_clean)
    margin_shift = abs(margin_pert - margin_clean)
    max_score_shift = max(shift_A, shift_B)

    print(f"  clean:     score(A)={s_A_clean:+.4f}  score(B)={s_B_clean:+.4f}  margin={margin_clean:+.4f}", flush=True)
    print(f"  perturbed: score(A)={s_A_pert:+.4f}  score(B)={s_B_pert:+.4f}  margin={margin_pert:+.4f}", flush=True)
    print(f"  shift_A={shift_A:.4f}  shift_B={shift_B:.4f}  Δmargin={margin_shift:.4f}", flush=True)

    report_check(
        f"perturbation at ||row|| scale moves score by ≥ {DIR_SENSITIVITY_MIN_NATS:.2f} nats",
        max_score_shift >= DIR_SENSITIVITY_MIN_NATS,
        f"max(shift_A,shift_B)={max_score_shift:.4f}",
        fails,
    )


def check_3_strong_text_injection(model, embed_layer, tok, device, fails):
    banner("Stage 0.5 / Check 3 — Strong-text-injection ('the answer is A' → prefer A)")
    pieces = build_scoring_pieces(tok, device)

    inject_text = "the answer is A"
    inj_ids = tok(inject_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    K_T = embed_layer(inj_ids)

    s_A = score_candidate(model, embed_layer, K_T, pieces, 0)
    s_B = score_candidate(model, embed_layer, K_T, pieces, 1)
    margin = s_A - s_B
    print(f"  inject={inject_text!r}  score(' A')={s_A:+.4f}  score(' B')={s_B:+.4f}  margin={margin:+.4f}", flush=True)

    # Cross-check against the OPPOSITE injection: should flip the preference.
    inj2_ids = tok("the answer is B", return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    K_T2 = embed_layer(inj2_ids)
    s_A2 = score_candidate(model, embed_layer, K_T2, pieces, 0)
    s_B2 = score_candidate(model, embed_layer, K_T2, pieces, 1)
    margin2 = s_A2 - s_B2
    print(f"  inject='the answer is B'  score(' A')={s_A2:+.4f}  score(' B')={s_B2:+.4f}  margin={margin2:+.4f}", flush=True)

    report_check(
        f"score(' A') - score(' B') ≥ {STRONG_INJECT_MIN_MARGIN} nats when injecting 'the answer is A'",
        margin >= STRONG_INJECT_MIN_MARGIN,
        f"got margin={margin:.4f}",
        fails,
    )
    report_check(
        "Opposite injection flips preference (margin('B'-cue) < margin('A'-cue))",
        margin2 < margin,
        f"margin_A_inj={margin:.4f}, margin_B_inj={margin2:.4f}",
        fails,
    )


def check_4_bookend(tok, fails):
    banner("Stage 0.5 / Check 4 — <|vision_start|> / <|vision_end|> bookend + whitespace")
    bookend_strs = ["<|vision_start|>", "<|vision_end|>"]
    ids = tok.convert_tokens_to_ids(bookend_strs)
    print(f"  convert_tokens_to_ids({bookend_strs}) = {ids}", flush=True)
    report_check(
        f"bookend IDs are {EXPECTED_BOOKEND_IDS}",
        tuple(ids) == EXPECTED_BOOKEND_IDS,
        f"got {tuple(ids)}",
        fails,
    )

    # Whitespace round-trip on a chat template that uses the bookends.
    template = (
        "<|im_start|>user\n"
        "Decide A or B. <|vision_start|>here is the audio<|vision_end|> "
        "Make your choice.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    enc = tok(template, add_special_tokens=False).input_ids
    decoded = tok.decode(enc, skip_special_tokens=False)
    print(f"  template (repr): {template!r}", flush=True)
    print(f"  decoded  (repr): {decoded!r}", flush=True)

    report_check(
        "tok-decode round-trip preserves the templated string verbatim",
        decoded == template,
        f"len(template)={len(template)}, len(decoded)={len(decoded)}",
        fails,
    )

    # Each bookend should be exactly one token under add_special_tokens=False.
    for s, expect_id in zip(bookend_strs, EXPECTED_BOOKEND_IDS):
        toks = tok(s, add_special_tokens=False).input_ids
        ok = (len(toks) == 1) and (toks[0] == expect_id)
        report_check(
            f"{s} tokenizes to a single id == {expect_id}",
            ok,
            f"got {toks}",
            fails,
        )


def main() -> int:
    fails: list = []
    print(f"transformers version: {__import__('transformers').__version__}", flush=True)
    print(f"torch version:        {torch.__version__}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES: {__import__('os').environ.get('CUDA_VISIBLE_DEVICES','?')}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        report_check("CUDA available", False, "running on CPU; design requires GPU 6", fails)
        return 1

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(QWEN3_MODEL)
    model = (
        AutoModelForCausalLM.from_pretrained(QWEN3_MODEL, torch_dtype=torch.float32)
        .eval()
        .to(device)
    )
    embed = model.get_input_embeddings()
    print(f"Loaded {QWEN3_MODEL} in {time.time()-t0:.1f}s "
          f"(d_llm={embed.weight.shape[1]}, vocab={embed.weight.shape[0]})", flush=True)

    check_1_round_trip(model, embed, tok, device, fails)
    check_2_direction_sensitivity(model, embed, tok, device, fails)
    check_3_strong_text_injection(model, embed, tok, device, fails)
    check_4_bookend(tok, fails)

    banner("Stage 0.5 summary")
    if not fails:
        print("  ALL Stage 0.5 sanity checks PASS.", flush=True)
        return 0
    print(f"  {len(fails)} Stage 0.5 check(s) FAILED:", flush=True)
    for f in fails:
        print(f"    - {f}", flush=True)
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
