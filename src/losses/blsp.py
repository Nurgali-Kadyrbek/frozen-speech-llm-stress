"""L_BLSP loss — task cross-entropy + KL distillation between speech and text
branches (PRE_IMPLEMENTATION_DESIGN §6.1).

Layout per training sample:

    speech_prefix = [E_left, E_vis_start, K_speech, E_vis_end, E_right]
    text_prefix   = [E_left, E_vis_start, K_text,   E_vis_end, E_right]
    speech_full   = speech_prefix ⧺ E_response
    text_full     = text_prefix   ⧺ E_response

Both branches forward through frozen Qwen3-8B with teacher forcing on the
gold response. Speech-branch backprop flows ONLY into the adapter; the LLM
is frozen. Text-branch is computed under no_grad.

L_task = mean cross-entropy of `response_ids` under speech-branch logits
         (positions of `response_ids` in `speech_full`).
L_KL   = mean over response positions of KL(p_text || p_speech).

L_R0   = L_task + λ_KL · L_KL.

Both teacher-force (no free generation) so position alignment is trivial.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass
class BLSPInput:
    """Pre-tokenized halves for one training sample, ready for embed lookups."""
    left_ids:        torch.Tensor   # (T_left,)
    right_ids:       torch.Tensor   # (T_right,)
    response_ids:    torch.Tensor   # (T_resp,)
    transcript_ids:  torch.Tensor   # (T_text,) — for K_text in the text branch
    vision_start_id: int
    vision_end_id:   int


@torch.no_grad()
def _build_text_prefix(embed_layer, b: BLSPInput) -> torch.Tensor:
    vs = embed_layer(torch.tensor([b.vision_start_id], device=b.left_ids.device))
    ve = embed_layer(torch.tensor([b.vision_end_id],   device=b.left_ids.device))
    K_text = embed_layer(b.transcript_ids)               # (T_text, d_llm)
    pieces = [embed_layer(b.left_ids), vs, K_text, ve, embed_layer(b.right_ids)]
    return torch.cat(pieces, dim=0)


def _build_speech_prefix(embed_layer, K_speech: torch.Tensor, b: BLSPInput) -> torch.Tensor:
    """`K_speech` carries grad through the adapter; everything else is frozen."""
    with torch.no_grad():
        vs = embed_layer(torch.tensor([b.vision_start_id], device=K_speech.device))
        ve = embed_layer(torch.tensor([b.vision_end_id],   device=K_speech.device))
        E_left  = embed_layer(b.left_ids)
        E_right = embed_layer(b.right_ids)
    return torch.cat([E_left, vs, K_speech, ve, E_right], dim=0)


def _slice_response_logits(logits_full_seq: torch.Tensor, T_response: int) -> torch.Tensor:
    """Pick the logits that PREDICT response tokens (offset by 1).

    For a sequence of length T_total ending in T_response response tokens, the
    logits at positions [T_total - T_response - 1 : T_total - 1] predict the
    tokens at positions [T_total - T_response : T_total].
    """
    T_total = logits_full_seq.shape[0]
    start = T_total - T_response - 1
    end   = T_total - 1
    return logits_full_seq[start:end]   # (T_response, vocab)


def compute_blsp_loss(
    *,
    llm,                                # frozen Qwen3
    embed_layer,                        # llm.get_input_embeddings()
    K_speech: torch.Tensor,             # (T_audio_k, d_llm) — adapter output (with grad)
    inp: BLSPInput,
    lambda_kl: float = 1.0,
) -> dict:
    """Returns {'L_task': tensor, 'L_KL': tensor, 'L_R0': tensor}.

    Designed for microbatch=1 to keep position alignment trivial. Caller
    handles gradient accumulation across multiple BLSPInput instances.
    """
    device = K_speech.device
    response_ids = inp.response_ids
    T_resp = response_ids.shape[0]

    # ---- speech branch (with grad through adapter) ----
    speech_prefix = _build_speech_prefix(embed_layer, K_speech, inp)
    with torch.no_grad():
        E_response = embed_layer(response_ids)
    speech_full = torch.cat([speech_prefix, E_response], dim=0).unsqueeze(0)
    attn_speech = torch.ones(1, speech_full.shape[1], dtype=torch.long, device=device)
    out_speech = llm(inputs_embeds=speech_full, attention_mask=attn_speech)
    logits_speech_resp = _slice_response_logits(out_speech.logits[0], T_resp)
    log_p_speech = F.log_softmax(logits_speech_resp.float(), dim=-1)

    # ---- text branch (no grad — teacher) ----
    with torch.no_grad():
        text_prefix = _build_text_prefix(embed_layer, inp)
        text_full = torch.cat([text_prefix, embed_layer(response_ids)], dim=0).unsqueeze(0)
        attn_text = torch.ones(1, text_full.shape[1], dtype=torch.long, device=device)
        out_text = llm(inputs_embeds=text_full, attention_mask=attn_text)
        logits_text_resp = _slice_response_logits(out_text.logits[0], T_resp)
        log_p_text = F.log_softmax(logits_text_resp.float(), dim=-1)
        p_text = log_p_text.exp()

    # ---- losses ----
    # L_task: NLL of correct response tokens under speech branch.
    L_task = -log_p_speech[torch.arange(T_resp, device=device), response_ids].mean()

    # L_KL: KL(p_text || p_speech) — text is teacher.
    # KL = sum_v p_text * (log p_text - log p_speech). Constant w.r.t. adapter params.
    L_KL = (p_text * (log_p_text - log_p_speech)).sum(dim=-1).mean()

    L_R0 = L_task + lambda_kl * L_KL
    return {"L_task": L_task, "L_KL": L_KL, "L_R0": L_R0}


def free_branch_outputs(*tensors) -> None:
    """Best-effort early release of intermediate logits to keep peak memory low."""
    for t in tensors:
        if t is None:
            continue
        try:
            del t
        except Exception:
            pass
