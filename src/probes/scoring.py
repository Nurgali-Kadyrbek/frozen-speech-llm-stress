"""Constrained-answer scoring via inputs_embeds.

Per PRE_IMPLEMENTATION_DESIGN.md §3.5: build the full sequence
[E_left, K, E_right, E_delim, E_answer_i] via inputs_embeds, run a single
forward pass per candidate, take the mean log-prob over the answer-token
span. Mean (not sum) so length doesn't bias the comparison.

Used by Stage 1a (oracle, K = embed(transcript)) and Stage 2+ (K = adapter
output). Stage 1a does NOT pass any audio prefix — left/right halves form a
complete ChatML chat plus the delimiter, so K is empty here. The function
accepts an optional audio_embed for forward compatibility.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def tokenize_no_specials(tokenizer, text: str, device: str) -> torch.Tensor:
    """Tokenize without injecting BOS/EOS — required when stitching chat halves.

    HF tokenizers return float32 for empty strings (`shape=(1, 0)`); force long
    so downstream `embed_tokens(...)` doesn't reject the index tensor.
    """
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return ids.to(dtype=torch.long, device=device)


@torch.no_grad()
def mean_logprob_candidates(
    model,
    embed_layer,
    *,
    left_ids: torch.Tensor,
    right_ids: torch.Tensor,
    delim_ids: torch.Tensor,
    candidates: Sequence[torch.Tensor],
    audio_embed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a 1D tensor of mean log-prob per candidate.

    Sequence layout:
        [E(left_ids), audio_embed?, E(right_ids), E(delim_ids), E(candidate_i)]

    The candidate occupies positions [start, start+T_c). Logits at position p−1
    predict the token at position p; the slice logits[start−1 : start−1+T_c]
    is the per-token next-token distribution over the candidate span.
    """
    pieces = [embed_layer(left_ids)]
    if audio_embed is not None:
        pieces.append(audio_embed)
    pieces.extend([embed_layer(right_ids), embed_layer(delim_ids)])
    base_emb = torch.cat(pieces, dim=0)              # (T_base, d)
    T_base = base_emb.shape[0]

    scores = []
    for cand_ids in candidates:
        cand_emb = embed_layer(cand_ids)
        full_emb = torch.cat([base_emb, cand_emb], dim=0).unsqueeze(0)
        T_total = full_emb.shape[1]
        attn = torch.ones(1, T_total, dtype=torch.long, device=full_emb.device)
        logits = model(inputs_embeds=full_emb, attention_mask=attn).logits[0]

        T_c = cand_ids.shape[0]
        cand_start = T_total - T_c
        pred_logits = logits[cand_start - 1 : cand_start - 1 + T_c].float()
        log_probs = F.log_softmax(pred_logits, dim=-1)
        scores.append(log_probs[torch.arange(T_c, device=log_probs.device), cand_ids].mean())

    return torch.stack(scores)
