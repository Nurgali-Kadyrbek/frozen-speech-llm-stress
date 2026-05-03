"""L_R1 structural counterfactual loss components (Stage 3.2).

Architecture (kickoff §3.2):

  L_R1 = L_BLSP                          (carried over from blsp.py)
       + λ_cf       · L_cf               (1.0)
       + λ_artifact · L_artifact         (1.0)
       + λ_cond     · L_cond_pred        (0.5)
       + λ_NCE      · L_NCE_cond         (0.5)

Design choices from kickoff §3.2:

  L_cf : single-token-margin form. The training prompt is the 2-AFC probe
         layout (`A) {y_phi_a}\\nB) {y_phi_a'}\\nAnswer:`); response is the
         single correct answer letter. L_cf reads the answer-position logit
         from the SAME forward pass already computed for L_BLSP — zero extra
         Qwen forwards. Restricted softmax over {tok_id_A, tok_id_B}.

  L_artifact : asymmetric KL across paired Expresso members. The first
               microbatch (a) caches detached log_p_a; the second (a'')
               retrieves the cache and computes max(0, KL(p_a || p_a'') −
               δ_artifact). Gradient flows only through K_a''; per-step
               role assignment can be flipped to keep the contribution
               symmetric across an epoch.

  L_cond_pred : two MLP heads jointly trained on each cf-pair sample:
                C_phi_full receives concat(pool(A(H(a))), restricted_T(a)),
                C_phi_T_only receives restricted_T(a) only. T-only acc
                gates the contamination claim — if it climbs above 0.30
                (chance ≈ 0.077), the lexical-only path is solving Φ.

                restricted_T(a) = 8192-dim hashed unigram counts (L2
                normalized) concatenated with the 2-d scalar
                [n_tokens, n_words] vector. Total restricted_T dim = 8194.

  L_NCE_cond : within-pair InfoNCE with 1 positive + 1 negative.
               Anchor = pool(A(H(a))).
               Positive  = embed_layer(tok("emphasis position: {Φ(a)}")).mean(0)
               Negative  = embed_layer(tok("emphasis position: {Φ(a')}")).mean(0)
               Temperature τ = 0.07. Cosine similarity over d_llm.
               No actual word substitution — we use the abstract integer
               position, so the contrastive loss cannot leak via token
               overlap with K_a's transcript portion (kickoff requirement).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Shared helpers ---------- #

WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _tokens_of(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text or "")]


def restricted_t_features(transcript: str, n_words: int, n_buckets: int = 8192) -> np.ndarray:
    """Build the 8194-dim restricted_T(a) vector.

    8192-dim hashed unigram counts (L2-normalized) ‖ [n_tokens, n_words].
    Hash uses sha256 to be stable across Python versions and processes.
    """
    toks = _tokens_of(transcript)
    feats = np.zeros(n_buckets + 2, dtype=np.float32)
    for t in toks:
        h = hashlib.sha256(t.encode()).digest()
        idx = int.from_bytes(h[:4], "big") % n_buckets
        feats[idx] += 1.0
    nrm = np.linalg.norm(feats[:n_buckets])
    if nrm > 0:
        feats[:n_buckets] /= nrm
    feats[n_buckets]     = float(len(toks))    # n_tokens
    feats[n_buckets + 1] = float(n_words)      # n_words (canonical word count)
    return feats


# ---------- L_cf: single-token margin from L_BLSP forward ---------- #

def compute_l_cf_from_logits(
    *,
    response_position_logits: torch.Tensor,    # (vocab,) — logits at the position predicting the answer letter
    correct_letter_id: int,
    wrong_letter_id: int,
) -> torch.Tensor:
    """L_cf = -log( exp(s_correct) / (exp(s_A) + exp(s_B)) ) over 2-AFC.

    We deliberately use the raw logits at the answer position and mask all
    tokens except the two AFC letters. Equivalent to a 2-class CE over
    {tok_A, tok_B}.
    """
    # log_softmax with mask: take only the two letter logits and re-normalize.
    s_correct = response_position_logits[correct_letter_id]
    s_wrong   = response_position_logits[wrong_letter_id]
    # Stable: log-sum-exp of the two then subtract correct.
    lse = torch.logsumexp(torch.stack([s_correct, s_wrong]), dim=0)
    return (lse - s_correct).float()


# ---------- L_artifact: asymmetric KL on response axis ---------- #

def compute_l_artifact(
    *,
    detached_log_p_a:     torch.Tensor,    # (T_resp, vocab) — from cached fwd of K_a, no_grad
    log_p_a_prime:        torch.Tensor,    # (T_resp, vocab) — current fwd of K_a'', WITH grad
    delta_artifact:       float = 0.10,
) -> torch.Tensor:
    """ℓ_artifact = max(0, KL(p_a || p_a'') − δ).

    KL is averaged over the response axis. Gradient flows only through
    log_p_a_prime; detached_log_p_a is treated as a constant.
    """
    # If response lengths mismatch (different tokenizations across pair members),
    # we truncate to the shorter prefix. With the artifact pair-mate ASR task,
    # response is the transcript so this should match exactly.
    T = min(detached_log_p_a.shape[0], log_p_a_prime.shape[0])
    p_a       = detached_log_p_a[:T].exp()
    log_p_a   = detached_log_p_a[:T]
    log_p_app = log_p_a_prime[:T]
    # KL(p_a || p_a'') = sum_v p_a (log p_a - log p_a''); mean over response axis.
    kl = (p_a * (log_p_a - log_p_app)).sum(dim=-1).mean()
    return F.relu(kl - delta_artifact).float()


# ---------- L_cond_pred: 2-layer MLP heads on (pool(A(H)), restricted_T) ---------- #

class CPhi(nn.Module):
    """Two-layer MLP joint head: concat(pool(A(H)), restricted_T) → Φ."""

    def __init__(self, *, d_in: int, hidden: int = 256, n_classes: int = 13, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


def compute_l_cond_pred(
    *,
    c_phi_full:   CPhi,
    c_phi_t_only: CPhi,
    pooled_A_H:   torch.Tensor,    # (d_llm,) — mean pool over time of A(H(a))
    restricted_T: torch.Tensor,    # (d_T,)  — hashed-unigram features
    phi_label:    int,             # stress index (0..n_max-1)
    n_max_classes: int = 13,
) -> dict:
    """Compute C_phi_full and C_phi_T_only logits + cross-entropy losses.

    Both heads are trained with the same target. The T-only head is a
    diagnostic (kickoff §3.2 requirement): if its accuracy on probe-eval
    rises above ~0.30, the conditional task is contaminated by lexical
    correlations and L_cond_pred should be reduced or dropped.
    """
    label = torch.tensor([phi_label], dtype=torch.long, device=pooled_A_H.device)
    label = torch.clamp(label, 0, n_max_classes - 1)

    # Full head: concat(pool(A(H)), restricted_T)
    full_in = torch.cat([pooled_A_H.unsqueeze(0), restricted_T.unsqueeze(0)], dim=1)
    logits_full = c_phi_full(full_in)
    L_full = F.cross_entropy(logits_full, label)

    # T-only head: restricted_T alone
    t_only_in = restricted_T.unsqueeze(0)
    logits_t = c_phi_t_only(t_only_in)
    L_t_only = F.cross_entropy(logits_t, label)

    with torch.no_grad():
        pred_full = logits_full.argmax(dim=-1).item()
        pred_t    = logits_t.argmax(dim=-1).item()
    return {
        "L_cond_pred":       L_full,    # main loss (gradients to A and C_phi_full)
        "L_t_only_aux":      L_t_only,  # diagnostic-only (gradients to C_phi_t_only)
        "pred_full":         int(pred_full),
        "pred_t_only":       int(pred_t),
        "label":             int(phi_label),
    }


# ---------- L_NCE_cond: within-pair InfoNCE on abstract Φ-position descriptions ---------- #

def make_phi_description_embeds(
    *,
    embed_layer,                 # llm.get_input_embeddings(), frozen
    tokenizer,                   # Qwen3 tokenizer
    phi_indices:    list[int],
    device:         torch.device,
    dtype:          torch.dtype = torch.float32,
    template:       str = "emphasis position: {idx}",
) -> torch.Tensor:
    """Return (N, d_llm) — mean pool over token embeds of `template.format(idx=phi)`
    for each phi in `phi_indices`.

    No grad through the LLM embeddings (frozen). Default template is the
    abstract-position form (Stage-3 original). Stage 3.6 deprecated this in
    favor of word-substitution descriptions (see make_phi_word_embed)
    because abstract-template embeddings have ≥0.99 cos sim across phis,
    structurally killing the L_NCE-cond margin (kickoff B1).
    """
    out_rows = []
    with torch.no_grad():
        for phi in phi_indices:
            text = template.format(idx=int(phi))
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
            ids = ids.to(dtype=torch.long, device=device)
            emb = embed_layer(ids).float().mean(dim=0)
            out_rows.append(emb)
    return torch.stack(out_rows, dim=0).to(dtype=dtype)


def make_phi_word_embed(
    *,
    embed_layer, tokenizer, word: str, device: torch.device,
    dtype: torch.dtype = torch.float32,
    template: str = "stress on {word}",
) -> torch.Tensor:
    """Return mean-pool embedding for a single word-substitution description.

    Stage 3.6 / B1: replaces abstract-template L_NCE-cond descriptions.
    Per-pair caching is required because the description depends on
    (transcript_id, phi_index) — not just phi_index.
    """
    text = template.format(word=word)
    with torch.no_grad():
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        ids = ids.to(dtype=torch.long, device=device)
        emb = embed_layer(ids).float().mean(dim=0)
    return emb.to(dtype=dtype)


def build_phi_word_embed_cache(
    *,
    cf_pairs,                  # Iterable[CfPair]
    stress_by_audio_id: dict,  # {audio_id: {words: [...], ...}}
    embed_layer, tokenizer, device: torch.device,
    template: str = "stress on {word}",
) -> dict[tuple[str, int], torch.Tensor]:
    """Pre-compute one (transcription_id, phi_index) → d_llm embedding map.

    Used by stage3_train.py to look up q_pos / q_neg per cf-pair member
    without per-step tokenization.
    """
    cache: dict[tuple[str, int], torch.Tensor] = {}
    for cp in cf_pairs:
        ra = stress_by_audio_id.get(cp.audio_id_a)
        if ra is None:
            continue
        words = ra.get("words") or []
        for phi in (cp.stress_index_a, cp.stress_index_a_prime):
            key = (cp.transcription_id, int(phi))
            if key in cache:
                continue
            word = words[phi] if 0 <= phi < len(words) else ""
            cache[key] = make_phi_word_embed(
                embed_layer=embed_layer, tokenizer=tokenizer,
                word=word, device=device, template=template,
            )
    return cache


def compute_l_nce_cond(
    *,
    pooled_A_H: torch.Tensor,    # (d_llm,) — pool(A(H(a)))
    q_pos:      torch.Tensor,    # (d_llm,) — embed of "emphasis position: phi_a"  (no grad)
    q_neg:      torch.Tensor,    # (d_llm,) — embed of "emphasis position: phi_a'" (no grad)
    tau:        float = 0.07,
) -> torch.Tensor:
    """1-pos 1-neg InfoNCE. Cosine similarity, temperature τ.

    L_NCE = -log( exp(sim_pos/τ) / (exp(sim_pos/τ) + exp(sim_neg/τ)) )
          = log(1 + exp((sim_neg − sim_pos)/τ))   (binary contrastive sigmoid)
    """
    k = F.normalize(pooled_A_H.unsqueeze(0), dim=-1)
    qp = F.normalize(q_pos.unsqueeze(0), dim=-1)
    qn = F.normalize(q_neg.unsqueeze(0), dim=-1)
    sim_pos = (k * qp).sum(dim=-1) / tau
    sim_neg = (k * qn).sum(dim=-1) / tau
    # L = -log(exp(s_pos) / (exp(s_pos) + exp(s_neg)))
    lse = torch.logsumexp(torch.stack([sim_pos, sim_neg], dim=0), dim=0)
    return (lse - sim_pos).squeeze().float()


# ---------- Per-component grad-norm logging utility ---------- #

def per_component_grad_norm(
    params, loss_components: dict, retain_graph: bool = True
) -> dict[str, float]:
    """Compute ‖∇L_i‖ for each component i in `loss_components`.

    Calls `torch.autograd.grad(L_i, params, retain_graph=True)` per
    component. Heavyweight (one extra backward per component); call only
    every N steps from the training loop.
    """
    grad_norms = {}
    params = [p for p in params if p.requires_grad]
    for name, L in loss_components.items():
        if not torch.is_tensor(L) or L.requires_grad is False or L.dim() != 0:
            continue
        try:
            grads = torch.autograd.grad(
                L, params, retain_graph=retain_graph, allow_unused=True,
            )
            total = 0.0
            for g in grads:
                if g is not None:
                    total += float(g.float().norm().item() ** 2)
            grad_norms[name] = total ** 0.5
        except RuntimeError:
            grad_norms[name] = float("nan")
    return grad_norms


# ---------- Per-pair artifact cache (for cross-microbatch state) ---------- #

@dataclass
class ArtifactCacheEntry:
    detached_log_p: torch.Tensor    # (T_resp, vocab) — fp32, detached
    response_ids:   torch.Tensor    # (T_resp,)


class ArtifactPairCache:
    """One-shot cache: filled on first member, consumed by second.

    Keyed by an integer pair_id assigned at sampling time. The training
    loop must process pair members in the order (a) → (a'') within the
    same grad_accum cycle; otherwise L_artifact is computed without a
    matching reference and silently skipped.
    """

    def __init__(self):
        self._d: dict[int, ArtifactCacheEntry] = {}

    def put(self, pair_id: int, entry: ArtifactCacheEntry) -> None:
        self._d[pair_id] = entry

    def consume(self, pair_id: int) -> ArtifactCacheEntry | None:
        return self._d.pop(pair_id, None)

    def clear(self) -> None:
        self._d.clear()
