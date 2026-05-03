"""Probe-K heads for Stage 1b/1c.

Two probe-head families per the session prompt:
  - LinearProbe:  pool(H) → masked-softmax over N_max stress positions.
                  Acts as the SELECTOR for layer locking (1b).
  - MLPProbe:     pool(H) → 256-d hidden (GeLU) → masked-softmax.
                  Reported as upper-bound diagnostic only.

Cells with weighted-sum {layers} use a learnable softmax-weighted combination
across pre-pooled per-layer vectors. This is mathematically equivalent to
`pool(sum_l softmax(w_l) · H_l)` because mean-pool is linear.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Layer-cell selector module ---------- #

class LayerCell(nn.Module):
    """Reduce per-layer pooled vectors `(B, L_used, d)` to `(B, d)`.

    Modes:
      - 'single':   pick a single layer by index (no learnable params).
      - 'wsum':     learnable softmax across L_used layers.
    """

    def __init__(self, mode: str, n_layers_used: int):
        super().__init__()
        assert mode in {"single", "wsum"}
        self.mode = mode
        self.n_layers_used = n_layers_used
        if mode == "wsum":
            # init to uniform — softmax(zeros) is uniform.
            self.weights = nn.Parameter(torch.zeros(n_layers_used))
        else:
            self.register_parameter("weights", None)

    def forward(self, pooled_layers: torch.Tensor) -> torch.Tensor:
        # pooled_layers: (B, L_used, d)
        if self.mode == "single":
            # caller selects the single layer ahead of time
            assert pooled_layers.shape[1] == 1, f"expected (B,1,d), got {tuple(pooled_layers.shape)}"
            return pooled_layers[:, 0, :]
        w = F.softmax(self.weights, dim=0)  # (L_used,)
        return torch.einsum("l,bld->bd", w, pooled_layers)


# ---------- Probe heads ---------- #

class LinearHead(nn.Module):
    def __init__(self, d_in: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPHead(nn.Module):
    def __init__(self, d_in: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------- End-to-end probe (cell + head) ---------- #

class ProbeK(nn.Module):
    def __init__(self, *, d_in: int, n_classes: int,
                 cell_mode: str, n_layers_used: int, head: str):
        super().__init__()
        self.cell = LayerCell(cell_mode, n_layers_used)
        if head == "linear":
            self.head = LinearHead(d_in, n_classes)
        elif head == "mlp2":
            self.head = MLPHead(d_in, n_classes)
        else:
            raise ValueError(head)

    def forward(self, pooled_layers: torch.Tensor) -> torch.Tensor:
        feat = self.cell(pooled_layers)        # (B, d)
        return self.head(feat)                 # (B, N_max)


# ---------- Training + evaluation ---------- #

@dataclass
class FitConfig:
    epochs: int = 80
    lr: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 1e-4
    seed: int = 0


def masked_softmax_logits(logits: torch.Tensor, n_words: torch.Tensor) -> torch.Tensor:
    """Set logits[i, k] = -inf for k >= n_words[i]."""
    B, K = logits.shape
    arange = torch.arange(K, device=logits.device).unsqueeze(0).expand(B, K)  # (B, K)
    mask = arange < n_words.unsqueeze(1)  # (B, K)
    return logits.masked_fill(~mask, float("-inf"))


def fit_probe(probe: ProbeK,
              X_train: torch.Tensor, y_train: torch.Tensor, n_words_train: torch.Tensor,
              X_eval: torch.Tensor, y_eval: torch.Tensor, n_words_eval: torch.Tensor,
              cfg: FitConfig, device: str) -> dict:
    torch.manual_seed(cfg.seed)
    probe.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = X_train.shape[0]

    best = {"epoch": -1, "eval_acc": -1.0}
    history = []

    for ep in range(cfg.epochs):
        probe.train()
        idx = torch.randperm(n)
        losses = []
        for i in range(0, n, cfg.batch_size):
            j = idx[i : i + cfg.batch_size]
            xb = X_train[j].to(device)
            yb = y_train[j].to(device)
            nb = n_words_train[j].to(device)
            logits = probe(xb)
            ml = masked_softmax_logits(logits, nb)
            loss = F.cross_entropy(ml, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        probe.eval()
        with torch.no_grad():
            logits = probe(X_eval.to(device))
            ml = masked_softmax_logits(logits, n_words_eval.to(device))
            preds = ml.argmax(dim=-1).cpu()
            eval_acc = (preds == y_eval).float().mean().item()
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "eval_acc": eval_acc})
        if eval_acc > best["eval_acc"]:
            best = {"epoch": ep, "eval_acc": eval_acc}

    return {"history": history, "best": best, "final_eval_acc": history[-1]["eval_acc"]}


@torch.no_grad()
def predict(probe: ProbeK, X: torch.Tensor, n_words: torch.Tensor, device: str,
            *, dtype: torch.dtype | None = None,
            noise_sigma: float = 0.0, noise_seed: int = 0) -> torch.Tensor:
    """Predict argmax with optional fp16 cast / additive Gaussian noise on inputs."""
    probe.eval()
    x = X.clone()
    if dtype is not None:
        x = x.to(dtype).to(X.dtype)   # round-trip through dtype to simulate quantization
    if noise_sigma > 0:
        g = torch.Generator(device="cpu").manual_seed(noise_seed)
        x = x + torch.randn(x.shape, generator=g) * noise_sigma
    logits = probe(x.to(device))
    ml = masked_softmax_logits(logits, n_words.to(device))
    return ml.argmax(dim=-1).cpu()


def accuracy(preds: torch.Tensor, y: torch.Tensor) -> float:
    return (preds == y).float().mean().item()


def within_transcript_accuracy(
    preds: torch.Tensor, y: torch.Tensor, transcript_ids: list[str],
    candidate_lookup: dict[str, list[int]],
) -> float:
    """For each item, restrict argmax to the transcript's candidate stress positions
    (drawn from the lookup) and compare to y.
    """
    # `preds` is the argmax over the full N_max range; it ignores the candidate
    # restriction. Recompute restricted argmax from probe outputs separately.
    raise NotImplementedError("Use within_transcript_accuracy_from_logits instead.")


@torch.no_grad()
def within_transcript_argmax(probe: ProbeK, X: torch.Tensor,
                             transcript_ids: list[str],
                             candidate_lookup: dict[str, list[int]],
                             device: str) -> torch.Tensor:
    """Restrict argmax to the given transcript's candidate stress positions."""
    probe.eval()
    logits = probe(X.to(device))    # (B, N_max)
    K = logits.shape[1]
    out = torch.empty(logits.shape[0], dtype=torch.long)
    for i, tid in enumerate(transcript_ids):
        cands = candidate_lookup.get(tid, [])
        if not cands:
            out[i] = logits[i].argmax().cpu()
            continue
        scores = logits[i, cands]   # (n_cand,)
        out[i] = cands[int(scores.argmax().item())]
    return out
