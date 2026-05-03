"""BLSP-class adapter: WavLM-L16 hidden states → Qwen3-8B input embeddings.

Architecture (Stage-2 kickoff §2.0, post-Stage-1 dimensions for Qwen3-8B):

    H ∈ (B, T_s, d_enc=1024) from WavLM-L16
    → Conv1d(d_enc → 4·d_enc=4096, kernel=4, stride=4) over time
    → reshape to (B, T_s/4, 4096)
    → MLP-2: 4096 → 2·4096=8192 → d_llm=4096, GeLU, residual after layer 1
      (residual sums the post-layer-1 output back into linear-2's output;
       both endpoints are 4096-dim, so the skip is dimensionally clean)
    → final RMSNorm with learnable scale
    → add learnable modality-type embedding (1, 1, d_llm), broadcast over T_k
    → output (B, T_k=T_s/4, d_llm=4096)

Init recipe per session prompt rule (R2):
    - hidden layers (conv + linear-1) : kaiming-normal
    - last linear (linear-2) std       : std_8B / sqrt(d_llm)
    - RMSNorm scale                    : 1.0 (smoke retunes if needed)
    - modality token                   : N(0, std_8B)
    - std_8B is read from
      `Qwen/Qwen3-8B`.get_input_embeddings().weight.std() at runtime;
      the 1.7B value (0.0345) does NOT transfer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AdapterConfig:
    d_enc: int = 1024            # WavLM-Large d_enc
    d_llm: int = 4096            # Qwen3-8B d_llm
    conv_kernel: int = 4
    conv_stride: int = 4
    mlp_hidden_mult: int = 2     # hidden = mlp_hidden_mult * d_llm = 8192
    last_linear_std: float = 1e-4   # placeholder; overridden via init_scales(...)
    rmsnorm_init_scale: float = 1.0
    modality_token_std: float = 0.02


class RMSNorm(nn.Module):
    """RMSNorm on the last dim with a learnable scalar per channel.

    Output: x / RMS(x) * scale, where scale is initialized to a constant.
    """
    def __init__(self, d: int, init_scale: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.full((d,), float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS along last dim
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).to(x.dtype) * self.scale


class BLSPAdapter(nn.Module):
    """Conv4× → 2-layer MLP w/ residual → RMSNorm → +modality-token."""

    def __init__(self, cfg: AdapterConfig):
        super().__init__()
        self.cfg = cfg

        d_in = cfg.d_enc
        d_post_conv = 4 * cfg.d_enc       # 4096
        d_hidden = cfg.mlp_hidden_mult * cfg.d_llm  # 8192
        d_out = cfg.d_llm

        # Conv1d expects (B, C_in, T); we transpose at runtime.
        self.conv = nn.Conv1d(d_in, d_post_conv,
                              kernel_size=cfg.conv_kernel,
                              stride=cfg.conv_stride,
                              padding=0)
        self.linear1 = nn.Linear(d_post_conv, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_out)
        self.norm = RMSNorm(d_out, init_scale=cfg.rmsnorm_init_scale)
        self.modality_token = nn.Parameter(torch.zeros(1, 1, d_out))

        self._init_weights()

    # -- Initialization ------------------------------------------------------ #

    def _init_weights(self) -> None:
        # Kaiming-normal for conv + first linear (GeLU-friendly).
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="relu")
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)

        # Last linear initialized small per session prompt (R2). Caller MUST
        # call set_last_linear_std(std_8B / sqrt(d_llm)) before training to
        # bind the final scale to the actual model.
        nn.init.normal_(self.linear2.weight, mean=0.0, std=self.cfg.last_linear_std)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

        # Modality token: small Gaussian.
        nn.init.normal_(self.modality_token, mean=0.0, std=self.cfg.modality_token_std)

    def set_last_linear_std(self, std: float) -> None:
        with torch.no_grad():
            self.linear2.weight.normal_(mean=0.0, std=std)
            if self.linear2.bias is not None:
                self.linear2.bias.zero_()

    def set_rmsnorm_scale(self, scale: float) -> None:
        with torch.no_grad():
            self.norm.scale.fill_(float(scale))

    def set_modality_token_std(self, std: float) -> None:
        with torch.no_grad():
            self.modality_token.normal_(mean=0.0, std=std)

    # -- Forward ------------------------------------------------------------- #

    def forward(self, H: torch.Tensor, valid_T_s: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
          H:          (B, T_s, d_enc) hidden states from WavLM-L16.
          valid_T_s:  (B,) optional valid frame counts for the conv input.
                      If supplied, we return a `valid_T_k` mask too.

        Returns:
          K:          (B, T_k, d_llm) adapter output with modality token added.
          valid_T_k:  (B,) ceiled-down valid token counts post-conv (or None).
        """
        # (B, T_s, d_enc) → (B, d_enc, T_s) for Conv1d
        x = H.transpose(1, 2)                       # (B, d_enc, T_s)
        x = self.conv(x)                             # (B, 4*d_enc, T_k)
        x = x.transpose(1, 2)                        # (B, T_k, 4*d_enc)

        # 2-layer MLP with residual after first layer.
        h1 = F.gelu(self.linear1(x))                 # (B, T_k, d_hidden)
        h2 = self.linear2(h1)                        # (B, T_k, d_llm)
        # Residual from the first layer's input space requires matching dim.
        # Per spec: 'residual after layer 1' — we add a projected shortcut from
        # the post-conv 4*d_enc=4096 input back into the d_llm=4096 output. With
        # 4*d_enc == d_llm == 4096 the connection is a clean identity.
        residual = x if x.shape[-1] == h2.shape[-1] else None
        if residual is not None:
            h2 = h2 + residual

        z = self.norm(h2)                            # (B, T_k, d_llm)
        z = z + self.modality_token                  # broadcast over T_k

        valid_T_k = None
        if valid_T_s is not None:
            # Conv1d with stride=k, no padding, kernel=k → T_k = floor((T_s - k)/k + 1)
            #                                                 = floor(T_s/k) when T_s % k == 0,
            #                                                 = floor((T_s - k)/k + 1) otherwise.
            k = self.cfg.conv_kernel
            s = self.cfg.conv_stride
            valid_T_k = torch.clamp((valid_T_s - k) // s + 1, min=1).to(torch.long)
        return z, valid_T_k

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
