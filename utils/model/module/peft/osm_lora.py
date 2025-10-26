# utils/model/module/peft/osm_lora.py
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    LoRA wrapper for nn.Linear.
    Forward: y = (W + (alpha/r) * B @ A) x + b
      A: [r, in_features], B: [out_features, r]
    """
    def __init__(self, base: nn.Linear, rank: int, alpha: float, sparsity: float = 1.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.has_bias = base.bias is not None

        # base
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=True)
        if self.has_bias:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=True)
        else:
            self.register_parameter("bias", None)

        # LoRA 參數
    self.rank = int(rank)
    self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)
    self.sparsity = float(sparsity)

        if self.rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.rank, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
            # binary masks for OSM sparsity (element-wise). 1: active, 0: pruned
            mask_A = torch.ones(self.rank, self.in_features)
            mask_B = torch.ones(self.out_features, self.rank)
            if 0.0 <= self.sparsity < 1.0:
                # keep ratio = sparsity (e.g., 0.5 keeps 50%)
                mask_A.bernoulli_(self.sparsity)
                mask_B.bernoulli_(self.sparsity)
            self.register_buffer("mask_A", mask_A)
            self.register_buffer("mask_B", mask_B)
            self.reset_parameters_lora()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.register_buffer("mask_A", None)
            self.register_buffer("mask_B", None)

        # merge 狀態
        self._merged: bool = False
        self._weight_backup: Optional[torch.Tensor] = None

    def reset_parameters_lora(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank == 0:
            return F.linear(x, self.weight, self.bias)
        if self._merged:
            return F.linear(x, self.weight, self.bias)
        # 正常 LoRA 推論： base + scaling * (B @ A)
        if self.mask_A is not None and self.mask_B is not None:
            lora_A_eff = self.lora_A * self.mask_A
            lora_B_eff = self.lora_B * self.mask_B
        else:
            lora_A_eff = self.lora_A
            lora_B_eff = self.lora_B
        delta = lora_B_eff @ lora_A_eff  # [out, in]
        return F.linear(x, self.weight + self.scaling * delta, self.bias)

    @torch.no_grad()
    def merge(self):
        """把 LoRA 增量暫時合併進 weight（推論 0 開銷）。"""
        if self.rank == 0 or self._merged:
            return
        # 確保裝置一致
        dev = self.weight.device
        if self.mask_A is not None and self.mask_B is not None:
            lora_A_eff = (self.lora_A * self.mask_A).to(dev)
            lora_B_eff = (self.lora_B * self.mask_B).to(dev)
        else:
            lora_A_eff = self.lora_A.to(dev)
            lora_B_eff = self.lora_B.to(dev)
        delta = (lora_B_eff @ lora_A_eff)
        self._weight_backup = self.weight.detach().clone()
        self.weight.add_(self.scaling * delta)
        self._merged = True

    @torch.no_grad()
    def unmerge(self):
        """還原 merge 前的 weight。"""
        if self.rank == 0 or not self._merged:
            return
        if self._weight_backup is not None:
            self.weight.data.copy_(self._weight_backup)
            self._weight_backup = None
        self._merged = False

    @torch.no_grad()
    def set_sparsity(self, sparsity: float):
        """Reset binary masks according to sparsity in [0,1]."""
        self.sparsity = float(sparsity)
        if self.rank == 0:
            return
        self.mask_A.fill_(1.0)
        self.mask_B.fill_(1.0)
        if 0.0 <= self.sparsity < 1.0:
            self.mask_A.bernoulli_(self.sparsity)
            self.mask_B.bernoulli_(self.sparsity)
