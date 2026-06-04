#!/usr/bin/env python3
"""DATE model components.

This module implements the discriminator used by DATE. The generator is kept in
the training loop because DATE supports either a learned MLM generator or the
random generator used in the paper's ablations.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class DateDiscriminator(nn.Module):
    """Transformer discriminator with RTD and RMD prediction heads."""

    def __init__(self, model_name_or_path: str, num_mask_patterns: int, dropout: float = 0.1):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name_or_path)
        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=config)
        hidden_size = int(config.hidden_size)
        self.rtd_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )
        self.rmd_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_mask_patterns),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        rtd_logits = self.rtd_head(hidden)
        cls_hidden = hidden[:, 0]
        rmd_logits = self.rmd_head(cls_hidden)
        return rtd_logits, rmd_logits
