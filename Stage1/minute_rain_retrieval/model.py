"""Transformer that maps a variable number of PHY observations to one minute label."""
from __future__ import annotations

import torch
from torch import nn


class EncoderBlock(nn.Module):
    """Pre-norm encoder block with an explicitly boolean padding mask."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens)
        # A distinct view avoids the PyTorch 2.0 native MHA path, which
        # incorrectly canonicalizes a boolean padding mask to float and warns.
        value = normalized.view_as(normalized)
        attended, _ = self.attention(
            normalized, normalized, value,
            key_padding_mask=padding_mask, need_weights=False,
        )
        tokens = tokens + self.dropout1(attended)
        return tokens + self.feed_forward(self.norm2(tokens))


class MinuteRainTransformer(nn.Module):
    def __init__(self, input_dim: int, num_satellites: int, d_model: int = 192,
                 num_heads: int = 8, num_layers: int = 3, d_ff: int = 512,
                 dropout: float = 0.1, max_points: int = 256,
                 snr_quality_mode: str = "none", snr_threshold_db: float = -10.0,
                 snr_gate_temperature_db: float = 2.0):
        super().__init__()
        if snr_quality_mode not in {"none", "hard_mask", "soft_gate"}:
            raise ValueError(f"unsupported SNR quality mode: {snr_quality_mode}")
        if snr_gate_temperature_db <= 0:
            raise ValueError("snr_gate_temperature_db must be positive")
        self.snr_quality_mode = snr_quality_mode
        self.snr_threshold_db = float(snr_threshold_db)
        self.snr_gate_temperature_db = float(snr_gate_temperature_db)
        sat_dim = min(32, d_model // 4)
        numeric_dim = d_model - sat_dim
        self.numeric_projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, numeric_dim), nn.GELU()
        )
        self.quality_projection = (
            nn.Linear(1, numeric_dim, bias=False)
            if snr_quality_mode == "soft_gate" else None
        )
        self.satellite_embedding = nn.Embedding(num_satellites + 1, sat_dim, padding_idx=0)
        self.fusion = nn.Linear(d_model, d_model)
        self.summary_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Parameter(torch.zeros(1, max_points + 1, d_model))
        self.encoder = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)
        self.rain_classifier = nn.Linear(d_model, 1)
        self.amount_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(d_model // 2, 1), nn.Softplus())
        nn.init.normal_(self.summary_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def _quality_weights(self, raw_snr_db: torch.Tensor | None,
                         valid_mask: torch.Tensor) -> torch.Tensor:
        if self.snr_quality_mode == "none":
            return valid_mask.to(dtype=torch.float32)
        if raw_snr_db is None:
            raise ValueError(
                f"raw_snr_db is required when snr_quality_mode={self.snr_quality_mode}"
            )
        if self.snr_quality_mode == "hard_mask":
            return (raw_snr_db >= self.snr_threshold_db).to(dtype=torch.float32)
        return torch.sigmoid(
            (raw_snr_db - self.snr_threshold_db) / self.snr_gate_temperature_db
        )

    def forward(self, features: torch.Tensor, satellite_ids: torch.Tensor,
                valid_mask: torch.Tensor,
                raw_snr_db: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        quality = self._quality_weights(raw_snr_db, valid_mask)
        numeric_features = features
        if self.snr_quality_mode == "soft_gate":
            # Gate only radio observations and their dry-reference differences.
            # Geometry, weather, image, and relative-time context remain available.
            numeric_features = features.clone()
            numeric_features[..., :4] = numeric_features[..., :4] * quality.unsqueeze(-1)
            numeric_features[..., -4:] = numeric_features[..., -4:] * quality.unsqueeze(-1)
        numeric_tokens = self.numeric_projection(numeric_features)
        if self.quality_projection is not None:
            numeric_tokens = numeric_tokens + self.quality_projection(quality.unsqueeze(-1))
        point_tokens = torch.cat([
            numeric_tokens, self.satellite_embedding(satellite_ids)
        ], dim=-1)
        point_tokens = self.fusion(point_tokens)
        summary = self.summary_token.expand(features.shape[0], -1, -1)
        tokens = torch.cat([summary, point_tokens], dim=1)
        tokens = tokens + self.position_embedding[:, :tokens.shape[1]]
        attention_valid_mask = valid_mask
        if self.snr_quality_mode == "hard_mask":
            attention_valid_mask = valid_mask & quality.bool()
        padding_mask = torch.cat([
            torch.zeros(features.shape[0], 1, dtype=torch.bool, device=features.device),
            ~attention_valid_mask,
        ], dim=1)
        for layer in self.encoder:
            tokens = layer(tokens, padding_mask)
        minute_state = self.output_norm(tokens[:, 0])
        rain_logit = self.rain_classifier(minute_state).squeeze(-1)
        conditional_amount = self.amount_head(minute_state).squeeze(-1)
        prediction = torch.sigmoid(rain_logit) * conditional_amount
        return {
            "prediction": prediction,
            "rain_logit": rain_logit,
            "conditional_amount": conditional_amount,
            "quality_weight": quality,
            "attention_valid_mask": attention_valid_mask,
        }
