"""
Patch-based Encoder-Decoder Transformer (Pass-based 版本)

每个样本是一个卫星过境片段（pass），关键变化：
- 加入 Satellite Embedding（含未知卫星槽位用于冷启动）
- 支持 attention mask 处理 padding
- 整段过境输出标签（pass_rainfall_mm, wind_speed, wind_direction）
- 可消融的 Channel Attention（two-stage：先时间attn，后通道attn）
"""
import math
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Channel-mixing patching：全部特征拼接后投影到 d_model。"""

    def __init__(self, input_dim: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(input_dim * patch_len, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x: (B, T, C), mask: (B, T) bool
        B, T, C = x.shape
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.bool()
        x_p = x.unfold(1, self.patch_len, self.stride)   # (B, N, C, P)
        x_p = x_p.permute(0, 1, 3, 2).contiguous()       # (B, N, P, C)
        N = x_p.shape[1]
        x_p = x_p.reshape(B, N, -1)
        x_p = self.norm(self.proj(x_p))

        if mask is not None:
            m_p = mask.unfold(1, self.patch_len, self.stride)  # (B, N, P)
            patch_mask = m_p.any(dim=-1).bool()
        else:
            patch_mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        return x_p, patch_mask


class GroupedPatchEmbedding(nn.Module):
    """
    分组 patching：每个特征组（link/position/weather）独立 embedding。
    输出 (B, N, G, d_model)，G = 特征组数。
    """

    def __init__(self, group_dims: list, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.group_dims = group_dims  # e.g. [4, 6, 3]
        self.patch_len = patch_len
        self.stride = stride
        self.projs = nn.ModuleList([
            nn.Linear(g * patch_len, d_model) for g in group_dims
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in group_dims
        ])

    def forward(self, x, mask=None):
        # x: (B, T, sum(group_dims))
        B, T, _ = x.shape
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.bool()
        # 切分通道
        chunks = torch.split(x, self.group_dims, dim=-1)
        group_tokens = []
        for chunk, proj, norm in zip(chunks, self.projs, self.norms):
            # chunk: (B, T, g)
            x_p = chunk.unfold(1, self.patch_len, self.stride)  # (B, N, g, P)
            x_p = x_p.permute(0, 1, 3, 2).contiguous()          # (B, N, P, g)
            N = x_p.shape[1]
            x_p = x_p.reshape(B, N, -1)                         # (B, N, P*g)
            x_p = norm(proj(x_p))                               # (B, N, d)
            group_tokens.append(x_p)
        # 堆叠为 (B, N, G, d)
        out = torch.stack(group_tokens, dim=2)

        if mask is not None:
            m_p = mask.unfold(1, self.patch_len, self.stride)
            patch_mask = m_p.any(dim=-1).bool()  # (B, N)
        else:
            patch_mask = torch.ones(B, out.size(1), dtype=torch.bool, device=x.device)
        return out, patch_mask


class ModalEncoder(nn.Module):
    """Small modality-specific residual encoder, deliberately compact for small data."""
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 2), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x):
        return x + self.net(x)


class ConditionalLayerNorm(nn.Module):
    """FuXi-inspired bounded adaLN conditioning."""
    def __init__(self, d_model: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, 2 * d_model))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, condition):
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        while scale.ndim < x.ndim:
            scale, shift = scale.unsqueeze(1), shift.unsqueeze(1)
        return self.norm(x) * (1.0 + 0.1 * torch.tanh(scale)) + shift


class GroupAttentionPatchEmbedding(nn.Module):
    """
    物理分组 attention patching：先按 feature group 编码，再在每个 patch 内做组间 attention。

    输出 (B, N, d_model)，可直接接标准时间维 EncoderLayer。
    """

    def __init__(
        self,
        group_dims: list,
        patch_len: int,
        stride: int,
        d_model: int,
        group_hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.group_dims = list(group_dims)
        self.patch_len = patch_len
        self.stride = stride
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(g * patch_len, group_hidden_dim),
                nn.LayerNorm(group_hidden_dim),
                nn.GELU(),
            )
            for g in self.group_dims
        ])
        self.group_embed = nn.Parameter(torch.zeros(len(self.group_dims), group_hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=group_hidden_dim,
            nhead=n_heads,
            dim_feedforward=group_hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.group_encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(group_hidden_dim),
            nn.Linear(group_hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )
        nn.init.normal_(self.group_embed, std=0.02)

    def forward(self, x, mask=None):
        # x: (B, T, sum(group_dims))
        B, T, _ = x.shape
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.bool()
        chunks = torch.split(x, self.group_dims, dim=-1)
        group_tokens = []
        N = None
        for chunk, proj in zip(chunks, self.group_projs):
            x_p = chunk.unfold(1, self.patch_len, self.stride)  # (B, N, g, P)
            x_p = x_p.permute(0, 1, 3, 2).contiguous()          # (B, N, P, g)
            N = x_p.shape[1]
            x_p = x_p.reshape(B * N, -1)
            group_tokens.append(proj(x_p))
        groups = torch.stack(group_tokens, dim=1)
        groups = groups + self.group_embed.unsqueeze(0).to(dtype=groups.dtype)
        groups = self.group_encoder(groups)
        out = self.out(groups.mean(dim=1)).reshape(B, N, -1)

        if mask is not None:
            m_p = mask.unfold(1, self.patch_len, self.stride)
            patch_mask = m_p.any(dim=-1).bool()
        else:
            patch_mask = torch.ones(B, out.size(1), dtype=torch.bool, device=x.device)
        return out, patch_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class SummaryEmbedding(nn.Module):
    """Pass-level statistics token for weak rain attenuation signals."""

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim * 6, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x, mask):
        # x: (B, T, C), mask: (B, T)
        valid = mask.unsqueeze(-1).to(dtype=x.dtype)
        count = valid.sum(dim=1).clamp_min(1.0)
        x_masked = x * valid
        mean = x_masked.sum(dim=1) / count
        centered = (x - mean.unsqueeze(1)) * valid
        std = torch.sqrt((centered.square().sum(dim=1) / count).clamp_min(1e-6))
        x_min = x.masked_fill(~mask.unsqueeze(-1), float("inf")).amin(dim=1)
        x_max = x.masked_fill(~mask.unsqueeze(-1), float("-inf")).amax(dim=1)
        x_min = torch.where(torch.isfinite(x_min), x_min, mean)
        x_max = torch.where(torch.isfinite(x_max), x_max, mean)
        x_range = x_max - x_min
        lengths = mask.long().sum(dim=1).clamp_min(1)
        first = x[:, 0]
        last_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, x.size(-1))
        last = x.gather(1, last_idx).squeeze(1)
        slope = (last - first) / lengths.sub(1).clamp_min(1).unsqueeze(-1).to(x.dtype)
        stats = torch.cat([mean, std, x_min, x_max, x_range, slope], dim=-1)
        return self.proj(stats).unsqueeze(1)


class EncoderLayer(nn.Module):
    """标准 Encoder 层：仅时间维 self-attention"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, condition=None):
        if key_padding_mask is not None and key_padding_mask.dtype != torch.bool:
            key_padding_mask = key_padding_mask.bool()
        h = self.norm1(x)
        # PyTorch 2.0.1 eval fast path warns on bool key_padding_mask internally.
        # Cloning value preserves self-attention math while using the stable path.
        v = h if self.training else h.clone()
        a, _ = self.self_attn(h, h, v, key_padding_mask=key_padding_mask)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class TwoStageEncoderLayer(nn.Module):
    """
    Two-stage Encoder 层：先时间维 attention，后通道维 attention。

    输入: (B, N, G, d)  N=patches, G=channel groups
    - Stage 1 (Time): 对每个 group 独立做 patch 间 attention
    - Stage 2 (Channel): 对每个 patch 位置独立做 group 间 attention
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout, batch_first=True)
        self.channel_attn = nn.MultiheadAttention(d_model, n_heads,
                                                  dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # x: (B, N, G, d), key_padding_mask: (B, N)
        if key_padding_mask is not None and key_padding_mask.dtype != torch.bool:
            key_padding_mask = key_padding_mask.bool()
        B, N, G, D = x.shape

        # Stage 1: 时间维 attention，每个group独立
        # reshape: (B, N, G, d) → (B*G, N, d)
        h = self.norm1(x).permute(0, 2, 1, 3).reshape(B * G, N, D)
        # 每个group共享同样的patch_mask，重复G次
        if key_padding_mask is not None:
            time_mask = key_padding_mask.unsqueeze(1).expand(-1, G, -1).reshape(B * G, N)
            time_mask = time_mask.bool()
        else:
            time_mask = None
        v = h if self.training else h.clone()
        a, _ = self.time_attn(h, h, v, key_padding_mask=time_mask)
        a = a.reshape(B, G, N, D).permute(0, 2, 1, 3)  # (B, N, G, d)
        x = x + self.drop(a)

        # Stage 2: 通道维 attention，每个patch位置独立
        # reshape: (B, N, G, d) → (B*N, G, d)
        h = self.norm2(x).reshape(B * N, G, D)
        c, _ = self.channel_attn(h, h, h)  # 通道间无需padding mask
        c = c.reshape(B, N, G, D)
        x = x + self.drop(c)

        # FFN
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc_out, enc_padding_mask=None):
        if enc_padding_mask is not None and enc_padding_mask.dtype != torch.bool:
            enc_padding_mask = enc_padding_mask.bool()
        h = self.norm1(x)
        a, _ = self.self_attn(h, h, h)
        x = x + self.drop(a)
        h = self.norm2(x)
        c, _ = self.cross_attn(h, enc_out, enc_out,
                               key_padding_mask=enc_padding_mask)
        x = x + self.drop(c)
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x


class PatchEncoderDecoder(nn.Module):
    """
    Pass-based Patch Encoder-Decoder Transformer.

    支持三种 Encoder 模式：
    - Channel-mixing（默认）：13维特征拼接做patch embedding
    - Two-stage attention（可选）：分组patch embedding + 时间attn + 通道attn
    - Group attention（可选）：分组patch embedding + 组间attn + 标准时间attn

    通过 cfg.model.fusion_mode 切换，便于消融对比。
    """

    def __init__(self, cfg: dict):
        super().__init__()
        mc = cfg["model"]
        input_dim = mc["input_dim"]
        patch_len = mc["patch_len"]
        stride = mc["stride"]
        d_model = mc["d_model"]
        n_heads = mc["n_heads"]
        d_ff = mc["d_ff"]
        dropout = mc["dropout"]
        e_layers = mc["e_layers"]
        d_layers = mc["d_layers"]
        num_satellites = mc["num_satellites"]
        sat_emb_dim = mc.get("sat_emb_dim", 16)

        legacy_channel = mc.get("use_channel_attention", False)
        self.fusion_mode = mc.get("fusion_mode")
        if self.fusion_mode is None:
            self.fusion_mode = "cw" if legacy_channel else "cm"
        self.fusion_mode = str(self.fusion_mode).lower()
        if self.fusion_mode not in {"cm", "cw", "ga"}:
            raise ValueError(f"unsupported model.fusion_mode={self.fusion_mode}; expected cm/cw/ga")
        self.use_channel_attn = self.fusion_mode == "cw"
        self.use_summary_token = mc.get("use_summary_token", True)
        self.use_modal_encoders = mc.get("use_modal_encoders", True)
        self.use_conditioning = mc.get("use_conditioning", True)
        self.use_quality_gating = mc.get("use_quality_gating", True)
        self.feature_group_dims = mc.get("feature_group_dims", [6, 6, 3])
        assert sum(self.feature_group_dims) == input_dim, \
            f"feature_group_dims sum {sum(self.feature_group_dims)} != input_dim {input_dim}"

        n_targets = (len(cfg["targets"]["primary"])
                     + len(cfg["targets"].get("auxiliary", [])))

        self.patch_len = patch_len
        self.stride = stride
        self.n_targets = n_targets

        # Satellite Embedding：索引0=未知卫星
        self.sat_embedding = nn.Embedding(num_satellites, sat_emb_dim)
        self.sat_proj = nn.Linear(sat_emb_dim, d_model)
        condition_input_dim = int(mc.get("condition_input_dim", 10))
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_input_dim + d_model, d_model), nn.SiLU(),
            nn.LayerNorm(d_model),
        )
        self.input_adaln = ConditionalLayerNorm(d_model, d_model)

        # Encoder（根据消融开关选择）
        if self.fusion_mode == "cw":
            self.enc_patch_embed = GroupedPatchEmbedding(
                self.feature_group_dims, patch_len, stride, d_model
            )
            self.encoder_layers = nn.ModuleList([
                TwoStageEncoderLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(e_layers)
            ])
            # 通道融合：将 (B, N, G, d) 聚合为 (B, N, d) 供 decoder cross-attn 使用
            self.channel_fuse = nn.Linear(len(self.feature_group_dims) * d_model, d_model)
        elif self.fusion_mode == "ga":
            self.enc_patch_embed = GroupAttentionPatchEmbedding(
                self.feature_group_dims,
                patch_len,
                stride,
                d_model,
                group_hidden_dim=int(mc.get("group_hidden_dim", 128)),
                n_heads=int(mc.get("group_attention_heads", 4)),
                n_layers=int(mc.get("group_attention_layers", 1)),
                dropout=float(mc.get("group_attention_dropout", dropout)),
            )
            self.encoder_layers = nn.ModuleList([
                EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)
            ])
            self.channel_fuse = None
        else:
            self.enc_patch_embed = PatchEmbedding(input_dim, patch_len, stride, d_model)
            self.encoder_layers = nn.ModuleList([
                EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)
            ])
            self.channel_fuse = None

        n_groups = len(self.feature_group_dims)
        self.modal_encoders = nn.ModuleList([
            ModalEncoder(d_model, dropout) for _ in range(n_groups)
        ])
        self.quality_gate = nn.Sequential(
            nn.Linear(d_model + 1, max(d_model // 2, 16)), nn.SiLU(),
            nn.Linear(max(d_model // 2, 16), 1),
        )

        self.enc_pos = PositionalEncoding(d_model, dropout=dropout)
        self.enc_norm = nn.LayerNorm(d_model)
        self.summary_embed = (
            SummaryEmbedding(input_dim, d_model) if self.use_summary_token else None
        )

        # Decoder：n_targets 个可学习 query token
        self.target_queries = nn.Parameter(torch.randn(n_targets, d_model) * 0.02)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(d_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)

        # 输出头：雨量直接输出物理空间的非负 mm 值。
        self.rainfall_head = nn.Linear(d_model, 1)
        self.rain_cls_head = nn.Linear(d_model, 1)
        self.aux_heads = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(max(n_targets - 1, 0))
        ])
        self.aux_target_names = list(cfg["targets"].get("auxiliary", []))
        # rainfall, classification and auxiliary tasks; bounded in compute_loss.
        self.task_log_vars = nn.Parameter(torch.zeros(2 + len(self.aux_heads)))
        self.nonnegative_rainfall = mc.get("nonnegative_rainfall", True)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features, mask, satellite_idx, condition=None, modal_quality=None):
        """
        features: (B, T, input_dim)
        mask: (B, T) bool  True=真实数据, False=padding
        satellite_idx: (B,) long
        """
        if mask.dtype != torch.bool:
            mask = mask.bool()
        B = features.size(0)
        sat_emb = self.sat_proj(self.sat_embedding(satellite_idx))  # (B, d)
        if condition is None:
            condition = features.new_zeros(B, 10)
        cond = self.condition_encoder(torch.cat([condition, sat_emb], dim=-1))

        if self.fusion_mode == "cw":
            # Two-stage 模式
            enc_in, patch_mask = self.enc_patch_embed(features, mask)  # (B, N, G, d)
            if self.use_modal_encoders:
                enc_in = torch.stack([
                    encoder(enc_in[:, :, i]) for i, encoder in enumerate(self.modal_encoders)
                ], dim=2)
            if self.use_quality_gating and modal_quality is not None:
                quality = modal_quality[:, None, :, None].expand(-1, enc_in.size(1), -1, -1)
                gate = torch.sigmoid(self.quality_gate(torch.cat([enc_in, quality], dim=-1)))
                enc_in = enc_in * gate * quality
            # 加 satellite embedding（广播到所有 patch 和 group）
            enc_in = enc_in + sat_emb.unsqueeze(1).unsqueeze(1)
            if self.use_conditioning:
                enc_in = self.input_adaln(enc_in, cond)
            # 加位置编码（在时间维上）：先 reshape 到 (B*G, N, d) 加完再 reshape 回去
            B_, N, G, D = enc_in.shape
            enc_in_t = enc_in.permute(0, 2, 1, 3).reshape(B_ * G, N, D)
            enc_in_t = self.enc_pos(enc_in_t)
            enc_in = enc_in_t.reshape(B_, G, N, D).permute(0, 2, 1, 3)

            enc_pad_mask = ~patch_mask
            enc_out = enc_in
            for layer in self.encoder_layers:
                enc_out = layer(enc_out, key_padding_mask=enc_pad_mask)
            # 通道融合: (B, N, G, d) → (B, N, d)
            enc_out = enc_out.reshape(B, N, G * D)
            enc_out = self.channel_fuse(enc_out)
            enc_out = self.enc_norm(enc_out)
        else:
            # Channel-mixing / group-attention 模式
            enc_in, patch_mask = self.enc_patch_embed(features, mask)  # (B, N, d)
            enc_in = enc_in + sat_emb.unsqueeze(1)
            if self.use_conditioning:
                enc_in = self.input_adaln(enc_in, cond)
            enc_in = self.enc_pos(enc_in)
            enc_pad_mask = ~patch_mask
            enc_out = enc_in
            for layer in self.encoder_layers:
                enc_out = layer(enc_out, key_padding_mask=enc_pad_mask)
            enc_out = self.enc_norm(enc_out)

        if self.use_summary_token:
            summary = self.summary_embed(features, mask)
            enc_out = torch.cat([summary, enc_out], dim=1)
            enc_pad_mask = torch.cat(
                [
                    torch.zeros(B, 1, dtype=torch.bool, device=enc_pad_mask.device),
                    enc_pad_mask,
                ],
                dim=1,
            )


        # Decoder
        dec_in = self.target_queries.unsqueeze(0).expand(B, -1, -1)
        dec_out = dec_in
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, enc_padding_mask=enc_pad_mask)
        dec_out = self.dec_norm(dec_out)

        # 输出头
        rain_repr = dec_out[:, 0]
        rainfall = self.rainfall_head(rain_repr)
        if self.nonnegative_rainfall:
            rainfall = torch.nn.functional.softplus(rainfall)
        rain_logit = self.rain_cls_head(rain_repr).squeeze(-1)
        if self.aux_heads:
            aux_values = []
            for i, (name, head) in enumerate(zip(self.aux_target_names, self.aux_heads)):
                value = head(dec_out[:, i + 1])
                if name in {"rain_rate_mean", "rain_rate_max"}:
                    # Auxiliary labels are standardized; do not force positivity here.
                    pass
                elif name == "rainy_ratio":
                    # Also standardized during training, so retain an unconstrained head.
                    pass
                aux_values.append(value)
            auxiliary = torch.cat(aux_values, dim=-1)
        else:
            auxiliary = None
        return rainfall, auxiliary, rain_logit
