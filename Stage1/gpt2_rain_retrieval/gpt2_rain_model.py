"""GPT2 backbone baseline for Stage1 pass-level rainfall retrieval."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GPT2Model


class PatchEmbedding(nn.Module):
    """Project multivariate pass patches into GPT2 hidden space."""

    def __init__(self, input_dim: int, patch_len: int, stride: int, hidden_size: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(input_dim * patch_len, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        bsz, _, channels = x.shape
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.bool()
        patches = x.unfold(1, self.patch_len, self.stride)  # (B, N, C, P)
        patches = patches.permute(0, 1, 3, 2).contiguous()  # (B, N, P, C)
        n_patches = patches.shape[1]
        patches = patches.reshape(bsz, n_patches, channels * self.patch_len)
        embeds = self.norm(self.proj(patches))

        if mask is not None:
            patch_mask = mask.unfold(1, self.patch_len, self.stride).any(dim=-1).bool()
        else:
            patch_mask = torch.ones(bsz, n_patches, dtype=torch.bool, device=x.device)
        return embeds, patch_mask


class GroupAttentionPatchEmbedding(nn.Module):
    """Encode each physical feature group before projecting patches to GPT2."""

    def __init__(
        self,
        input_dim: int,
        feature_group_dims: list[int],
        patch_len: int,
        stride: int,
        hidden_size: int,
        group_hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        if sum(feature_group_dims) != input_dim:
            raise ValueError(
                f"feature_group_dims sum to {sum(feature_group_dims)}, expected input_dim={input_dim}"
            )
        self.patch_len = patch_len
        self.stride = stride
        self.feature_group_dims = list(feature_group_dims)
        self.group_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim * patch_len, group_hidden_dim),
                    nn.LayerNorm(group_hidden_dim),
                    nn.GELU(),
                )
                for dim in self.feature_group_dims
            ]
        )
        self.group_embed = nn.Parameter(torch.zeros(len(self.feature_group_dims), group_hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=group_hidden_dim,
            nhead=num_heads,
            dim_feedforward=group_hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.group_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(group_hidden_dim),
            nn.Linear(group_hidden_dim, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        nn.init.normal_(self.group_embed, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        bsz, _, _ = x.shape
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.bool()
        patches = x.unfold(1, self.patch_len, self.stride)  # (B, N, C, P)
        patches = patches.permute(0, 1, 3, 2).contiguous()  # (B, N, P, C)
        n_patches = patches.shape[1]
        patches = patches.reshape(bsz * n_patches, self.patch_len, -1)

        group_tokens = []
        start = 0
        for dim, proj in zip(self.feature_group_dims, self.group_projs):
            group = patches[:, :, start : start + dim].reshape(bsz * n_patches, self.patch_len * dim)
            group_tokens.append(proj(group))
            start += dim
        groups = torch.stack(group_tokens, dim=1)
        groups = groups + self.group_embed.unsqueeze(0).to(dtype=groups.dtype)
        groups = self.group_encoder(groups)
        embeds = self.out(groups.mean(dim=1)).reshape(bsz, n_patches, -1)

        if mask is not None:
            patch_mask = mask.unfold(1, self.patch_len, self.stride).any(dim=-1).bool()
        else:
            patch_mask = torch.ones(bsz, n_patches, dtype=torch.bool, device=x.device)
        return embeds, patch_mask


class SummaryEmbedding(nn.Module):
    """Pass-level statistics token, matching the small Stage1 model."""

    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim * 6, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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


class GPT2RainRegressor(nn.Module):
    """Patch-to-GPT2 regression baseline.

    GPT2 receives continuous patch embeddings through ``inputs_embeds``. The
    default experiment freezes GPT2 and trains only the numeric adapters and
    regression heads, making it a controlled comparison against the small
    Stage1 Transformer.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        mc = cfg["model"]
        input_dim = int(mc["input_dim"])
        gpt2_dir = mc["gpt2_model_dir"]
        gpt2_kwargs = {
            "local_files_only": bool(mc.get("local_files_only", True)),
        }
        torch_dtype = mc.get("torch_dtype", "float32")
        if torch_dtype == "float16":
            gpt2_kwargs["torch_dtype"] = torch.float16
        elif torch_dtype == "bfloat16":
            gpt2_kwargs["torch_dtype"] = torch.bfloat16

        self.gpt2 = GPT2Model.from_pretrained(gpt2_dir, **gpt2_kwargs)
        gpt2_layers = int(mc.get("gpt2_layers", 6))
        if gpt2_layers > 0:
            self.gpt2.h = self.gpt2.h[:gpt2_layers]
        hidden_size = int(self.gpt2.config.n_embd)

        if bool(mc.get("use_group_attention", False)):
            self.patch_embed = GroupAttentionPatchEmbedding(
                input_dim=input_dim,
                feature_group_dims=list(mc["feature_group_dims"]),
                patch_len=int(mc["patch_len"]),
                stride=int(mc["stride"]),
                hidden_size=hidden_size,
                group_hidden_dim=int(mc.get("group_hidden_dim", 128)),
                num_heads=int(mc.get("group_attention_heads", 4)),
                num_layers=int(mc.get("group_attention_layers", 1)),
                dropout=float(mc.get("group_attention_dropout", mc.get("dropout", 0.1))),
            )
        else:
            self.patch_embed = PatchEmbedding(
                input_dim=input_dim,
                patch_len=int(mc["patch_len"]),
                stride=int(mc["stride"]),
                hidden_size=hidden_size,
            )
        self.use_summary_token = bool(mc.get("use_summary_token", True))
        self.summary_embed = SummaryEmbedding(input_dim, hidden_size) if self.use_summary_token else None

        sat_emb_dim = int(mc.get("sat_emb_dim", 16))
        self.sat_embedding = nn.Embedding(int(mc["num_satellites"]), sat_emb_dim)
        self.sat_proj = nn.Linear(sat_emb_dim, hidden_size)

        dropout = float(mc.get("dropout", 0.1))
        n_aux = len(cfg["targets"].get("auxiliary", []))
        self.head_norm = nn.LayerNorm(hidden_size)
        self.head_drop = nn.Dropout(dropout)
        self.rainfall_head = nn.Linear(hidden_size, 1)
        self.rain_cls_head = nn.Linear(hidden_size, 1)
        self.aux_head = nn.Linear(hidden_size, n_aux) if n_aux > 0 else None
        self.nonnegative_rainfall = bool(mc.get("nonnegative_rainfall", True))

        self._set_gpt2_trainability(str(mc.get("freeze_gpt2", "all")).lower())

    def _set_gpt2_trainability(self, mode: str) -> None:
        if mode in {"false", "none", "0"}:
            for name, param in self.gpt2.named_parameters():
                param.requires_grad = True
                if name.startswith("wte."):
                    # We feed numerical soft tokens through inputs_embeds, so
                    # GPT2's text token embedding table is never used.
                    param.requires_grad = False
            return
        if mode in {"ln_wpe", "layernorm_position"}:
            for name, param in self.gpt2.named_parameters():
                param.requires_grad = ("ln" in name) or ("wpe" in name)
            return
        if mode in {"all", "true", "1", "frozen"}:
            for param in self.gpt2.parameters():
                param.requires_grad = False
            return
        raise ValueError(f"Unsupported freeze_gpt2 mode: {mode}")

    def forward(self, features: torch.Tensor, mask: torch.Tensor, satellite_idx: torch.Tensor):
        if mask.dtype != torch.bool:
            mask = mask.bool()
        bsz = features.size(0)
        patch_embeds, patch_mask = self.patch_embed(features, mask)
        sat_emb = self.sat_proj(self.sat_embedding(satellite_idx)).unsqueeze(1)
        patch_embeds = patch_embeds + sat_emb

        if self.use_summary_token:
            summary = self.summary_embed(features, mask) + sat_emb
            inputs_embeds = torch.cat([summary, patch_embeds], dim=1)
            attention_mask = torch.cat(
                [
                    torch.ones(bsz, 1, dtype=torch.long, device=features.device),
                    patch_mask.long(),
                ],
                dim=1,
            )
        else:
            inputs_embeds = patch_embeds
            attention_mask = patch_mask.long()

        hidden = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        ).last_hidden_state

        valid = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        pooled = self.head_drop(self.head_norm(pooled))
        rainfall = self.rainfall_head(pooled)
        if self.nonnegative_rainfall:
            rainfall = torch.nn.functional.softplus(rainfall)
        rain_logit = self.rain_cls_head(pooled).squeeze(-1)
        auxiliary = self.aux_head(pooled) if self.aux_head is not None else None
        return rainfall, auxiliary, rain_logit
