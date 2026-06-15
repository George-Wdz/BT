from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

BT_ROOT = Path(__file__).resolve().parents[4]
STAGE1_ROOT = BT_ROOT / "Stage1"
if str(STAGE1_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE1_ROOT))

STAGE1_MODEL_ROOT = STAGE1_ROOT / "rain_retrieval" / "model"
if str(STAGE1_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE1_MODEL_ROOT))

from vision_weather.models import WeatherClassifier  # noqa: E402
from models.patch_encoder_decoder import PatchEncoderDecoder  # noqa: E402


class FrozenWeatherVisionEncoder(nn.Module):
    """Frozen image encoder backed by the existing weather classifier."""

    def __init__(
        self,
        *,
        weights: str,
        device: Optional[torch.device | str] = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        ckpt = torch.load(Path(weights).expanduser(), map_location="cpu")
        class_names = ckpt.get("class_names", [])
        if not class_names:
            raise ValueError(f"checkpoint has no class_names: {weights}")

        self.class_names = list(class_names)
        self.image_size = int(ckpt.get("image_size", 224))
        self.model = WeatherClassifier(
            num_classes=len(self.class_names),
            dropout=float(ckpt.get("dropout", 0.2)),
            resnet_width=int(ckpt.get("resnet_width", 32)),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.out_dim = int(self.model.encoder.out_dim)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad_(False)
            self.model.eval()

        if device is not None:
            self.model.to(device)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.extract_features(pixel_values)

    @torch.inference_mode()
    def classify(self, pixel_values: torch.Tensor) -> dict:
        logits = self.model(pixel_values)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = probs.argmax(dim=-1)
        return {
            "logits": logits,
            "probs": probs,
            "pred_idx": pred_idx,
            "pred_label": [self.class_names[int(i)] for i in pred_idx.detach().cpu()],
        }


class FeatureToSoftPromptProjector(nn.Module):
    """Map one encoder feature vector into a short sequence of LLM soft tokens."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_tokens: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_tokens = int(num_tokens)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.num_tokens * self.output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.net(features)
        return out.view(features.shape[0], self.num_tokens, self.output_dim)


class FrozenStage1RainEncoder(nn.Module):
    """Frozen Stage1 rainfall retrieval model used as a pass-level encoder."""

    def __init__(
        self,
        *,
        checkpoint_dir: str,
        device: Optional[torch.device | str] = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        checkpoint_path = self.checkpoint_dir / "checkpoint.pth"
        meta_path = self.checkpoint_dir / "meta.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Stage1 checkpoint not found: {checkpoint_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Stage1 meta not found: {meta_path}")

        self.meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        self.cfg = self.meta["cfg"]
        self.model = PatchEncoderDecoder(self.cfg)
        state = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.out_dim = int(self.cfg["model"]["d_model"]) + 2

        if freeze:
            for param in self.model.parameters():
                param.requires_grad_(False)
            self.model.eval()

        if device is not None:
            self.model.to(device)

    def _encode(self, features: torch.Tensor, mask: torch.Tensor, satellite_idx: torch.Tensor):
        model = self.model
        batch_size = features.size(0)
        sat_emb = model.sat_proj(model.sat_embedding(satellite_idx))

        if model.use_channel_attn:
            enc_in, patch_mask = model.enc_patch_embed(features, mask)
            enc_in = enc_in + sat_emb.unsqueeze(1).unsqueeze(1)
            bsz, num_patches, num_groups, hidden = enc_in.shape
            enc_in_t = enc_in.permute(0, 2, 1, 3).reshape(bsz * num_groups, num_patches, hidden)
            enc_in_t = model.enc_pos(enc_in_t)
            enc_in = enc_in_t.reshape(bsz, num_groups, num_patches, hidden).permute(0, 2, 1, 3)

            enc_pad_mask = ~patch_mask
            enc_out = enc_in
            for layer in model.encoder_layers:
                enc_out = layer(enc_out, key_padding_mask=enc_pad_mask)
            enc_out = enc_out.reshape(batch_size, num_patches, num_groups * hidden)
            enc_out = model.channel_fuse(enc_out)
            enc_out = model.enc_norm(enc_out)
        else:
            enc_in, patch_mask = model.enc_patch_embed(features, mask)
            enc_in = enc_in + sat_emb.unsqueeze(1)
            enc_in = model.enc_pos(enc_in)
            enc_pad_mask = ~patch_mask
            enc_out = enc_in
            for layer in model.encoder_layers:
                enc_out = layer(enc_out, key_padding_mask=enc_pad_mask)
            enc_out = model.enc_norm(enc_out)

        if model.use_summary_token:
            summary = model.summary_embed(features, mask)
            enc_out = torch.cat([summary, enc_out], dim=1)
            enc_pad_mask = torch.cat(
                [
                    torch.zeros(batch_size, 1, dtype=torch.bool, device=enc_pad_mask.device),
                    enc_pad_mask,
                ],
                dim=1,
            )
        return enc_out, enc_pad_mask

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        satellite_idx: torch.Tensor,
    ) -> torch.Tensor:
        enc_out, enc_pad_mask = self._encode(features, mask, satellite_idx)
        valid = (~enc_pad_mask).unsqueeze(-1).to(dtype=enc_out.dtype)
        pooled = (enc_out * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        rainfall, _, rain_logit = self.model(features, mask, satellite_idx)
        rainfall_feature = torch.log1p(rainfall.clamp_min(0.0))
        rain_prob = torch.sigmoid(rain_logit).unsqueeze(-1)
        return torch.cat([pooled, rainfall_feature, rain_prob], dim=-1)

    @torch.inference_mode()
    def predict(self, features: torch.Tensor, mask: torch.Tensor, satellite_idx: torch.Tensor) -> dict:
        rainfall, auxiliary, rain_logit = self.model(features, mask, satellite_idx)
        return {
            "rainfall_mm": rainfall.squeeze(-1),
            "rain_probability": torch.sigmoid(rain_logit),
            "auxiliary": auxiliary,
        }


class TaskRouter(nn.Module):
    """Base interface for future LoRA/adapter routing modules."""

    def forward(self, *args, **kwargs):  # pragma: no cover - interface placeholder
        raise NotImplementedError


class KeywordTaskRouter:
    """Non-trainable bootstrap router used before a learned router is available."""

    def __init__(self, task_keywords: dict[str, list[str]]) -> None:
        self.task_keywords = task_keywords

    def route(self, text: str, default_task: str = "general") -> str:
        lowered = text.lower()
        for task_name, keywords in self.task_keywords.items():
            if any(keyword.lower() in lowered for keyword in keywords):
                return task_name
        return default_task
