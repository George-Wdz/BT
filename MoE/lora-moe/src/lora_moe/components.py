from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

LLAMA_FACTORY_ROOT = Path("/home/wdz/LLaMA-Factory")
if str(LLAMA_FACTORY_ROOT) not in sys.path:
    sys.path.insert(0, str(LLAMA_FACTORY_ROOT))

from leo_model.vision.models import WeatherClassifier  # noqa: E402


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

