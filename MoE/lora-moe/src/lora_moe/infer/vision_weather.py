from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_moe.components import FeatureToSoftPromptProjector, FrozenWeatherVisionEncoder
from lora_moe.datasets import IMAGE_SUFFIXES, WeatherInstructionDataset, preprocess_image
from lora_moe.train.vision_weather_lora import build_prompt_parts, dtype_from_name


LABEL_TEXT = {
    "sunny": "晴天",
    "cloudy": "多云",
    "rain": "下雨",
}
DEFAULT_VISION_WEIGHTS = (
    "/home/wdz/BT/Stage1/vision_weather/weights/"
    "20260605_182036_weather_cls_more_cloudy_gpu_30ep_best_model.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vision-weather LoRA inference.")
    parser.add_argument("--model-dir", default="/home/wdz/BT/MoE/models/Qwen2.5-14B-Instruct")
    parser.add_argument("--adapter-dir", default="")
    parser.add_argument("--projector-path", default="")
    parser.add_argument("--output-dir", default="/home/wdz/BT/MoE/lora-moe/outputs/vision_weather_lora_qv_v1")
    parser.add_argument("--use-best", action="store_true")
    parser.add_argument("--vision-weights", default=DEFAULT_VISION_WEIGHTS)
    parser.add_argument("--split-root", default="/home/wdz/BT/Stage1/vision_weather/data/split")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--image", default="")
    parser.add_argument("--image-dir", default="")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save-jsonl", default="")
    return parser.parse_args()


def first_parameter_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def resolve_artifacts(args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = Path(args.output_dir).expanduser()
    if args.adapter_dir:
        adapter_dir = Path(args.adapter_dir).expanduser()
    else:
        adapter_dir = output_dir / "best" / "adapter" if args.use_best else output_dir / "adapter"

    if args.projector_path:
        projector_path = Path(args.projector_path).expanduser()
    else:
        projector_path = output_dir / "best" / "projector.pt" if args.use_best else output_dir / "projector.pt"

    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter_dir not found: {adapter_dir}")
    if not projector_path.exists():
        raise FileNotFoundError(f"projector_path not found: {projector_path}")
    return adapter_dir, projector_path


def load_projector(path: Path, output_dim: int, device: torch.device, dtype: torch.dtype):
    ckpt = torch.load(path, map_location="cpu")
    projector = FeatureToSoftPromptProjector(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        output_dim=output_dim,
        num_tokens=int(ckpt["num_tokens"]),
        dropout=0.0,
    )
    projector.load_state_dict(ckpt["state_dict"])
    projector.to(device, dtype=dtype)
    projector.eval()
    return projector, ckpt


def iter_image_paths(args: argparse.Namespace, image_size: int, class_names: list[str]) -> Iterable[dict]:
    if args.image:
        path = Path(args.image).expanduser()
        yield {"image_path": str(path), "label": None, "answer": None}
        return

    if args.image_dir:
        root = Path(args.image_dir).expanduser()
        paths = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        for path in paths[: args.limit if args.limit > 0 else None]:
            yield {"image_path": str(path), "label": None, "answer": None}
        return

    dataset = WeatherInstructionDataset(
        split_root=args.split_root,
        split=args.split,
        image_size=image_size,
        class_names=class_names,
        max_samples=args.limit,
    )
    for idx in range(len(dataset)):
        yield dataset[idx]


def parse_prediction(text: str) -> str | None:
    lowered = text.lower()
    if "多云" in text or "cloudy" in lowered:
        return "cloudy"
    if "晴天" in text or "晴朗" in text or "sunny" in lowered:
        return "sunny"
    if "下雨" in text or "雨天" in text or "rain" in lowered:
        return "rain"
    return None


@torch.inference_mode()
def generate_one(
    *,
    model,
    tokenizer,
    vision_encoder: FrozenWeatherVisionEncoder,
    projector,
    image_path: str,
    prefix_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    input_device: torch.device,
    torch_dtype: torch.dtype,
    max_new_tokens: int,
    temperature: float,
) -> dict:
    pixel_values = preprocess_image(Path(image_path), vision_encoder.image_size)
    pixel_values = pixel_values.unsqueeze(0).to(input_device, dtype=torch.float32)
    features = vision_encoder(pixel_values).to(input_device, dtype=torch_dtype)
    visual_embeds = projector(features).to(dtype=model.get_input_embeddings().weight.dtype)

    embedding = model.get_input_embeddings()
    prefix_embeds = embedding(prefix_ids.to(input_device)).unsqueeze(0)
    suffix_embeds = embedding(suffix_ids.to(input_device)).unsqueeze(0)
    inputs_embeds = torch.cat([prefix_embeds, visual_embeds, suffix_embeds], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], device=input_device, dtype=torch.long)

    do_sample = temperature > 0
    output_ids = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    cls = vision_encoder.classify(pixel_values)
    return {
        "image_path": image_path,
        "generated_text": text,
        "pred_label_from_text": parse_prediction(text),
        "vision_encoder_label": cls["pred_label"][0],
        "vision_encoder_probs": {
            name: round(float(cls["probs"][0, idx].detach().cpu()), 6)
            for idx, name in enumerate(vision_encoder.class_names)
        },
    }


def main() -> None:
    args = parse_args()
    adapter_dir, projector_path = resolve_artifacts(args)
    torch_dtype = dtype_from_name(args.dtype)

    print(f"[INFO] adapter_dir={adapter_dir}", flush=True)
    print(f"[INFO] projector_path={projector_path}", flush=True)
    print(f"[INFO] loading tokenizer/base model: {args.model_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    input_device = first_parameter_device(model.get_input_embeddings())
    projector, projector_ckpt = load_projector(projector_path, model.config.hidden_size, input_device, torch_dtype)
    vision_weights = args.vision_weights or str(projector_ckpt["vision_weights"])
    vision_encoder = FrozenWeatherVisionEncoder(weights=vision_weights, device=input_device, freeze=True)
    vision_encoder.eval()

    prefix_ids, suffix_ids = build_prompt_parts(tokenizer)
    rows = []
    correct = 0
    total_labeled = 0
    samples = list(iter_image_paths(args, vision_encoder.image_size, vision_encoder.class_names))
    for sample in tqdm(samples, desc="infer"):
        row = generate_one(
            model=model,
            tokenizer=tokenizer,
            vision_encoder=vision_encoder,
            projector=projector,
            image_path=sample["image_path"],
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            input_device=input_device,
            torch_dtype=torch_dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        row["label"] = sample.get("label")
        if row["label"] is not None:
            total_labeled += 1
            row["correct"] = row["pred_label_from_text"] == row["label"]
            correct += int(row["correct"])
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    if total_labeled:
        print(
            json.dumps(
                {
                    "total_labeled": total_labeled,
                    "correct": correct,
                    "accuracy": round(correct / total_labeled, 6),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if args.save_jsonl:
        save_path = Path(args.save_jsonl).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[INFO] saved jsonl: {save_path}", flush=True)


if __name__ == "__main__":
    main()
