from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_moe.components import FeatureToSoftPromptProjector, FrozenWeatherVisionEncoder
from lora_moe.datasets import WeatherInstructionDataset, weather_collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train vision-weather projector + Qwen LoRA.")
    parser.add_argument("--model-dir", default="/home/wdz/BT/MoE/models/Qwen2.5-14B-Instruct")
    parser.add_argument("--split-root", default="/home/wdz/BT/Stage1/vision_weather/data/split")
    parser.add_argument(
        "--vision-weights",
        default="/home/wdz/BT/Stage1/vision_weather/weights/20260605_182036_weather_cls_more_cloudy_gpu_30ep_best_model.pt",
    )
    parser.add_argument("--output-dir", default="/home/wdz/BT/MoE/lora-moe/outputs/vision_weather_lora_v1")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=128)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-visual-tokens", type=int, default=8)
    parser.add_argument("--projector-hidden-dim", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,v_proj",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def first_parameter_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def tokenize_no_special(tokenizer, text: str) -> torch.Tensor:
    return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def build_prompt_parts(tokenizer):
    prefix = (
        "<|im_start|>system\n"
        "你是气象图像识别助手。请根据用户上传的图片判断天气。"
        "回答要自然、简洁，不要提到视觉token、编码器、特征、模型内部表示等实现细节。\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
    )
    suffix = "请识别这张图片的天气，只回答一句中文结论。<|im_end|>\n<|im_start|>assistant\n"
    return tokenize_no_special(tokenizer, prefix), tokenize_no_special(tokenizer, suffix)


def build_inputs_embeds_and_labels(
    *,
    model,
    tokenizer,
    projector,
    vision_features: torch.Tensor,
    answers: list[str],
    prefix_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    input_device: torch.device,
):
    embedding = model.get_input_embeddings()
    dtype = embedding.weight.dtype

    prefix_ids = prefix_ids.to(input_device)
    suffix_ids = suffix_ids.to(input_device)
    prefix_embeds = embedding(prefix_ids).unsqueeze(0)
    suffix_embeds = embedding(suffix_ids).unsqueeze(0)

    visual_embeds = projector(vision_features.to(input_device, dtype=next(projector.parameters()).dtype))
    visual_embeds = visual_embeds.to(dtype=dtype)

    per_sample_embeds = []
    per_sample_labels = []
    max_len = 0
    for idx, answer in enumerate(answers):
        target_ids = tokenize_no_special(tokenizer, answer + "<|im_end|>").to(input_device)
        target_embeds = embedding(target_ids).unsqueeze(0)
        embeds = torch.cat(
            [
                prefix_embeds,
                visual_embeds[idx : idx + 1],
                suffix_embeds,
                target_embeds,
            ],
            dim=1,
        )
        ignore_len = prefix_ids.numel() + visual_embeds.shape[1] + suffix_ids.numel()
        labels = torch.cat(
            [
                torch.full((ignore_len,), -100, dtype=torch.long, device=input_device),
                target_ids,
            ],
            dim=0,
        )
        per_sample_embeds.append(embeds[0])
        per_sample_labels.append(labels)
        max_len = max(max_len, embeds.shape[1])

    batch_size = len(per_sample_embeds)
    hidden_size = per_sample_embeds[0].shape[-1]
    inputs_embeds = torch.zeros(batch_size, max_len, hidden_size, device=input_device, dtype=dtype)
    attention_mask = torch.zeros(batch_size, max_len, device=input_device, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, device=input_device, dtype=torch.long)

    for idx, embeds in enumerate(per_sample_embeds):
        length = embeds.shape[0]
        inputs_embeds[idx, :length] = embeds
        attention_mask[idx, :length] = 1
        labels[idx, :length] = per_sample_labels[idx]

    return inputs_embeds, attention_mask, labels


def save_projector(
    *,
    projector,
    path: Path,
    args: argparse.Namespace,
    vision_encoder: FrozenWeatherVisionEncoder,
) -> None:
    torch.save(
        {
            "state_dict": projector.state_dict(),
            "input_dim": projector.input_dim,
            "hidden_dim": args.projector_hidden_dim,
            "output_dim": projector.output_dim,
            "num_tokens": projector.num_tokens,
            "vision_weights": args.vision_weights,
            "class_names": vision_encoder.class_names,
            "image_size": vision_encoder.image_size,
        },
        path,
    )


def save_training_artifacts(
    *,
    model,
    tokenizer,
    projector,
    save_dir: Path,
    args: argparse.Namespace,
    vision_encoder: FrozenWeatherVisionEncoder,
    state: dict,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir / "adapter")
    tokenizer.save_pretrained(save_dir / "adapter")
    save_projector(
        projector=projector,
        path=save_dir / "projector.pt",
        args=args,
        vision_encoder=vision_encoder,
    )
    with (save_dir / "train_state.json").open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


@torch.inference_mode()
def evaluate_loss(
    *,
    model,
    tokenizer,
    projector,
    vision_encoder: FrozenWeatherVisionEncoder,
    loader: DataLoader,
    prefix_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    input_device: torch.device,
    torch_dtype: torch.dtype,
) -> float:
    model_was_training = model.training
    projector_was_training = projector.training
    model.eval()
    projector.eval()

    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(input_device, dtype=torch.float32, non_blocking=True)
        features = vision_encoder(pixel_values).to(input_device, dtype=torch_dtype)
        inputs_embeds, attention_mask, labels = build_inputs_embeds_and_labels(
            model=model,
            tokenizer=tokenizer,
            projector=projector,
            vision_features=features,
            answers=batch["answer"],
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            input_device=input_device,
        )
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        total_loss += float(outputs.loss.detach().cpu())
        total_batches += 1

    if model_was_training:
        model.train()
    if projector_was_training:
        projector.train()
    return total_loss / max(1, total_batches)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = dtype_from_name(args.dtype)
    print(f"[INFO] loading tokenizer: {args.model_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    print(f"[INFO] loading Qwen: device_map={args.device_map} dtype={args.dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    input_device = first_parameter_device(model.get_input_embeddings())
    print(f"[INFO] input_device={input_device}", flush=True)

    vision_encoder = FrozenWeatherVisionEncoder(
        weights=args.vision_weights,
        device=input_device,
        freeze=True,
    )
    vision_encoder.eval()

    projector = FeatureToSoftPromptProjector(
        input_dim=vision_encoder.out_dim,
        hidden_dim=args.projector_hidden_dim,
        output_dim=model.config.hidden_size,
        num_tokens=args.num_visual_tokens,
        dropout=0.0,
    ).to(input_device, dtype=torch_dtype)
    projector.train()

    dataset = WeatherInstructionDataset(
        split_root=args.split_root,
        split="train",
        image_size=vision_encoder.image_size,
        class_names=vision_encoder.class_names,
        max_samples=args.max_train_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=weather_collate,
        pin_memory=True,
    )
    print(f"[INFO] train_samples={len(dataset)} batch_size={args.batch_size}", flush=True)

    val_dataset = None
    val_loader = None
    if args.eval_steps > 0 or args.early_stopping_patience > 0:
        val_dataset = WeatherInstructionDataset(
            split_root=args.split_root,
            split="val",
            image_size=vision_encoder.image_size,
            class_names=vision_encoder.class_names,
            max_samples=args.max_val_samples,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=weather_collate,
            pin_memory=True,
        )
        print(f"[INFO] val_samples={len(val_dataset)} eval_steps={args.eval_steps}", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad] + list(projector.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    prefix_ids, suffix_ids = build_prompt_parts(tokenizer)
    global_step = 0
    running_loss = 0.0
    best_val_loss = None
    bad_eval_count = 0
    stop_training = False
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            pixel_values = batch["pixel_values"].to(input_device, dtype=torch.float32, non_blocking=True)
            with torch.no_grad():
                features = vision_encoder(pixel_values)
            features = features.to(input_device, dtype=torch_dtype)

            inputs_embeds, attention_mask, labels = build_inputs_embeds_and_labels(
                model=model,
                tokenizer=tokenizer,
                projector=projector,
                vision_features=features,
                answers=batch["answer"],
                prefix_ids=prefix_ids,
                suffix_ids=suffix_ids,
                input_device=input_device,
            )
            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            running_loss += float(loss.detach().cpu()) * args.grad_accum_steps

            if step % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                avg_loss = running_loss / max(1, global_step)
                progress.set_postfix({"loss": f"{avg_loss:.4f}", "global_step": global_step})

                state = {
                    "global_step": global_step,
                    "epoch": epoch + 1,
                    "train_loss_avg": avg_loss,
                    "best_val_loss": best_val_loss,
                    "bad_eval_count": bad_eval_count,
                }
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_dir = output_dir / "checkpoints" / f"step_{global_step:06d}"
                    print(f"[INFO] saving checkpoint: {checkpoint_dir}", flush=True)
                    save_training_artifacts(
                        model=model,
                        tokenizer=tokenizer,
                        projector=projector,
                        save_dir=checkpoint_dir,
                        args=args,
                        vision_encoder=vision_encoder,
                        state=state,
                    )

                if args.eval_steps > 0 and val_loader is not None and global_step % args.eval_steps == 0:
                    val_loss = evaluate_loss(
                        model=model,
                        tokenizer=tokenizer,
                        projector=projector,
                        vision_encoder=vision_encoder,
                        loader=val_loader,
                        prefix_ids=prefix_ids,
                        suffix_ids=suffix_ids,
                        input_device=input_device,
                        torch_dtype=torch_dtype,
                    )
                    improved = best_val_loss is None or val_loss < best_val_loss - args.early_stopping_min_delta
                    if improved:
                        best_val_loss = val_loss
                        bad_eval_count = 0
                        best_dir = output_dir / "best"
                        print(f"[INFO] new best val_loss={val_loss:.6f}; saving {best_dir}", flush=True)
                        save_training_artifacts(
                            model=model,
                            tokenizer=tokenizer,
                            projector=projector,
                            save_dir=best_dir,
                            args=args,
                            vision_encoder=vision_encoder,
                            state={**state, "val_loss": val_loss, "best_val_loss": best_val_loss},
                        )
                    else:
                        bad_eval_count += 1
                        print(
                            f"[INFO] val_loss={val_loss:.6f}; best={best_val_loss:.6f}; "
                            f"bad_eval_count={bad_eval_count}",
                            flush=True,
                        )
                    if args.early_stopping_patience > 0 and bad_eval_count >= args.early_stopping_patience:
                        print("[INFO] early stopping triggered", flush=True)
                        stop_training = True
                        break

                if args.max_steps and global_step >= args.max_steps:
                    break

        if stop_training or (args.max_steps and global_step >= args.max_steps):
            break

    print(f"[INFO] saving LoRA adapter and projector to {output_dir}", flush=True)
    save_training_artifacts(
        model=model,
        tokenizer=tokenizer,
        projector=projector,
        save_dir=output_dir,
        args=args,
        vision_encoder=vision_encoder,
        state={
            "global_step": global_step,
            "train_loss_avg": running_loss / max(1, global_step),
            "best_val_loss": best_val_loss,
            "bad_eval_count": bad_eval_count,
            "stopped_early": stop_training,
        },
    )
    with (output_dir / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print("[INFO] done", flush=True)


if __name__ == "__main__":
    main()
