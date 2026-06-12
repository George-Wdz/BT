from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_moe.components import FeatureToSoftPromptProjector, FrozenStage1RainEncoder
from lora_moe.datasets import (
    Stage1RainInstructionDataset,
    build_stage1_metadata_prompt,
    build_stage1_rain_answer,
    stage1_rain_collate,
)
from lora_moe.train.vision_weather_lora import dtype_from_name, first_parameter_device, tokenize_no_special


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage1 rainfall retrieval projector + Qwen LoRA.")
    parser.add_argument("--model-dir", default="/home/wdz/BT/MoE/models/Qwen2.5-14B-Instruct")
    parser.add_argument(
        "--stage1-checkpoint-dir",
        default=(
            "/home/wdz/BT/Stage1/model/checkpoints/"
            "pass_dataset_rain_retrieval_compare_channels_compare_cm_cw_20260612_1140_cm/"
            "stage1_cm_dm256_df512_eh8_el3_dl2_pl8_st4_bs32_lr0.0001_itr0"
        ),
    )
    parser.add_argument("--pass-dataset-path", default="")
    parser.add_argument("--output-dir", default="/home/wdz/BT/MoE/lora-moe/outputs/stage1_rain_lora_qv_v1")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--max-val-samples", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-stage1-tokens", type=int, default=8)
    parser.add_argument("--projector-hidden-dim", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj")
    parser.add_argument("--no-rain-threshold", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_system_prompt() -> str:
    return (
        "<|im_start|>system\n"
        "你是卫星链路降雨反演助手。请根据用户给出的过境元信息和卫星链路反演专家表示，"
        "用中文给出本次卫星过境的降雨量。回答要包含卫星ID、过境起止时间和反演结论，"
        "不要提到 token、编码器、LoRA、特征向量或模型内部实现。\n"
        "<|im_end|>\n"
    )


def build_prompt_texts(batch: dict) -> tuple[list[str], list[str]]:
    prefixes = []
    suffixes = []
    system_prompt = build_system_prompt()
    for satellite_id, pass_start, pass_end, points in zip(
        batch["satellite_id"],
        batch["pass_start"],
        batch["pass_end"],
        batch["points"],
    ):
        metadata = build_stage1_metadata_prompt(
            satellite_id=int(satellite_id),
            pass_start=str(pass_start),
            pass_end=str(pass_end),
            points=int(points),
        )
        prefixes.append(f"{system_prompt}<|im_start|>user\n{metadata}\n")
        suffixes.append("\n请给出本次卫星过境的反演降雨量和简要判断。<|im_end|>\n<|im_start|>assistant\n")
    return prefixes, suffixes


def _template_index(satellite_id: int, pass_start: str, seed: int) -> int:
    raw = f"{seed}:{satellite_id}:{pass_start}".encode("utf-8")
    return zlib.crc32(raw)


def build_answers_from_stage1_prediction(
    *,
    batch: dict,
    pred_rainfall: torch.Tensor,
    rain_probability: torch.Tensor,
    no_rain_threshold: float,
    seed: int,
) -> list[str]:
    pred_values = pred_rainfall.detach().cpu().reshape(-1).tolist()
    prob_values = rain_probability.detach().cpu().reshape(-1).tolist()
    answers = []
    for idx, (satellite_id, pass_start, pass_end, points) in enumerate(
        zip(batch["satellite_id"], batch["pass_start"], batch["pass_end"], batch["points"])
    ):
        answers.append(
            build_stage1_rain_answer(
                satellite_id=int(satellite_id),
                pass_start=str(pass_start),
                pass_end=str(pass_end),
                points=int(points),
                pred_rainfall_mm=float(pred_values[idx]),
                rain_probability=float(prob_values[idx]),
                no_rain_threshold=no_rain_threshold,
                template_idx=_template_index(int(satellite_id), str(pass_start), seed),
            )
        )
    return answers


def build_inputs_embeds_and_labels(
    *,
    model,
    tokenizer,
    projector,
    stage1_features: torch.Tensor,
    prompt_prefixes: list[str],
    prompt_suffixes: list[str],
    answers: list[str],
    input_device: torch.device,
):
    embedding = model.get_input_embeddings()
    dtype = embedding.weight.dtype

    stage1_embeds = projector(stage1_features.to(input_device, dtype=next(projector.parameters()).dtype))
    stage1_embeds = stage1_embeds.to(dtype=dtype)

    per_sample_embeds = []
    per_sample_labels = []
    max_len = 0
    for idx, answer in enumerate(answers):
        prefix_ids = tokenize_no_special(tokenizer, prompt_prefixes[idx]).to(input_device)
        suffix_ids = tokenize_no_special(tokenizer, prompt_suffixes[idx]).to(input_device)
        prefix_embeds = embedding(prefix_ids).unsqueeze(0)
        suffix_embeds = embedding(suffix_ids).unsqueeze(0)
        target_ids = tokenize_no_special(tokenizer, answer + "<|im_end|>").to(input_device)
        target_embeds = embedding(target_ids).unsqueeze(0)
        embeds = torch.cat(
            [
                prefix_embeds,
                stage1_embeds[idx : idx + 1],
                suffix_embeds,
                target_embeds,
            ],
            dim=1,
        )
        ignore_len = prefix_ids.numel() + stage1_embeds.shape[1] + suffix_ids.numel()
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
    stage1_encoder: FrozenStage1RainEncoder,
) -> None:
    torch.save(
        {
            "state_dict": projector.state_dict(),
            "input_dim": projector.input_dim,
            "hidden_dim": args.projector_hidden_dim,
            "output_dim": projector.output_dim,
            "num_tokens": projector.num_tokens,
            "stage1_checkpoint_dir": args.stage1_checkpoint_dir,
            "pass_dataset_path": args.pass_dataset_path or stage1_encoder.cfg["data"]["pass_dataset_path"],
            "stage1_cfg": stage1_encoder.cfg,
            "no_rain_threshold": args.no_rain_threshold,
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
    stage1_encoder: FrozenStage1RainEncoder,
    state: dict,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir / "adapter")
    tokenizer.save_pretrained(save_dir / "adapter")
    save_projector(
        projector=projector,
        path=save_dir / "projector.pt",
        args=args,
        stage1_encoder=stage1_encoder,
    )
    with (save_dir / "train_state.json").open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


@torch.inference_mode()
def evaluate_loss(
    *,
    model,
    tokenizer,
    projector,
    stage1_encoder: FrozenStage1RainEncoder,
    loader: DataLoader,
    input_device: torch.device,
    torch_dtype: torch.dtype,
    no_rain_threshold: float,
    seed: int,
) -> float:
    model_was_training = model.training
    projector_was_training = projector.training
    model.eval()
    projector.eval()

    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        features = batch["features"].to(input_device, dtype=torch.float32, non_blocking=True)
        mask = batch["mask"].to(input_device, dtype=torch.bool, non_blocking=True)
        satellite_idx = batch["satellite_idx"].to(input_device, dtype=torch.long, non_blocking=True)
        encoded = stage1_encoder(features, mask, satellite_idx).to(input_device, dtype=torch_dtype)
        pred = stage1_encoder.predict(features, mask, satellite_idx)
        answers = build_answers_from_stage1_prediction(
            batch=batch,
            pred_rainfall=pred["rainfall_mm"],
            rain_probability=pred["rain_probability"],
            no_rain_threshold=no_rain_threshold,
            seed=seed,
        )
        prompt_prefixes, prompt_suffixes = build_prompt_texts(batch)
        inputs_embeds, attention_mask, labels = build_inputs_embeds_and_labels(
            model=model,
            tokenizer=tokenizer,
            projector=projector,
            stage1_features=encoded,
            prompt_prefixes=prompt_prefixes,
            prompt_suffixes=prompt_suffixes,
            answers=answers,
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

    stage1_encoder = FrozenStage1RainEncoder(
        checkpoint_dir=args.stage1_checkpoint_dir,
        device=input_device,
        freeze=True,
    )
    stage1_encoder.eval()

    projector = FeatureToSoftPromptProjector(
        input_dim=stage1_encoder.out_dim,
        hidden_dim=args.projector_hidden_dim,
        output_dim=model.config.hidden_size,
        num_tokens=args.num_stage1_tokens,
        dropout=0.0,
    ).to(input_device, dtype=torch_dtype)
    projector.train()

    dataset = Stage1RainInstructionDataset(
        checkpoint_dir=args.stage1_checkpoint_dir,
        split="train",
        pass_dataset_path=args.pass_dataset_path,
        max_samples=args.max_train_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=stage1_rain_collate,
        pin_memory=True,
    )
    print(f"[INFO] train_samples={len(dataset)} batch_size={args.batch_size}", flush=True)

    val_loader = None
    if args.eval_steps > 0 or args.early_stopping_patience > 0:
        val_dataset = Stage1RainInstructionDataset(
            checkpoint_dir=args.stage1_checkpoint_dir,
            split="val",
            pass_dataset_path=args.pass_dataset_path,
            max_samples=args.max_val_samples,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=stage1_rain_collate,
            pin_memory=True,
        )
        print(f"[INFO] val_samples={len(val_dataset)} eval_steps={args.eval_steps}", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad] + list(projector.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    global_step = 0
    running_loss = 0.0
    best_val_loss = None
    bad_eval_count = 0
    stop_training = False
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            features = batch["features"].to(input_device, dtype=torch.float32, non_blocking=True)
            mask = batch["mask"].to(input_device, dtype=torch.bool, non_blocking=True)
            satellite_idx = batch["satellite_idx"].to(input_device, dtype=torch.long, non_blocking=True)
            with torch.no_grad():
                encoded = stage1_encoder(features, mask, satellite_idx)
                pred = stage1_encoder.predict(features, mask, satellite_idx)
            encoded = encoded.to(input_device, dtype=torch_dtype)
            answers = build_answers_from_stage1_prediction(
                batch=batch,
                pred_rainfall=pred["rainfall_mm"],
                rain_probability=pred["rain_probability"],
                no_rain_threshold=args.no_rain_threshold,
                seed=args.seed,
            )
            prompt_prefixes, prompt_suffixes = build_prompt_texts(batch)

            inputs_embeds, attention_mask, labels = build_inputs_embeds_and_labels(
                model=model,
                tokenizer=tokenizer,
                projector=projector,
                stage1_features=encoded,
                prompt_prefixes=prompt_prefixes,
                prompt_suffixes=prompt_suffixes,
                answers=answers,
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
                        stage1_encoder=stage1_encoder,
                        state=state,
                    )

                if args.eval_steps > 0 and val_loader is not None and global_step % args.eval_steps == 0:
                    val_loss = evaluate_loss(
                        model=model,
                        tokenizer=tokenizer,
                        projector=projector,
                        stage1_encoder=stage1_encoder,
                        loader=val_loader,
                        input_device=input_device,
                        torch_dtype=torch_dtype,
                        no_rain_threshold=args.no_rain_threshold,
                        seed=args.seed,
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
                            stage1_encoder=stage1_encoder,
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

    print(f"[INFO] saving Stage1 LoRA adapter and projector to {output_dir}", flush=True)
    save_training_artifacts(
        model=model,
        tokenizer=tokenizer,
        projector=projector,
        save_dir=output_dir,
        args=args,
        stage1_encoder=stage1_encoder,
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
