#!/usr/bin/env python3
"""Train the GPT2 baseline on Stage1 rainfall retrieval NPZ datasets."""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parent
STAGE1_MODEL_ROOT = ROOT.parent / "rain_retrieval" / "model"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STAGE1_MODEL_ROOT))

from data.data_factory import data_provider, validate_feature_config
from gpt2_rain_model import GPT2RainRegressor
from utils.tools import adjust_learning_rate, test, vali


SCIENTIFIC_FLOAT = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)[eE][+-]?\d+$")


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return rank() == 0


def log(message: str) -> None:
    if is_main():
        print(message)


@contextmanager
def suppress_non_main_stdout():
    if is_main():
        yield
        return
    old_stdout = sys.stdout
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def setup_distributed() -> tuple[torch.device, int]:
    if "WORLD_SIZE" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, 0
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return device, local_rank


def cleanup_distributed() -> None:
    if is_dist():
        dist.destroy_process_group()


def _parse_override_value(raw: str):
    value = yaml.safe_load(raw)
    if isinstance(value, str) and SCIENTIFIC_FLOAT.match(value):
        return float(value)
    return value


def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set '{item}'. Expected key=value.")
        key, raw_value = item.split("=", 1)
        path = key.split(".")
        cursor = cfg
        for part in path[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                raise KeyError(f"Unknown config key '{key}'.")
            cursor = cursor[part]
        leaf = path[-1]
        if not isinstance(cursor, dict) or leaf not in cursor:
            raise KeyError(f"Unknown config key '{key}'.")
        cursor[leaf] = _parse_override_value(raw_value)


def _coerce_numeric(cfg: dict) -> None:
    d = cfg.get("data", {})
    if isinstance(d.get("rain_filter_min"), str):
        d["rain_filter_min"] = float(d["rain_filter_min"])
    sat_ids = d.get("satellite_filter_ids")
    if isinstance(sat_ids, int):
        d["satellite_filter_ids"] = [sat_ids]
    elif isinstance(sat_ids, str):
        d["satellite_filter_ids"] = [
            int(x.strip()) for x in sat_ids.split(",") if x.strip()
        ]

    b = cfg.get("dry_baseline", {})
    for key in (
        "rain_threshold",
        "image_rain_prob_threshold",
        "min_sunny_prob",
        "time_scale_hours",
        "time_weight",
        "position_weight",
    ):
        if isinstance(b.get(key), str):
            b[key] = float(b[key])

    m = cfg.get("model", {})
    for key in (
        "input_dim",
        "max_seq_len",
        "patch_len",
        "stride",
        "num_satellites",
        "sat_emb_dim",
        "gpt2_layers",
        "group_hidden_dim",
        "group_attention_heads",
        "group_attention_layers",
    ):
        if key in m and isinstance(m[key], str):
            m[key] = int(m[key])
    for key in ("dropout", "group_attention_dropout"):
        if isinstance(m.get(key), str):
            m[key] = float(m[key])

    t = cfg["training"]
    for key in (
        "lr",
        "weight_decay",
        "rainfall_loss_weight",
        "rain_threshold",
        "rainy_loss_weight",
        "rain_classification_loss_weight",
        "rain_classification_pos_weight",
        "auxiliary_loss_weight",
        "grad_clip",
        "decay_fac",
    ):
        if key in t and isinstance(t[key], str):
            t[key] = float(t[key])
    for key in ("epochs", "batch_size", "patience", "iterations", "seed", "tmax"):
        if key in t and isinstance(t[key], str):
            t[key] = int(t[key])


def compute_loss(rain_pred, aux_pred, rain_logit, labels, labels_phys, cfg):
    rain_true = labels_phys[:, 0]
    rain_threshold = cfg["training"].get("rain_threshold", 1e-6)
    rainy = (rain_true > rain_threshold).float()
    sample_weight = 1.0 + rainy * cfg["training"].get("rainy_loss_weight", 0.0)
    per_sample = nn.functional.smooth_l1_loss(
        rain_pred.squeeze(-1), rain_true, reduction="none"
    )
    loss_rain = (per_sample * sample_weight).mean()
    total = cfg["training"].get("rainfall_loss_weight", 1.0) * loss_rain

    cls_weight = cfg["training"].get("rain_classification_loss_weight", 0.0)
    if cls_weight > 0:
        pos_weight = torch.tensor(
            [cfg["training"].get("rain_classification_pos_weight", 1.0)],
            device=rain_logit.device,
        )
        loss_cls = nn.functional.binary_cross_entropy_with_logits(
            rain_logit, rainy, pos_weight=pos_weight
        )
        total = total + cls_weight * loss_cls

    if aux_pred is not None and labels.shape[-1] > 1:
        loss_aux = nn.MSELoss()(aux_pred, labels[:, 1:])
        total = total + cfg["training"].get("auxiliary_loss_weight", 1.0) * loss_aux
    return total, loss_rain


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_setting(cfg: dict, itr: int) -> str:
    m = cfg["model"]
    t = cfg["training"]
    freeze = str(m.get("freeze_gpt2", "all")).replace("/", "_")
    group = "ga" if bool(m.get("use_group_attention", False)) else "flat"
    return (
        f"gpt2_l{m.get('gpt2_layers', 'all')}_{group}_frz{freeze}"
        f"_pl{m['patch_len']}_st{m['stride']}_bs{t['batch_size']}"
        f"_lr{t['lr']}_itr{itr}"
    )


def parameter_summary_lines(model: nn.Module) -> list[str]:
    def count_params(module: nn.Module) -> tuple[int, int]:
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return trainable, total

    trainable, total = count_params(model)
    lines = [
        f"Model parameters: trainable={trainable:,} total={total:,} trainable%={100 * trainable / max(total, 1):.4f}"
    ]

    gpt2_trainable, gpt2_total = count_params(model.gpt2)
    adapter_modules = [
        ("patch_embed", model.patch_embed),
        ("summary_embed", model.summary_embed),
        ("sat_embedding", model.sat_embedding),
        ("sat_proj", model.sat_proj),
        ("head_norm", model.head_norm),
        ("rainfall_head", model.rainfall_head),
        ("rain_cls_head", model.rain_cls_head),
        ("aux_head", model.aux_head),
    ]
    adapter_trainable = 0
    adapter_total = 0
    adapter_parts = []
    for name, module in adapter_modules:
        if module is None:
            continue
        part_trainable, part_total = count_params(module)
        adapter_trainable += part_trainable
        adapter_total += part_total
        adapter_parts.append(f"{name}={part_trainable:,}/{part_total:,}")

    lines.append(
        "Parameter split: "
        f"gpt2_trainable={gpt2_trainable:,}/{gpt2_total:,}, "
        f"non_gpt2_trainable={adapter_trainable:,}/{adapter_total:,}"
    )
    lines.append("Trainable non-GPT2 modules: " + ", ".join(adapter_parts))

    gpt2_trainable_names = [
        f"{name}={param.numel():,}"
        for name, param in model.gpt2.named_parameters()
        if param.requires_grad
    ]
    if gpt2_trainable_names:
        preview = ", ".join(gpt2_trainable_names[:24])
        if len(gpt2_trainable_names) > 24:
            preview += f", ... (+{len(gpt2_trainable_names) - 24} tensors)"
        lines.append("Trainable GPT2 tensors: " + preview)
    else:
        lines.append("Trainable GPT2 tensors: none")
    return lines


def train_one_epoch(model, loader, optimizer, cfg, device):
    model.train()
    total, total_rain, n_batches = 0.0, 0.0, 0
    grad_clip = cfg["training"].get("grad_clip", 1.0)
    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device).bool()
        sat_idx = batch["satellite_idx"].to(device).long()
        labels = batch["labels"].to(device)
        labels_phys = batch["labels_phys"].to(device)

        optimizer.zero_grad()
        rain_pred, aux_pred, rain_logit = model(features, mask, sat_idx)
        loss, rain_loss = compute_loss(
            rain_pred, aux_pred, rain_logit, labels, labels_phys, cfg
        )
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=grad_clip,
            )
        optimizer.step()
        total += loss.item()
        total_rain += rain_loss.item()
        n_batches += 1
    return total / max(n_batches, 1), total_rain / max(n_batches, 1)


def reduce_mean(value: float, device: torch.device) -> float:
    if not is_dist():
        return value
    tensor = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size()
    return float(tensor.item())


def sync_bool(value: bool, device: torch.device) -> bool:
    if not is_dist():
        return value
    tensor = torch.tensor([1 if value else 0], dtype=torch.int64, device=device)
    dist.broadcast(tensor, src=0)
    return bool(tensor.item())


def ddp_train_loader(dataset, cfg: dict, seed: int):
    if not is_dist():
        return None, None
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size(),
        rank=rank(),
        shuffle=True,
        seed=seed,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        sampler=sampler,
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 0),
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, sampler


def run_one_iter(cfg: dict, itr: int, device: torch.device):
    set_seed(cfg["training"]["seed"] + itr)
    setting = build_setting(cfg, itr)
    save_dir = Path(cfg["checkpoints"]) / setting
    if is_main():
        save_dir.mkdir(parents=True, exist_ok=True)
    if is_dist():
        dist.barrier()
    log(f"\n========== itr {itr} | setting: {setting} ==========")

    with suppress_non_main_stdout():
        train_ds, train_loader, sat_mapper, scaler_X, scaler_y, split = data_provider(cfg, flag="train")
        val_ds, val_loader, _, _, _, _ = data_provider(
            cfg, flag="val", sat_mapper=sat_mapper, scaler_X=scaler_X, scaler_y=scaler_y, cached_split=split
        )
        test_ds, test_loader, _, _, _, _ = data_provider(
            cfg, flag="test", sat_mapper=sat_mapper, scaler_X=scaler_X, scaler_y=scaler_y, cached_split=split
        )
    if len(train_ds) == 0:
        log("Train set is empty, skip this iter.")
        return None
    ddp_loader, ddp_sampler = ddp_train_loader(train_ds, cfg, cfg["training"]["seed"] + itr)
    if ddp_loader is not None:
        train_loader = ddp_loader
        if is_main() and cfg["training"].get("use_rainy_sampler", False):
            log("DDP enabled: using DistributedSampler; rainy_sample_weight is still applied through the loss, not the sampler.")

    cfg["model"]["num_satellites"] = max(
        int(cfg["model"]["num_satellites"]), sat_mapper.num_satellites
    )
    log(f"Satellite mapper: {sat_mapper.num_satellites} slots")

    model = GPT2RainRegressor(cfg).to(device)
    for line in parameter_summary_lines(model):
        log(line)
    if is_dist():
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    if cfg["training"].get("use_cosine", True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["training"].get("tmax", 20), eta_min=1e-8
        )
    else:
        scheduler = None

    best_score = None
    val_loss_min = float("inf")
    patience = int(cfg["training"]["patience"])
    early_counter = 0
    early_stop = False
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        if ddp_sampler is not None:
            ddp_sampler.set_epoch(epoch)
        t0 = time.time()
        tr_loss, tr_rain = train_one_epoch(model, train_loader, optimizer, cfg, device)
        tr_loss = reduce_mean(tr_loss, device)
        tr_rain = reduce_mean(tr_rain, device)
        val_loss = vali(model, val_loader, compute_loss, cfg, device) if len(val_ds) > 0 else tr_loss
        val_loss = reduce_mean(val_loss, device)
        if scheduler is not None:
            scheduler.step()
        else:
            adjust_learning_rate(optimizer, epoch, cfg)
        lr = optimizer.param_groups[0]["lr"]
        log(
            f"Epoch {epoch:03d} | Train {tr_loss:.4f} (rain {tr_rain:.4f}) "
            f"| Val {val_loss:.4f} | LR {lr:.2e} | {time.time() - t0:.1f}s"
        )
        if is_main():
            score = -val_loss
            if best_score is None or score >= best_score:
                if best_score is None:
                    old = "inf"
                else:
                    old = f"{val_loss_min:.6f}"
                print(f"Validation loss decreased ({old} --> {val_loss:.6f}). Saving model ...")
                state_model = model.module if isinstance(model, DDP) else model
                torch.save(state_model.state_dict(), save_dir / "checkpoint.pth")
                best_score = score
                val_loss_min = val_loss
                early_counter = 0
            else:
                early_counter += 1
                print(f"EarlyStopping counter: {early_counter} out of {patience}")
                early_stop = early_counter >= patience
        early_stop = sync_bool(early_stop, device)
        if early_stop:
            log("Early stopping triggered.")
            break

    ckpt = save_dir / "checkpoint.pth"
    if is_dist():
        dist.barrier()
    state_model = model.module if isinstance(model, DDP) else model
    if ckpt.exists():
        state_model.load_state_dict(torch.load(ckpt, map_location=device))
        log(f"Loaded best checkpoint from {ckpt}")
    else:
        log("No checkpoint found, evaluating last-epoch model.")

    with suppress_non_main_stdout():
        results = test(state_model, test_loader, scaler_y, cfg, device) if len(test_ds) > 0 else {}
    if is_main():
        torch.save(
            {
                "cfg": cfg,
                "scaler_X": scaler_X,
                "scaler_y": scaler_y,
                "sat_mapper": sat_mapper.id_to_idx,
                "model_type": "gpt2_rain",
            },
            save_dir / "meta.pt",
        )
    if is_dist():
        dist.barrier()
    return itr, setting, results, save_dir


def main(cfg_path: str, overrides: list[str] | None = None, dry_run: bool = False) -> None:
    device, local_rank = setup_distributed()
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    apply_overrides(cfg, overrides or [])
    _coerce_numeric(cfg)
    validate_feature_config(cfg)

    log(f"Using device: {device}")
    if is_dist():
        log(f"DDP enabled: world_size={world_size()} local_rank={local_rank}")
    if overrides and is_main():
        log("Applied config overrides:")
        for item in overrides:
            print(f"  {item}")
    if dry_run:
        if is_main():
            print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        return

    all_results = []
    for itr in range(cfg["training"].get("iterations", 1)):
        result = run_one_iter(cfg, itr, device)
        if result:
            all_results.append(result)
    if not all_results:
        log("No iterations produced results.")
        return

    if not is_main():
        return

    print("\n========== Summary across iterations ==========")
    summary_path = Path(cfg["checkpoints"]) / "iteration_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "setting", "target", "mae", "mse", "checkpoint_dir"])
        writer.writeheader()
        for iteration, setting, metrics, ckpt_dir in all_results:
            for target, values in metrics.items():
                writer.writerow(
                    {
                        "iteration": iteration,
                        "setting": setting,
                        "target": target,
                        "mae": values.get("mae"),
                        "mse": values.get("mse"),
                        "checkpoint_dir": str(ckpt_dir),
                    }
                )
    target_names = list(all_results[0][2].keys())
    for target in target_names:
        vals = np.array([r[2][target]["mae"] for r in all_results], dtype=np.float64)
        mses = np.array([r[2][target]["mse"] for r in all_results], dtype=np.float64)
        print(f"[{target}] mae={vals.mean():.4f}±{vals.std():.4f} mse={mses.mean():.4f}±{mses.std():.4f}")
    print(f"Saved iteration summary to {summary_path}")

    rain_rows = [
        (metrics["pass_rainfall_mm"]["mae"], ckpt_dir)
        for _, _, metrics, ckpt_dir in all_results
        if "pass_rainfall_mm" in metrics
    ]
    if rain_rows:
        best_mae, best_ckpt = min(rain_rows, key=lambda x: x[0])
        best_path = Path(cfg["checkpoints"]) / "best_iteration_checkpoint.txt"
        best_path.write_text(str(best_ckpt) + "\n")
        print(f"best_iteration_checkpoint={best_ckpt} target=pass_rainfall_mm mae={best_mae:.7g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        main(args.config, args.set, args.dry_run)
    finally:
        cleanup_distributed()
