"""
Stage1 训练入口：Pass-based Patch Encoder-Decoder Transformer

工程结构对齐 Stage2/GPT4TS/Long-term_Forecasting/main.py：
- 多 itr loop（每个 itr 独立 seed/scaler/sat_mapper）
- train / vali / test 三阶段
- EarlyStopping + 学习率调度
- 反标准化后报告 pass_rainfall_mm/wind_speed/wind_direction 的 MAE/MSE
"""
import sys
import time
import random
import argparse
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from data.data_factory import data_provider
from models.patch_encoder_decoder import PatchEncoderDecoder
from utils.tools import EarlyStopping, adjust_learning_rate, vali, test


SCIENTIFIC_FLOAT = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)[eE][+-]?\d+$")


def compute_loss(rain_pred, aux_pred, rain_logit, labels, labels_phys, cfg):
    """
    rain_pred: (B, 1)
    aux_pred:  (B, n_aux) or None
    labels:    standardized labels, used for auxiliary targets
    labels_phys: physical labels, used for rainfall in mm
    """
    w = cfg["training"]["rainfall_loss_weight"]
    rain_true = labels_phys[:, 0]
    rain_threshold = cfg["training"].get("rain_threshold", 1e-6)
    rainy = (rain_true > rain_threshold).float()
    sample_weight = 1.0 + rainy * cfg["training"].get("rainy_loss_weight", 0.0)
    per_sample = nn.functional.smooth_l1_loss(
        rain_pred.squeeze(-1), rain_true, reduction="none"
    )
    loss_rain = (per_sample * sample_weight).mean()
    total = w * loss_rain
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_setting(cfg: dict, itr: int) -> str:
    m = cfg["model"]
    t = cfg["training"]
    chan = "ca" if m.get("use_channel_attention", False) else "cm"
    return (
        f"stage1_{chan}_dm{m['d_model']}_df{m['d_ff']}_eh{m['n_heads']}"
        f"_el{m['e_layers']}_dl{m['d_layers']}_pl{m['patch_len']}_st{m['stride']}"
        f"_bs{t['batch_size']}_lr{t['lr']}_itr{itr}"
    )


def train_one_epoch(model, loader, optimizer, cfg, device):
    model.train()
    total, total_rain, n = 0.0, 0.0, 0
    grad_clip = cfg["training"].get("grad_clip", 1.0)
    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total += loss.item()
        total_rain += rain_loss.item()
        n += 1
    return total / max(n, 1), total_rain / max(n, 1)


def run_one_iter(cfg: dict, itr: int, device):
    set_seed(cfg["training"]["seed"] + itr)

    setting = build_setting(cfg, itr)
    save_dir = Path(cfg["checkpoints"]) / setting
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== itr {itr} | setting: {setting} ==========")

    # 1) train / val / test loader（train 阶段拟合 scaler 和 sat_mapper）
    train_ds, train_loader, sat_mapper, scaler_X, scaler_y, split = data_provider(
        cfg, flag="train"
    )
    val_ds, val_loader, _, _, _, _ = data_provider(
        cfg, flag="val", sat_mapper=sat_mapper,
        scaler_X=scaler_X, scaler_y=scaler_y, cached_split=split,
    )
    test_ds, test_loader, _, _, _, _ = data_provider(
        cfg, flag="test", sat_mapper=sat_mapper,
        scaler_X=scaler_X, scaler_y=scaler_y, cached_split=split,
    )

    if len(train_ds) == 0:
        print("Train set is empty, skip this iter.")
        return None

    # 同步 mapper 大小到 cfg（确保 embedding 表足够大）
    cfg["model"]["num_satellites"] = max(
        cfg["model"]["num_satellites"], sat_mapper.num_satellites
    )
    print(f"Satellite mapper: {sat_mapper.num_satellites} slots "
          f"(1 unknown + {len(sat_mapper.id_to_idx)} known)")

    # 2) model & optimizer
    model = PatchEncoderDecoder(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    if cfg["training"].get("use_cosine", True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["training"].get("tmax", 20), eta_min=1e-8,
        )
    else:
        scheduler = None

    early_stopping = EarlyStopping(
        patience=cfg["training"]["patience"], verbose=True,
    )

    # 3) train loop
    epochs = cfg["training"]["epochs"]
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_rain = train_one_epoch(model, train_loader, optimizer, cfg, device)

        if len(val_ds) > 0:
            val_loss = vali(model, val_loader, compute_loss, cfg, device)
        else:
            val_loss = tr_loss  # 验证集为空时退化到训练 loss（避免崩溃）

        if scheduler is not None:
            scheduler.step()
        else:
            adjust_learning_rate(optimizer, epoch, cfg)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | Train {tr_loss:.4f} (rain {tr_rain:.4f}) "
              f"| Val {val_loss:.4f} | LR {lr:.2e} | {time.time()-t0:.1f}s")

        early_stopping(val_loss, model, str(save_dir))
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # 4) test：加载 best checkpoint
    ckpt = save_dir / "checkpoint.pth"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded best checkpoint from {ckpt}")
    else:
        print("No checkpoint found, evaluating last-epoch model.")

    if len(test_ds) > 0:
        results = test(model, test_loader, scaler_y, cfg, device)
    else:
        print("Test set empty, skipping test.")
        results = {}

    # 保存附加信息（scaler / sat_mapper）便于后续推理
    torch.save({
        "cfg": cfg,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "sat_mapper": sat_mapper.id_to_idx,
    }, save_dir / "meta.pt")

    return results


def _coerce_numeric(cfg: dict):
    """YAML 1.2 不会把 '1e-4' 当成 float。手动把训练超参里的字符串数值转换。"""
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
    for k in (
        "rain_threshold",
        "image_rain_prob_threshold",
        "time_scale_hours",
        "time_weight",
        "position_weight",
    ):
        if isinstance(b.get(k), str):
            b[k] = float(b[k])

    t = cfg["training"]
    for k in (
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
        if k in t and isinstance(t[k], str):
            t[k] = float(t[k])
    for k in ("epochs", "batch_size", "warmup_epochs", "patience",
              "iterations", "seed", "tmax"):
        if k in t and isinstance(t[k], str):
            t[k] = int(t[k])


def _parse_override_value(raw: str):
    """Parse --set values with YAML semantics and scientific-float fallback."""
    value = yaml.safe_load(raw)
    if isinstance(value, str) and SCIENTIFIC_FLOAT.match(value):
        return float(value)
    return value


def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """Apply dotted-key overrides, e.g. training.lr=5e-5.

    Only existing keys are accepted. This keeps experiment commands explicit and
    catches typos such as `train.lr` before a long training run starts.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set '{item}'. Expected key=value.")
        key, raw_value = item.split("=", 1)
        path = key.split(".")
        if any(not part for part in path):
            raise ValueError(f"Invalid --set key '{key}'.")

        cursor = cfg
        for part in path[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                raise KeyError(f"Unknown config key '{key}'.")
            cursor = cursor[part]

        leaf = path[-1]
        if not isinstance(cursor, dict) or leaf not in cursor:
            raise KeyError(f"Unknown config key '{key}'.")
        cursor[leaf] = _parse_override_value(raw_value)


def main(cfg_path: str, overrides: list[str] | None = None, dry_run: bool = False):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    overrides = overrides or []
    apply_overrides(cfg, overrides)
    _coerce_numeric(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if overrides:
        print("Applied config overrides:")
        for item in overrides:
            print(f"  {item}")
    if dry_run:
        print("\nResolved config:")
        print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        return

    iters = cfg["training"].get("iterations", 1)
    all_results = []
    for ii in range(iters):
        res = run_one_iter(cfg, ii, device)
        if res:
            all_results.append(res)

    if not all_results:
        print("No iterations produced results.")
        return

    # 汇总：按 target 聚合 mae/mse 的均值与方差
    print("\n========== Summary across iterations ==========")
    target_names = list(all_results[0].keys())
    metric_names = ["mae", "mse"]
    for tn in target_names:
        line = [f"[{tn}]"]
        for mn in metric_names:
            vals = np.array([r[tn][mn] for r in all_results])
            line.append(f"{mn}={vals.mean():.4f}±{vals.std():.4f}")
        print(" ".join(line))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).parent / "configs" / "default.yaml"))
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override an existing YAML config key, e.g. --set training.lr=5e-5",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved config and exit before training.")
    args = parser.parse_args()
    main(args.config, args.set, args.dry_run)
