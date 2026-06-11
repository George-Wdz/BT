"""
训练辅助工具：早停、学习率调度、验证、测试。

参考 Stage2/GPT4TS/Long-term_Forecasting/utils/tools.py，
针对 Stage1 的 batch dict 形式（features/mask/satellite_idx/labels）
和多目标输出（pass_rainfall_mm + wind_speed + wind_direction）做了改造。
"""
import numpy as np
import torch
from .metrics import metric


def adjust_learning_rate(optimizer, epoch, cfg):
    lradj = cfg["training"].get("lradj", "type1")
    base_lr = cfg["training"]["lr"]
    decay_fac = cfg["training"].get("decay_fac", 0.9)

    if lradj == "type1":
        lr_adjust = {epoch: base_lr if epoch < 3 else base_lr * (decay_fac ** ((epoch - 3) // 1))}
    elif lradj == "type2":
        lr_adjust = {epoch: base_lr * (decay_fac ** ((epoch - 1) // 1))}
    elif lradj == "type4":
        lr_adjust = {epoch: base_lr * (decay_fac ** (epoch // 1))}
    else:
        lr_adjust = {epoch: base_lr if epoch < 3 else base_lr * (0.9 ** ((epoch - 3) // 1))}

    if epoch in lr_adjust:
        new_lr = lr_adjust[epoch]
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr
        print(f"Updating learning rate to {new_lr}")


class EarlyStopping:
    """按 val_loss 保存最优模型；连续 patience 轮无改进则触发早停。"""

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), f"{path}/checkpoint.pth")
        self.val_loss_min = val_loss


def _move_batch(batch, device):
    return {
        "features": batch["features"].to(device),
        "mask": batch["mask"].to(device),
        "satellite_idx": batch["satellite_idx"].to(device).long(),
        "labels": batch["labels"].to(device),
        "labels_phys": batch["labels_phys"].to(device),
    }


def vali(model, vali_loader, loss_fn, cfg, device):
    """验证：返回平均总损失（与训练损失同口径，用于 early stopping）。"""
    model.eval()
    total_loss = []
    with torch.no_grad():
        for batch in vali_loader:
            b = _move_batch(batch, device)
            rain_pred, aux_pred, rain_logit = model(
                b["features"], b["mask"], b["satellite_idx"]
            )
            loss, _ = loss_fn(
                rain_pred, aux_pred, rain_logit, b["labels"], b["labels_phys"], cfg
            )
            total_loss.append(loss.item())
    model.train()
    return float(np.average(total_loss)) if total_loss else float("inf")


def test(model, test_loader, scaler_y, cfg, device):
    """
    测试：收集 (rain_pred, aux_pred, labels)，反标准化后逐目标计算指标。
    返回 dict: {target_name: {'mae','mse','rmse','mape','mspe','smape','nd'}}
    """
    model.eval()
    rains, auxs, ys = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            b = _move_batch(batch, device)
            rain_pred, aux_pred, rain_logit = model(
                b["features"], b["mask"], b["satellite_idx"]
            )
            rains.append(rain_pred.detach().cpu().numpy())            # (B, 1)
            if aux_pred is not None:
                auxs.append(aux_pred.detach().cpu().numpy())          # (B, 2)
            ys.append(b["labels"].detach().cpu().numpy())             # (B, 3)

    if not rains:
        print("test loader empty")
        return {}

    rain_pred_phys = np.concatenate(rains, axis=0)                     # (N, 1), already mm
    y_norm = np.concatenate(ys, axis=0)                                # (N, 3)
    if auxs:
        aux_pred = np.concatenate(auxs, axis=0)                        # (N, 2)
        pred_norm = np.concatenate(
            [np.zeros((aux_pred.shape[0], 1), dtype=aux_pred.dtype), aux_pred],
            axis=1,
        )
    else:
        pred_norm = None

    # 反标准化到物理量
    true_phys = scaler_y.inverse_transform(y_norm)
    if pred_norm is not None:
        pred_aux_phys = scaler_y.inverse_transform(pred_norm)
        pred_phys = pred_aux_phys.copy()
        pred_phys[:, 0] = rain_pred_phys.squeeze(-1)
    else:
        pred_phys = np.zeros_like(true_phys)
        pred_phys[:, 0] = rain_pred_phys.squeeze(-1)

    target_names = list(cfg["targets"]["primary"]) + list(cfg["targets"].get("auxiliary", []))
    results = {}
    for i, name in enumerate(target_names):
        p = pred_phys[:, i]
        t = true_phys[:, i]
        mae, mse = metric(p, t)
        results[name] = {"mae": mae, "mse": mse}
        print(f"[{name}] mae={mae:.4f} mse={mse:.4f}")
    return results
