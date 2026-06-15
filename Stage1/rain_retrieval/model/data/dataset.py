"""
Pass-based PyTorch Dataset

每个样本是一个卫星过境片段（pass），包含：
- satellite_id: 卫星ID（用于 embedding）
- features: 链路 + 位置 + 地面气象的拼接特征序列 (T, 13)
- length: 真实序列长度（用于 mask）
- labels: 过境时段的气象标签 (3,) [pass_rainfall_mm, wind_speed, wind_direction]

由于不同过境长度不同，使用 padding 对齐到固定长度，配合 attention mask
告诉模型哪些位置是真实数据、哪些是填充。
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import List, Dict


# 默认输入特征维度: link(6) + position(6) + ground_weather(3) = 15
# 可选 image_weather(4): prob_sunny/prob_cloudy/prob_rain/image_available
FEATURE_DIMS = {"link": 6, "position": 6, "ground_weather": 3}
INPUT_DIM = sum(FEATURE_DIMS.values())

BASE_LABEL_INDEX = {
    "pass_rainfall_mm": 0,
    "wind_speed": 1,
    "wind_direction": 2,
}
META_LABEL_KEYS = {
    "rain_rate_mean",
    "rain_rate_max",
    "rainy_ratio",
}


class SatelliteIDMapper:
    """将卫星ID映射为连续索引（含未知ID的冷启动槽位）"""

    UNKNOWN_IDX = 0  # 索引 0 保留给未知卫星

    def __init__(self, known_ids: List[int]):
        # 索引 0 = unknown，从 1 开始分配已知卫星
        self.id_to_idx = {sat_id: i + 1 for i, sat_id in enumerate(sorted(known_ids))}
        self.num_satellites = len(known_ids) + 1  # +1 for unknown

    def __call__(self, sat_id: int) -> int:
        return self.id_to_idx.get(sat_id, self.UNKNOWN_IDX)


class PassDataset(Dataset):
    """以卫星过境片段为基本单元的 Dataset。

    构造时仅接受已切分好的 passes 子集（train/val/test 之一）。
    train 集合传 fit_scalers=True 拟合 scaler；val/test 复用 train 的 scaler。
    """

    def __init__(self, passes: List[Dict], sat_mapper: SatelliteIDMapper,
                 max_len: int = 256,
                 scaler_X: StandardScaler = None,
                 scaler_y: StandardScaler = None,
                 fit_scalers: bool = False,
                 extra_feature_keys: List[str] = None,
                 target_names: List[str] = None):
        self.max_len = max_len
        self.sat_mapper = sat_mapper
        self.extra_feature_keys = extra_feature_keys or []
        self.target_names = target_names or [
            "pass_rainfall_mm", "wind_speed", "wind_direction"
        ]

        # 拼接每个 pass 的特征序列
        self.features = []   # List of (T_i, 13)
        self.lengths = []
        self.sat_indices = []
        labels_list = []

        for p in passes:
            parts = [
                p["link_features"],         # (T, 6)
                p["position_features"],     # (T, 6)
                p["ground_weather"],        # (T, 3)
            ]
            if "image_weather" in p:
                parts.append(p["image_weather"])  # (T, 4)
            for key in self.extra_feature_keys:
                if key not in p:
                    raise KeyError(f"pass is missing optional feature group: {key}")
                parts.append(p[key])
            feat = np.concatenate(parts, axis=1).astype(np.float32)   # (T, C)
            # 截断超长过境
            if len(feat) > max_len:
                feat = feat[:max_len]
            self.features.append(feat)
            self.lengths.append(len(feat))
            self.sat_indices.append(sat_mapper(p["satellite_id"]))
            labels_list.append(self._select_labels(p))

        self.labels_phys = np.stack(labels_list).astype(np.float32)  # (N, 3)

        # 标准化（按特征维度，使用所有真实step）
        all_feat = np.concatenate(self.features, axis=0)  # (sum_T, 13)
        if fit_scalers:
            self.scaler_X = StandardScaler().fit(all_feat)
            self.scaler_y = StandardScaler().fit(self.labels_phys)
        else:
            assert scaler_X is not None and scaler_y is not None, \
                "val/test 必须传入 train 集拟合的 scaler_X / scaler_y"
            self.scaler_X = scaler_X
            self.scaler_y = scaler_y

        # 应用标准化
        self.features = [self.scaler_X.transform(f).astype(np.float32)
                         for f in self.features]
        self.labels = self.scaler_y.transform(self.labels_phys).astype(np.float32)
        self.input_dim = self.features[0].shape[1] if self.features else INPUT_DIM

    def _select_labels(self, p: Dict) -> np.ndarray:
        base = np.asarray(p["labels"], dtype=np.float32)
        meta = p.get("label_meta", {})
        values = []
        for name in self.target_names:
            if name in BASE_LABEL_INDEX:
                idx = BASE_LABEL_INDEX[name]
                if idx >= len(base):
                    raise KeyError(f"base label '{name}' is unavailable in this pass")
                value = float(base[idx])
            elif name in META_LABEL_KEYS:
                if name not in meta:
                    raise KeyError(f"label_meta is missing auxiliary target: {name}")
                value = float(meta[name])
            else:
                raise KeyError(f"unknown target name: {name}")
            if not np.isfinite(value):
                value = 0.0
            values.append(value)
        return np.asarray(values, dtype=np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        T = len(feat)
        # Padding 到 max_len，未填充位置为 0
        padded = np.zeros((self.max_len, self.input_dim), dtype=np.float32)
        padded[:T] = feat
        # mask: True = 真实数据, False = padding
        mask = np.zeros(self.max_len, dtype=bool)
        mask[:T] = True

        return {
            "features": torch.from_numpy(padded),     # (max_len, 13)
            "mask": torch.from_numpy(mask),            # (max_len,)
            "length": T,
            "satellite_idx": self.sat_indices[idx],
            "labels": torch.from_numpy(self.labels[idx]),  # (3,)
            "labels_phys": torch.from_numpy(self.labels_phys[idx]),  # (3,)
        }
