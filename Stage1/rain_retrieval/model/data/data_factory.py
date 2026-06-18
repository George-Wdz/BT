"""
Stage1 数据入口工厂。

参考 Stage2/GPT4TS/Long-term_Forecasting/data_provider/data_factory.py 的接口，
但适配 pass-based 数据：
- 第一次调用（flag='train'）：加载/构建全部 passes，按时间排序后顺序切 train/val/test，
  在 train 子集上拟合 scaler_X / scaler_y，并基于 train 子集生成 sat_mapper。
- 后续调用（flag='val'/'test'）：必须传入 train 阶段返回的 sat_mapper、scaler_X、scaler_y。
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import PassDataset, SatelliteIDMapper
from data.preprocessing import build_pass_dataset


_PASSES_CACHE = {}  # 按数据路径和过滤条件缓存已加载的 passes，避免重复 I/O

FEATURE_GROUP_KEYS = {
    "link": "link_features",
    "position": "position_features",
    "ground_weather": "ground_weather",
    "image_weather": "image_weather",
    "dry_delta": "link_dry_delta",
    "dry_delta_summary": "link_dry_delta_summary",
}

FEATURE_GROUP_DIMS = {
    "position": 6,
    "ground_weather": 3,
    "image_weather": 4,
}


def _parse_satellite_filter(raw_ids) -> set[int]:
    if raw_ids in (None, "", []):
        return set()
    if isinstance(raw_ids, str):
        raw_ids = [x.strip() for x in raw_ids.split(",") if x.strip()]
    return {int(x) for x in raw_ids}


def _parse_feature_groups(raw_groups) -> list[str]:
    if raw_groups in (None, "", []):
        return ["link", "position", "ground_weather", "image_weather", "dry_delta"]
    if isinstance(raw_groups, str):
        text = raw_groups.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        raw_groups = [x.strip() for x in text.split(",") if x.strip()]
    groups = [str(g).strip() for g in raw_groups if str(g).strip()]
    unknown = [g for g in groups if g not in FEATURE_GROUP_KEYS]
    if unknown:
        raise ValueError(f"Unknown feature groups: {unknown}. Available: {sorted(FEATURE_GROUP_KEYS)}")
    return groups


def enabled_feature_groups(cfg: dict) -> list[str]:
    return _parse_feature_groups(cfg.get("features", {}).get("enabled_groups"))


def feature_group_dims(cfg: dict, groups: list[str] | None = None) -> list[int]:
    groups = groups or enabled_feature_groups(cfg)
    dims = []
    link_dim = len(cfg.get("features", {}).get("link", []))
    if link_dim <= 0:
        link_dim = int(cfg["model"]["feature_group_dims"][0])
    for group in groups:
        if group in ("link", "dry_delta"):
            dims.append(link_dim)
        elif group == "dry_delta_summary":
            dims.append(link_dim * 6)
        else:
            dims.append(FEATURE_GROUP_DIMS[group])
    return dims


def validate_feature_config(cfg: dict) -> None:
    groups = enabled_feature_groups(cfg)
    dims = feature_group_dims(cfg, groups)
    input_dim = int(cfg["model"]["input_dim"])
    configured_dims = list(cfg["model"].get("feature_group_dims", []))
    if sum(dims) != input_dim:
        raise ValueError(
            f"features.enabled_groups={groups} imply input_dim={sum(dims)}, "
            f"but model.input_dim={input_dim}"
        )
    if configured_dims and list(configured_dims) != dims:
        raise ValueError(
            f"features.enabled_groups={groups} imply feature_group_dims={dims}, "
            f"but model.feature_group_dims={configured_dims}"
        )
    if "dry_delta" in groups and not cfg.get("dry_baseline", {}).get("enabled", False):
        raise ValueError("features.enabled_groups includes dry_delta but dry_baseline.enabled is false")
    if "dry_delta_summary" in groups and not cfg.get("dry_baseline", {}).get("add_summary", False):
        raise ValueError("features.enabled_groups includes dry_delta_summary but dry_baseline.add_summary is false")
    if "image_weather" in groups and not cfg.get("image_weather", {}).get("enabled", False):
        raise ValueError("features.enabled_groups includes image_weather but image_weather.enabled is false")


def _optional_feature_keys(cfg: dict) -> list[str]:
    keys = []
    baseline_cfg = cfg.get("dry_baseline", {})
    groups = enabled_feature_groups(cfg)
    if baseline_cfg.get("enabled", False):
        if "dry_delta" in groups:
            keys.append("link_dry_delta")
        if "dry_delta_summary" in groups and baseline_cfg.get("add_summary", False):
            keys.append("link_dry_delta_summary")
    return keys


def _copy_pass_with_feature(p: Dict, key: str, value: np.ndarray) -> Dict:
    q = dict(p)
    q[key] = value.astype(np.float32)
    return q


def _pass_center(p: Dict) -> pd.Timestamp:
    timestamps = pd.DatetimeIndex(p["timestamps"])
    return timestamps[0] + (timestamps[-1] - timestamps[0]) / 2


def _safe_time_hours(a: pd.Timestamp, b: pd.Timestamp) -> float:
    return abs((a - b).total_seconds()) / 3600.0


def _mean_vector(p: Dict, key: str) -> np.ndarray:
    arr = np.asarray(p[key], dtype=np.float32)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(arr, axis=0)
    return np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _delta_summary_features(delta: np.ndarray) -> np.ndarray:
    """Repeat pass-level delta statistics at each timestep."""
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(delta, axis=0)
        std = np.nanstd(delta, axis=0)
        min_v = np.nanmin(delta, axis=0)
        max_v = np.nanmax(delta, axis=0)
    range_v = max_v - min_v
    if len(delta) > 1:
        slope = (delta[-1] - delta[0]) / float(len(delta) - 1)
    else:
        slope = np.zeros(delta.shape[1], dtype=np.float32)
    stats = np.concatenate([mean, std, min_v, max_v, range_v, slope], axis=0)
    stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.repeat(stats.reshape(1, -1), repeats=len(delta), axis=0)


def _image_rain_probability(p: Dict) -> float | None:
    if "image_weather" not in p:
        return None
    image = np.asarray(p["image_weather"], dtype=np.float32)
    if image.ndim != 2 or image.shape[1] < 3:
        return None
    return float(np.nanmean(image[:, 2]))


def _is_dry_baseline_candidate(p: Dict, baseline_cfg: dict, threshold: float) -> bool:
    """Return whether a train pass can be used to estimate clear-sky link state."""
    if float(p["labels"][0]) > threshold:
        return False

    if baseline_cfg.get("exclude_instant_rain", True):
        meta = p.get("label_meta", {})
        rain_rate_max = float(meta.get("rain_rate_max", 0.0) or 0.0)
        if rain_rate_max > threshold:
            return False

    if baseline_cfg.get("exclude_image_rain", True):
        prob = _image_rain_probability(p)
        image_available = int(p.get("label_meta", {}).get("image_available", 0) or 0)
        if image_available and prob is not None:
            limit = float(baseline_cfg.get("image_rain_prob_threshold", 0.2))
            if prob >= limit:
                return False
    return True


def attach_train_dry_baseline(
    train_passes: List[Dict],
    val_passes: List[Dict],
    test_passes: List[Dict],
    cfg: dict,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Append same-satellite dry-baseline delta features.

    Baselines are computed from train dry passes only. Val/test labels are not
    used to build baseline values.
    """
    baseline_cfg = cfg.get("dry_baseline", {})
    if not baseline_cfg.get("enabled", False):
        return train_passes, val_passes, test_passes

    threshold = float(baseline_cfg.get("rain_threshold", cfg["training"].get("rain_threshold", 1e-6)))
    method = str(baseline_cfg.get("method", "mean")).lower()
    add_summary = bool(baseline_cfg.get("add_summary", False))
    link_dim = len(cfg.get("features", {}).get("link", []))
    if link_dim <= 0:
        link_dim = int(cfg["model"]["feature_group_dims"][0])

    dry_train = [
        p for p in train_passes
        if _is_dry_baseline_candidate(p, baseline_cfg, threshold)
    ]
    if not dry_train:
        print("Dry baseline disabled at runtime: no dry train passes available.")
        return train_passes, val_passes, test_passes

    def add_features(p: Dict, baseline: np.ndarray) -> Dict:
        link = np.asarray(p["link_features"], dtype=np.float32)
        if link.shape[1] != link_dim:
            raise ValueError(
                f"dry_baseline link_dim mismatch: link_features has {link.shape[1]}, "
                f"config expects {link_dim}"
            )
        delta = link - baseline.reshape(1, -1)
        q = _copy_pass_with_feature(p, "link_dry_delta", delta)
        if add_summary:
            q["link_dry_delta_summary"] = _delta_summary_features(delta).astype(np.float32)
        return q

    if method == "mean":
        by_sat: dict[int, list[np.ndarray]] = {}
        global_parts = []
        for p in dry_train:
            link = np.asarray(p["link_features"], dtype=np.float32)
            by_sat.setdefault(int(p["satellite_id"]), []).append(link)
            global_parts.append(link)

        global_baseline = np.concatenate(global_parts, axis=0).mean(axis=0)
        sat_baseline = {
            sat_id: np.concatenate(parts, axis=0).mean(axis=0)
            for sat_id, parts in by_sat.items()
        }

        def apply_mean(split: List[Dict]) -> List[Dict]:
            out = []
            missing_sat = 0
            for p in split:
                baseline = sat_baseline.get(int(p["satellite_id"]))
                if baseline is None:
                    baseline = global_baseline
                    missing_sat += 1
                out.append(add_features(p, baseline))
            if missing_sat:
                print(f"Dry baseline used global fallback for {missing_sat} passes")
            return out

        print(
            f"Attached train dry baseline deltas: method=mean, satellites={len(sat_baseline)}, "
            f"link_dim={link_dim}, add_summary={add_summary}, candidates={len(dry_train)}, "
            f"threshold={threshold:g}"
        )
        return apply_mean(train_passes), apply_mean(val_passes), apply_mean(test_passes)

    if method != "matched":
        raise ValueError(f"Unsupported dry_baseline.method: {method}")

    candidates = []
    for p in dry_train:
        candidates.append({
            "pass": p,
            "satellite_id": int(p["satellite_id"]),
            "center": _pass_center(p),
            "link_mean": _mean_vector(p, "link_features"),
            "position_mean": _mean_vector(p, "position_features"),
        })

    pos_matrix = np.stack([c["position_mean"] for c in candidates], axis=0)
    pos_center = pos_matrix.mean(axis=0)
    pos_scale = pos_matrix.std(axis=0)
    pos_scale[pos_scale < 1e-6] = 1.0
    for c in candidates:
        c["position_z"] = (c["position_mean"] - pos_center) / pos_scale

    by_sat_candidates: dict[int, list[dict]] = {}
    for c in candidates:
        by_sat_candidates.setdefault(c["satellite_id"], []).append(c)

    time_scale = max(float(baseline_cfg.get("time_scale_hours", 72.0)), 1e-6)
    time_weight = float(baseline_cfg.get("time_weight", 1.0))
    position_weight = float(baseline_cfg.get("position_weight", 1.0))
    global_baseline = np.stack([c["link_mean"] for c in candidates], axis=0).mean(axis=0)

    def select_baseline(p: Dict) -> tuple[np.ndarray, bool]:
        sat_id = int(p["satellite_id"])
        pool = by_sat_candidates.get(sat_id, candidates)
        used_global = sat_id not in by_sat_candidates
        if len(pool) > 1:
            filtered = [c for c in pool if c["pass"] is not p]
            if filtered:
                pool = filtered
        center = _pass_center(p)
        pos_z = (_mean_vector(p, "position_features") - pos_center) / pos_scale
        best = None
        best_score = None
        norm = max(np.sqrt(len(pos_z)), 1.0)
        for c in pool:
            time_score = _safe_time_hours(center, c["center"]) / time_scale
            position_score = float(np.linalg.norm(pos_z - c["position_z"]) / norm)
            score = time_weight * time_score + position_weight * position_score
            if best_score is None or score < best_score:
                best = c
                best_score = score
        if best is None:
            return global_baseline, True
        return best["link_mean"], used_global

    def apply_matched(split: List[Dict]) -> List[Dict]:
        out = []
        missing_sat = 0
        for p in split:
            baseline, used_global = select_baseline(p)
            if used_global:
                missing_sat += 1
            out.append(add_features(p, baseline))
        if missing_sat:
            print(f"Dry baseline used global fallback for {missing_sat} passes")
        return out

    print(
        f"Attached train dry baseline deltas: method=matched, satellites={len(by_sat_candidates)}, "
        f"dry_candidates={len(candidates)}, link_dim={link_dim}, add_summary={add_summary}, "
        f"time_scale_hours={time_scale:g}, time_weight={time_weight:g}, "
        f"position_weight={position_weight:g}, threshold={threshold:g}"
    )
    return apply_matched(train_passes), apply_matched(val_passes), apply_matched(test_passes)


def load_all_passes(cfg: dict) -> List[Dict]:
    """加载或构建全部 passes（首次调用做 build，之后从 npz 缓存读）。"""
    pass_path = cfg["data"]["pass_dataset_path"]
    min_rainfall = float(cfg["data"].get("rain_filter_min", 0.0) or 0.0)
    satellite_ids = _parse_satellite_filter(cfg["data"].get("satellite_filter_ids"))
    cache_key = (pass_path, min_rainfall, tuple(sorted(satellite_ids)))
    if cache_key in _PASSES_CACHE:
        return _PASSES_CACHE[cache_key]

    if Path(pass_path).exists():
        print(f"Loading cached pass dataset: {pass_path}")
        npz = np.load(pass_path, allow_pickle=True)
        passes = list(npz["passes"])
    else:
        passes = build_pass_dataset(
            db_path=cfg["data"]["db_path"],
            output_path=pass_path,
            feature_cols=cfg.get("features"),
            strict_source_filters=cfg["data"].get("strict_source_filters", False),
            image_weather_cfg=cfg.get("image_weather"),
        )

    if min_rainfall > 0:
        before = len(passes)
        passes = [p for p in passes if float(p["labels"][0]) > min_rainfall]
        print(f"Filtered passes by rain_filter_min>{min_rainfall:g}: "
              f"{len(passes)} / {before} kept")

    if satellite_ids:
        before = len(passes)
        passes = [p for p in passes if int(p["satellite_id"]) in satellite_ids]
        print(f"Filtered passes by satellite_filter_ids={sorted(satellite_ids)}: "
              f"{len(passes)} / {before} kept")

    _PASSES_CACHE[cache_key] = passes
    return passes


def split_passes_by_time(
    passes: List[Dict],
    split: List[float],
    val_strategy: str = "time",
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """按 pass 起始时间排序后顺序切分为 train/val/test。"""
    assert abs(sum(split) - 1.0) < 1e-6, f"data_split must sum to 1, got {split}"

    def start_ts(p):
        ts = p["timestamps"][0]
        return pd.Timestamp(ts)

    sorted_passes = sorted(passes, key=start_ts)
    n = len(sorted_passes)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    if val_strategy == "stratified_all":
        rainy = [p for p in sorted_passes if float(p["labels"][0]) > 1e-6]
        dry = [p for p in sorted_passes if float(p["labels"][0]) <= 1e-6]
        rng = np.random.default_rng(seed)
        rng.shuffle(rainy)
        rng.shuffle(dry)

        def split_group(group: List[Dict]) -> tuple[list[Dict], list[Dict], list[Dict]]:
            n_group = len(group)
            n_group_train = int(n_group * split[0])
            n_group_val = int(n_group * split[1])
            return (
                group[:n_group_train],
                group[n_group_train:n_group_train + n_group_val],
                group[n_group_train + n_group_val:],
            )

        rainy_train, rainy_val, rainy_test = split_group(rainy)
        dry_train, dry_val, dry_test = split_group(dry)
        train = sorted(rainy_train + dry_train, key=start_ts)
        val = sorted(rainy_val + dry_val, key=start_ts)
        test = sorted(rainy_test + dry_test, key=start_ts)
    elif val_strategy == "stratified_before_test":
        test = sorted_passes[n_train + n_val:]
        pool = sorted_passes[:n_train + n_val]
        rainy = [p for p in pool if float(p["labels"][0]) > 1e-6]
        dry = [p for p in pool if float(p["labels"][0]) <= 1e-6]
        rng = np.random.default_rng(seed)
        rng.shuffle(rainy)
        rng.shuffle(dry)
        n_val_rainy = min(len(rainy), max(1, round(n_val * len(rainy) / max(len(pool), 1))))
        n_val_dry = n_val - n_val_rainy
        val = rainy[:n_val_rainy] + dry[:n_val_dry]
        train = rainy[n_val_rainy:] + dry[n_val_dry:]
        train = sorted(train, key=start_ts)
        val = sorted(val, key=start_ts)
    else:
        train = sorted_passes[:n_train]
        val = sorted_passes[n_train:n_train + n_val]
        test = sorted_passes[n_train + n_val:]
    return train, val, test


def data_provider(cfg: dict,
                  flag: str,
                  sat_mapper: Optional[SatelliteIDMapper] = None,
                  scaler_X=None, scaler_y=None,
                  cached_split: Optional[Tuple[List, List, List]] = None):
    """
    构建一个数据集和对应的 DataLoader。

    Args:
        cfg: 配置字典
        flag: 'train' / 'val' / 'test'
        sat_mapper / scaler_X / scaler_y: val/test 阶段必传，由 train 阶段生成
        cached_split: 可选预切分结果 (train_passes, val_passes, test_passes)，避免重复排序

    Returns:
        dataset, loader, sat_mapper, scaler_X, scaler_y, split_passes
        其中 split_passes 是 (train, val, test) 元组，可用于后续调用复用。
    """
    assert flag in ("train", "val", "test"), f"unknown flag: {flag}"

    # 切分（首次调用做，之后复用）
    if cached_split is None:
        all_passes = load_all_passes(cfg)
        train_passes, val_passes, test_passes = split_passes_by_time(
            all_passes,
            cfg["data"]["data_split"],
            val_strategy=cfg["data"].get("val_strategy", "time"),
            seed=cfg["training"].get("seed", 42),
        )
        train_passes, val_passes, test_passes = attach_train_dry_baseline(
            train_passes, val_passes, test_passes, cfg
        )
        print(f"Split: total={len(all_passes)}, "
              f"train={len(train_passes)}, val={len(val_passes)}, test={len(test_passes)}")
    else:
        train_passes, val_passes, test_passes = cached_split

    flag_passes = {"train": train_passes, "val": val_passes, "test": test_passes}[flag]

    # train 阶段：拟合 scaler 和 sat_mapper
    if flag == "train":
        known_ids = sorted(set(int(p["satellite_id"]) for p in flag_passes))
        sat_mapper = SatelliteIDMapper(known_ids)
        fit = True
    else:
        assert sat_mapper is not None and scaler_X is not None and scaler_y is not None, \
            f"flag='{flag}' 必须传入 train 阶段生成的 sat_mapper / scaler_X / scaler_y"
        fit = False

    dataset = PassDataset(
        flag_passes, sat_mapper,
        max_len=cfg["model"]["max_seq_len"],
        scaler_X=scaler_X, scaler_y=scaler_y, fit_scalers=fit,
        extra_feature_keys=_optional_feature_keys(cfg),
        feature_groups=enabled_feature_groups(cfg),
        target_names=list(cfg["targets"]["primary"]) + list(cfg["targets"].get("auxiliary", [])),
    )

    if flag == "train":
        scaler_X = dataset.scaler_X
        scaler_y = dataset.scaler_y

    sampler = None
    shuffle = (flag == "train")
    if flag == "train" and cfg["training"].get("use_rainy_sampler", False):
        rainy_weight = cfg["training"].get("rainy_sample_weight", 1.0)
        weights = [
            rainy_weight if float(p["labels"][0]) > cfg["training"].get("rain_threshold", 1e-6) else 1.0
            for p in flag_passes
        ]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    drop_last = False  # pass 数量少，不丢
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg["data"].get("num_workers", 0),
        drop_last=drop_last,
    )

    print(f"[{flag}] {len(dataset)} samples, {len(loader)} batches")
    return dataset, loader, sat_mapper, scaler_X, scaler_y, (train_passes, val_passes, test_passes)
