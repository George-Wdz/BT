from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_moe.components import FeatureToSoftPromptProjector, FrozenStage1RainEncoder
from lora_moe.datasets import (
    build_stage1_metadata_prompt,
    normalize_stage1_rainfall,
    stage1_rain_collate,
    stage1_rain_level,
)
from lora_moe.train.stage1_rain_lora import dtype_from_name


STAGE1_MODEL_ROOT = Path("/home/wdz/BT/Stage1/rain_retrieval/model")
if str(STAGE1_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE1_MODEL_ROOT))

from data.data_factory import (  # noqa: E402
    _delta_summary_features,
    _image_rain_probability,
    _is_dry_baseline_candidate,
    _mean_vector,
    _optional_feature_keys,
    _pass_center,
    split_passes_by_time,
)
from data.dataset import PassDataset, SatelliteIDMapper  # noqa: E402
from data.db import load_ground_weather, load_weather_station  # noqa: E402
from data.preprocessing import (  # noqa: E402
    IMAGE_WEATHER_COLS,
    load_image_weather_predictions,
    merge_ground_weather,
    segment_passes,
)


DEFAULT_MODEL_DIR = "/home/wdz/BT/MoE/models/Qwen2.5-14B-Instruct"
DEFAULT_OUTPUT_DIR = "/home/wdz/BT/MoE/lora-moe/outputs/stage1_rain_lora_a2b2_v1"
DEFAULT_SENSOR_DB_PATH = "/home/wdz/satellite_data/satellite_data.db"
DEFAULT_IMAGE_WEATHER_CSV = "/home/wdz/BT/Stage1/rain_retrieval/data/camera_labels/latest_weather_labels_slim.csv"


class GenerateRequest(BaseModel):
    prompt: str = "请根据最新卫星链路反演结果，说明当前降雨情况。"
    max_new_tokens: int = Field(default=80, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0)
    task_mode: str = "auto"


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    full_text: str
    input_tokens: int
    output_tokens: int
    modality_status: str
    stage1_inversion: dict
    artifacts: dict


def token_ids(tokenizer, text: str) -> torch.Tensor:
    return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def resolve_artifacts(output_dir: str, adapter_dir: str, projector_path: str, use_best: bool) -> tuple[Path, Path]:
    root = Path(output_dir).expanduser()
    adapter = Path(adapter_dir).expanduser() if adapter_dir else root / ("best/adapter" if use_best else "adapter")
    projector = (
        Path(projector_path).expanduser()
        if projector_path
        else root / ("best/projector.pt" if use_best else "projector.pt")
    )
    if not adapter.exists():
        raise FileNotFoundError(f"adapter not found: {adapter}")
    if not projector.exists():
        raise FileNotFoundError(f"projector not found: {projector}")
    return adapter, projector


def latest_db_time(db_path: Path) -> Optional[datetime]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        row = conn.execute("SELECT max(localTime) FROM phy_data").fetchone()
    if not row or not row[0]:
        return None
    return datetime.fromisoformat(str(row[0]).replace("Z", "+00:00")).replace(tzinfo=None)


def read_recent_phy(db_path: Path, link_cols: list[str], lookback_hours: float) -> pd.DataFrame:
    select_cols = ", ".join(link_cols)
    predicates = " AND ".join(f"{col} IS NOT NULL" for col in link_cols)
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        latest = pd.read_sql_query("SELECT max(localTime) AS latest FROM phy_data", conn)["latest"].iloc[0]
        if latest is None:
            return pd.DataFrame()
        latest_ts = pd.to_datetime(latest, format="ISO8601")
        start_ts = latest_ts - pd.Timedelta(hours=lookback_hours)
        query = f"""
            SELECT localTime, satelliteId, earthStationId, {select_cols}
            FROM phy_data
            WHERE localTime >= ? AND {predicates}
            ORDER BY localTime
        """
        df = pd.read_sql_query(query, conn, params=[start_ts.isoformat()])
    if df.empty:
        return df
    df["earthStationId"] = 0
    df["localTime"] = pd.to_datetime(df["localTime"], format="ISO8601")
    return df.set_index("localTime").sort_index()


def read_recent_position(
    db_path: Path,
    pos_cols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    select_cols = ", ".join(pos_cols)
    start = start - pd.Timedelta(minutes=10)
    end = end + pd.Timedelta(minutes=10)
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        query = f"""
            SELECT localTime, satId, {select_cols}
            FROM position_data
            WHERE localTime >= ? AND localTime <= ?
            ORDER BY localTime
        """
        df = pd.read_sql_query(query, conn, params=[start.isoformat(), end.isoformat()])
    if df.empty:
        return df
    df["localTime"] = pd.to_datetime(df["localTime"], format="ISO8601")
    return df.set_index("localTime")[pos_cols].sort_index()


class Stage1DryBaseline:
    """Frozen dry-baseline state built once from the checkpoint training split."""

    def __init__(self, cfg: dict, pass_dataset_path: str):
        self.cfg = cfg
        self.pass_dataset_path = Path(pass_dataset_path).expanduser()
        self.method = str(cfg.get("dry_baseline", {}).get("method", "mean")).lower()
        self.enabled = bool(cfg.get("dry_baseline", {}).get("enabled", False))
        self.add_summary = bool(cfg.get("dry_baseline", {}).get("add_summary", False))
        self.link_dim = len(cfg.get("features", {}).get("link", []))
        if self.link_dim <= 0:
            self.link_dim = int(cfg["model"]["feature_group_dims"][0])
        self.state = self._build()

    def _build(self) -> dict:
        if not self.enabled:
            return {"mode": "disabled"}
        npz = np.load(self.pass_dataset_path, allow_pickle=True)
        passes = list(npz["passes"])
        train_passes, _, _ = split_passes_by_time(
            passes,
            self.cfg["data"]["data_split"],
            val_strategy=self.cfg["data"].get("val_strategy", "time"),
            seed=self.cfg["training"].get("seed", 42),
        )
        baseline_cfg = self.cfg.get("dry_baseline", {})
        threshold = float(baseline_cfg.get("rain_threshold", self.cfg["training"].get("rain_threshold", 1e-6)))
        dry_train = [
            p for p in train_passes
            if _is_dry_baseline_candidate(p, baseline_cfg, threshold)
        ]
        if not dry_train:
            return {"mode": "disabled", "reason": "no_dry_train_passes"}

        if self.method == "mean":
            by_sat: dict[int, list[np.ndarray]] = {}
            global_parts = []
            for p in dry_train:
                link = np.asarray(p["link_features"], dtype=np.float32)
                by_sat.setdefault(int(p["satellite_id"]), []).append(link)
                global_parts.append(link)
            global_baseline = np.concatenate(global_parts, axis=0).mean(axis=0).astype(np.float32)
            sat_baseline = {
                sat_id: np.concatenate(parts, axis=0).mean(axis=0).astype(np.float32)
                for sat_id, parts in by_sat.items()
            }
            return {
                "mode": "mean",
                "threshold": threshold,
                "dry_candidates": len(dry_train),
                "satellites": len(sat_baseline),
                "global_baseline": global_baseline,
                "sat_baseline": sat_baseline,
            }

        if self.method == "matched":
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
            return {
                "mode": "matched",
                "threshold": threshold,
                "dry_candidates": len(dry_train),
                "satellites": len(by_sat_candidates),
                "candidates": candidates,
                "by_sat_candidates": by_sat_candidates,
                "global_baseline": np.stack([c["link_mean"] for c in candidates], axis=0).mean(axis=0).astype(np.float32),
                "pos_center": pos_center,
                "pos_scale": pos_scale,
                "time_scale_hours": max(float(baseline_cfg.get("time_scale_hours", 72.0)), 1e-6),
                "time_weight": float(baseline_cfg.get("time_weight", 1.0)),
                "position_weight": float(baseline_cfg.get("position_weight", 1.0)),
            }

        raise ValueError(f"Unsupported dry_baseline.method for online service: {self.method}")

    def describe(self) -> dict:
        state = self.state
        return {
            "enabled": self.enabled,
            "mode": state.get("mode"),
            "method": self.method,
            "add_summary": self.add_summary,
            "link_dim": self.link_dim,
            "dry_candidates": state.get("dry_candidates"),
            "satellites": state.get("satellites"),
            "pass_dataset_path": str(self.pass_dataset_path),
        }

    def _select_mean_baseline(self, p: dict) -> tuple[np.ndarray, bool]:
        sat_id = int(p["satellite_id"])
        baseline = self.state["sat_baseline"].get(sat_id)
        if baseline is None:
            return self.state["global_baseline"], True
        return baseline, False

    def _select_matched_baseline(self, p: dict) -> tuple[np.ndarray, bool]:
        sat_id = int(p["satellite_id"])
        pool = self.state["by_sat_candidates"].get(sat_id, self.state["candidates"])
        used_global = sat_id not in self.state["by_sat_candidates"]
        center = _pass_center(p)
        pos_z = (_mean_vector(p, "position_features") - self.state["pos_center"]) / self.state["pos_scale"]
        best = None
        best_score = None
        norm = max(np.sqrt(len(pos_z)), 1.0)
        for c in pool:
            time_score = abs((center - c["center"]).total_seconds()) / 3600.0 / self.state["time_scale_hours"]
            position_score = float(np.linalg.norm(pos_z - c["position_z"]) / norm)
            score = self.state["time_weight"] * time_score + self.state["position_weight"] * position_score
            if best_score is None or score < best_score:
                best = c
                best_score = score
        if best is None:
            return self.state["global_baseline"], True
        return best["link_mean"], used_global

    def apply(self, passes: list[dict]) -> list[dict]:
        if not self.enabled or self.state.get("mode") == "disabled":
            return passes
        out = []
        for p in passes:
            if self.state["mode"] == "mean":
                baseline, _ = self._select_mean_baseline(p)
            else:
                baseline, _ = self._select_matched_baseline(p)
            link = np.asarray(p["link_features"], dtype=np.float32)
            delta = link - baseline.reshape(1, -1)
            q = dict(p)
            q["link_dry_delta"] = delta.astype(np.float32)
            if self.add_summary:
                q["link_dry_delta_summary"] = _delta_summary_features(delta).astype(np.float32)
            out.append(q)
        return out


class Stage1OnlineFeatureBuilder:
    def __init__(
        self,
        *,
        cfg: dict,
        meta: dict,
        db_path: str,
        image_weather_csv: str,
        image_tolerance: str,
        dry_baseline: Stage1DryBaseline,
    ) -> None:
        self.cfg = cfg
        self.meta = meta
        self.db_path = Path(db_path).expanduser()
        self.image_weather_csv = Path(image_weather_csv).expanduser() if image_weather_csv else None
        self.image_tolerance = image_tolerance
        self.dry_baseline = dry_baseline
        self._image_cache_mtime = None
        self._image_cache = None

        mapper = SatelliteIDMapper([])
        mapper.id_to_idx = {int(k): int(v) for k, v in meta["sat_mapper"].items()}
        mapper.num_satellites = max(mapper.id_to_idx.values(), default=0) + 1
        self.sat_mapper = mapper

    def _image_weather(self):
        if not self.image_weather_csv or not self.image_weather_csv.exists():
            return None
        mtime = self.image_weather_csv.stat().st_mtime
        if self._image_cache is None or self._image_cache_mtime != mtime:
            self._image_cache = load_image_weather_predictions(str(self.image_weather_csv))
            self._image_cache_mtime = mtime
        return self._image_cache

    def _attach_online_features(self, passes: list[dict]) -> list[dict]:
        if not passes:
            return []
        weather_cols = list(self.cfg.get("features", {}).get("ground_weather", ["temperature", "humidity", "pressure"]))
        start = min(pd.DatetimeIndex(p["timestamps"])[0] for p in passes) - pd.Timedelta(minutes=10)
        end = max(pd.DatetimeIndex(p["timestamps"])[-1] for p in passes) + pd.Timedelta(minutes=10)
        gw = load_ground_weather(str(self.db_path), start_time=start.isoformat(), end_time=end.isoformat())
        ws = load_weather_station(str(self.db_path), start_time=start.isoformat(), end_time=end.isoformat())
        ground_weather = merge_ground_weather(gw, ws)
        image_weather = self._image_weather()
        image_tol = pd.Timedelta(self.image_tolerance)

        out = []
        for p in passes:
            idx = pd.DatetimeIndex(p["timestamps"])
            if ground_weather.empty:
                continue
            gw_aligned = ground_weather[weather_cols].reindex(idx, method="nearest", tolerance=pd.Timedelta("60s"))
            if gw_aligned.isna().any().any():
                gw_aligned = gw_aligned.ffill().bfill()
            if gw_aligned.isna().any().any():
                continue

            meta = {
                "pass_start": idx[0],
                "pass_end": idx[-1],
                "weather_rows": 0,
                "rain_rate_mean": 0.0,
                "rain_rate_max": 0.0,
                "rainy_ratio": 0.0,
            }
            if not ws.empty:
                ws_in_range = ws.loc[idx[0] : idx[-1]]
                if len(ws_in_range):
                    rain_rate = pd.to_numeric(ws_in_range["rainfall"], errors="coerce")
                    meta.update({
                        "weather_rows": int(len(ws_in_range)),
                        "rain_rate_mean": float(rain_rate.mean()) if rain_rate.notna().any() else 0.0,
                        "rain_rate_max": float(rain_rate.max()) if rain_rate.notna().any() else 0.0,
                        "rainy_ratio": float((rain_rate > 0).mean()) if rain_rate.notna().any() else 0.0,
                    })

            image_vec = np.zeros(len(IMAGE_WEATHER_COLS), dtype=np.float32)
            if image_weather is not None and not image_weather.empty:
                center = idx[0] + (idx[-1] - idx[0]) / 2
                nearest_pos = image_weather.index.get_indexer([center], method="nearest", tolerance=image_tol)[0]
                if nearest_pos >= 0:
                    row = image_weather.iloc[nearest_pos]
                    image_vec = row[IMAGE_WEATHER_COLS].to_numpy(dtype=np.float32)
                    meta["image_available"] = 1
                    meta["image_time_delta_s"] = abs((image_weather.index[nearest_pos] - center).total_seconds())
                else:
                    meta["image_available"] = 0
                    meta["image_time_delta_s"] = None

            out.append({
                **p,
                "ground_weather": gw_aligned.values.astype(np.float32),
                "image_weather": np.repeat(image_vec.reshape(1, -1), len(idx), axis=0).astype(np.float32),
                "labels": np.zeros(3, dtype=np.float32),
                "label_meta": meta,
            })
        return out

    def build_latest_passes(
        self,
        *,
        lookback_hours: float,
        pass_gap_threshold_s: float,
        min_pass_points: int,
        max_passes: int,
    ) -> list[dict]:
        feature_cfg = self.cfg.get("features", {})
        link_cols = list(feature_cfg.get("link", ["phyRssi", "rssi", "snr", "lastCniValue"]))
        pos_cols = list(feature_cfg.get("position", [
            "longitude", "latitude", "satAltitude", "posLongitude", "posLatitude", "altitude"
        ]))
        phy = read_recent_phy(self.db_path, link_cols, lookback_hours)
        if phy.empty:
            return []
        pos = read_recent_position(self.db_path, pos_cols, phy.index.min(), phy.index.max())
        if pos.empty:
            return []

        passes = segment_passes(
            phy,
            pos,
            link_cols=link_cols,
            pos_cols=pos_cols,
            gap_threshold=pass_gap_threshold_s,
            min_points=min_pass_points,
        )
        passes = sorted(passes, key=lambda p: pd.DatetimeIndex(p["timestamps"])[-1], reverse=True)
        if max_passes > 0:
            passes = passes[:max_passes]
        passes = self._attach_online_features(passes)
        return self.dry_baseline.apply(passes)

    def to_dataset(self, passes: list[dict]) -> PassDataset:
        return PassDataset(
            passes,
            self.sat_mapper,
            max_len=self.cfg["model"]["max_seq_len"],
            scaler_X=self.meta["scaler_X"],
            scaler_y=self.meta["scaler_y"],
            fit_scalers=False,
            extra_feature_keys=_optional_feature_keys(self.cfg),
            target_names=list(self.cfg["targets"]["primary"]) + list(self.cfg["targets"].get("auxiliary", [])),
        )


class Stage1RainServiceRunner:
    def __init__(
        self,
        *,
        model_dir: str,
        output_dir: str,
        adapter_dir: str,
        projector_path: str,
        use_best: bool,
        db_path: str,
        image_weather_csv: str,
        image_tolerance: str,
        host_device_map: str,
        dtype: str,
        poll_interval_s: float,
        stale_after_s: float,
        lookback_hours: float,
        max_passes: int,
        pass_gap_threshold_s: float,
        min_pass_points: int,
        no_rain_threshold: float,
    ) -> None:
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.adapter_dir, self.projector_path = resolve_artifacts(output_dir, adapter_dir, projector_path, use_best)
        self.db_path = Path(db_path).expanduser()
        self.image_weather_csv = image_weather_csv
        self.image_tolerance = image_tolerance
        self.device_map = host_device_map
        self.dtype = dtype
        self.poll_interval_s = poll_interval_s
        self.stale_after_s = stale_after_s
        self.lookback_hours = lookback_hours
        self.max_passes = max_passes
        self.pass_gap_threshold_s = pass_gap_threshold_s
        self.min_pass_points = min_pass_points
        self.no_rain_threshold = no_rain_threshold
        self.lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.cache = self._no_latest_state(None, "not_started")

        torch_dtype = dtype_from_name(dtype)
        print(f"Loading tokenizer from {model_dir}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        print(f"Loading Qwen base from {model_dir} with device_map={host_device_map}, dtype={dtype}...", flush=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=host_device_map,
            low_cpu_mem_usage=True,
        )
        print(f"Loading Stage1 A2B2 LoRA adapter from {self.adapter_dir}...", flush=True)
        self.model = PeftModel.from_pretrained(base_model, self.adapter_dir)
        self.model.eval()
        self.input_device = next(self.model.get_input_embeddings().parameters()).device

        projector_ckpt = torch.load(self.projector_path, map_location="cpu")
        self.projector = FeatureToSoftPromptProjector(
            input_dim=int(projector_ckpt["input_dim"]),
            hidden_dim=int(projector_ckpt["hidden_dim"]),
            output_dim=self.model.config.hidden_size,
            num_tokens=int(projector_ckpt["num_tokens"]),
            dropout=0.0,
        )
        self.projector.load_state_dict(projector_ckpt["state_dict"])
        self.projector.to(self.input_device, dtype=torch_dtype)
        self.projector.eval()
        self.num_stage1_tokens = int(projector_ckpt["num_tokens"])
        self.stage1_checkpoint_dir = str(projector_ckpt["stage1_checkpoint_dir"])
        self.pass_dataset_path = str(projector_ckpt.get("pass_dataset_path") or "")

        print(f"Loading frozen Stage1 encoder from {self.stage1_checkpoint_dir}...", flush=True)
        self.stage1_encoder = FrozenStage1RainEncoder(
            checkpoint_dir=self.stage1_checkpoint_dir,
            device=self.input_device,
            freeze=True,
        )
        self.stage1_encoder.eval()
        if not self.pass_dataset_path:
            self.pass_dataset_path = self.stage1_encoder.cfg["data"]["pass_dataset_path"]

        print("Building frozen dry-baseline state...", flush=True)
        self.dry_baseline = Stage1DryBaseline(self.stage1_encoder.cfg, self.pass_dataset_path)
        self.feature_builder = Stage1OnlineFeatureBuilder(
            cfg=self.stage1_encoder.cfg,
            meta=self.stage1_encoder.meta,
            db_path=str(self.db_path),
            image_weather_csv=image_weather_csv,
            image_tolerance=image_tolerance,
            dry_baseline=self.dry_baseline,
        )
        print(f"Dry baseline: {self.dry_baseline.describe()}", flush=True)
        print("Stage1 rainfall LoRA service loaded.", flush=True)

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, name="stage1-rain-worker", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def _set_cache(self, state: dict) -> None:
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        with self.lock:
            self.cache = state

    def _no_latest_state(self, latest_time: Optional[datetime], reason: str) -> dict:
        age_s = None
        if latest_time is not None:
            age_s = max(0.0, (datetime.now() - latest_time).total_seconds())
        return {
            "status": "no_latest_satellite_pass",
            "message": "无最新卫星过境",
            "reason": reason,
            "latest_phy_time": latest_time.isoformat(sep=" ", timespec="seconds") if latest_time else None,
            "latest_phy_age_s": round(age_s, 3) if age_s is not None else None,
            "pred_rainfall_mm": None,
            "reported_rainfall_mm": None,
            "recent_passes": [],
        }

    @torch.inference_mode()
    def update_once(self) -> dict:
        latest_time = latest_db_time(self.db_path)
        if latest_time is None:
            return self._no_latest_state(None, "phy_data_empty")
        age_s = (datetime.now() - latest_time).total_seconds()
        if age_s > self.stale_after_s:
            return self._no_latest_state(latest_time, "phy_data_stale")

        passes = self.feature_builder.build_latest_passes(
            lookback_hours=self.lookback_hours,
            pass_gap_threshold_s=self.pass_gap_threshold_s,
            min_pass_points=self.min_pass_points,
            max_passes=self.max_passes,
        )
        if not passes:
            return self._no_latest_state(latest_time, "no_valid_pass_after_preprocessing")

        dataset = self.feature_builder.to_dataset(passes)
        batch = stage1_rain_collate([dataset[i] | {
            "satellite_id": int(passes[i]["satellite_id"]),
            "pass_start": str(pd.DatetimeIndex(passes[i]["timestamps"])[0]),
            "pass_end": str(pd.DatetimeIndex(passes[i]["timestamps"])[-1]),
            "points": int(len(pd.DatetimeIndex(passes[i]["timestamps"]))),
            "true_rainfall_mm": 0.0,
            "answer": "",
        } for i in range(len(dataset))])
        features = batch["features"].to(self.input_device, dtype=torch.float32)
        mask = batch["mask"].to(self.input_device, dtype=torch.bool)
        satellite_idx = batch["satellite_idx"].to(self.input_device, dtype=torch.long)

        stage1_features = self.stage1_encoder(features, mask, satellite_idx)
        stage1_embeds = self.projector(stage1_features.to(self.input_device, dtype=next(self.projector.parameters()).dtype))
        stage1_embeds = stage1_embeds.to(dtype=self.model.get_input_embeddings().weight.dtype)
        pred = self.stage1_encoder.predict(features, mask, satellite_idx)
        rainfall = pred["rainfall_mm"].detach().cpu().numpy().reshape(-1).astype(float)
        rain_prob = pred["rain_probability"].detach().cpu().numpy().reshape(-1).astype(float)

        rows = []
        for p, y, prob in zip(passes, rainfall, rain_prob):
            ts = pd.DatetimeIndex(p["timestamps"])
            image = p.get("image_weather")
            raw_rainfall = float(y)
            display_rainfall = normalize_stage1_rainfall(raw_rainfall, self.no_rain_threshold)
            row = {
                "satellite_id": int(p["satellite_id"]),
                "pass_start": str(ts[0]),
                "pass_end": str(ts[-1]),
                "points": int(len(ts)),
                "pred_rainfall_mm": round(raw_rainfall, 6),
                "reported_rainfall_mm": round(float(display_rainfall), 6),
                "rainfall_level": stage1_rain_level(raw_rainfall, self.no_rain_threshold),
                "rain_probability": round(float(prob), 6),
            }
            if image is not None:
                image = np.asarray(image, dtype=np.float64)
                if image.ndim == 2 and image.shape[1] >= 4:
                    row.update({
                        "prob_sunny": round(float(np.nanmean(image[:, 0])), 6),
                        "prob_cloudy": round(float(np.nanmean(image[:, 1])), 6),
                        "prob_rain": round(float(np.nanmean(image[:, 2])), 6),
                        "image_available": bool(np.nanmax(image[:, 3]) > 0),
                    })
            rows.append(row)

        return {
            "status": "active",
            "message": "最新卫星过境反演已更新",
            "latest_phy_time": latest_time.isoformat(sep=" ", timespec="seconds"),
            "latest_phy_age_s": round(max(0.0, age_s), 3),
            "pred_rainfall_mm": rows[0]["pred_rainfall_mm"],
            "reported_rainfall_mm": rows[0]["reported_rainfall_mm"],
            "rainfall_level": rows[0]["rainfall_level"],
            "rain_probability": rows[0]["rain_probability"],
            "recent_passes": rows,
            "stage1_soft_embeds": stage1_embeds[:1].detach(),
        }

    def _run_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._set_cache(self.update_once())
            except Exception as exc:
                self._set_cache({
                    "status": "error",
                    "message": "链路反演失败",
                    "error": repr(exc),
                    "pred_rainfall_mm": None,
                    "reported_rainfall_mm": None,
                    "recent_passes": [],
                })
            self.stop_event.wait(self.poll_interval_s)

    def latest(self) -> dict:
        with self.lock:
            state = dict(self.cache)
        state.pop("stage1_soft_embeds", None)
        return state

    def _route_stage1(self, prompt: str, task_mode: str) -> bool:
        mode = (task_mode or "auto").strip().lower()
        if mode in ("stage1", "inversion", "rain", "rainfall"):
            return True
        if mode in ("text", "none", "off"):
            return False
        text = prompt.lower()
        keywords = ("stage1", "反演", "链路", "当前降雨", "当前雨量", "rainfall inversion", "link")
        return any(k in text for k in keywords)

    def _prompt_embeds(self, prompt: str, stage1_embeds: torch.Tensor, latest_pass: dict):
        user_prompt = prompt.strip() or "请根据最新卫星链路反演结果，说明当前降雨情况。"
        metadata = build_stage1_metadata_prompt(
            satellite_id=int(latest_pass["satellite_id"]),
            pass_start=str(latest_pass["pass_start"]),
            pass_end=str(latest_pass["pass_end"]),
            points=int(latest_pass["points"]),
        )
        prefix = (
            "<|im_start|>system\n"
            "你是卫星链路降雨反演助手。你会收到最新过境元信息和卫星链路反演专家表示。"
            "请基于专家表示回答本次过境降雨量，回答要包含卫星ID、过境起止时间和反演结论。"
            "不要展开推理依据，不要补充没有依据的天气描述。"
            "不要提到 token、编码器、LoRA、特征向量或模型内部实现。除非用户明确要求英文，否则使用中文。\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{metadata}\n"
        )
        suffix = f"\n用户问题：{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        embedding = self.model.get_input_embeddings()
        prefix_ids = token_ids(self.tokenizer, prefix).to(self.input_device)
        suffix_ids = token_ids(self.tokenizer, suffix).to(self.input_device)
        prefix_embeds = embedding(prefix_ids).unsqueeze(0)
        suffix_embeds = embedding(suffix_ids).unsqueeze(0)
        inputs_embeds = torch.cat([prefix_embeds, stage1_embeds.to(self.input_device), suffix_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=self.input_device, dtype=torch.long)
        return inputs_embeds, attention_mask, prefix + suffix

    @torch.inference_mode()
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        use_stage1 = self._route_stage1(request.prompt, request.task_mode)
        state = self.cache
        if not use_stage1:
            state = {"status": "text_only", "message": "未触发卫星链路反演专家"}
        elif state.get("status") == "not_started":
            state = self.update_once()
            self._set_cache(state)

        if use_stage1 and state.get("status") == "no_latest_satellite_pass":
            latest = self.latest()
            return GenerateResponse(
                prompt=request.prompt,
                generated_text="无最新卫星过境",
                full_text="无最新卫星过境",
                input_tokens=0,
                output_tokens=0,
                modality_status="stage1_no_latest_satellite_pass",
                stage1_inversion=latest,
                artifacts=self.artifacts(),
            )
        if use_stage1 and state.get("status") == "error":
            latest = self.latest()
            return GenerateResponse(
                prompt=request.prompt,
                generated_text="链路反演失败",
                full_text="链路反演失败",
                input_tokens=0,
                output_tokens=0,
                modality_status="stage1_error",
                stage1_inversion=latest,
                artifacts=self.artifacts(),
            )

        do_sample = request.temperature > 0
        if use_stage1:
            stage1_embeds = state["stage1_soft_embeds"]
            latest_pass = (state.get("recent_passes") or [{}])[0]
            inputs_embeds, attention_mask, model_prompt = self._prompt_embeds(
                request.prompt,
                stage1_embeds,
                latest_pass,
            )
            with self.model_lock:
                output_ids = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=request.max_new_tokens,
                    do_sample=do_sample,
                    temperature=request.temperature if do_sample else None,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            input_len = inputs_embeds.shape[1]
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            full_text = generated_text
            modality = "stage1_encoder_projector_lora_connected"
        else:
            prompt = request.prompt.strip()
            prompt_for_model = prompt
            if getattr(self.tokenizer, "chat_template", None):
                prompt_for_model = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            inputs = self.tokenizer(prompt_for_model, return_tensors="pt").to(self.input_device)
            with self.model_lock:
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    do_sample=do_sample,
                    temperature=request.temperature if do_sample else None,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            input_len = int(inputs["input_ids"].shape[-1])
            generated_ids = output_ids[0, input_len:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            modality = "text_only"

        return GenerateResponse(
            prompt=request.prompt,
            generated_text=generated_text,
            full_text=full_text,
            input_tokens=int(input_len),
            output_tokens=int(output_ids.shape[-1]),
            modality_status=modality,
            stage1_inversion=self.latest(),
            artifacts=self.artifacts(),
        )

    def artifacts(self) -> dict:
        return {
            "model_dir": self.model_dir,
            "output_dir": self.output_dir,
            "adapter_dir": str(self.adapter_dir),
            "projector_path": str(self.projector_path),
            "stage1_checkpoint_dir": self.stage1_checkpoint_dir,
            "pass_dataset_path": self.pass_dataset_path,
            "db_path": str(self.db_path),
            "device_map": self.device_map,
            "dtype": self.dtype,
            "input_device": str(self.input_device),
            "num_stage1_tokens": self.num_stage1_tokens,
            "dry_baseline": self.dry_baseline.describe(),
        }


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>卫星链路降雨反演</title>
  <style>
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#f6f7f9; color:#1f2933; }
    h1, main { width:min(1100px,calc(100vw - 32px)); margin:24px auto; }
    main { display:grid; grid-template-columns:380px 1fr; gap:16px; }
    section { background:#fff; border:1px solid #d8dee8; border-radius:8px; padding:16px; }
    label { display:block; color:#6b7280; font-size:13px; margin-bottom:6px; }
    textarea,input,select { width:100%; border:1px solid #d8dee8; border-radius:6px; padding:10px; font:inherit; background:#fff; }
    textarea { min-height:160px; line-height:1.5; resize:vertical; }
    .field { margin-top:14px; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px; }
    button { width:100%; height:42px; margin-top:16px; border:0; border-radius:6px; background:#0f766e; color:#fff; font:inherit; font-weight:650; cursor:pointer; }
    button:hover { background:#0b5f59; }
    button:disabled { opacity:.65; cursor:wait; }
    .answer { min-height:360px; white-space:pre-wrap; line-height:1.6; border:1px solid #d8dee8; border-radius:6px; padding:14px; background:#fbfcfd; }
    .meta { margin-top:12px; color:#6b7280; font-size:13px; line-height:1.6; white-space:pre-wrap; }
    .status { margin-top:12px; color:#6b7280; font-size:13px; min-height:20px; }
    @media (max-width:820px) { main { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <h1>卫星链路降雨反演</h1>
  <main>
    <section>
      <div class="field" style="margin-top:0">
        <label for="prompt">输入内容</label>
        <textarea id="prompt">请根据最新卫星链路反演结果，说明当前降雨情况。</textarea>
      </div>
      <div class="field">
        <label for="taskMode">路由</label>
        <select id="taskMode">
          <option value="auto" selected>自动判断</option>
          <option value="stage1">强制链路反演</option>
          <option value="text">只用文本</option>
        </select>
      </div>
      <div class="row">
        <div>
          <label for="maxNewTokens">输出 tokens</label>
          <input id="maxNewTokens" type="number" min="1" max="512" value="80" />
        </div>
        <div>
          <label for="temperature">temperature</label>
          <input id="temperature" type="number" min="0" step="0.1" value="0.0" />
        </div>
      </div>
      <button id="send">发送</button>
      <button id="refresh" type="button">刷新反演缓存</button>
      <div class="status" id="status"></div>
    </section>
    <section>
      <label>模型输出</label>
      <div class="answer" id="answer"></div>
      <div class="meta" id="meta"></div>
    </section>
  </main>
  <script>
    const send = document.getElementById("send");
    const refresh = document.getElementById("refresh");
    const statusEl = document.getElementById("status");
    const answer = document.getElementById("answer");
    const meta = document.getElementById("meta");
    async function showLatest() {
      const res = await fetch("/stage1/latest");
      const data = await res.json();
      meta.textContent = `stage1_inversion:\n${JSON.stringify(data, null, 2)}`;
    }
    refresh.addEventListener("click", async () => {
      refresh.disabled = true; statusEl.textContent = "刷新中...";
      try {
        const res = await fetch("/stage1/tick", {method:"POST"});
        const data = await res.json();
        meta.textContent = `stage1_inversion:\n${JSON.stringify(data, null, 2)}`;
        statusEl.textContent = "已刷新";
      } catch (err) {
        statusEl.textContent = String(err);
      } finally {
        refresh.disabled = false;
      }
    });
    send.addEventListener("click", async () => {
      send.disabled = true; statusEl.textContent = "生成中..."; answer.textContent = "";
      try {
        const payload = {
          prompt: document.getElementById("prompt").value,
          task_mode: document.getElementById("taskMode").value,
          max_new_tokens: Number(document.getElementById("maxNewTokens").value),
          temperature: Number(document.getElementById("temperature").value)
        };
        const res = await fetch("/generate", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
        answer.textContent = data.generated_text || "";
        meta.textContent =
          `input_tokens: ${data.input_tokens}\n` +
          `output_tokens: ${data.output_tokens}\n` +
          `modality_status: ${data.modality_status}\n\n` +
          `stage1_inversion:\n${JSON.stringify(data.stage1_inversion || {}, null, 2)}\n\n` +
          `artifacts:\n${JSON.stringify(data.artifacts || {}, null, 2)}`;
        statusEl.textContent = "完成";
      } catch (err) {
        statusEl.textContent = "请求失败"; answer.textContent = String(err);
      } finally {
        send.disabled = false;
      }
    });
    showLatest();
  </script>
</body>
</html>
"""


def create_app(runner: Stage1RainServiceRunner) -> FastAPI:
    app = FastAPI(title="LoRA-MoE Satellite Link Rain API", version="0.1.0")

    @app.on_event("startup")
    def startup_event():
        runner.start()

    @app.on_event("shutdown")
    def shutdown_event():
        runner.stop()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.get("/health")
    def health():
        return {"status": "ok", **runner.artifacts()}

    @app.get("/stage1/latest")
    def latest():
        return runner.latest()

    @app.post("/stage1/tick")
    def tick():
        state = runner.update_once()
        runner._set_cache(state)
        return runner.latest()

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest):
        return runner.generate(request)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--adapter-dir", default="")
    parser.add_argument("--projector-path", default="")
    parser.add_argument("--use-best", action="store_true", default=False)
    parser.add_argument("--db-path", default=DEFAULT_SENSOR_DB_PATH)
    parser.add_argument("--image-weather-csv", default=DEFAULT_IMAGE_WEATHER_CSV)
    parser.add_argument("--image-tolerance", default="10min")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--poll-interval-s", type=float, default=30.0)
    parser.add_argument("--stale-after-s", type=float, default=180.0)
    parser.add_argument("--lookback-hours", type=float, default=4.0)
    parser.add_argument("--max-passes", type=int, default=8)
    parser.add_argument("--pass-gap-threshold-s", type=float, default=60.0)
    parser.add_argument("--min-pass-points", type=int, default=10)
    parser.add_argument("--no-rain-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    runner = Stage1RainServiceRunner(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        adapter_dir=args.adapter_dir,
        projector_path=args.projector_path,
        use_best=args.use_best,
        db_path=args.db_path,
        image_weather_csv=args.image_weather_csv,
        image_tolerance=args.image_tolerance,
        host_device_map=args.device_map,
        dtype=args.dtype,
        poll_interval_s=args.poll_interval_s,
        stale_after_s=args.stale_after_s,
        lookback_hours=args.lookback_hours,
        max_passes=args.max_passes,
        pass_gap_threshold_s=args.pass_gap_threshold_s,
        min_pass_points=args.min_pass_points,
        no_rain_threshold=args.no_rain_threshold,
    )
    app = create_app(runner)
    print(f"Serving LoRA-MoE Stage1 rainfall FastAPI on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
