# 辅助工具

本目录保存不属于日常训练入口、但用于数据维护和实验分析的命令行工具。所有命令均从`Stage1/minute_rain_retrieval`目录执行。

## analysis

| 脚本 | 用途 |
|---|---|
| `analyze_inference_consistency.py` | 统计各终端反演精度及共同分钟的一致性 |
| `analyze_snr_quality.py` | 分析SNR分布和不同门槛下的分钟样本保留率 |
| `analyze_terminal_correlations.py` | 比较新旧终端在共同时间和卫星上的链路特征相关性 |

## data

| 脚本 | 用途 |
|---|---|
| `archive_raw_sources.py` | 按处理后数据的时间范围归档原始数据库记录 |
| `derive_feature_ablation_dataset.py` | 从固定NPZ生成特征消融版本 |
| `derive_point_threshold_datasets.py` | 保持划分不变，生成不同PHY点数门槛的数据版本 |
| `resplit_dataset_by_time.py` | 按显式时间边界重划train/val/test |

## history

| 脚本 | 用途 |
|---|---|
| `backfill_raw_history.py` | 直接读取三终端数据库并离线重算历史反演结果 |
| `backfill_service_history.py` | 通过已运行的HTTP服务触发指定日期范围回填 |

历史工具会写SQLite或调用在线服务，不属于公开数据上的离线训练流程。

## export

| 脚本 | 用途 |
|---|---|
| `export_current_model_results.py` | 导出当前模型评估、降雨样本和台风时段结果 |
| `export_typhoon_results.py` | 从历史库导出台风时段及整体有雨记录 |

## 使用方式

每个脚本均提供独立参数说明，例如：

```bash
python tools/analysis/analyze_snr_quality.py --help
python tools/data/resplit_dataset_by_time.py --help
python tools/history/backfill_raw_history.py --help
python tools/export/export_current_model_results.py --help
```

这些工具不会被`train.py`隐式调用。标准训练只依赖项目根目录中的数据、模型和训练模块。
