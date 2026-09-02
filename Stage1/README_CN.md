# Stage1 一分钟降雨反演

Stage1是一个多源LEO卫星链路分钟降雨反演原型。系统将雨量计锚点前一分钟内的卫星链路、星地几何、地面温湿度气压和天空图像天气概率编码为变长序列，输出该分钟累计降雨量及降雨概率。当前固定使用分钟反演，旧过境级反演已移除。

## Reviewer最短路径

```bash
cd /home/wdz/BT/Stage1/minute_rain_retrieval
python check_delivery.py --training-only
python -m pytest -q
```

同事接手的默认范围是固定数据上的离线训练、验证和测试。8041依赖持续回传的本地数据库、照片和部署权重，仅用于数据服务器本地运行，不属于克隆后的复现要求。

## 安装

支持Linux和Python 3.10/3.11。GPU不是单元测试和CPU推理的必要条件；GPU运行需要安装与本机驱动匹配的PyTorch和torchvision。

```bash
cd /home/wdz/BT
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# GPU机器可先按本机CUDA版本安装torch和torchvision。
python -m pip install -r Stage1/requirements.txt
```

当前验证环境为PyTorch 2.0.1、torchvision 0.15.2、NumPy 1.25.0、Pandas 2.0.3和FastAPI 0.99.1。Git包含可直接训练和验证的固定数据版本；完整在线服务仍需要本地权重、照片和SQLite。

## 核心目录

| 路径 | 内容 |
|---|---|
| `minute_rain_retrieval/` | 数据构建、Transformer训练、归档、测试和在线服务 |
| `minute_rain_retrieval/data/reproducible_v1/` | Git内置的完整NPZ、固定划分和视觉标签 |
| `minute_rain_retrieval/data/archive/` | 已审核的规范分钟数据集及固定划分 |
| `minute_rain_retrieval/weights/deployed/` | 8041加载的三套模型权重 |
| `vision_weather/` | 天空图像三分类模型和在线标签生成 |
| `data/camera/` | 原始照片和视觉标签 |
| `data/source_backups/` | 三终端工控机恢复数据 |
| `rainfall_dashboard/` | FastAPI页面、ECharts和链路分析接口 |
| `terminal_002_rain_retrieval/`, `terminal_003_rain_retrieval/` | 新终端配置和Z-score适配器 |
| `satellite_identity/` | 终端卫星ID与物理卫星映射分析 |
| `link_reliability_analysis/` | 原始链路质量和长期趋势产物 |

## 输入与输出

训练输入是SQLite中的`phy_data`、`position_data`、`weather_data`、`weather_station`和可选视觉标签CSV。默认001终端ID为`01-31-0005-0001`，标签按`weather_station.rainfall * 0.1`解释为锚点前一分钟累计雨量。

```bash
cd /home/wdz/BT/Stage1/minute_rain_retrieval
bash scripts/run_minute_rain_workflow.sh
```

默认从当前数据库重建数据并训练。每次运行生成唯一`run_id`：

- `data/training_runs/<run_id>/`：完整NPZ、独立train/val/test、索引、摘要和manifest；
- `outputs/training_runs/<run_id>/`：`best.pt`、`metrics.json`、三套预测CSV和有雨样本CSV。

使用Git内置数据直接训练和评估：

```bash
python train.py \
  --dataset-path data/reproducible_v1/minute_rainfall_full.npz \
  --output-dir outputs/reproduction \
  --epochs 80 \
  --batch-size 64 \
  --max-train-dry-ratio 3 \
  --selection-metric balanced_mae
```

完整输入契约、参数和输出说明见 [分钟项目README](minute_rain_retrieval/README_CN.md)。

## 交付文档

- [设计说明](minute_rain_retrieval/docs/DESIGN_CN.md)
- [测试说明](minute_rain_retrieval/docs/TESTING_CN.md)
- [风险与限制](minute_rain_retrieval/docs/RISKS_CN.md)
- [3～5分钟Demo](minute_rain_retrieval/docs/DEMO_CN.md)
- [8041权重部署与回滚](minute_rain_retrieval/docs/DEPLOYMENT_CN.md)
- [数据和目录清理记录](minute_rain_retrieval/CLEANUP_REPORT_CN.md)
