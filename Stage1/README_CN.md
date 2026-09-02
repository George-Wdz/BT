# Stage1：一分钟降雨反演

Stage1研究利用多源LEO卫星链路观测估计分钟级降雨。每个样本对应一个雨量计锚点前的60秒窗口，输入包括卫星链路、星地几何、地面温湿度气压和天空图像天气概率，模型输出该窗口的累计降雨量与降雨概率。

当前实现采用分钟级建模，旧的卫星过境级实现已移除。

## 环境要求

- Linux
- Python 3.10或3.11
- CPU可用于数据检查、单元测试和模型推理
- GPU训练需要与驱动匹配的PyTorch和torchvision

```bash
git clone https://github.com/George-Wdz/BT.git
cd BT

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# GPU环境可先按照PyTorch官方说明安装匹配CUDA的torch和torchvision。
python -m pip install -r Stage1/requirements.txt
```

项目已在Python 3.11.15、PyTorch 2.0.1、torchvision 0.15.2、NumPy 1.25.0、Pandas 2.0.3和FastAPI 0.99.1环境中验证。

## 快速验证

```bash
cd Stage1/minute_rain_retrieval
python check_delivery.py --training-only
python -m pytest -q
```

`--training-only`检查Python依赖和仓库内置数据，不要求本地采集数据库或在线服务权重。

## 复现实验

仓库内置`stratified_all/seed=42`固定数据版本，可直接训练：

```bash
python train.py \
  --dataset-path data/reproducible_v1/minute_rainfall_full.npz \
  --output-dir outputs/reproduction \
  --epochs 80 \
  --batch-size 64 \
  --max-train-dry-ratio 3 \
  --selection-metric balanced_mae
```

训练器读取完整NPZ中的固定`split`字段。`data/reproducible_v1/train`、`val`和`test`是对应划分的独立导出版本，用于检查样本及索引。

主要输出包括：

```text
outputs/reproduction/
  best.pt
  metrics.json
  train_predictions.csv
  val_predictions.csv
  test_predictions.csv
  val_rainy_predictions.csv
  test_rainy_predictions.csv
  val_test_rainy_predictions.csv
```

## 从原始数据构建

拥有符合字段约定的SQLite采集数据库时，可通过工作流重新构建数据并训练：

```bash
bash scripts/run_minute_rain_workflow.sh \
  --db-path /path/to/satellite_data.db \
  --image-csv /path/to/weather_labels.csv
```

工作流为每次运行生成唯一`run_id`，数据归档至`data/training_runs/<run_id>/`，模型和评估结果写入`outputs/training_runs/<run_id>/`。数据库字段、时间对齐和参数说明见[分钟反演详细文档](minute_rain_retrieval/README_CN.md)。

## 目录结构

| 路径 | 内容 |
|---|---|
| `minute_rain_retrieval/` | 数据构建、Transformer模型、训练、评估与测试 |
| `minute_rain_retrieval/data/reproducible_v1/` | 固定完整NPZ、独立划分、索引和视觉标签 |
| `vision_weather/` | 天空图像三分类模型与标签生成 |
| `rainfall_dashboard/` | FastAPI/ECharts可视化实现 |
| `terminal_002_rain_retrieval/`、`terminal_003_rain_retrieval/` | 新终端配置和链路域适配参数 |
| `satellite_identity/` | 终端星历ID与物理卫星映射分析 |
| `link_reliability_analysis/` | 链路质量与长期趋势分析 |

## 在线服务范围

仓库保留在线推理和可视化实现，但服务依赖持续回传的SQLite、相机照片、三套分钟模型权重和视觉分类权重。这些运行制品不在Git中，因此在线服务不是默认复现实验的一部分。

## 文档

- [分钟反演详细使用说明](minute_rain_retrieval/README_CN.md)
- [设计说明](minute_rain_retrieval/docs/DESIGN_CN.md)
- [测试说明](minute_rain_retrieval/docs/TESTING_CN.md)
- [风险与限制](minute_rain_retrieval/docs/RISKS_CN.md)
- [3～5分钟演示顺序](minute_rain_retrieval/docs/DEMO_CN.md)
- [本地在线权重部署与回滚](minute_rain_retrieval/docs/DEPLOYMENT_CN.md)
