# 一分钟降雨反演：运行与交付说明

## 项目是什么

本目录是Stage1唯一的降雨反演实现。雨量计时刻`g`的`rainfall * 0.1`表示`(g-60 s, g]`内累计降雨量，模型使用该窗口内全部有效卫星观测输出一个雨量标量和一个降雨概率。输出不按PHY点逐点累加。

当前模型输入为20维：基础16维包括链路4维、星地几何4维、温湿度气压3维、视觉天气4维和相对时间1维；训练集估计的链路dry delta再增加4维。模型是3层、8头、隐藏维度192的Transformer编码器，最大256个点。

## 环境与安装

- Linux；当前Shell、cron和服务脚本未适配Windows。
- Python 3.10或3.11。
- CPU可以构建数据、运行测试和推理；当前8041使用CUDA。
- 完整交付约需要30 GB可用空间，主要是照片、规范原始库和恢复备份；代码本身远小于该体积。

```bash
cd /home/wdz/BT
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Stage1/requirements.txt
```

GPU环境应先安装与驱动/CUDA匹配的PyTorch和torchvision，再安装其余依赖。执行：

```bash
cd Stage1/minute_rain_retrieval
python check_delivery.py --db-path /home/wdz/satellite_data/satellite_data.db
```

## 必需输入

### 训练数据源

默认数据库为`/home/wdz/satellite_data/satellite_data.db`，至少包含：

| 表 | 实际使用字段 |
|---|---|
| `phy_data` | `localTime`, `satelliteId`, `phyRssi`, `rssi`, `snr`, `lastCniValue`, `terminalId` |
| `position_data` | `localTime`, `satId`, 地面经纬高、卫星经纬高、`ecefPx/Py/Pz`, `terminalId` |
| `weather_data` | `timestamp`, `temperature`, `humidity`, `pressure`, `terminalId` |
| `weather_station` | `datetime`, `rainfall`, `terminalId` |

002/003在线读取还使用`phy_bb_data`和`phy_rssi_data`。位置必须与PHY具有相同卫星ID，并在默认5秒容差内最近邻匹配。视觉CSV至少包含`timestamp`、`prob_sunny`、`prob_cloudy`和`prob_rain`。

### Git内置可复现数据

仓库包含`data/reproducible_v1/`，克隆后无需采集数据库即可训练和评估：

```text
minute_rainfall_full.npz       # 训练器实际读取，内部含固定split标记
train/minute_rainfall_train.*  # 训练划分审计副本
val/minute_rainfall_val.*      # 验证划分审计副本
test/minute_rainfall_test.*    # 测试划分审计副本
camera_weather_labels.csv      # 本版本使用的图像分类标签，不含原始图像
dataset_manifest.json          # 数据版本、来源和划分说明
```

完整NPZ与三个审计副本来自同一个固定`stratified_all/seed=42`版本。训练器需要完整NPZ；单独的划分文件不能替代它。

### 不在Git中的在线运行制品

完整服务需随交付包提供：

- `weights/deployed/{position_model,no_position_fallback,new_terminal_transfer}/best.pt`；
- `../vision_weather/weights/20260623_133102_weather_cls_rain_station_balanced_100ep_best_model.pt`；
- `../data/camera/images/`及`../data/camera/labels/`；
- 002/003的`config.yaml`和`adapter.json`；
- 实时采集SQLite；恢复备份仅用于补历史，不是启动实时推理的唯一数据源。

## 构建与训练

克隆后直接复现训练，不需要原始SQLite：

```bash
cd Stage1/minute_rain_retrieval
python check_delivery.py --training-only
python train.py \
  --dataset-path data/reproducible_v1/minute_rainfall_full.npz \
  --output-dir outputs/reproduction \
  --epochs 80 \
  --batch-size 64 \
  --max-train-dry-ratio 3 \
  --selection-metric balanced_mae
```

从本地采集SQLite重新构建数据时，编辑`scripts/run_minute_rain_workflow.sh`中的路径和参数后执行：

```bash
cd /home/wdz/BT/Stage1/minute_rain_retrieval
bash scripts/run_minute_rain_workflow.sh
```

默认配置：重建数据、001终端、每分钟至少3个PHY点、位置/气象容差5秒、图像容差600秒、`stratified_all`、seed 42、训练集无雨/有雨最多3:1、80 epochs、balanced MAE选优。当前部署没有启用SNR硬掩码。

复用已有NPZ：

```bash
bash scripts/run_minute_rain_workflow.sh \
  --rebuild-dataset 0 \
  --dataset-path data/archive/minute_rainfall_v1_20260825_minphy3_stratified_seed42/processed/minute_rainfall_full.npz
```

复用NPZ不会按命令行重新划分；要更换`time`、`stratified_all`或`event_holdout`必须重建数据。

## 输出在哪里

每次workflow生成：

```text
data/training_runs/<run_id>/
  run_manifest.json
  processed/minute_rainfall_full.{npz,index.csv,summary.json}
  splits/train/minute_rainfall_train.*
  splits/val/minute_rainfall_val.*
  splits/test/minute_rainfall_test.*

outputs/training_runs/<run_id>/
  best.pt
  metrics.json
  train_predictions.csv
  val_predictions.csv
  test_predictions.csv
  val_rainy_predictions.csv
  test_rainy_predictions.csv
  val_test_rainy_predictions.csv
```

规范数据版本包含26,409个分钟样本，其中1,256个有雨样本；固定划分为18,486/3,961/3,962。完整数据、`SNR >= -10 dB`强链路视图和`SNR >= -25 dB`雨天鲁棒视图均位于`data/archive`。

## 本地在线服务

该部分仅面向持续接收终端数据的本地服务器，不是同事离线复现的必要步骤。Git不包含在线服务需要的实时SQLite、照片和部署权重。

先修改`MoE/lora-moe/scripts/serve_three_terminal_minute_rain_demo.sh`中的数据库、权重、相机和备份路径，然后：

```bash
cd /home/wdz/BT/MoE/lora-moe
PYTHON=/home/wdz/BT/.venv/bin/python \
  bash scripts/serve_three_terminal_minute_rain_demo.sh
```

服务默认监听`0.0.0.0:8041`。浏览器访问`http://<服务器IP>:8041`，健康检查为：

```bash
curl -fsS http://127.0.0.1:8041/health
```

输入来自实时SQLite和相机目录，不是通过页面上传。后台每30秒刷新视觉标签和最近链路，推理结果写入`MoE/lora-moe/data/runtime/rain_retrieval_history.sqlite3`；时间轴缓存写入同目录下`timeline_cache/`。历史表名为兼容旧库仍叫`rain_retrieval_passes`，其中当前模型记录的语义是一分钟窗口。

主要查询接口：

```text
GET /health
GET /api/rainfall?date=YYYY-MM-DD&max_passes=500&recompute=false
GET /api/timeline?start=<ISO时间>&end=<ISO时间>&resolution_minutes=1
GET /api/rainy-dates
GET /api/history/stats
```

## 测试

```bash
cd /home/wdz/BT/Stage1/minute_rain_retrieval
python -m pytest -q
```

当前结果为`7 passed`。详细覆盖范围、服务检查和未覆盖项见 [测试说明](docs/TESTING_CN.md)。

## 进一步阅读

- [设计说明](docs/DESIGN_CN.md)
- [风险与限制](docs/RISKS_CN.md)
- [3～5分钟Demo](docs/DEMO_CN.md)
- [8041权重部署与回滚](docs/DEPLOYMENT_CN.md)
- [SNR质量审查](analysis/snr_quality_review_20260902.md)
