# Stage1: Pass 级降雨反演

Stage1 负责把一次卫星过境期间的星地链路观测反演为该过境窗口内的降雨量。它是“当前/近实时反演”，不是未来预测；未来预测放在 Stage2。

## 1. 输入

唯一源数据是 SQLite 数据库：

```text
/home/wdz/satellite_data/satellite_data.db
```

使用表：

| 表 | 用途 |
| --- | --- |
| `phy_data` | 链路序列：`phyRssi`、`rssi`、`snr`、`lastCniValue`、`freqOffset`、`td` |
| `position_data` | 卫星和终端位置 |
| `weather_data` | 输入气象特征：温度、湿度、气压 |
| `weather_station` | 标签和补充气象：累计雨量、瞬时雨强、风速、风向 |

模型单个样本是一段卫星过境 pass。输入特征维度固定为：

```text
link(6) + position(6) + ground_weather(3) = 15
```

链路特征顺序：

```text
phyRssi, rssi, snr, lastCniValue, freqOffset, td
```

## 2. 标签

主标签：

```text
pass_rainfall_mm = rainfall_cumulative(pass_end) - rainfall_cumulative(pass_start)
```

辅助标签：

```text
wind_speed
wind_direction
```

瞬时 `rainfall` 波动较大，不直接作为主回归目标；它只落到 `pass_dataset.index.csv` 中作为雨强诊断字段：

```text
rain_rate_mean
rain_rate_max
rainy_ratio
```

## 3. 处理流程

1. 只读打开 SQLite 数据库。
2. 从 `phy_data` 读取链路样本。默认信任采集端 `/home/wdz/satellite_data/server.py` 的清洗结果，只保证本次实验选中的输入特征非空。
   - 如需在训练端重复过滤 `satelliteId=4294967295`、`snr=255`、`freqOffset=0`、`td=0`，设置 `data.strict_source_filters=true`
3. 从 `position_data` 读取位置样本。默认信任采集端清洗结果；训练端仍会在最近邻对齐后剔除缺失关键位置字段的时间点。
4. 按 `satelliteId` 和时间间隔切分 pass：
   - 同一卫星连续观测间隔小于 60 秒视为同一次过境
   - 少于 10 个链路点的 pass 丢弃
5. 对齐位置数据：
   - 以每个 phy 时间戳为基准
   - 从 `position_data` 做最近邻匹配
   - 容忍窗口为 5 秒
   - 缺失关键位置字段的时间点会被丢弃
6. 对齐温湿压输入：
   - 以每个 phy 时间戳为基准
   - 优先使用 `weather_data`
   - 缺口由 `weather_station` 的温湿压补齐
   - 最近邻匹配容忍窗口为 60 秒
7. 构建 pass 级标签：
   - 用 `weather_station.rainfall_cumulative` 在 `pass_start` 和 `pass_end` 两个边界做日内线性插值
   - 差分得到 `pass_rainfall_mm`
   - pass 跨 0 点或累计雨量边界缺失时丢弃
   - `wind_speed` / `wind_direction` 使用 pass 窗口内均值
8. 生成训练缓存和审计文件。

## 4. 输出

默认输出目录：

```text
Stage1/model/data/
```

| 文件 | 用途 |
| --- | --- |
| `pass_dataset.npz` | 训练直接读取的压缩 pass 数据 |
| `pass_dataset.index.csv` | 人可读 pass 清单：卫星 ID、过境起止时间、标签、雨强统计 |
| `pass_dataset.summary.json` | 本次 DB 截取摘要：源表范围、标签口径、pass 数、rainy pass 数 |

最近一次构建验证：

```text
有效 pass: 1858
卫星数: 191
rainy pass: 51 / 1858
pass 长度: min=10, median=34, p90=105, max=166
```

## 5. 模型

模型入口：

```text
Stage1/model/models/patch_encoder_decoder.py
```

支持两种 encoder 逻辑：

| 模式 | 配置 | 说明 |
| --- | --- | --- |
| channel-mixing | `model.use_channel_attention=false` | 15 维特征拼接后做 patch embedding |
| channel-wise / two-stage | `model.use_channel_attention=true` | link / position / weather 分组 embedding，先时间 attention，再通道 attention |

标签维度保持 3：

```text
pass_rainfall_mm, wind_speed, wind_direction
```

因此切换两种 encoder 不影响输出头。

## 6. 训练

默认训练：

```bash
cd /home/wdz/BT/Stage1/model
bash scripts/train_default.sh
```

首次训练会自动构建 `pass_dataset.npz`。如果要强制从 DB 重建：

```bash
REBUILD_CACHE=1 bash scripts/train_default.sh
```

只检查配置，不启动训练：

```bash
DRY_RUN=1 LR=5e-5 BATCH_SIZE=16 bash scripts/train_default.sh
```

常用超参覆盖：

```bash
LR=5e-5 BATCH_SIZE=16 PATCH_LEN=4 STRIDE=2 bash scripts/train_default.sh
```

小规模网格实验：

```bash
PATCH_LENS="4 8" STRIDES="2 4" LRS="1e-4 5e-5" ITERATIONS=1 bash scripts/train_sweep.sh
```

也可以直接使用 `--set`：

```bash
python3 main.py \
  --config configs/default.yaml \
  --set training.lr=5e-5 \
  --set model.patch_len=4 \
  --set model.stride=2
```

### 消融实验脚本

为了复现实验和改条件，`Stage1/model/scripts/train_experiments.sh` 封装了常用组合：

```bash
cd /home/wdz/BT/Stage1/model

# 15维输入，channel-mixing
bash scripts/train_experiments.sh baseline_cm

# 15维输入，channel-wise / two-stage attention
bash scripts/train_experiments.sh baseline_ca

# 去掉 freqOffset/td，只保留 phyRssi/rssi/snr/lastCniValue；
# 同时去掉 wind_speed/wind_direction 辅助任务，只训练 pass_rainfall_mm
REBUILD_CACHE=1 bash scripts/train_experiments.sh link4_rainonly_cm
REBUILD_CACHE=1 bash scripts/train_experiments.sh link4_rainonly_ca

# 单阶段改进版：非负雨量输出 + 降雨分类辅助头 + rainy pass 加权/采样
bash scripts/train_experiments.sh link4_cls_cm

# 在上面基础上，额外加入 pass 级统计 summary token：
# mean/std/min/max/range/slope，便于捕捉雨衰弱信号
bash scripts/train_experiments.sh link4_cls_summary_cm
```

常用覆盖：

```bash
ITERATIONS=1 EPOCHS=50 LR=5e-5 bash scripts/train_experiments.sh link4_rainonly_cm
```

如果继续改输入特征，需要同步设置：

```bash
EXTRA_SET="features.link=[phyRssi,rssi,snr,lastCniValue] model.input_dim=13 model.feature_group_dims=[4,6,3] targets.auxiliary=[]" \
PASS_DATASET_PATH=/home/wdz/BT/Stage1/model/data/pass_dataset_custom.npz \
CHECKPOINTS=/home/wdz/BT/Stage1/model/checkpoints_stage1_custom \
REBUILD_CACHE=1 \
bash scripts/train_default.sh
```

注意：不同输入特征应使用不同的 `PASS_DATASET_PATH` 和 `CHECKPOINTS`，避免 15 维缓存/checkpoint 与新特征实验混用。

默认 `data.strict_source_filters=false`，即信任 `/home/wdz/satellite_data/server.py` 入库前的清洗；Stage1 只要求当前实验实际选中的输入特征非空。若需要复查历史脏数据，可以加：

```bash
EXTRA_SET="data.strict_source_filters=true" REBUILD_CACHE=1 bash scripts/train_experiments.sh baseline_cm
```

当前雨量样本极不均衡，建议保留：

```yaml
data:
  val_strategy: "stratified_before_test"
training:
  use_rainy_sampler: true
  rainy_sample_weight: 20.0
  rainy_loss_weight: 20.0
  rain_classification_loss_weight: 0.5
  rain_classification_pos_weight: 20.0
model:
  nonnegative_rainfall: true
```

快速相关性分析显示，单个链路指标与 `pass_rainfall_mm` 的线性相关性都较弱；雨/无雨差异相对更明显的是 `phyRssi_std`、`rssi_min`、`phyRssi_mean`、`rssi_std`、`phyRssi_range`、`lastCniValue_*` 等 pass 级统计。因此 `link4_cls_summary_cm` 会显式加入统计 token，但它可能提高 rainy recall 的同时增加 dry 样本误差，需要和 `link4_cls_cm` 对比选择。

### Rainy-only 训练

如果怀疑大量无雨 pass 导致模型过拟合到“预测 0”，可以只保留有雨 pass 做单独训练：

```bash
cd /home/wdz/BT/Stage1/model

# link4 + 图像天气概率，channel-mixing，只训练 rainy pass
ITERATIONS=3 BATCH_SIZE=16 EPOCHS=120 PATIENCE=25 \
bash scripts/train_experiments.sh link4_img_rainonly_cm

# link4 + 图像天气概率，channel-wise / two-stage attention，只训练 rainy pass
ITERATIONS=3 BATCH_SIZE=16 EPOCHS=120 PATIENCE=25 \
bash scripts/train_experiments.sh link4_img_rainonly_ca
```

rainy-only 的过滤阈值由 `RAIN_FILTER_MIN` 控制，默认是：

```text
RAIN_FILTER_MIN=1e-6
```

例如只保留累计雨量大于 `0.05mm` 的 pass：

```bash
RAIN_FILTER_MIN=0.05 ITERATIONS=3 BATCH_SIZE=16 EPOCHS=120 PATIENCE=25 \
bash scripts/train_experiments.sh link4_img_rainonly_cm
```

这个过滤是训练启动时的可选项，不会改写已有 `pass_dataset_*.npz` 缓存。通用启动脚本也支持：

```bash
RAIN_FILTER_MIN=1e-6 bash scripts/train_default.sh
```

### 单卫星/卫星组诊断训练

为了判断跨卫星差异是否掩盖雨衰信号，可以只保留指定 `satelliteId` 训练：

```bash
cd /home/wdz/BT/Stage1/model

# 单颗卫星
SATELLITE_IDS=1558 ITERATIONS=1 EPOCHS=80 \
bash scripts/train_experiments.sh link4_img_cls_cm

# 多颗卫星组成一个小组
SATELLITE_IDS=264,268,272 ITERATIONS=1 EPOCHS=80 \
bash scripts/train_experiments.sh link4_img_cls_cm
```

如果只看 rainy pass：

```bash
SATELLITE_IDS=264,268,272 RAIN_FILTER_MIN=1e-6 ITERATIONS=1 EPOCHS=120 BATCH_SIZE=8 \
bash scripts/train_experiments.sh link4_img_rainonly_cm
```

当前数据中，单颗卫星的 rainy pass 很少：过境最多的卫星只有约 28 个 pass，且 rainy 通常只有 0-2 个。因此单卫星训练更适合作为诊断，不适合作为最终模型。更实际的方案是先做同卫星 clear-sky baseline 差分，再在全量数据上训练。

### Dry Baseline 差分特征

`dry_baseline` 会在训练启动后，用 train split 的 dry pass 计算每颗卫星的链路均值 baseline，然后给输入追加 4 维差分：

```text
phyRssi_delta_to_dry_baseline
rssi_delta_to_dry_baseline
snr_delta_to_dry_baseline
lastCniValue_delta_to_dry_baseline
```

注意：baseline 只使用 train split 的 dry pass，避免使用 val/test 标签造成泄漏。

全量数据实验：

```bash
cd /home/wdz/BT/Stage1/model

DRY_BASELINE_ENABLED=true ITERATIONS=3 EPOCHS=100 \
bash scripts/train_experiments.sh link4_img_drybase_cm
```

指定卫星组实验：

```bash
SATELLITE_IDS=264,268,272 DRY_BASELINE_ENABLED=true \
ITERATIONS=3 EPOCHS=120 BATCH_SIZE=8 PATIENCE=25 \
bash scripts/train_experiments.sh link4_img_drybase_cm
```

对应不带 dry baseline 的对照：

```bash
SATELLITE_IDS=264,268,272 \
ITERATIONS=3 EPOCHS=120 BATCH_SIZE=8 PATIENCE=25 \
bash scripts/train_experiments.sh link4_img_cls_cm
```

### 相机图片同步

相机端当前会把最新 `capture_*.jpg` 同步到服务器同事账号目录：

```text
mjh@192.168.1.94:/home/mjh/WorkSpace/Weather-Platform/backend/camera/
```

Stage1 提供了从该目录拉取图片到本机训练目录的脚本：

```bash
cd /home/wdz/BT/Stage1/model
bash scripts/sync_camera_from_mjh.sh
```

默认本机保存目录：

```text
/home/wdz/BT/Stage1/data/camera
```

脚本默认只拉取远端最新一张图片。远端目录目前只保留最新约 60 张，因此本机目录应作为长期归档目录，定时拉取后不要删除本地历史图片。

推荐定时任务每分钟使用 `MODE=all`，补拉远端当前仍保留的所有图片。`rsync` 只会传输本地缺失或变化的文件，开销较小，而且比只拉最新一张更不容易在任务短暂停顿时产生断层。

```bash
* * * * * cd /home/wdz/BT/Stage1/model && MODE=all bash scripts/sync_camera_from_mjh.sh >> /home/wdz/BT/Stage1/logs/camera_sync.log 2>&1
```

如果服务短暂停过，也可以补拉远端当前还保留的所有 `capture_*.jpg`：

```bash
MODE=all bash scripts/sync_camera_from_mjh.sh
```

修改远端或本地路径：

```bash
REMOTE_HOST=192.168.1.94 \
REMOTE_USER=mjh \
REMOTE_DIR=/home/mjh/WorkSpace/Weather-Platform/backend/camera \
LOCAL_DIR=/home/wdz/BT/Stage1/data/camera \
bash scripts/sync_camera_from_mjh.sh
```

当前本机和 `mjh` 在同一服务器网络中，已使用 `192.168.1.94` 跑通 SSH 密钥访问，不需要 Tailscale。

### 相机图片天气标签

图片归档在 Stage1 中，但天气分类推理仍调用已有视觉模型：

```text
视觉模型代码：/home/wdz/LLaMA-Factory/leo_model/vision
视觉训练数据：/home/wdz/LLaMA-Factory/leo_model/data/vision_weather
Stage1 图片归档：/home/wdz/BT/Stage1/data/camera
Stage1 天气标签：/home/wdz/BT/Stage1/data/camera_labels
```

生成当前归档图片的天气概率标签：

```bash
cd /home/wdz/BT/Stage1/model
bash scripts/predict_camera_weather.sh
```

脚本会调用：

```text
/home/wdz/LLaMA-Factory/leo_model/vision/predict_weather_labels.py
```

默认使用 vision `weights/` 目录下最新的 `.pt` 权重，输出：

```text
/home/wdz/BT/Stage1/data/camera_labels/<timestamp>_weather_labels.csv
/home/wdz/BT/Stage1/data/camera_labels/latest_weather_labels.csv
```

CSV 中包含：

```text
timestamp, pred_label, confidence, prob_sunny, prob_cloudy, prob_rain
```

Stage1 训练实际只使用：

```text
timestamp, prob_sunny, prob_cloudy, prob_rain
```

并在数据构建时自动增加：

```text
image_available
```

`image_path`、`file_name` 等审计列不会进入模型。为了减少文件体积和避免误用，`predict_camera_weather.sh` 会额外生成：

```text
/home/wdz/BT/Stage1/data/camera_labels/latest_weather_labels_slim.csv
```

slim 文件只保留：

```text
timestamp, pred_label, pred_idx, confidence, prob_sunny, prob_cloudy, prob_rain
```

如果要固定某个权重：

```bash
WEIGHTS=/home/wdz/LLaMA-Factory/leo_model/vision/weights/20260605_182036_weather_cls_more_cloudy_gpu_30ep_best_model.pt \
bash scripts/predict_camera_weather.sh
```

推荐定时流程是先同步图片，再生成最新标签：

```bash
* * * * * cd /home/wdz/BT/Stage1/model && bash scripts/sync_camera_from_mjh.sh && bash scripts/predict_camera_weather.sh >> /home/wdz/BT/Stage1/logs/camera_weather.log 2>&1
```

## 7. 配置

核心配置文件：

```text
Stage1/model/configs/default.yaml
```

关键项：

```yaml
data:
  db_path: "/home/wdz/satellite_data/satellite_data.db"
  pass_dataset_path: "/home/wdz/BT/Stage1/model/data/pass_dataset.npz"
  data_split: [0.7, 0.2, 0.1]
  strict_source_filters: false

model:
  input_dim: 15
  max_seq_len: 256
  patch_len: 8
  stride: 4
  use_channel_attention: false
  feature_group_dims: [6, 6, 3]

targets:
  primary: [pass_rainfall_mm]
  auxiliary: [wind_speed, wind_direction]
```

不要随意修改：

- `model.input_dim` 必须等于 `15`
- `model.feature_group_dims` 必须等于 `[6, 6, 3]`
- `features.*` 需要和 `preprocessing.py`、`dataset.py` 同步

## 8. 雨天诊断

`diagnose_rainy.py` 用于检查 test split 中 rainy pass 的真值和预测值，并对比 zero/mean baseline。脚本会读取每个 checkpoint 自带的模型配置，因此同一目录下可以同时诊断 channel-mixing 和 channel-wise 结果。

```bash
cd /home/wdz/BT/Stage1/model
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -u scripts/diagnose_rainy.py --config configs/default.yaml
```

可选参数：

```bash
python3 -u scripts/diagnose_rainy.py \
  --config configs/default.yaml \
  --checkpoints /home/wdz/BT/Stage1/model/checkpoints_stage1_15f_passrain \
  --pattern '*itr0' \
  --rain-threshold 1e-6
```

注意事项：

- 当前 feature schema 是 15 维；旧 13 维 checkpoint 会被跳过，不能和当前 `pass_dataset.npz` 混用。
- 如果 DB 新增数据后重建了 `pass_dataset.npz`，诊断的是“旧 checkpoint 在新缓存 test split 上的表现”，不等价于训练时 test 指标。
- 当前顺序切分是 `0.7/0.2/0.1`，如果最后 10% 时间段没有降雨，rainy 诊断会显示 test 中无 rainy pass；这时应重新训练或专门做雨天子集评估。

## 9. 验证

```bash
python3 -m compileall -q /home/wdz/BT/Stage1/model
python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path('/home/wdz/BT/Stage1/model')))
from data.preprocessing import build_pass_dataset
passes = build_pass_dataset(
    '/home/wdz/satellite_data/satellite_data.db',
    '/tmp/stage1_pass_dataset_check.npz',
)
print(len(passes))
PY
```

## 10. 安全边界

- 数据库只读打开：`file:{db_path}?mode=ro`
- 训练脚本不修改、删除、建表或 vacuum 原始数据库
- 生成物限制在 `Stage1/model/data/`、`Stage1/model/checkpoints/`、`Stage1/model/logs/`
- 旧标签口径训练出的 checkpoint 只适合历史对照，不应直接和新 `pass_rainfall_mm` 口径混用
