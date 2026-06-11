# Stage1 降雨反演改进计划

更新时间：2026-06-04

## 1. 当前目标

Stage1 的目标是基于一次卫星过境 pass 内的星地链路、卫星位置、地面气象等观测，反演该过境窗口内的累计降雨量：

```text
pass_rainfall_mm = rainfall_cumulative(pass_end) - rainfall_cumulative(pass_start)
```

该任务是当前/近实时反演，不是未来预测。未来预测仍归 Stage2。

## 2. 当前数据和基线结论

当前可用数据来自：

```text
/home/wdz/satellite_data/satellite_data.db
```

已构建的新数据集大致情况：

```text
总 pass: 2358-2361
rainy pass: 87
test pass: 237
test rainy pass: 36
```

现有基线表现：

| 实验 | 输入 | 训练目标 | 整体 MAE | rainy MAE | 结论 |
| --- | --- | --- | ---: | ---: | --- |
| baseline channel-mixing | 15 维 | 雨量 + 风速/风向 | 0.1003 | 0.6189 | 基本预测接近 0 |
| baseline channel-wise | 15 维 | 雨量 + 风速/风向 | 0.1029 | 0.6150 | 未显著改善 |
| link4 rain-only | 13 维，去掉 fd/td | 只训雨量 | 0.0972 | 0.6185 | dry 更准，但 rainy 仍接近 0 |
| link4 cls | link4 + 分类头 | 雨量 + 是否降雨 | 0.0983 | 0.5963 | 分类头能识别部分雨样本 |
| link4 cls summary | link4 + 统计 token | 雨量 + 是否降雨 | 0.1038 | 0.5891 | rainy 稍好，dry 误差增加 |

关键问题：

1. 数据极不均衡，rainy pass 占比很低。
2. 原时间顺序切分中，val 集没有 rainy pass，导致 early stopping 选择“预测无雨”的模型。
3. test 最后时间段存在更强降雨，最大约 4.12mm；train 最大约 1.74mm，存在分布外外推。
4. 单个链路指标与雨量的线性相关较弱，不能只依赖简单绝对值。

## 3. 已完成的代码改进

已在 Stage1 中完成以下改动：

1. 雨量输出非负化：`softplus(rainfall_head)`。
2. 新增是否降雨分类辅助头。
3. 新增 rainy pass 损失加权。
4. 新增 rainy pass 采样权重。
5. 验证集切分改为：

```yaml
data:
  val_strategy: "stratified_before_test"
```

含义是 test 仍保留最后 10% 时间段，train/val 在 test 前按雨/无雨分层，避免 val 全无雨。

6. 支持信任采集端清洗：

```yaml
data:
  strict_source_filters: false
```

`/home/wdz/satellite_data/server.py` 已在入库前过滤无效卫星、无效 SNR、无效 fd/td、重复数据等。Stage1 默认只保证本次实验实际使用的输入特征非空。

7. 新增实验脚本：

```text
/home/wdz/BT/Stage1/model/scripts/train_experiments.sh
```

推荐命令：

```bash
cd /home/wdz/BT/Stage1/model

ITERATIONS=3 bash scripts/train_experiments.sh link4_cls_cm
ITERATIONS=3 bash scripts/train_experiments.sh link4_cls_summary_cm
```

## 4. 链路指标分析结论

基于 pass 级统计，单个链路指标与 `pass_rainfall_mm` 的 Pearson/Spearman 相关性都不强。

雨/无雨差异相对更明显的统计特征包括：

```text
phyRssi_std
rssi_min
phyRssi_mean
rssi_std
phyRssi_range
lastCniValue_mean/max/min/range/std
snr_mean
```

这说明雨衰信号更可能体现在同一 pass 内的波动、极值、范围和相对变化，而不是单个时间点的绝对值。

## 5. 下一步优先方案

### 5.1 同卫星无雨基线差分

不同卫星配置、链路预算、轨道几何和终端观测条件不同，跨卫星直接比较绝对 `RSSI/SNR/phyRssi` 会引入强偏置。

当前模型有 satellite embedding，但它只是学习卫星 ID 偏置，不等价于显式建模“同一颗卫星在雨/非雨环境下的链路差异”。

建议新增同卫星 clear-sky baseline 特征：

```text
current_snr - same_satellite_clear_sky_snr_baseline
current_rssi - same_satellite_clear_sky_rssi_baseline
current_phyRssi - same_satellite_clear_sky_phyRssi_baseline
current_lastCniValue - same_satellite_clear_sky_cni_baseline
```

baseline 选择策略：

1. 同一颗卫星。
2. 无雨 pass。
3. 尽量匹配相似卫星高度/经纬度/过境时长/时间邻近。
4. 对每个特征计算 mean/std/min/max/range/slope 的差分。

优先级：最高。

### 5.2 图像天气分类特征接入

已有图像分类模型输出三类：

```text
晴天 / 多云 / 下雨
```

理论上会增强效果，尤其是当前链路信号弱、雨样本少的情况下。

建议不要只接硬标签，而是接概率：

```text
p_sunny, p_cloudy, p_rain
```

这一路线可以视为“伪多模态”或“轻量多模态”：不把原始图像直接送入大模型，也不使用完整 ViT/CLIP 视觉编码器，而是先用一个专门的图像天气分类模型把图像压缩成天气状态特征，再与链路、位置、气象特征融合。

推荐理由：

1. 当前图像类别简单，仅包括多云、阴天、下雨三类。
2. 图像场景相似，完整 ViT 端到端训练容易过重且过拟合。
3. Stage1 的核心目标是数值雨量反演，不是开放域图像理解。
4. 分类概率 `p_sunny/p_cloudy/p_rain` 比图像 token 更容易解释和验证。
5. 工程风险低，能快速判断图像天气先验是否对 rainy recall 有帮助。

对 Stage1 的用法：

```text
link + position + ground_weather + image_weather_probs
```

注意：

1. 图像时间要和 pass 对齐，优先取 pass 中心时间最近的分类结果。
2. 图像分类用于补充“是否有雨/天气状态”，雨量大小仍以链路和气象站监督为主。
3. 图像输出建议保存成表：

```text
timestamp, p_sunny, p_cloudy, p_rain, image_path/model_version
```

优先级：高。

### 5.2.1 伪多模态接入方式

推荐先采用概率特征接入：

```text
image -> weather image classifier -> [p_sunny, p_cloudy, p_rain]
```

然后按 pass 对齐：

```text
pass_center_time -> nearest image prediction within tolerance
```

最终输入：

```text
link encoder input:
  phyRssi, rssi, snr, lastCniValue, same-satellite deltas

position encoder input:
  satellite/terminal position and geometry

weather encoder input:
  temperature, humidity, pressure

image-weather encoder input:
  p_sunny, p_cloudy, p_rain
```

如果后续分类模型能导出倒数第二层 embedding，也可以增加：

```text
image_weather_embedding
```

但第一版建议只用三类概率，因为它更稳定、更易排查。

不推荐第一版直接使用 ViT 的原因：

1. 样本量不足时，ViT embedding 可能带来大量无关视觉信息。
2. 图像类别很少，复杂视觉编码器的收益未必覆盖训练和调参成本。
3. 当前反演瓶颈主要是雨样本少、同卫星基线差分不足，而不是视觉语义理解不足。

建议图像分类结果表结构：

```text
image_weather_predictions
  id
  timestamp
  p_sunny
  p_cloudy
  p_rain
  pred_label
  image_path
  model_name
  model_version
  created_at
```

Stage1 数据构建时只读该表，不依赖图像文件本身。

### 5.2.2 当前 vision 模型检查结论

检查目录：

```text
/home/wdz/LLaMA-Factory/leo_model/vision
```

当前代码包含：

1. `models.py`：轻量 TinyResNet 天气分类器，支持 `extract_features(pixel_values)` 导出中间特征。
2. `predict_weather_labels.py`：离线批量推理脚本，递归扫描图片目录，输出 CSV。
3. `train_weather_classifier.py` / `eval_weather_classifier.py`：训练和评估脚本。
4. `weights/`：已有训练权重。

当前权重类别为：

```text
sunny / cloudy / rain
```

这与当前目标“晴天、多云、下雨”一致，可以直接作为第一版图像天气先验。最近较好的评估结果显示：

```text
overall accuracy ≈ 0.97
rain recall ≈ 0.90
rain precision ≈ 0.40
rain F1 ≈ 0.55
```

含义是：模型对“下雨”比较敏感，但仍存在一定误报。因此 Stage1 不应只使用硬标签 `pred_label`，而应使用概率 `p_sunny/p_cloudy/p_rain`，让雨量反演模型自己学习图像先验的可信程度。

当前离线推理输出已经接近可用，建议统一输出字段：

```text
timestamp
image_path
pred_label
confidence
p_sunny
p_cloudy
p_rain
model_name
model_version
weight_path
created_at
```

如果需要与 `/home/wdz/satellite_data/satellite_data.db` 中的链路、位置、气象数据长期对齐，建议新增 SQLite 表：

```sql
CREATE TABLE IF NOT EXISTS image_weather_predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL,
  image_path TEXT NOT NULL UNIQUE,
  pred_label TEXT,
  confidence REAL,
  p_sunny REAL,
  p_cloudy REAL,
  p_rain REAL,
  model_name TEXT,
  model_version TEXT,
  weight_path TEXT,
  created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_image_weather_timestamp
ON image_weather_predictions(timestamp);
```

不建议把原始图片二进制直接写进 SQLite。原始图片保留为文件，SQLite 只保存路径、时间戳、分类概率和模型版本，避免数据库膨胀，也方便后续用新权重重新离线推理。

训练使用方式：

1. 采集端持续保存图片文件，文件名或元数据必须包含时间戳。
2. 图像分类脚本定期或离线扫描新增图片，写入 CSV 或 SQLite。
3. Stage1 构建 pass 数据集时，按 `pass_center_time` 查找最近的图像分类结果。
4. 若最近图片与 pass 中心时间差超过阈值，例如 5-10 分钟，则该 pass 的图像特征置为缺失或默认值，并增加 `image_available` 标志。
5. 第一版只接 `p_sunny/p_cloudy/p_rain/image_available`，暂不接图片 embedding。

结论：当前 vision 模型满足第一版接入需求，不需要改成 ViT，也不需要在线端到端多模态训练。需要补的主要是“预测结果落库/CSV 标准化”和“Stage1 时间对齐读取”。

当前 vision 相关路径：

```text
视觉分类模型代码：/home/wdz/LLaMA-Factory/leo_model/vision
视觉分类训练数据：/home/wdz/LLaMA-Factory/leo_model/data/vision_weather
Stage1 相机图片归档：/home/wdz/BT/Stage1/data/camera
```

不建议把 vision 模型代码和训练数据整体搬进 Stage1。原因是 vision 模型本身是一个独立任务，放在 LLaMA-Factory/leo_model 下更清晰；Stage1 只需要读取它的推理产物，例如 CSV 或 SQLite 表中的 `p_sunny/p_cloudy/p_rain`。这样可以避免两个项目之间复制权重、复制训练数据和版本不一致。

推荐目录职责：

1. `/home/wdz/LLaMA-Factory/leo_model/vision`：继续负责图像分类模型训练、评估、离线推理。
2. `/home/wdz/LLaMA-Factory/leo_model/data/vision_weather`：继续作为图像分类的标注训练集。
3. `/home/wdz/BT/Stage1/data/camera`：保存从服务器同步回来的实时相机图片归档。
4. `/home/wdz/satellite_data/satellite_data.db` 或 Stage1 CSV：保存图像分类推理结果，供 Stage1 按时间对齐读取。

如果后续要工程化，可以在 Stage1 中只增加轻量 wrapper 脚本，调用 vision 的 `predict_weather_labels.py` 对 `/home/wdz/BT/Stage1/data/camera` 进行推理，并把输出写成标准 CSV/SQLite。

已增加 Stage1 wrapper：

```text
/home/wdz/BT/Stage1/model/scripts/predict_camera_weather.sh
```

该脚本不复制 vision 代码，只调用：

```text
/home/wdz/LLaMA-Factory/leo_model/vision/predict_weather_labels.py
```

输入和输出：

```text
输入图片：/home/wdz/BT/Stage1/data/camera
输出标签：/home/wdz/BT/Stage1/data/camera_labels/latest_weather_labels.csv
```

这样 Stage1 后续只需要对齐 `latest_weather_labels.csv` 或导入 SQLite 后的 `image_weather_predictions` 表。

评估时需要做消融：

```text
link4_cls
link4_cls + image_probs
link4_cls + same_satellite_delta
link4_cls + same_satellite_delta + image_probs
```

重点观察：

```text
rainy recall
rainy MAE
dry false alarm
overall MAE/MSE
大雨样本 top-k
```

### 5.3 损失和评估继续改进

继续保留单阶段训练，不做两步训练。

建议保留：

```yaml
model:
  nonnegative_rainfall: true

training:
  use_rainy_sampler: true
  rainy_sample_weight: 20.0
  rainy_loss_weight: 20.0
  rain_classification_loss_weight: 0.5
  rain_classification_pos_weight: 20.0
```

评估必须同时报告：

```text
overall MAE/MSE
rainy MAE
dry MAE
zero baseline MAE
rain classification precision/recall
大雨样本 top-k 的 true/pred/prob
```

只看 overall MAE 会被 dry 样本掩盖。

## 6. LLaMA-MoE 权重评估

已下载权重：

```text
/home/wdz/BT/MoE/models/LLaMA-MoE-v1-3_5B-2_8
```

目录大小约 13GB，包含：

```text
config.json
configuration_llama_moe.py
modeling_llama_moe_hf.py
tokenizer.model
pytorch_model-00001-of-00002.bin
pytorch_model-00002-of-00002.bin
```

已通过 `example.py` 验证可正常推理生成文本。

对当前项目的作用边界：

1. 对“数值雨量反演”没有直接帮助。雨量反演核心仍依赖结构化时序 encoder、同卫星差分特征、图像天气先验和监督标签。
2. 可用于后续解释/报告/对话。例如：

```text
输入：模型反演结果 + 链路变化 + 图像天气分类 + 气象站上下文
输出：自然语言解释、告警报告、人工可读结论
```

3. 不建议现在把 LLaMA-MoE 作为唯一主干直接承担数值反演。当前瓶颈是 rainy 样本稀缺、同卫星差分特征和多源特征表达，不是语言生成能力。
4. 后续可考虑：

```text
Stage1 encoder 输出数值结果
↓
规则/模板整理为结构化文本
↓
LLaMA-MoE 生成解释报告
```

## 7. 多编码器接主模型方案

结论先行：可以做多个对应的编码器再接主模型，并且不一定需要改动原始 LLaMA-MoE 权重。是否需要改原模型，取决于“主模型”承担什么职责：

1. 如果主模型只负责解释和生成报告，可以完全不改 LLaMA-MoE 权重。
2. 如果主模型要直接参与雨量反演，可以冻结 LLaMA-MoE，只训练外部编码器、projector 和回归头。
3. 如果希望 LLaMA-MoE 内部 attention/FFN 真正参与多模态融合，需要改模型 forward 或 remote code，并可能做 LoRA/Adapter 微调。
4. 如果要把 LLaMA-MoE 改造成“结构化多模态 MoE”，则需要改模型结构和训练流程，工作量最大，不建议作为第一阶段。

### 7.1 多编码器设计

建议按数据源拆编码器：

```text
link encoder:
  输入 phyRssi/rssi/snr/lastCniValue 及其同卫星 clear-sky delta

position encoder:
  输入 satellite longitude/latitude/altitude、terminal position、pass geometry

weather encoder:
  输入 temperature/humidity/pressure

image weather encoder:
  输入 p_cloudy/p_overcast/p_rain，或图像模型中间 embedding

satellite embedding:
  输入 satellite_id，用于建模不同卫星配置差异
```

融合方式建议从简单到复杂推进：

```text
方案 A: concatenate + MLP/Transformer fusion
方案 B: modality tokens + cross-attention
方案 C: modality-specific experts + router
方案 D: 接入 LLaMA-MoE token embedding 空间
```

当前最推荐的是 A/B，不建议直接上 D。

### 7.2 方案 A：结构化多编码器，不接 LLaMA-MoE 主干

结构：

```text
link encoder
position encoder
weather encoder
image-prob encoder
satellite embedding
        ↓
fusion transformer / attention pooling
        ↓
rainfall regression head + rain classification head
```

优点：

1. 不依赖 LLaMA-MoE，训练快。
2. 直接优化雨量 MAE/rainy MAE。
3. 最容易判断每个模态是否有效。
4. 不需要改原模型权重。

缺点：

1. 不具备自然语言解释能力。
2. 对复杂跨模态推理能力有限。

适用阶段：当前阶段首选。

### 7.3 方案 B：结构化多编码器 + LLaMA-MoE 只做报告生成

结构：

```text
多编码器 Stage1 模型
  输出：rainfall_mm, rain_prob, key_features, image_weather_probs
        ↓
模板化结构化文本
        ↓
冻结 LLaMA-MoE
        ↓
解释报告 / 对话回答
```

例子：

```text
输入给 LLaMA-MoE：
卫星 543 在 2026-06-04 13:xx 的 pass 中，模型估计累计降雨 0.42mm；
rain_prob=0.71；phyRssi_std 低于同卫星晴空基线；p_rain=0.83。
请生成一段简洁的业务解释。
```

优点：

1. 不改 LLaMA-MoE 权重。
2. 可以马上复用 `/home/wdz/BT/MoE/models/LLaMA-MoE-v1-3_5B-2_8`。
3. 风险低，不影响数值反演模型。

缺点：

1. LLaMA-MoE 不直接提升雨量数值预测。
2. 解释质量依赖前端结构化结果。

适用阶段：当 Stage1 数值模型可用后，用于展示和交互。

### 7.4 方案 C：多编码器输出 soft tokens，接入冻结 LLaMA-MoE

结构：

```text
link encoder          → link tokens
position encoder      → position tokens
weather encoder       → weather tokens
image encoder/probs   → image/weather tokens
satellite embedding   → satellite token
        ↓
projector: encoder_dim → LLaMA-MoE hidden_size
        ↓
[soft tokens] + [text prompt tokens]
        ↓
冻结 LLaMA-MoE
        ↓
回归头 / 文本输出
```

这里不需要改原始 LLaMA-MoE 权重，但需要写一个 wrapper model：

```python
class SatelliteMoEWrapper(nn.Module):
    def __init__(self, llama_moe, encoders, projectors):
        ...

    def forward(structured_inputs, input_ids):
        soft_embeds = projectors(encoders(structured_inputs))
        text_embeds = llama_moe.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([soft_embeds, text_embeds], dim=1)
        outputs = llama_moe(inputs_embeds=inputs_embeds, ...)
```

训练方式：

```text
冻结 LLaMA-MoE 全部参数
训练 encoders + projectors + rainfall head
可选训练少量 LoRA/Adapter
```

雨量输出不建议直接让 LLaMA-MoE 生成数字字符串，而应加数值 head：

```text
last hidden state / soft token hidden state → rainfall_head → softplus → rainfall_mm
```

优点：

1. 不必修改原始 `.bin` 权重。
2. 可以让 LLaMA-MoE 参与跨模态 token 融合。
3. 可同时输出数值和解释。

缺点：

1. 需要改代码写 wrapper 和训练 loop。
2. 显存占用大，训练速度慢。
3. 数据量小，冻结 LLaMA-MoE 时收益未必明显。
4. 如果不设计好回归头，语言模型容易只学到文本模式，不会提升数值反演。

适用阶段：结构化多编码器确认有效后，再尝试。

### 7.5 方案 D：修改 LLaMA-MoE remote code，原生支持多模态输入

做法：

1. 修改 `modeling_llama_moe_hf.py` 或 llama-moe 仓库中的模型代码。
2. 在 forward 中新增 structured inputs：

```python
forward(
    input_ids=None,
    inputs_embeds=None,
    link_features=None,
    position_features=None,
    weather_features=None,
    image_features=None,
    ...
)
```

3. 在模型内部构造 soft tokens，再拼到 text embeddings 前面。
4. 保存新的 config 字段和 adapter/projector 权重。

是否需要改原始权重：

```text
不一定需要改原始 LLaMA-MoE 参数；
但需要新增 encoders/projectors/head 参数；
需要保存新的 checkpoint；
需要维护自定义 remote code。
```

优点：

1. 推理接口统一。
2. 更像完整多模态模型。

缺点：

1. 工程复杂度高。
2. 以后升级权重或换 Qwen 时迁移成本高。
3. 当前数据量不足时容易过拟合。

适用阶段：不建议近期做。

### 7.6 方案 E：在 LLaMA-MoE 原模型上做 LoRA/Adapter 微调

结构：

```text
多编码器 soft tokens + text prompt
        ↓
LLaMA-MoE + LoRA/Adapter
        ↓
rainfall head + text head
```

是否改原始权重：

```text
原始权重可以保持不变；
训练产生 LoRA/Adapter 权重和外部编码器权重。
```

优点：

1. 比全量微调省显存。
2. 可能提升跨模态融合。
3. 保留原始权重，便于回滚。

缺点：

1. 样本少时 LoRA 也可能过拟合。
2. 训练复杂度明显高于纯 Stage1 encoder。
3. 评估必须防止语言模型“看起来会解释，但数值没提升”。

适用阶段：在同卫星差分和图像概率有效后再尝试。

### 7.7 方案对比

| 方案 | 是否改 LLaMA-MoE 原权重 | 是否需要新增模型代码 | 是否直接提升数值反演 | 推荐程度 |
| --- | --- | --- | --- | --- |
| A 结构化多编码器 | 否 | 少量 | 是 | 最高 |
| B LLaMA-MoE 只生成报告 | 否 | 少量 | 否 | 高 |
| C soft tokens 接冻结 LLaMA-MoE | 否 | 中等 | 可能 | 中 |
| D 改 remote code 原生多模态 | 不改原权重但改模型结构 | 高 | 可能 | 低 |
| E LoRA/Adapter 微调 | 不改 base，新增 LoRA | 中高 | 可能 | 中低 |

当前建议：

```text
第一阶段：方案 A
第二阶段：方案 B
第三阶段：如果 A 的特征已经有效，再尝试方案 C/E
暂不做：方案 D
```

## 8. 是否必须在原模型权重上改动

不必须。

推荐原则：

1. 原始 LLaMA-MoE 权重保持只读，不直接改 `.bin`。
2. 新增结构化编码器、projector、rainfall head、classification head，单独保存。
3. 如果需要语言模型参与，优先用 wrapper + frozen LLaMA-MoE。
4. 若微调语言模型，优先 LoRA/Adapter，不做全量微调。
5. 只有在模型产品化、接口需要统一时，才考虑修改 remote code。

推荐 checkpoint 组织：

```text
checkpoints/
  stage1_structured_encoder/
    encoder.pt
    config.yaml
    scaler.pkl

  stage1_llama_moe_wrapper/
    encoders.pt
    projectors.pt
    rainfall_head.pt
    adapter_or_lora.pt
    base_model_path.txt  # 指向 /home/wdz/BT/MoE/models/LLaMA-MoE-v1-3_5B-2_8
```

这样可以做到：

```text
base LLaMA-MoE 不动
业务模型可独立训练、替换、回滚
后续换 Qwen 或 DeepSeek 时也能复用 encoders/projectors 的大部分设计
```

## 9. 推荐近期执行顺序

1. 跑完整三轮改进版：

```bash
cd /home/wdz/BT/Stage1/model
ITERATIONS=3 bash scripts/train_experiments.sh link4_cls_cm
ITERATIONS=3 bash scripts/train_experiments.sh link4_cls_summary_cm
```

2. 实现同卫星 clear-sky baseline 差分特征。
3. 训练图像三分类模型，并导出 `p_cloudy/p_overcast/p_rain`。
4. 将图像概率接入 Stage1 数据集。
5. 对比：

```text
link4_cls
link4_cls + same_satellite_delta
link4_cls + image_probs
link4_cls + same_satellite_delta + image_probs
```

6. 实现结构化多编码器融合模型，即方案 A。
7. 如果数值反演达到可用水平，接 LLaMA-MoE 做报告生成，即方案 B。
8. 如果结构化多编码器有效且仍需进一步提升，再尝试 soft tokens 接冻结 LLaMA-MoE，即方案 C。
9. 暂不修改 LLaMA-MoE 原始权重和 remote code，除非进入产品化统一接口阶段。
