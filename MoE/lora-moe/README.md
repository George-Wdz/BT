# LoRA-MoE 路线 B 原型

本目录用于实现路线 B：在强中文基座模型上，用多个 LoRA/Adapter 承接不同任务，并逐步加入路由机制。当前先从视觉天气识别任务开始，目标是验证“视觉编码器输出 -> 统一 token -> Qwen -> 中文回答”的链路。

## 当前结论

现有天气图像分类模型可以先作为视觉编码器使用，不建议一开始重新训练。

理由：

- 现有模型已经能稳定输出 `sunny/cloudy/rain`，说明它的 CNN 特征已经包含天气判别信息。
- `WeatherClassifier` 里已经有 `extract_features()`，可以直接取分类头之前的特征；当前权重实际输出为 256 维，具体维度由 checkpoint 里的 `resnet_width` 决定。
- 路线 B 的第一步不是重新做视觉识别，而是让 LLM 学会理解视觉 encoder 的连续特征。
- 先冻结视觉 encoder，可以把训练风险集中在 projector 和 LoRA 上，显存、数据量、调试难度都更低。

当前首版结构：

```text
image
  -> frozen WeatherClassifier.encoder
  -> feature vector
  -> trainable projector
  -> visual soft tokens
  -> Qwen2.5-14B-Instruct + vision_weather LoRA
  -> 中文天气识别回答
```

这一步仍然不是完整 MoE-LoRA，但它是后续 MoE-LoRA 的第一个专家适配器。

## 和方案 A 的区别

方案 A 是服务层工具调用：

```text
图片 -> 分类器 -> 分类概率文本 -> Qwen 总结
```

路线 B 是模型输入层融合：

```text
图片 -> 视觉编码器 -> 连续向量 token -> Qwen + LoRA
```

路线 B 不再把分类结果先转成文本，而是把 encoder 特征投影成 Qwen 可以读取的软 token。后续 Stage1、Stage2 也可以采用同样形式：

```text
Stage1 encoder -> projector -> soft tokens -> Qwen + stage1 LoRA
Stage2 encoder -> projector -> soft tokens -> Qwen + stage2 LoRA
Vision encoder -> projector -> soft tokens -> Qwen + vision LoRA
```

## 目录结构

```text
lora-moe/
  README.md
  configs/
    vision_weather_lora.yaml
  scripts/
    train_vision_weather_lora.sh
  src/
    lora_moe/
      __init__.py
      components.py
      datasets.py
      train/
        vision_weather_lora.py
```

约定：

- `scripts/` 只放 `.sh` 启动脚本，负责设置环境变量、日志路径、常用超参。
- `src/lora_moe/` 放可复用 Python 包代码。
- `src/lora_moe/components.py` 先集中放 encoder、projector、router 类，便于早期快速演进；如果后续类数量明显变多，再拆成 `encoders/`、`projectors/`、`routers/` 子包。
- `src/lora_moe/train/` 放训练入口，使用 `python -m lora_moe.train.xxx` 启动。

## 第一阶段目标

先做一个最小可训练 Demo：

输入一张天气图片，模型输出中文类别说明，例如：

```text
这张图像的天气是晴天。
```

训练时：

- Qwen 主干冻结。
- 视觉分类器默认冻结。
- 只训练：
  - 视觉 projector；
  - Qwen 上的 `vision_weather` LoRA。

这样做不会破坏 Qwen 的语言能力。

## 默认路径

当前默认使用：

```text
Qwen:
/home/wdz/BT/MoE/models/Qwen2.5-14B-Instruct

视觉分类权重:
/home/wdz/LLaMA-Factory/leo_model/vision/weights/20260605_182036_weather_cls_more_cloudy_gpu_30ep_best_model.pt

图像 split:
/home/wdz/LLaMA-Factory/leo_model/data/vision_weather/split
```

当前 split 大致为：

```text
train/rain    824
train/sunny   3000
train/cloudy  3000
val/rain      103
val/sunny     300
val/cloudy    300
```

注意：test split 里 sunny/cloudy 数量很大，暂时不要把 test 当训练集混进去。

## 训练命令

先用很小样本做 smoke test：

```bash
cd /home/wdz/BT/MoE/lora-moe

CUDA_VISIBLE_DEVICES=0,1 \
conda run --no-capture-output -n smoe bash scripts/train_vision_weather_lora.sh
```

也可以用环境变量覆盖常用参数：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MAX_TRAIN_SAMPLES=64 \
MAX_STEPS=20 \
OUTPUT_DIR=/home/wdz/BT/MoE/lora-moe/outputs/vision_weather_lora_smoke \
conda run --no-capture-output -n smoe bash scripts/train_vision_weather_lora.sh
```

如果 smoke test 能跑通，再扩大：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUTPUT_DIR=/home/wdz/BT/MoE/lora-moe/outputs/vision_weather_lora_v1 \
MAX_TRAIN_SAMPLES=0 \
MAX_STEPS=0 \
GRAD_ACCUM_STEPS=16 \
EPOCHS=1 \
conda run --no-capture-output -n smoe bash scripts/train_vision_weather_lora.sh
```

## 超参数怎么选

首轮目标是“链路跑通 + 不破坏 Qwen 语言能力”，不是追求一次训练到最优。建议按三档推进。

### 1. Smoke test

用于确认代码、显存、多卡切分、反向传播都没问题。

```bash
CUDA_VISIBLE_DEVICES=0,1
MAX_TRAIN_SAMPLES=64
MAX_STEPS=20
BATCH_SIZE=1
GRAD_ACCUM_STEPS=8
LORA_R=8
NUM_VISUAL_TOKENS=8
LEARNING_RATE=2e-4
```

判断标准：

- 能完成训练并保存 `adapter/` 和 `projector.pt`。
- loss 能正常下降或至少不是 NaN。
- GPU 不 OOM。

如果 OOM：

- 优先增加 GPU 数量，例如 `CUDA_VISIBLE_DEVICES=0,1,2,3`。
- 其次确认当前默认是 `q_proj,v_proj`；如果仍然 OOM，再减少 GPU 上其他进程或考虑 4bit QLoRA。
- 再考虑安装 `bitsandbytes` 做 4bit QLoRA。

### 2. 小规模有效训练

用于观察模型是否真的学会视觉 token 到中文天气回答。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
MAX_TRAIN_SAMPLES=512
MAX_STEPS=200
BATCH_SIZE=1
GRAD_ACCUM_STEPS=16
LORA_R=8
LORA_ALPHA=16
NUM_VISUAL_TOKENS=8
LEARNING_RATE=2e-4
```

如果训练后回答仍完全不看图像：

- 先确认推理时加载了同一个 `projector.pt` 和 LoRA adapter。
- 把 `MAX_STEPS` 提到 `500`。
- 把 `NUM_VISUAL_TOKENS` 从 `8` 提到 `16`。
- 如果仍不行，再把 `LORA_R` 从 `8` 提到 `16`。

### 3. 完整首版训练

用于得到可演示的 vision_weather 专家适配器。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
MAX_TRAIN_SAMPLES=0
MAX_STEPS=0
EPOCHS=1
BATCH_SIZE=1
GRAD_ACCUM_STEPS=16
LORA_R=8
LORA_ALPHA=16
NUM_VISUAL_TOKENS=8
LEARNING_RATE=2e-4
```

当前数据量不大，先不要多轮训练。若训练集表现很好但新图片回答变差，通常是过拟合，应优先：

- `EPOCHS` 保持 1。
- `LORA_DROPOUT` 从 `0.05` 提到 `0.1`。
- 增加回答模板多样性，而不是继续加训练轮数。

### 关键参数解释

`BATCH_SIZE`：
每一步实际送进 Qwen 的图片数。14B 显存压力大，先固定为 `1`。

`GRAD_ACCUM_STEPS`：
梯度累积步数。有效 batch 约等于：

```text
effective_batch = BATCH_SIZE * GRAD_ACCUM_STEPS
```

当前推荐有效 batch 在 `8-16`，所以 `BATCH_SIZE=1` 时设 `8` 或 `16`。

`MAX_STEPS`：
限制优化步数。调试时比 `EPOCHS` 更可控。设为 `0` 表示不限制，按 `EPOCHS` 跑。

`MAX_TRAIN_SAMPLES`：
限制训练样本数量。调试用 `64/512`，正式训练用 `0`。

`LEARNING_RATE`：
只训练 LoRA 和 projector，`2e-4` 是合理起点。若 loss 抖动、输出变差或出现 NaN，降到 `1e-4`。

`NUM_VISUAL_TOKENS`：
视觉特征映射成多少个软 token。三分类天气任务信息量不大，`8` 足够；如果后续要描述云量、雨强、能见度，可试 `16`。

`PROJECTOR_HIDDEN_DIM`：
projector MLP 的隐层宽度。当前视觉 encoder 输出 256 维，Qwen hidden size 较大，`1024` 是保守选择。一般不优先调这个。

`LORA_R`：
LoRA 的秩，决定适配器容量和显存。推荐：

```text
r=8   首选，省显存，适合简单视觉天气识别
r=16  如果 r=8 欠拟合，再尝试
r=32  暂时不建议，容易增加显存和过拟合风险
```

`LORA_ALPHA`：
LoRA 缩放参数，通常取 `2 * LORA_R`。例如 `r=8, alpha=16`。

`LORA_TARGET_MODULES`：
LoRA 注入到哪些 Qwen 线性层。当前默认是业内最常见、最保守的 Q/V 配置：

```text
q_proj,v_proj
```

这会在每层注意力的 Q、V 投影矩阵旁边增加 LoRA A/B 矩阵，例如：

```text
q_proj.lora_A / q_proj.lora_B
v_proj.lora_A / v_proj.lora_B
```

这样做的优点是显存省、训练稳定、对基座语言能力影响小，适合当前三分类视觉天气识别首版。

如果 Q/V 学不动，可以按下面顺序逐步扩大：

```text
q_proj,v_proj,o_proj
q_proj,k_proj,v_proj,o_proj
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

最后一档是更强的全 linear LoRA/QLoRA 风格，但对小数据任务更容易过拟合，也更容易让回答风格变窄。当前不建议默认使用。

旧版 smoke test 曾使用：

```text
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

那次 `trainable params` 约为 `34,406,400`，比现在 Q/V 默认更多。改成 Q/V 后，可训练参数会明显减少。

`DTYPE`：
4090 推荐 `bfloat16`。如果环境或驱动不支持，再试 `float16`。

`DEVICE_MAP`：
14B bf16 需要多卡，推荐 `auto`。除非做量化或模型很小，否则不要单卡硬塞。

## 显存建议

Qwen2.5-14B-Instruct 的 bf16 权重大约 28GB，单张 4090 放不下。

建议：

- smoke test：先用 2 张 4090。
- 正式 LoRA：优先用 4 张 4090。
- 如果显存仍不够，再安装 `bitsandbytes`，改成 4bit QLoRA。

当前脚本先支持 bf16 + `device_map=auto`。4bit QLoRA 是下一步增强，不在首版强依赖。

## Checkpoint 和早停

训练产物默认保存在 `OUTPUT_DIR`，例如：

```text
/home/wdz/BT/MoE/lora-moe/outputs/vision_weather_lora_smoke/
  adapter/          # PEFT LoRA adapter，里面保存 q_proj/v_proj 的 LoRA A/B 矩阵
  projector.pt      # 视觉特征到 Qwen soft tokens 的 projector
  train_args.json   # 本次训练参数
  train_state.json  # 当前训练状态，包含 global_step、loss、早停状态等
```

如果 `SAVE_STEPS>0`，会额外保存中途 checkpoint：

```text
OUTPUT_DIR/
  checkpoints/
    step_000010/
      adapter/
      projector.pt
      train_state.json
```

当前脚本默认：

```bash
SAVE_STEPS=10
```

所以 smoke test 在第 10、20... 个优化步会保存中途 checkpoint。不过你刚才那次 smoke test 用的是旧脚本，还没有中途 checkpoint，只保存了最终：

```text
outputs/vision_weather_lora_smoke/adapter/
outputs/vision_weather_lora_smoke/projector.pt
outputs/vision_weather_lora_smoke/train_args.json
```

早停默认关闭：

```bash
EVAL_STEPS=0
EARLY_STOPPING_PATIENCE=0
```

如果要开启早停，可以这样：

```bash
EVAL_STEPS=50 \
EARLY_STOPPING_PATIENCE=3 \
MAX_VAL_SAMPLES=128 \
conda run --no-capture-output -n smoe bash scripts/train_vision_weather_lora.sh
```

含义：

- 每 50 个优化步在 val split 上算一次 `val_loss`。
- 如果连续 3 次验证没有明显改善，就停止训练。
- `MAX_VAL_SAMPLES=128` 是为了避免验证太慢。

首版建议：smoke test 不开早停；小规模有效训练可以开；完整训练如果只跑 1 个 epoch，早停不是必须。

## 为什么不是直接用 logits

可以把分类器 logits/probability 转成文本交给 Qwen，但那还是方案 A 的工具调用风格。

路线 B 更应该让模型读取连续特征：

- `logits` 只有 3 维，信息太少，只表示分类结果。
- encoder feature 有更多图像语义信息，后续可以扩展到更多视觉任务。
- projector 统一输出 Qwen hidden size 的 soft tokens，和 Stage1/Stage2 的 encoder 路线一致。

首版仍会保留 label 监督，即训练目标是中文天气标签回答；但输入给 Qwen 的不是标签文本，而是视觉特征 token。

## 后续 MoE-LoRA 路线

路线 B 推荐分三步：

1. 单专家 LoRA
   - `vision_weather`：图像天气识别。
   - `stage1_inversion`：链路/卫星过境降雨反演。
   - `stage2_forecast`：长期天气预测。

2. 多 LoRA 手动切换
   - 根据任务类型手动 `set_adapter()`。
   - 这一步已经能支撑论文/项目演示里的“多任务适配器”。

3. Router 训练
   - 输入文本 query + modality metadata + encoder summary。
   - 输出选择哪个 LoRA，或给多个 LoRA 分配权重。
   - 初期建议 Top-1，后续再做 Top-k 混合。

真正的 LoRA-MoE 不是删除 Qwen 的 FFN，也不是重训基座模型，而是在冻结基座上增加多个可训练低秩专家和路由。

## 当前风险

- 这个视觉 encoder 是轻量 CNN，能力只覆盖当前三分类天气任务；如果后续要复杂图像理解，需要换 CLIP/SigLIP/EVA 等更强视觉编码器。
- 首版训练目标很简单，可能学成“根据视觉 token 输出类别”，不代表已经具备复杂视觉推理。
- 如果 LoRA 训练数据只包含固定模板回答，模型会倾向模板化输出；后续需要加入多样化指令。
- 训练时必须控制上下文长度，软 token 数量不要一开始设太大，推荐 `8` 或 `16`。

## 当前推荐

先不要重训视觉分类模型。先冻结它，训练 projector + LoRA。

只有出现以下情况，才考虑重新训练或升级视觉 encoder：

- 新验证集上三分类准确率明显下降；
- 需要识别更细粒度天气，例如小雨/中雨/大雨、雾、雪；
- 需要从图像中提取云量、能见度、地面积水等连续变量；
- 需要支持更复杂的视觉问答，而不只是天气分类。
