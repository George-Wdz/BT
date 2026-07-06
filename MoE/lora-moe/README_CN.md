# LoRA-MoE：多编码器多 LoRA 原型

[English](README.md) | [中文](README_CN.md)

本目录保存项目自有的参数高效多模态适配代码。它是 LoRA-MoE 路线的早期原型：当前包含 `vision_weather` 专家和 `stage1_rain` 专家，分别用仓库内冻结编码器生成 soft tokens，再输入 Qwen 语言模型。

## 范围

| 项 | 说明 |
| --- | --- |
| 当前任务 | 相机图像天气识别；Stage1 卫星链路降雨反演 |
| 视觉类别 | `sunny`、`cloudy`、`rain` |
| 视觉编码器 | 冻结的 `Stage1/vision_weather/WeatherClassifier` encoder，对应 A1B1 |
| Stage1 编码器 | 冻结的 `PatchEncoderDecoder` 反演模型，对应 A2B2 |
| Projector | 可训练 MLP，将冻结编码器特征映射为 soft tokens |
| 语言模型 | Qwen2.5-14B-Instruct，除 LoRA 外冻结 |
| LoRA 注入层 | `q_proj`、`v_proj` |

该模块与 Stage1 数据构建中直接使用图像分类器导出概率的路径不同。它用于验证模型输入层的多模态适配，不直接替代完整降雨反演流程。

## 结构

```text
image
  -> frozen WeatherClassifier encoder
  -> visual feature vector
  -> trainable projector
  -> visual soft tokens
  -> Qwen2.5-14B-Instruct + vision_weather LoRA
  -> 中文天气回答
```

训练时只更新 projector 和 LoRA 参数。视觉编码器和基座语言模型保持冻结。

Stage1 反演专家结构：

```text
satellite pass features
  -> frozen Stage1 PatchEncoderDecoder
  -> pass-level retrieval feature vector
  -> trainable projector
  -> Stage1 soft tokens
  -> Qwen2.5-14B-Instruct + stage1_rain LoRA
  -> 中文降雨量回答
```

Stage1 首版训练目标很窄：让 Qwen 根据反演 token 输出“本次卫星过境降雨量约为 X 毫米”。回答目标使用冻结 Stage1 小模型自己的预测值，而不是雨量计真实标签，这样训练和在线推理保持一致。雨量计分辨率按 `--no-rain-threshold 0.05` 处理，预测值低于该阈值时按无雨/0mm 表达。这不代表 Stage1 小模型本身已经足够准确；后续 Stage1 权重更新后，应重新训练对应 projector 和 A2B2 LoRA。

## 目录

```text
MoE/lora-moe/
  configs/
  scripts/
  src/lora_moe/
    components.py
    datasets.py
    train/
    infer/
    serve/
```

生成的 adapter、projector 权重、日志和 checkpoint 暂不纳入本仓库。轻量 `Stage1/vision_weather` 默认分类权重已纳入仓库，因为视觉专家本地复现需要匹配的编码器 checkpoint。

## 训练

### 视觉天气 A1B1

Smoke test：

```bash
cd /home/wdz/BT/MoE/lora-moe
conda run --no-capture-output -n smoe bash scripts/train_vision_weather_lora.sh
```

首版完整训练：

```bash
cd /home/wdz/BT/MoE/lora-moe
conda run --no-capture-output -n smoe bash scripts/train_vision_weather_lora.sh \
  --output-dir /home/wdz/BT/MoE/lora-moe/outputs/vision_weather_lora_v1 \
  --max-train-samples 0 \
  --max-steps 0 \
  --epochs 1 \
  --grad-accum-steps 16
```

### Stage1 反演 A2B2

Smoke test：

```bash
cd /home/wdz/BT/MoE/lora-moe
conda run --no-capture-output -n smoe bash scripts/train_stage1_rain_lora.sh
```

演示版训练：

```bash
cd /home/wdz/BT/MoE/lora-moe
conda run --no-capture-output -n smoe bash scripts/train_stage1_rain_lora.sh \
  --output-dir /home/wdz/BT/MoE/lora-moe/outputs/stage1_rain_lora_a2b2_v1 \
  --max-train-samples 0 \
  --max-steps 100 \
  --epochs 1 \
  --grad-accum-steps 16
```

默认 Stage1 checkpoint：

```text
/home/wdz/BT/Stage1/rain_retrieval/model/checkpoints/pass_dataset_rain_retrieval_20260612_1116/stage1_cm_dm256_df512_eh8_el3_dl2_pl8_st4_bs32_lr0.0001_itr0
```

默认 Stage1 pass 数据集：

```text
/home/wdz/BT/Stage1/rain_retrieval/model/data/datasets/pass_dataset_rain_retrieval_20260612_1116.npz
```

如果之后 Stage1 小模型重训好了，请在
`scripts/train_stage1_rain_lora.sh` 中更新 `--stage1-checkpoint-dir`、
`--pass-dataset-path` 和 `--output-dir`，或把这些命令行参数追加到脚本后重新训练 A2B2。

关键默认参数：

| 参数 | 默认值 |
| --- | --- |
| `--batch-size` | `1` |
| `--grad-accum-steps` | smoke test 用 `8`，较大训练用 `16` |
| `--num-visual-tokens` | `8` |
| `--num-stage1-tokens` | `8` |
| `--projector-hidden-dim` | `1024` |
| `--lora-r` / `--lora-alpha` | `8` / `16` |
| `--lora-dropout` | `0.05` |
| `--learning-rate` | `2e-4` |
| `--no-rain-threshold` | `0.05` |

## 推理

```bash
cd /home/wdz/BT/MoE/lora-moe
conda run --no-capture-output -n smoe bash scripts/infer_vision_weather.sh
```

可选输入：

```bash
bash scripts/infer_vision_weather.sh --image /path/to/image.jpg
bash scripts/infer_vision_weather.sh --image-dir /path/to/images
bash scripts/infer_vision_weather.sh --save-jsonl /tmp/vision_weather_predictions.jsonl
```

## 服务

当前服务按专家拆分部署，目的是先分别验证 A1B1 和 A2B2 的训练产物、在线特征构造和 GPU 显存占用。后续视觉、Stage1、Stage2 都稳定后，再合并成统一多专家 FastAPI，由一个入口做规则路由或学习路由。

视觉天气 A1B1：

```bash
cd /home/wdz/BT/MoE/lora-moe
conda run --no-capture-output -n smoe bash scripts/serve_vision_weather_fastapi.sh
```

Stage1 反演 A2B2：

```bash
cd /home/wdz/BT/MoE/lora-moe
conda run --no-capture-output -n smoe bash scripts/serve_stage1_rain_fastapi.sh \
  --output-dir /home/wdz/BT/MoE/lora-moe/outputs/stage1_rain_lora_a2b2_v1
```

默认端口：

```text
vision_weather: http://127.0.0.1:8010
stage1_rain:    http://127.0.0.1:8011
```

Stage1 A2B2 服务启动时会持久化加载：

- Qwen2.5-14B-Instruct；
- Stage1 A2B2 LoRA adapter；
- Stage1 projector；
- 冻结 Stage1 checkpoint；
- `meta.pt` 中的 `cfg`、`scaler_X`、`scaler_y`、`sat_mapper`；
- 从训练 split 固化出来的 dry baseline 状态。

后台线程默认每 30 秒读取 `/home/wdz/satellite_data/satellite_data.db` 的最近窗口数据，只构造最新 pass，不全库重建。没有最新卫星过境时，`/generate` 会直接返回“无最新卫星过境”。

关键在线参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--poll-interval-s` | `30` | 后台轮询间隔 |
| `--stale-after-s` | `180` | 最新 `phy_data` 超过该秒数视为无最新卫星过境 |
| `--lookback-hours` | `4` | 每次只读取最近几小时数据库 |
| `--max-passes` | `8` | 每次最多保留最近几个 pass |
| `--use-best` | 已启用 | 加载 `best/adapter` 和 `best/projector.pt` |
| `--db-path` | `/home/wdz/satellite_data/satellite_data.db` | 实时卫星数据库 |
| `--no-rain-threshold` | `0.05` | 低于该阈值的 Stage1 预测展示为无雨/0mm |

### 双普通 Qwen 互相交流 Demo

该 demo 用于验证“服务层多智能体/伪 MoE”的基本机制：两个普通 Qwen 常驻不同 GPU，Gateway 先调用 Qwen-A 生成初稿，再调用 Qwen-B 复核，最后把复核意见交回 Qwen-A 生成最终回答。它不加载 LoRA，不调用视觉或反演专家，只验证“大模型之间通过 API 把输出作为输入继续交互”的可行性。

浏览器界面采用左右双栏布局：左侧展示 Qwen-A 的初稿和终稿，右侧展示 Qwen-B 的复核意见，底部是用户输入框和推理参数。这样可以直观看到 A/B 的交互过程。

界面默认使用流式输出：底层普通 Qwen 服务通过 `/generate/stream` 逐 token 输出，Gateway 通过 `/debate/stream` 按协作模式转发事件，前端边接收边追加到左右对话框中。

当前保留两种协作模式：

- `B复核`：应用模式，流程固定为 `A 初稿 -> B 复核/补充 -> A 终稿`。后续多个专家 B 接入时，优先沿用这个模式。
- `A/B交互`：技术演示模式，`rounds` 表示 A/B 交互轮数。`rounds=1` 时流程是 `A 初稿 -> B 复核 -> A 终稿`；`rounds=3` 时流程是 `A 初稿 -> B 复核1 -> A 修订1 -> B 复核2 -> A 修订2 -> B 复核3 -> A 终稿`。

Gateway 的职责：

- 对外提供一个统一入口和 Ollama 兼容接口；
- 保存 Qwen-A/Qwen-B 的角色系统提示词；
- 控制调用顺序：应用模式为 A 初稿 -> B 复核 -> A 终稿，技术演示模式为 A/B 多轮交互；
- 维护短期记忆，把同一 `session_id` 最近若干轮问答摘要放回下一轮 prompt；
- 隔离底层 Qwen 服务，让 Qwen-A/Qwen-B 保持无状态通用文本生成器。

当前短期记忆保存在 Gateway 进程内，服务重启后会丢失。它只用于演示和调试；如果后续要稳定给前端长期使用，应改成 SQLite、Redis 或向量库持久化。

启动：

```bash
cd /home/wdz/BT/MoE/lora-moe
conda run --no-capture-output -n smoe bash scripts/serve_dual_qwen_demo.sh
```

默认资源和端口：

```text
Qwen-A:  GPU 0,1  http://127.0.0.1:8131
Qwen-B:  GPU 2,3  http://127.0.0.1:8132
Gateway:          http://0.0.0.0:8030
```

普通 FastAPI 调用：

```bash
curl -X POST http://127.0.0.1:8030/debate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"服务层多智能体和模型内部 MoE 有什么区别？","mode":"review_then_final","max_new_tokens":256,"temperature":0.0,"rounds":1}'
```

流式 FastAPI 调用：

```bash
curl -N -X POST http://127.0.0.1:8030/debate/stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"服务层多智能体和模型内部 MoE 有什么区别？","mode":"interactive","max_new_tokens":256,"temperature":0.0,"rounds":3}'
```

Ollama 兼容调用：

```bash
curl http://127.0.0.1:8030/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"dual-qwen-demo","prompt":"服务层多智能体和模型内部 MoE 有什么区别？","stream":false}'
```

如果前端只支持 Ollama 服务地址，把前端的 Ollama base URL 指到：

```text
http://服务器IP:8030
```

模型名使用：

```text
dual-qwen-demo
```

请求中可传入：

```json
{
  "session_id": "default",
  "use_memory": true,
  "reset_memory": false
}
```

`session_id` 用来区分不同会话；`reset_memory=true` 会在本轮调用前清空该会话短期记忆。

## 产物管理

以下产物暂不纳入本仓库：

- Qwen 基座权重；
- 除当前默认 `Stage1/vision_weather` checkpoint 外的新 WeatherClassifier 权重；
- Stage1 checkpoint 和 `meta.pt`；
- Stage1 NPZ 数据集；
- 在线卫星数据库；
- LoRA adapter 输出；
- `projector.pt`；
- 日志和中间 checkpoint；
- 图像数据集。

可共享模型产物后续可单独发布，例如存放在私有 Hugging Face Model 仓库或单位对象存储中。
