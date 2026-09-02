# 8041权重部署与回滚

## 边界

训练和评估只读NPZ并写入指定的`output-dir`，不会访问8041历史库，也不会自动替换在线权重。8041仅在服务启动时读取`weights/deployed/`下的权重，因此生成候选模型不会影响正在运行的服务。

## 1. 验证候选权重

训练完成后先检查：

```bash
cat outputs/training_runs/<run_id>/metrics.json
```

重点比较`val`和`test`的`mae`、`rainy_mae`、`precision`、`recall`及`f1`。不能仅根据训练集指标替换线上权重。

也可以使用候选目录中的`best.pt`重新导出评估结果：

```bash
python train.py \
  --dataset-path data/reproducible_v1/minute_rainfall_full.npz \
  --output-dir outputs/training_runs/<run_id> \
  --device cuda \
  --evaluate-only 1
```

## 2. 明确要替换的模型

8041使用三个权重，不能混用：

| 文件 | 用途 |
|---|---|
| `weights/deployed/position_model/best.pt` | 001终端且位置匹配成功 |
| `weights/deployed/no_position_fallback/best.pt` | 001终端缺少有效位置时回退 |
| `weights/deployed/new_terminal_transfer/best.pt` | 002/003终端迁移模型 |

本目录的标准001训练只生成第一类模型，不能直接覆盖后两类模型。

## 3. 备份并替换

以下示例仅替换001位置模型：

```bash
cd /home/wdz/BT/Stage1/minute_rain_retrieval
stamp=$(date +%Y%m%d_%H%M%S)
cp weights/deployed/position_model/best.pt \
  weights/deployed/position_model/best.pt.$stamp.bak
cp outputs/training_runs/<run_id>/best.pt \
  weights/deployed/position_model/best.pt
```

替换文件不会改变已经运行进程内的模型，必须重启8041才会生效。历史库不会因替换权重自动清空；若要比较模型版本，应使用新的历史库副本或明确执行离线回填，不要直接覆盖现有历史结果。

## 4. 重启与检查

先停止当前8041进程，再使用原来的screen或进程管理方式启动：

```bash
cd /home/wdz/BT/MoE/lora-moe
PYTHON=/home/wdz/BT/.venv/bin/python \
  bash scripts/serve_three_terminal_minute_rain_demo.sh
```

检查：

```bash
curl -fsS http://127.0.0.1:8041/health
curl -fsS 'http://127.0.0.1:8041/api/rainfall?date=2026-08-09&max_passes=10&recompute=false'
```

## 5. 回滚

停止8041，将备份文件复制回原路径，再启动服务。不要在服务仍运行时删除当前权重或历史数据库。
