# 测试说明

## 自动化测试覆盖

当前共有8个pytest用例，覆盖以下稳定契约：

- 位置只能与相同卫星ID匹配，不能跨卫星最近邻；
- 模型每个分钟样本只输出一个雨量和一个降雨logit；
- SNR硬掩码只排除低质量且原本有效的点；
- SNR软门控权重连续，且不删除上下文token；
- 动态无雨下采样保留全部有雨样本，并支持纯有雨训练；
- 分钟累计值按锚点去重，不按终端重复累加真实雨量；
- 三终端一致性只配对相同锚点时间。
- NumPy 2.x生成的固定NPZ可由已验证的NumPy 1.x环境加载。

## 执行方法

从分钟项目目录执行：

```bash
cd /home/wdz/BT/Stage1/minute_rain_retrieval
python -m pytest -q
```

运行交付前检查：

```bash
python check_delivery.py --db-path /home/wdz/satellite_data/satellite_data.db
```

克隆后只检查训练所需依赖和Git内置数据：

```bash
python check_delivery.py --training-only
```

只检查代码和Python依赖，不要求本地数据与权重：

```bash
python check_delivery.py --code-only
```

以下在线检查仅供持续接收数据的本地服务器使用，不属于同事离线验收要求：

```bash
curl -fsS http://127.0.0.1:8041/health
curl -fsS 'http://127.0.0.1:8041/api/rainfall?date=2026-08-09&max_passes=10&recompute=false'
```

`/health`应返回`status=ok`、三个terminal ID、模型版本、历史库统计和worker状态。第二条命令会读取历史；如果该日期尚未用当前模型完整物化，服务会执行整日推理后再写库。

## 当前结果

2026-09-02在`smoe`环境中执行结果：

```text
8 passed
delivery checks: 33 passed, 0 failed
```

同时通过：核心Python文件`py_compile`、两个Shell入口`bash -n`、规范归档复用评估以及8041实际健康检查。当前部署001有位置模型的测试集指标为：MAE 0.00581 mm、rainy MAE 0.07959 mm、F1 0.6184、recall 0.8360。该测试集采用`stratified_all`划分，不能解释为严格的未来时间泛化结果。

## 尚未覆盖

自动化测试尚未覆盖真实数据库上的端到端重建、浏览器E2E、并发压力、进程异常恢复、视觉模型精度和三终端长期在线稳定性。这些属于当前已知测试缺口，不应由8个单元测试替代。
