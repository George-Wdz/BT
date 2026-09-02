# 3～5分钟离线演示顺序

## 0:00～0:40：任务与样本定义

说明模型输入是雨量计锚点前一分钟内的多卫星变长序列，输出一个分钟累计降雨量和一个降雨概率。强调不是逐PHY点输出，也不是过境累计模型。

## 0:40～1:20：固定数据版本

打开`data/reproducible_v1/dataset_manifest.json`，展示：

- 共26,409个分钟样本，其中1,256个有雨样本；
- train/val/test为18,486/3,961/3,962；
- 相同卫星ID、5秒位置匹配、5秒气象容差和至少3个PHY点；
- 图像只开源分类标签，不包含原始照片。

说明`minute_rainfall_full.npz`是训练入口，三个独立划分目录用于审计。

## 1:20～2:10：代码主流程

按顺序展示：

1. `dataset.py`：只用训练集估计标准化和dry baseline；
2. `model.py`：20维输入、数值投影、卫星嵌入、Transformer和双输出头；
3. `train.py`：训练集动态无雨下采样、联合损失、验证集选优和三划分评估。

## 2:10～3:20：运行与输出

展示README中的正式训练命令。若时间有限，不现场等待80 epochs训练，可展示一次已完成运行目录：

```text
best.pt
metrics.json
train_predictions.csv
val_predictions.csv
test_predictions.csv
val_rainy_predictions.csv
test_rainy_predictions.csv
val_test_rainy_predictions.csv
```

打开`metrics.json`说明MAE、rainy MAE、precision、recall和F1；再打开`val_test_rainy_predictions.csv`展示时间、真实值、预测值和误差。

## 3:20～4:10：可复现性检查

现场执行：

```bash
python check_delivery.py --training-only
python -m pytest -q
```

完整性检查应报告Git内置数据存在，测试应通过8个稳定契约。

## 4:10～5:00：限制

主动说明：

- 当前固定划分是`stratified_all`，不能等同于严格未来时间泛化；
- 有雨样本占比较低，训练阶段使用动态无雨下采样；
- 在线服务依赖本地实时SQLite、照片和权重，不属于Git克隆后的复现范围；
- 该系统是固定站点研究原型，不应表述为跨地区生产级雨量计替代方案。
