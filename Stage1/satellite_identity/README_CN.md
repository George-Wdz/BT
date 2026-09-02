# 历史卫星 ID 统一与位置恢复审计

本目录分两阶段评估历史 LET 卫星 ID 能否统一到 0727 版本，并量化统一后可用于反演的数据。分析和候选生成工具只读访问原始 LET 文件及 `satellite_data.db`；名称以 `apply_` 或 `backfill_` 开头的脚本会先在线备份数据库，再执行明确的写入操作。

## 第一阶段：卫星身份统一

### 获取历史公开轨道历元

历史 LET ID 与物理卫星的可靠对应需要使用相同历元附近的历史
GP/TLE。先在 Space-Track 注册个人账号，然后运行：

```bash
cd /home/wdz/BT/Stage1/satellite_identity
python3 download_gp_history.py
```

脚本会在终端中询问 Space-Track 邮箱和密码；密码输入不可见，凭据不会
写入文件或命令行。随后脚本自动下载与四版 LET 有效历元对应的历史
GP CSV，并保存到 `analysis/history_gp/`。该目录已被 Git 忽略。

也可以预先通过环境变量传入凭据，但共享服务器上更推荐交互输入：

```bash
SPACETRACK_IDENTITY='account@example.com' \
SPACETRACK_PASSWORD='password' \
python3 download_gp_history.py
```

下载完成后应存在 `qianfan_gp_history_0401.csv`、`0429.csv`、`0611.csv`
和 `0727.csv`，以及记录各文件轨道历元数量的 `download_summary.json`。

下载完成后执行物理身份匹配：

```bash
python3 -m pip install -r requirements.txt
python3 match_historical_gp.py
```

安装新的 LET 文件后，必须把它作为新的生效区间加入，不能覆盖上一版本的
历史区间。以下命令示例表示新 LET 自北京时间 2026-08-20 09:40 起生效：

```bash
python3 match_historical_gp.py \
  --latest-let-path '/path/to/new-let.bin' \
  --latest-let-version 0820 \
  --latest-let-start-local-time '2026-08-20T09:40:00'
```

对应的轨道历史文件应保存为
`analysis/history_gp/qianfan_gp_history_0820.csv`。输出的规范 ID 字段及文件名
将随版本生成，例如 `canonical_0820_satellite_id` 和
`accepted_canonical_0820_mapping.csv`。

该步骤联合使用历史 GP/TLE、数据库 ECEF 位置、LET 轨道面和上海终端的
PHY 过境可见性。输出 `analysis/latest/historical_physical_mapping.csv`，
并将结果区分为 `accepted`、`provisional` 和 `unresolved`。程序只读访问
数据库，不会覆盖原始 `satelliteId`。

时间语义必须按数据源区分：PHY 可见性使用接收端北京时间 `localTime` 转
UTC；position 的 ECEF 坐标使用坐标对应的 `bdtTime`；LET 内部轨道点使用
LET 自带 BDT 历元。历史采集中 PHY 的 `bdtTime` 曾发生长时间卡滞，不能
替代实际链路接收时刻。

`analysis/latest/accepted_canonical_0727_mapping.csv` 仅保留源版本物理身份
和最新 0727 ID 均达到 `accepted` 的映射，可作为后续数据库视图或模型
预处理的安全白名单。其余候选仍保留在完整映射中供人工复核。

三终端协议 ID 一致性可独立审计：

```bash
python3 analyze_terminal_protocol_ids.py \
  --terminal-001-db /path/to/terminal001.sqlite3 \
  --terminal-002-db /path/to/terminal002.sqlite3 \
  --terminal-003-db /path/to/terminal003.sqlite3 \
  --operational-mapping analysis/latest/operational_canonical_0727_mapping.csv \
  --physical-mapping analysis/latest/historical_physical_mapping.csv \
  --output-dir analysis/terminal_protocol_audit
```

002/003 的协议 ID 按 `trackNo * 256 + phaseNo` 重建。脚本只把相同 ID 视为
直接对应；不同 ID 的同分钟共现仅导出为复核候选，不会自动写入运行映射。

经确认需要原地统一数据库时，执行 `apply_accepted_mapping.py`。程序先用
SQLite 在线备份生成完整快照，并另存逐行回滚信息；随后只更新高置信
白名单及其对应 LET 生效区间。不会应用 provisional 映射。

项目运行阶段还允许一种受限的连续性映射：同一数值ID已在其他版本获得
唯一物理身份，并且当前版本LET的轨道面与该身份一致时，可将
`numeric_continuity+historical_tle+let_orbit`纳入运行映射。它不会接受一般
的 `provisional` 或 `unresolved` 候选：

```bash
python3 build_operational_mapping.py
python3 apply_accepted_mapping.py \
  --mapping-path analysis/latest/operational_canonical_0727_mapping.csv
```

该映射对同一LET生效区间内的PHY和position使用同一规则，并在写库前生成
完整备份及逐行回滚表。

```bash
python3 build_identity_map.py
```

也可以显式指定文件，版本顺序即映射时间顺序：

```bash
python3 build_identity_map.py \
  --let 0401=/path/to/let_table0401.bin \
  --let 0429=/path/to/let_0429.bin \
  --let 0611=/path/to/let0611.bin \
  --let 0727=/path/to/let0727.bin
```

已验证的 LET 容器结构为：2 字节有效记录数，随后是 1296 个固定 173 字节槽位。每个有效槽位包含 4 字节卫星 ID、1 字节星历点数和最多 8 个 21 字节星历点。星历点前 4 字节可解释为小端 BDT 秒；剩余 17 字节尚无本地协议说明，因此仅保存十六进制值和 SHA-256，不解释为轨道参数。

映射采用保守规则：只有0727目录自身的身份按定义标记为 `accepted`。一个数字ID即使在后续LET版本中连续存在，也只能标记为 `provisional`，因为数字连续不等于物理卫星连续。若ID中途消失后再次出现，则可能发生编号复用，标记为 `unresolved`。在物理轨道验证通过前，`provisional` 和 `unresolved` 均不得用于覆盖数据库原始卫星ID。

新版新增 ID 不会与旧版删除 ID 强制配对。新增条目既可能来自新发射卫星，也可能来自历史卫星改号，仅凭目录集合无法区分二者。工具将其标记为 `new_catalog_entry_or_renumbered_identity_unresolved`，并仅报告目录净增长。

主要输出：

- `analysis/latest/satellite_id_mapping.csv`：原始 ID、0727 规范 ID、证据、状态和置信度；
- `analysis/latest/let_points_opaque.csv`：可追溯的 LET 星历点及未解析载荷；
- `analysis/latest/latest_id_provenance.csv`：0727 每个 ID 的首次出现版本和连续性状态；
- `analysis/latest/daily_position_catalog_scores.csv`：数据库每日位置 ID 与各 LET 版本的重合比例；
- `analysis/latest/let_ecef_validation_anchors.csv`：LET 历元与数据库 BDT 相差不超过 3 秒的同 ID ECEF 锚点，供后续轨道解码验证；
- `analysis/latest/mapping_summary.json`：版本变化和映射统计。

## 第二阶段：历史位置恢复审计

```bash
python3 assess_position_recovery.py
```

审计按当前 Stage1 规则，以相同卫星 ID 和 `bdtTime` 最近邻在 5 秒内匹配位置，并应用当前服务的非零位置字段约束。输出 `analysis/latest/position_recovery_audit.json`。

规范 ID 与历史 ECEF 位置是两个问题。ID 映射可以统一卫星 embedding、统计和 dry baseline，但不能凭空得到某个历史时刻的 ECEF 坐标。没有同时间位置记录的 PHY 数据，只有在 17 字节 LET 轨道载荷获得厂商字段说明并通过数据库 ECEF 验证后，才能利用对应历史 LET 传播并恢复位置。

## 原始位置保留与TLE修复

接收端现在保留位置原始值，仅拒绝无时间、无卫星ID以及同一终端同一卫星
1秒内的近重复记录。主库 `position_data` 保持原始21列结构；修复来源、原始
ID、NORAD编号和TLE历元保存在候选CSV及独立sidecar中。训练和在线推理仍应用
物理有效性过滤，因此未经修复的零值位置不会进入模型。

先审计原始CSV并生成按星历版本、卫星ID划分的质量统计和TLE下载时间窗：

```bash
python3 audit_raw_position_quality.py \
  --csv-path /home/wdz/BT/Stage1/position_data.csv
```

服务开始保留原始异常值后，可直接按终端和时间范围审计实时数据库：

```bash
python3 audit_raw_position_quality.py \
  --db-path /home/wdz/satellite_data/satellite_data.db \
  --terminal-id 01-31-0005-0001 \
  --start-time 2026-08-19
```

根据 `position_quality_by_id.csv` 中的建议日期补充历史TLE时，可覆盖默认下载
窗口。例如补充0727目录对应的完整采集区间：

```bash
python3 download_gp_history.py \
  --window 0727=2026-07-09:2026-08-19 \
  --chunk-days 7
```

`--window`可重复指定不同版本；同一次运行中每个版本只能出现一次。下载器默认
按7天拆分请求，逐块校验后去重合并，避免大时间窗请求超时。命令行结束日期
按包含当日解释，脚本会处理Space-Track右端排他的区间语义。成功分块缓存在
`analysis/history_gp/.chunks/`，后续中断重试会直接复用。

0401 LET记录的内部轨道历元早于实际采集期，因此物理身份匹配还需要覆盖
LET历元本身的历史TLE：

```bash
python3 download_gp_history.py \
  --window 0401=2026-01-10:2026-04-28 \
  --chunk-days 7
```

再生成可追溯的TLE修复候选：

```bash
python3 repair_raw_positions_from_tle.py \
  --csv-path /home/wdz/BT/Stage1/position_data.csv \
  --output-path analysis/position_recovery/repaired_positions.csv \
  --max-anchor-error-km 10 \
  --max-tle-age-days 7
```

修复要求物理身份为 `accepted`，并且具有误差不超过10 km的实测ECEF锚点，
或同时通过LET轨道面与同一内部ID多次PHY过境可见性验证；最近TLE历元距离
待修复时刻还必须不超过7天。输出CSV保留原始ID、NORAD编号、TLE历元和TLE
年龄。脚本只导出候选，不直接修改数据库；人工复核统计与轨道误差后再回填。

确认候选后，使用下列命令在线备份数据库并回填：

```bash
python3 apply_repaired_positions.py \
  --repaired-csv analysis/position_recovery/repaired_positions.csv
```

回填遵循数据库的 `localTime` 唯一约束：该时刻已有实测或重建有效位置时
直接跳过；已有无效原始位置时原位更新为通过验证的TLE位置；不存在该时刻时
才新增记录。整个回填在单一事务中执行，异常时自动回滚。

若历史接收端曾在入库前丢弃无效位置，可先将原始异常行以不可用于模型的
状态补回主库，供后续TLE修复：

```bash
python3 backfill_raw_invalid_positions.py \
  --csv-path /home/wdz/BT/Stage1/position_data.csv
```

脚本应用与PHY相同的版本化规范ID映射，并依赖 `localTime` 唯一索引保证重复
运行不会重复写入。原始ID和质量原因保留在审计输出中；模型读取层继续按物理
有效性规则排除异常记录，直到该时刻被经审核的TLE重建位置原位替换。

即使后续获得高置信映射，也应保留 `raw_satellite_id`，通过独立映射表或新增 `canonical_satellite_id` 字段使用统一身份，不应原地覆盖原始采集字段。

## 一键运行

```bash
bash run_two_stage_analysis.sh
```
