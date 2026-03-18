# Spatial Attention Heatmap 使用说明

本说明文档用于运行并分析 ST-GAT 空间注意力热图，覆盖以下内容：

- 单次热图可视化（支持关节名标签）
- `diag_mean / offdiag_mean / diag_over_offdiag` 三项统计
- 自动标注 top-k 跨关节连边（非对角）
- `time_index` 扫描导出（如 `0~19` 全部时间步）

> 说明：当前实现提取的是 **编码器最后一层 ST_GAT** 的空间注意力。  
> 时间维来自模型内部 DCT 特征长度（例如 20），并非原始视频帧索引。

---

## 1. 前置条件

1. 已准备数据集（`data/`）和模型权重（`results/.../models/*.p`）
2. Python 环境包含依赖：
   - `torch`
   - `numpy`
   - `matplotlib`
3. 在仓库根目录执行命令

---

## 2. 脚本总览

### 2.1 单次可视化脚本

```bash
python3 visualize_spatial_attention.py ...
```

功能：
- 导出某个动作样本在某个 `time_index`（或时间平均）的热图
- 保存注意力矩阵 `.npy`
- 输出并保存统计指标 + top-k 连边信息 `.json`
- 图中可直接标注 top-k 跨关节边（红框+数字）

### 2.2 时间扫描脚本

```bash
python3 visualize_spatial_attention_time_sweep.py ...
```

功能：
- 批量导出一段时间范围（如 `0~19`）的热图
- 每个时间步输出三项统计
- 汇总生成 `csv/json`

---

## 3. 快速开始

### 3.1 单张热图（示例）

```bash
python3 visualize_spatial_attention.py \
  --cfg humaneva \
  --checkpoint results/humaneva/models/0500.p \
  --action_keyword throwcatch \
  --time_index 10 \
  --topk_edges 5
```

### 3.2 运行 time_index 0~19 全部热图（示例）

```bash
python3 visualize_spatial_attention_time_sweep.py \
  --cfg humaneva \
  --checkpoint results/humaneva/models/0500.p \
  --action_keyword throwcatch \
  --time_start 0 \
  --time_end 19 \
  --topk_edges 5
```

### 3.3 指定 checkpoint 的两种方式

#### 方式 A：显式路径（推荐）

```bash
--checkpoint results/h36m/models/0500.p
```

#### 方式 B：用 `--iter` 让脚本自动拼路径

```bash
--iter 500
```

会按配置中的 `cfg.model_path` 自动定位，如 `results/h36m/models/0500.p`。

---

## 4. 输出文件说明

### 4.1 `visualize_spatial_attention.py`

默认输出目录：

```text
results/<cfg>/results/attention_maps/
```

常见输出：

- `*_attn.npy`：`[B, H, V, V]`（时间平均后的空间注意力，仍保留多头）
- `*_heatmap.png`：热图图像
- `*_stats.json`：统计与 top-k 边信息

`*_stats.json` 关键字段示例：

- `stats.diag_mean`
- `stats.offdiag_mean`
- `stats.diag_over_offdiag`
- `topk_cross_joint_edges`（每条边含 query/key 关节名、索引、权重）

### 4.2 `visualize_spatial_attention_time_sweep.py`

默认输出目录：

```text
results/<cfg>/results/attention_maps/time_sweep/
```

常见输出：

- `*_t00_heatmap.png ... *_t19_heatmap.png`
- `*_stats.csv`：每个时间步三项统计
- `*_stats_topk.json`：每个时间步 top-k 边详情
- `*_attn.npy`：`[B, H, V, V]`（空间平均注意力）

---

## 5. 指标解读（重点）

设热图矩阵为 `A`，大小 `V x V`：

- 行（Query）表示“哪个关节在看”
- 列（Key）表示“看哪个关节”
- 每行来自 softmax 归一化，因此行和接近 1

### 5.1 `diag_mean`

主对角线平均值：衡量“关节看自己”的强度。

### 5.2 `offdiag_mean`

非对角元素平均值：衡量“跨关节关注”的整体强度。

### 5.3 `diag_over_offdiag`

比值：

```text
diag_over_offdiag = diag_mean / offdiag_mean
```

典型解释：

- `> 1`：自关注相对更强
- `≈ 1`：自关注与跨关节关注接近
- `< 1`：跨关节关注整体更强（常见于强协同动作）

> 注意：对角线不一定必须最亮。  
> 对动作预测任务而言，非对角高亮往往更能说明模型捕捉到了功能协同（如手-头、对侧手脚等）。

---

## 6. top-k 跨关节连边如何看

`top-k` 默认从非对角元素中选权重最大的边：

- `QueryJoint -> KeyJoint`
- 权重越高，表示该 query 关节更依赖对应 key 关节信息

可用于：

1. 动作特异性分析（walk/run/smoking/throwcatch 的差异）
2. 模型诊断（是否塌缩到单关节或过于均匀）
3. 消融对比（不同模型/epoch/数据集）

---

## 7. 可选参数列表

### 7.1 `visualize_spatial_attention.py`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--cfg` | `h36m` | 数据集配置：`h36m` / `humaneva` |
| `--split` | `test` | 数据划分：`train` / `test` |
| `--gpu_index` | `0` | GPU 索引 |
| `--iter` | `500` | 未提供 `--checkpoint` 时使用 |
| `--checkpoint` | `None` | 模型权重路径（优先级高于 `--iter`） |
| `--action_keyword` | `walking` | 动作关键字（大小写不敏感） |
| `--sample_index` | `0` | 选择第几个匹配样本 |
| `--clip_start` | `None` | 片段起始位置，默认中间片段 |
| `--time_index` | `None` | 时间索引；`None` 表示时间平均 |
| `--seed` | `1` | 随机种子 |
| `--save_npy` | `None` | 自定义注意力 `.npy` 输出路径 |
| `--save_fig` | `None` | 自定义热图路径 |
| `--save_stats` | `None` | 自定义统计 JSON 路径 |
| `--hide_joint_labels` | `False` | 用索引替代关节名标签 |
| `--topk_edges` | `5` | 标注 top-k 非对角连边；`0` 关闭 |

### 7.2 `visualize_spatial_attention_time_sweep.py`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--cfg` | `h36m` | 数据集配置 |
| `--split` | `test` | 数据划分 |
| `--gpu_index` | `0` | GPU 索引 |
| `--iter` | `500` | 未提供 `--checkpoint` 时使用 |
| `--checkpoint` | `None` | 模型权重路径 |
| `--action_keyword` | `walking` | 动作关键字 |
| `--sample_index` | `0` | 第几个匹配样本 |
| `--clip_start` | `None` | 片段起始位置 |
| `--time_start` | `0` | 起始时间索引（可负数） |
| `--time_end` | `19` | 结束时间索引（可负数，含端点） |
| `--seed` | `1` | 随机种子 |
| `--topk_edges` | `5` | 每个时间步标注 top-k 连边 |
| `--hide_joint_labels` | `False` | 用索引替代关节名 |
| `--output_dir` | `None` | 输出目录，默认 `.../time_sweep/` |
| `--prefix` | `None` | 输出文件前缀 |
| `--save_npy` | `None` | 自定义注意力 `.npy` 路径 |

---

## 8. 常见问题

### Q1. 为什么对角线不是亮线？

A：不是错误。该注意力用于动作预测，模型会优先关注“对当前关节最有帮助”的关节，跨关节高亮是正常且有意义的。

### Q2. `time_index` 是原始帧吗？

A：不是。当前是模型内部特征时间维（DCT 相关），可理解为“时序特征索引”。

### Q3. 动作关键字怎么选？

A：脚本是子串匹配。HumanEva 常见可用关键字：`walking`, `jog`, `box`, `throwcatch`, `gestures`。

---

## 9. 推荐实验设置（可直接复现）

1. 固定 checkpoint，动作各取一个代表样本（walking / throwcatch / gestures）
2. 先看 `temporal_mean` 热图（`--time_index` 不传）
3. 再跑 `time_start=0, time_end=19` 观察随时间变化
4. 对比三项统计与 top-k 连边稳定性
5. 把图 + CSV/JSON 汇总进论文/附录

