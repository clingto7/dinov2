# extract_feature/ — DINOv2 特征提取与多样性分析工具

## 概述

使用 DINOv2 预训练模型从图像和视频中提取视觉特征的工具脚本集。包含基础特征提取、子集多样性分析、可视化绘图、本地数据集管理等功能。还包含 RoboTwin VLA 数据集的加载器、episode 多样性分析和基于多样性的数据选择 pipeline。用于 VLA 数据集分析、数据多样性评估、数据选取 baseline 等。

## 文件

### 核心特征提取脚本

| 文件 | 功能 | 关键参数 |
|------|------|---------|
| `extract_image_features.py` | 从图片批量提取 CLS/patch/中间层特征 | `--model`, `--image-dir`, `--demo`, `--output`, `--intermediate-layers` |
| `extract_video_features.py` | 从视频逐帧提取特征 + 时序/空间分析，支持多视频demo和视频间相似度矩阵。AV1 视频自动通过 PyAV 回退解码 | `--model`, `--video`, `--demo`, `--demo-names`, `--demo-count`, `--fps`, `--max-frames`, `--output`, `--output-dir` |

### 子集多样性分析脚本

| 文件 | 功能 | 关键参数 |
|------|------|---------|
| `analyze_image_subset_diversity.py` | 对图片集合生成子集，计算多样性指标，输出CSV+图表 | `--image-dir`, `--demo`, `--subset-size`, `--n-subsets`, `--strategy`, `--output-dir` |
| `analyze_video_subset_diversity.py` | 对视频集合生成子集，以temporal mean作为视频嵌入，同上分析 | `--video-dir`, `--demo`, `--subset-size`, `--n-subsets`, `--strategy`, `--fps`, `--max-frames`, `--output-dir` |

### RoboTwin VLA 数据集工具

| 文件 | 功能 | 关键参数 |
|------|------|---------|
| `robotwin_loader.py` | RoboTwin-LeRobot-v3.0 数据加载器：元数据解析、视频下载、episode 帧提取（AV1 via PyAV） | `--list-tasks`, `--task`, `--variant`, `--extract-episode`, `--target-fps`, `--max-frames` |
| `analyze_robotwin_diversity.py` | 对 RoboTwin task 的 episode 子集进行多样性分析：提取 DINOv2 CLS 特征 → temporal mean 嵌入 → 子集指标 | `--task`, `--demo`, `--subset-size`, `--n-subsets`, `--camera`, `--target-fps`, `--max-frames`, `--output-dir` |
| `select_episodes.py` | 基于多样性的 episode 选择：4种策略（greedy_maxdist, greedy_maxvar, kmeans, random）+ 策略对比 | `--features`, `--task`, `--demo`, `--n-select`, `--strategy`, `--compare-all`, `--output-dir` |

### 工具模块

| 文件 | 功能 |
|------|------|
| `diversity_metrics.py` | 多样性指标计算：pairwise cosine distance, L2 distance, feature variance |
| `subset_sampling.py` | 子集采样策略：random, sliding window |
| `plotting_utils.py` | 可视化：直方图、箱线图、PCA散点图、相似度热力图（matplotlib） |

### 数据集管理

| 文件 | 功能 | 关键参数 |
|------|------|---------|
| `manage_local_datasets.py` | 下载/管理本地demo数据集（COCO图片 + 多源视频） | `--list`, `--download images\|videos\|all`, `--status`, `--data-dir` |

### 文档

| 文件 | 功能 |
|------|------|
| `GUIDE.md` | 详细的脚本使用指南和DINOv2特征提取原理讲解 |

## 运行方式

```bash
# 必须设置 XFORMERS_DISABLED=1（推理不需要 xFormers）
# 使用 uv run 自动激活 .venv

# ====== 基础特征提取 ======

# 图像特征提取 - demo 模式
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_image_features.py --demo

# 图像特征提取 - 指定目录
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_image_features.py --image-dir <DIR> --output features.pt

# 视频特征提取 - demo 模式（单视频）
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_video_features.py --demo --demo-count 1 --max-frames 20

# 视频特征提取 - 多视频 demo（带视频间相似度矩阵）
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_video_features.py --demo --demo-count 3 --max-frames 10 --output-dir extract_feature/outputs/multi_video_demo

# 视频特征提取 - 指定视频
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_video_features.py --video <FILE> --fps 2 --output vid_feats.pt

# ====== 子集多样性分析 ======

# 图片多样性分析 - demo 模式
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/analyze_image_subset_diversity.py --demo --subset-size 3 --n-subsets 5 --output-dir extract_feature/outputs/image_diversity_demo

# 图片多样性分析 - 指定目录
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/analyze_image_subset_diversity.py --image-dir extract_feature/local_datasets/images --subset-size 5 --n-subsets 10 --output-dir extract_feature/outputs/image_diversity

# 视频多样性分析 - demo 模式
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/analyze_video_subset_diversity.py --demo --subset-size 2 --n-subsets 3 --max-frames 10 --output-dir extract_feature/outputs/video_diversity_demo

# 视频多样性分析 - 指定目录
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/analyze_video_subset_diversity.py --video-dir extract_feature/local_datasets/videos --subset-size 3 --n-subsets 5 --fps 1 --max-frames 15 --output-dir extract_feature/outputs/video_diversity

# ====== 数据集管理 ======

# 列出可用数据集
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/manage_local_datasets.py --list

# 查看下载状态
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/manage_local_datasets.py --status

# 下载所有 demo 数据
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/manage_local_datasets.py --download all

# 只下载图片
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/manage_local_datasets.py --download images

# ====== RoboTwin VLA 数据集 ======
# 注意: RoboTwin 命令需要取消 SOCKS 代理 (env -u ALL_PROXY -u all_proxy)

# 列出所有可用 task (50个)
env -u ALL_PROXY -u all_proxy XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/robotwin_loader.py --list-tasks

# 查看某 task 元数据
env -u ALL_PROXY -u all_proxy XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/robotwin_loader.py --task beat_block_hammer --variant clean

# 提取某 episode 的帧
env -u ALL_PROXY -u all_proxy XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/robotwin_loader.py --task beat_block_hammer --extract-episode 0 --target-fps 2 --max-frames 5 --output-dir extract_feature/outputs/robotwin_frames

# RoboTwin episode 多样性分析 - demo 模式 (beat_block_hammer, 前10个episode)
env -u ALL_PROXY -u all_proxy XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/analyze_robotwin_diversity.py --demo --subset-size 3 --n-subsets 5 --output-dir extract_feature/outputs/robotwin_diversity_demo

# RoboTwin episode 多样性分析 - 指定 task
env -u ALL_PROXY -u all_proxy XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/analyze_robotwin_diversity.py --task beat_block_hammer --variant clean --n-episodes 20 --subset-size 5 --n-subsets 10 --output-dir extract_feature/outputs/robotwin_diversity

# Episode 选择 - 从预计算特征 (先跑 analyze_robotwin_diversity.py)
env -u ALL_PROXY -u all_proxy XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/select_episodes.py --features extract_feature/outputs/robotwin_diversity_demo/features_and_metrics.pt --n-select 5 --strategy greedy_maxdist --compare-all --output-dir extract_feature/outputs/robotwin_selection

# Episode 选择 - demo 模式 (standalone, 自动提取特征)
env -u ALL_PROXY -u all_proxy XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/select_episodes.py --demo --n-select 5 --strategy greedy_maxdist --compare-all --output-dir extract_feature/outputs/robotwin_selection_demo
```

## 架构

### 特征提取流程

两个核心脚本共享相同模式：
1. **模型加载**: `torch.hub.load()` 从本地 repo 加载 → `DINOV2_REPO` 指向仓库根目录
2. **预处理**: Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet mean/std)
3. **特征提取**: 三种 API：
   - `model(x)` → CLS token `[B, D]`
   - `model.forward_features(x)` → dict(cls_token, patch_tokens, reg_tokens)
   - `model.get_intermediate_layers(x, n=k)` → 中间层特征
4. **统计分析**: 特征方差（数据多样性）、余弦相似度（帧间相似性/场景变化）

### 视频帧采样与 AV1 回退

`extract_video_features.py` 的 `sample_frames()` 函数实现了双后端采样策略：

1. **`_sample_frames_opencv()`**: 默认路径，使用 OpenCV `cv2.VideoCapture`，返回 BGR 帧 (numpy array)
2. **`_sample_frames_pyav()`**: 回退路径，使用 PyAV `av.open()`，返回 RGB 帧 (numpy array)
3. **`sample_frames()`**: 调度器 — 先 OpenCV，若获取 0 帧则自动切换 PyAV

返回值为 4-tuple: `(frames, actual_fps, info, is_rgb)`
- `is_rgb=False`: OpenCV 路径，帧为 BGR 格式
- `is_rgb=True`: PyAV 路径，帧已为 RGB 格式

`frames_to_batch()` 接受 `is_rgb` 参数，当 `is_rgb=True` 时跳过 BGR→RGB 转换。

**典型场景**: H.264/MJPEG 视频走 OpenCV，AV1 视频（如 LeRobot/RoboTwin）自动走 PyAV。

### 多样性分析流程

```
输入 (图片目录/视频目录/demo)
  ↓
特征提取 → 全部 CLS 特征 [N, D]
  ↓
子集采样 (random/sliding) → 多组索引
  ↓
对每个子集计算指标:
  - pairwise cosine distance (1 - cos_sim 的均值)
  - pairwise L2 distance (均值)
  - feature variance score (方差均值)
  ↓
输出:
  - metrics.csv (每行一个子集)
  - metric_histograms.png
  - metric_boxplots.png
  - pca_scatter.png
  - similarity_heatmap.png
  - features_and_metrics.pt
```

### 模块导入关系

```
analyze_image_subset_diversity.py
  ├── extract_image_features.py  (模型/特征提取/demo数据)
  ├── diversity_metrics.py       (指标计算)
  ├── subset_sampling.py         (子集生成)
  └── plotting_utils.py          (可视化)

analyze_video_subset_diversity.py
  ├── extract_video_features.py  (模型/视频处理/demo数据)
  ├── diversity_metrics.py
  ├── subset_sampling.py
  └── plotting_utils.py

analyze_robotwin_diversity.py
  ├── extract_video_features.py  (模型加载/特征提取)
  ├── robotwin_loader.py         (RoboTwin元数据/视频下载/帧提取)
  ├── diversity_metrics.py
  ├── subset_sampling.py
  └── plotting_utils.py

select_episodes.py
  ├── extract_video_features.py  (模型加载/特征提取)
  ├── robotwin_loader.py         (RoboTwin数据加载)
  ├── diversity_metrics.py
  └── plotting_utils.py

manage_local_datasets.py
  ├── extract_image_features.py  (DEMO_URLS)
  └── extract_video_features.py  (DEMO_VIDEOS)
```

## Demo 数据源

### 图片
- **COCO val2017** (10张): 猫、厨房、消防栓、大象、飞机、自行车、狗、风筝、滑雪、棒球

### 视频
- **Kinetics-mini**: 射箭 (HuggingFace)
- **UCF101-subset**: 婴儿爬行、篮球扣篮 (HuggingFace)
- **LeRobot**: 双臂机器人操作 (HuggingFace, VLA相关)
- **OpenCV samples**: 行人检测测试视频
- **BigBuckBunny**: 动画短片 (Blender Foundation, CC-BY)

## 可用模型

`dinov2_vits14` (21M, D=384), `dinov2_vitb14` (86M, D=768), `dinov2_vitl14` (300M, D=1024), `dinov2_vitg14` (1.1B, D=1536)。带 `_reg` 后缀的变体附加 4 个 register tokens。

## 输出格式

### 特征提取 (.pt 文件)

```python
# 图像
{"model": str, "image_names": list, "cls_features": [B,D], "patch_tokens": [B,N,D], ...}

# 视频 (单视频)
{"model": str, "video_info": dict, "cls_features": [N_frames,D], "patch_tokens": [N_frames,P,D],
 "temporal_mean": [D], "temporal_var": [D], ...}

# 视频 (多视频 summary)
{"model": str, "video_names": list, "similarity_matrix": [V,V], "temporal_mean_embeddings": [V,D], ...}
```

### 多样性分析 (输出目录)

```
output_dir/  (analyze_*_diversity.py)
  ├── metrics.csv                # 每行一个子集的指标
  ├── metric_histograms.png      # 指标分布直方图
  ├── metric_boxplots.png        # 指标箱线图
  ├── pca_scatter.png            # 全数据集 PCA 2D 散点图
  ├── similarity_heatmap.png     # 全数据集两两余弦相似度热力图
  └── features_and_metrics.pt    # 完整特征 + 指标 (torch 格式)
```

### Episode 选择 (输出目录)

```
output_dir/  (select_episodes.py)
  ├── selection_results.csv      # 每行一个策略的选择结果和指标
  ├── selected_episodes.json     # 每个策略选中的 episode ID 列表 (机器可读)
  ├── pca_scatter.png            # PCA 散点图，选中 episode 带 * 标记
  ├── similarity_heatmap.png     # 全数据集两两余弦相似度热力图
  └── selection_data.pt          # 完整嵌入 + 选择结果 (torch 格式)
```

## 目录结构

```
extract_feature/
  ├── AGENTS.md                           # 本文件
  ├── GUIDE.md                            # 详细使用指南
  ├── extract_image_features.py           # 图像特征提取
  ├── extract_video_features.py           # 视频特征提取 (含多视频demo, AV1 PyAV 回退)
  ├── analyze_image_subset_diversity.py   # 图像子集多样性分析
  ├── analyze_video_subset_diversity.py   # 视频子集多样性分析
  ├── robotwin_loader.py                  # RoboTwin 数据加载器 (HF Hub + PyAV)
  ├── analyze_robotwin_diversity.py       # RoboTwin episode 多样性分析
  ├── select_episodes.py                  # 基于多样性的 episode 选择 (4种策略)
  ├── diversity_metrics.py                # 多样性指标计算
  ├── subset_sampling.py                  # 子集采样策略
  ├── plotting_utils.py                   # 可视化工具
  ├── manage_local_datasets.py            # 数据集下载管理
  ├── local_datasets/                     # 本地数据集 (gitignored)
  │   ├── images/                         # COCO demo 图片
  │   └── videos/                         # 多源 demo 视频
  └── outputs/                            # 分析输出 (gitignored)
```

## 注意事项

- `ALL_PROXY= all_proxy=` — 使用 HuggingFace 时需取消 SOCKS 代理
- 首次运行会自动下载权重到 `~/.cache/torch/hub/checkpoints/`
- RTX 4060 (8GB) 推荐使用 `dinov2_vits14`，`vitl14` 需谨慎
- xFormers 警告可忽略（推理不需要）
- `local_datasets/` 和 `outputs/` 已在 `.gitignore` 中排除
- 所有脚本需通过 `PYTHONPATH=/home/m1zu/ws/dinov2` 运行以解析导入
- matplotlib 用于可视化，已加入 `pyproject.toml` 依赖
- AV1 视频回退: `extract_video_features.py` 的 `sample_frames()` 先尝试 OpenCV，若解码 0 帧则自动切换 PyAV (libdav1d)。返回值为 4-tuple `(frames, actual_fps, info, is_rgb)`，`is_rgb=True` 表示帧已是 RGB（无需 BGR→RGB 转换）。调用者需将 `is_rgb` 传给 `frames_to_batch()`。

## RoboTwin 数据集说明

- **数据源**: `hxma/RoboTwin-LeRobot-v3.0` (HuggingFace)，50 个机器人操作 task
- **格式**: LeRobot v3.0 格式，episode 边界存储在 parquet 文件中，视频为 AV1 编码的 MP4（所有 episode 拼接在一个文件中）
- **视频解码**: 必须使用 PyAV（OpenCV 不支持 AV1），帧返回为 PIL.Image.Image (RGB)
- **variant**: `clean`（50个demo）或 `randomized`（500个demo）
- **camera**: 3 个视角：`cam_high`, `cam_left_wrist`, `cam_right_wrist`
- **HF 缓存**: 下载的视频/元数据缓存在 `~/.cache/huggingface/hub/datasets--hxma--RoboTwin-LeRobot-v3.0/`
- **Episode 嵌入**: 每个 episode 提取帧 → DINOv2 CLS 特征 [N_frames, D] → temporal mean → [D]
- **选择策略**: `greedy_maxdist`（最大化最小距离）, `greedy_maxvar`（最大化方差）, `kmeans`（聚类中心选择）, `random`（随机基线）
- **Diversity ratio**: 选中子集的 cosine distance / 全集的 cosine distance，>1.0 说明选中的子集多样性高于全集平均
