# extract_feature/ — DINOv2 特征提取与多样性分析工具

## 概述

使用 DINOv2 预训练模型从图像和视频中提取视觉特征的工具脚本集。包含基础特征提取、子集多样性分析、可视化绘图、本地数据集管理等功能。用于 VLA 数据集分析、数据多样性评估、数据选取 pipeline 等。

## 文件

### 核心特征提取脚本

| 文件 | 功能 | 关键参数 |
|------|------|---------|
| `extract_image_features.py` | 从图片批量提取 CLS/patch/中间层特征 | `--model`, `--image-dir`, `--demo`, `--output`, `--intermediate-layers` |
| `extract_video_features.py` | 从视频逐帧提取特征 + 时序/空间分析，支持多视频demo和视频间相似度矩阵 | `--model`, `--video`, `--demo`, `--demo-names`, `--demo-count`, `--fps`, `--max-frames`, `--output`, `--output-dir` |

### 子集多样性分析脚本

| 文件 | 功能 | 关键参数 |
|------|------|---------|
| `analyze_image_subset_diversity.py` | 对图片集合生成子集，计算多样性指标，输出CSV+图表 | `--image-dir`, `--demo`, `--subset-size`, `--n-subsets`, `--strategy`, `--output-dir` |
| `analyze_video_subset_diversity.py` | 对视频集合生成子集，以temporal mean作为视频嵌入，同上分析 | `--video-dir`, `--demo`, `--subset-size`, `--n-subsets`, `--strategy`, `--fps`, `--max-frames`, `--output-dir` |

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
output_dir/
  ├── metrics.csv                # 每行一个子集的指标
  ├── metric_histograms.png      # 指标分布直方图
  ├── metric_boxplots.png        # 指标箱线图
  ├── pca_scatter.png            # 全数据集 PCA 2D 散点图
  ├── similarity_heatmap.png     # 全数据集两两余弦相似度热力图
  └── features_and_metrics.pt    # 完整特征 + 指标 (torch 格式)
```

## 目录结构

```
extract_feature/
  ├── AGENTS.md                           # 本文件
  ├── GUIDE.md                            # 详细使用指南
  ├── extract_image_features.py           # 图像特征提取
  ├── extract_video_features.py           # 视频特征提取 (含多视频demo)
  ├── analyze_image_subset_diversity.py   # 图像子集多样性分析
  ├── analyze_video_subset_diversity.py   # 视频子集多样性分析
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
