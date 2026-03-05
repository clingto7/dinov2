# 2026-03-05 DINOv2 环境搭建与特征提取脚本

## 任务概述

用户希望使用DINOv2模型熟悉视觉特征提取，最终目标是为VLA（Vision-Language-Action）机器人数据集搭建特征提取与分析pipeline。本次工作完成了DINOv2环境搭建、特征提取脚本的修复与验证、以及项目文档的建立。

## 工作内容

### 第一阶段（前一次会话，在dinov3仓库）

#### 1. DINOv3 AGENTS.md 生成
- 为DINOv3仓库生成了7个层级化AGENTS.md文件
- 根目录 + eval子目录 + layers子目录

#### 2. DINOv2 仓库克隆与AGENTS.md
- 从 `git@github.com:clingto7/dinov2.git` 克隆到 `/home/m1zu/ws/dinov2/`
- 生成了根目录 `AGENTS.md`

#### 3. Python环境搭建（dinov3仓库）
- 在dinov3仓库的 `.venv/` 创建了初始虚拟环境（后续迁移到dinov2本地）

#### 4. DINOv2 权重与API验证
- 确认DINOv2权重公开可访问（无需auth），HTTP 200
- 下载 ViT-S/14 权重（84MB）到 `~/.cache/torch/hub/checkpoints/`
- 验证三种API方法：
  - `model(x)` → CLS输出 `[1, 384]`
  - `model.forward_features(x)` → CLS + patch tokens
  - `model.get_intermediate_layers(x, n=4)` → 中间层特征

#### 5. 特征提取脚本编写
- 编写 `extract_image_features.py` — 图像特征提取（CLS/patch/中间层、余弦相似度、方差）
- 编写 `extract_video_features.py` — 视频特征提取（逐帧、时序统计、空间统计）
- 脚本原始路径在dinov3仓库的 `scripts/` 目录

### 第二阶段（当前会话，在dinov2仓库）

#### 6. UV环境搭建（dinov2本地）
- 在 `pyproject.toml` 中添加了 `[project]` 段，声明所有依赖
- 使用 `uv venv --python 3.12` 在 `/home/m1zu/ws/dinov2/.venv/` 创建本地虚拟环境
- `uv sync` 安装依赖：PyTorch 2.10.0+cu128, torchvision, omegaconf, opencv-python-headless 等51个包
- GPU验证通过：RTX 4060 Laptop GPU，CUDA可用

#### 7. 特征提取脚本修复
- 脚本已从 `scripts/` 移至 `extract_feature/` 目录
- **修复 `DINOV2_REPO` 路径**：`parent.parent.parent / "dinov2"` → `parent.parent`（因目录层级变化）
- 两个脚本均通过端到端验证：
  - `extract_image_features.py --demo`：下载5张COCO图片，提取CLS/patch特征，计算余弦相似度和数据集方差
  - `extract_video_features.py --demo --max-frames 10`：下载BigBuckBunny视频，采样10帧，计算时序/空间统计

#### 8. 子目录AGENTS.md生成
- `extract_feature/AGENTS.md` — 脚本用法、运行命令、输出格式
- `dinov2/models/AGENTS.md` — ViT backbone API、工厂函数、关键参数
- `dinov2/layers/AGENTS.md` — NN构建块、xFormers依赖信息
- `dinov2/hub/AGENTS.md` — torch.hub入口点、权重URL、使用方法

#### 9. AGENTS.md COMMANDS更新
- 将根目录 `AGENTS.md` 的COMMANDS部分从conda命令改为uv命令
- 添加了 `extract_feature/` 脚本的运行命令（含 `XFORMERS_DISABLED=1` 和 `PYTHONPATH`）

## 使用方法

```bash
# 图像特征提取 - demo模式
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_image_features.py --demo

# 图像特征提取 - 指定目录
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_image_features.py --image-dir /path/to/images --output features.pt

# 图像特征提取 - 中间层
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_image_features.py --demo --output features.pt --intermediate-layers 4

# 视频特征提取 - demo模式
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_video_features.py --demo --max-frames 20

# 视频特征提取 - 指定视频
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_video_features.py --video /path/to/video.mp4 --fps 2 --output vid_feats.pt

# 使用更大模型
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_image_features.py --demo --model dinov2_vitb14
```

## 关键发现

1. **DINOv3权重需要认证** — Meta下载链接403, HuggingFace 401 (Gated)
2. **DINOv2权重公开可用** — `dl.fbaipublicfiles.com/dinov2/` 无需认证
3. **代理问题** — `ALL_PROXY=socks://` 需要在使用HuggingFace时取消设置
4. **xFormers非必需** — 推理/特征提取不需要xFormers，只有训练需要；设置 `XFORMERS_DISABLED=1` 可消除警告
5. **DINOv2 patch_size=14**（非16） — 224×224输入产生16×16=256个patch

## 修改的文件

### 新增
- `/home/m1zu/ws/dinov2/.venv/` (Python 3.12虚拟环境, uv管理)
- `/home/m1zu/ws/dinov2/extract_feature/AGENTS.md`
- `/home/m1zu/ws/dinov2/dinov2/models/AGENTS.md`
- `/home/m1zu/ws/dinov2/dinov2/layers/AGENTS.md`
- `/home/m1zu/ws/dinov2/dinov2/hub/AGENTS.md`

### 修改
- `/home/m1zu/ws/dinov2/pyproject.toml` — 添加 `[project]` 段声明依赖
- `/home/m1zu/ws/dinov2/extract_feature/extract_image_features.py` — 修复 DINOV2_REPO 路径
- `/home/m1zu/ws/dinov2/extract_feature/extract_video_features.py` — 修复 DINOV2_REPO 路径
- `/home/m1zu/ws/dinov2/AGENTS.md` — 更新COMMANDS段为uv命令

### 历史（dinov3仓库，仅供参考）
- `/home/m1zu/ws/dinov3/AGENTS.md` 及6个子目录AGENTS.md
- `/home/m1zu/ws/dinov3/scripts/extract_image_features.py`（原始版本）
- `/home/m1zu/ws/dinov3/scripts/extract_video_features.py`（原始版本）

## 后续工作方向

- 搭建VLA数据集特征提取pipeline：一条数据 = 三个相机的视频 → DINOv2提取特征 → 跨数据条目分析
- 数据集多样性分析：计算特征方差等指标
- 可扩展架构：支持不同数据选取标准
