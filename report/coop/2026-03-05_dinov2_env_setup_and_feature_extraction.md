# 2026-03-05 DINOv2 环境与特征提取 — 协作文档

## 当前状态

DINOv2环境已在本仓库内搭建完毕（uv管理），两个特征提取脚本已修复并通过端到端测试。

## 环境信息

- **Python虚拟环境**: `/home/m1zu/ws/dinov2/.venv/` (Python 3.12.12, uv管理)
- **PyTorch**: 2.10.0+cu128
- **DINOv2仓库**: `/home/m1zu/ws/dinov2/`
- **GPU**: RTX 4060 Laptop GPU, 8GB VRAM
- **已下载权重**: `dinov2_vits14` (ViT-S/14, 21M params, embed_dim=384)
- **依赖管理**: `uv sync` 安装，依赖声明在 `pyproject.toml` 的 `[project]` 段

## 可用脚本

### `extract_feature/extract_image_features.py`
从图片提取DINOv2特征（CLS token, patch tokens, 中间层），支持批量处理、特征保存、数据集多样性统计。

```bash
# demo模式
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_image_features.py --demo

# 指定目录
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_image_features.py --image-dir <DIR> --output features.pt
```

### `extract_feature/extract_video_features.py`
从视频提取逐帧DINOv2特征，支持FPS控制、时序分析（帧间相似度、场景变化检测）、空间分析。

```bash
# demo模式
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_video_features.py --demo --max-frames 20

# 指定视频
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_video_features.py --video <FILE> --fps 2 --output vid_feats.pt
```

## 关键API模式

```python
import torch
model = torch.hub.load('/home/m1zu/ws/dinov2', 'dinov2_vits14', source='local')
model.eval().cuda()

# CLS token: model(x) → [B, 384]
# Full: model.forward_features(x) → dict with x_norm_clstoken, x_norm_patchtokens
# Intermediate: model.get_intermediate_layers(x, n=4, return_class_token=True)
```

## 注意事项

- 使用HuggingFace库时必须 `ALL_PROXY= all_proxy=` 取消SOCKS代理
- DINOv3权重需要认证（目前不可用），已改用DINOv2
- xFormers推理不需要；设置 `XFORMERS_DISABLED=1` 可消除警告
- 可用模型: `dinov2_vits14/vitb14/vitl14/vitg14` + `_reg` 变体
- RTX 4060 (8GB VRAM) 推荐使用 `dinov2_vits14`（~1GB VRAM）

## 项目文档结构

已生成的AGENTS.md文件：
- `/home/m1zu/ws/dinov2/AGENTS.md` — 根目录（含GUIDELINES、COMMANDS等）
- `/home/m1zu/ws/dinov2/extract_feature/AGENTS.md` — 特征提取脚本
- `/home/m1zu/ws/dinov2/dinov2/models/AGENTS.md` — ViT backbone
- `/home/m1zu/ws/dinov2/dinov2/layers/AGENTS.md` — NN构建块
- `/home/m1zu/ws/dinov2/dinov2/hub/AGENTS.md` — torch.hub入口

## 待完成

- VLA数据集特征提取pipeline（一条数据 = 三个相机视频 → 提取特征 → 跨条目分析多样性）
- 数据集多样性指标（特征方差等）
- 可扩展架构设计（支持不同数据选取标准）
