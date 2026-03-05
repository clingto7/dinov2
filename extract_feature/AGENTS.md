# extract_feature/ — DINOv2 特征提取脚本

## 概述

使用 DINOv2 预训练模型从图像和视频中提取视觉特征的工具脚本。用于 VLA 数据集分析、数据多样性评估、数据选取 pipeline 等。

## 文件

| 文件 | 功能 | 关键参数 |
|------|------|---------|
| `extract_image_features.py` | 从图片批量提取 CLS/patch/中间层特征 | `--model`, `--image-dir`, `--demo`, `--output`, `--intermediate-layers` |
| `extract_video_features.py` | 从视频逐帧提取特征 + 时序/空间分析 | `--model`, `--video`, `--demo`, `--fps`, `--max-frames`, `--output` |

## 运行方式

```bash
# 必须设置 XFORMERS_DISABLED=1（推理不需要 xFormers）
# 使用 uv run 自动激活 .venv

# 图像特征提取 - demo 模式
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_image_features.py --demo

# 图像特征提取 - 指定目录
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_image_features.py --image-dir <DIR> --output features.pt

# 视频特征提取 - demo 模式
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_video_features.py --demo --max-frames 20

# 视频特征提取 - 指定视频
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_video_features.py --video <FILE> --fps 2 --output vid_feats.pt
```

## 架构

两个脚本共享相同模式：
1. **模型加载**: `torch.hub.load()` 从本地 repo 加载 → `DINOV2_REPO` 指向仓库根目录
2. **预处理**: Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet mean/std)
3. **特征提取**: 三种 API：
   - `model(x)` → CLS token `[B, D]`
   - `model.forward_features(x)` → dict(cls_token, patch_tokens, reg_tokens)
   - `model.get_intermediate_layers(x, n=k)` → 中间层特征
4. **统计分析**: 特征方差（数据多样性）、余弦相似度（帧间相似性/场景变化）

## 可用模型

`dinov2_vits14` (21M, D=384), `dinov2_vitb14` (86M, D=768), `dinov2_vitl14` (300M, D=1024), `dinov2_vitg14` (1.1B, D=1536)。带 `_reg` 后缀的变体附加 4 个 register tokens。

## 输出格式 (.pt 文件)

```python
# 图像
{"model": str, "image_names": list, "cls_features": [B,D], "patch_tokens": [B,N,D], ...}

# 视频
{"model": str, "video_info": dict, "cls_features": [N_frames,D], "patch_tokens": [N_frames,P,D],
 "temporal_mean": [D], "temporal_var": [D], ...}
```

## 注意事项

- `ALL_PROXY= all_proxy=` — 使用 HuggingFace 时需取消 SOCKS 代理
- 首次运行会自动下载权重到 `~/.cache/torch/hub/checkpoints/`
- RTX 4060 (8GB) 推荐使用 `dinov2_vits14`，`vitl14` 需谨慎
- xFormers 警告可忽略（推理不需要）
