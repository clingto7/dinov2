# dinov2/hub/ — torch.hub 入口点

## 概述

提供 `torch.hub.load()` 的模型加载接口。每个文件定义一组模型构建函数，自动下载预训练权重。

## 文件

| 文件 | 内容 | 暴露的函数 |
|------|------|-----------|
| `backbones.py` | DINOv2 backbone 加载 | `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14` + `_reg` 变体 |
| `classifiers.py` | backbone + linear head | `dinov2_vit{s,b,l,g}14_lc`, `_reg_lc` 变体 |
| `depthers.py` | backbone + depth head | `dinov2_vit{s,b,l,g}14_{ld,dd}` (linear/DPT) |
| `dinotxt.py` | dino.txt 多模态模型 | `dinov2_vitl14_reg4_dinotxt_tet1280d20h24l` |
| `utils.py` | 工具函数 | `_DINOV2_BASE_URL`, `_make_dinov2_model_name`, `CenterPadding` |

## 子目录

| 目录 | 内容 |
|------|------|
| `cell_dino/` | Cell-DINO 生物学 backbone |
| `xray_dino/` | X-ray DINO backbone |
| `depth/` | 深度估计 head 模型 (DPT, BNHead) |
| `text/` | dino.txt 文本编码器 + 多模态模型 |

## 权重 URL 模式

```
https://dl.fbaipublicfiles.com/dinov2/dinov2_{arch}{patch_size}/dinov2_{arch}{patch_size}_pretrain.pth
# 例: https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
```

DINOv2 权重公开可下载（无需认证），首次使用自动缓存到 `~/.cache/torch/hub/checkpoints/`。

## 使用方式

```python
import torch

# 方式1: 从 GitHub 加载
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

# 方式2: 从本地 repo 加载（推荐，避免网络问题）
model = torch.hub.load('/home/m1zu/ws/dinov2', 'dinov2_vits14', source='local')
```

## 关键实现细节

- `_make_dinov2_model()` 在 `backbones.py` 中统一处理所有 backbone 变体
- 所有 DINOv2 模型使用 `patch_size=14`，`img_size=518`，`init_values=1.0`
- Register 模型额外传 `num_register_tokens=4`, `interpolate_antialias=True`, `interpolate_offset=0.0`
