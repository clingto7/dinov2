# dinov2/models/ — ViT Backbone 定义

## 概述

DINOv2 的核心模型定义。仅包含 Vision Transformer (ViT) 架构。

## 文件

| 文件 | 内容 |
|------|------|
| `vision_transformer.py` | `DinoVisionTransformer` 类 + 工厂函数 (vit_small/base/large/giant2) |

## DinoVisionTransformer 核心 API

```python
# 推理（CLS token 输出）
output = model(x)  # [B, D]

# 完整特征
out = model.forward_features(x)
# out["x_norm_clstoken"]     [B, D]        — 全局语义表示
# out["x_norm_patchtokens"]  [B, N, D]     — 每个 patch 的局部特征
# out["x_norm_regtokens"]    [B, R, D]     — register tokens（如有）
# out["x_prenorm"]           [B, 1+R+N, D] — LayerNorm 之前的原始输出
# out["masks"]               masks or None

# 中间层特征
feats = model.get_intermediate_layers(x, n=4, return_class_token=True, reshape=False, norm=True)
# → list of (patch_tokens [B,N,D], cls_token [B,D])
```

## 工厂函数

| 函数 | embed_dim | depth | num_heads | 对应模型 |
|------|-----------|-------|-----------|---------|
| `vit_small(patch_size=16)` | 384 | 12 | 6 | dinov2_vits14 |
| `vit_base(patch_size=16)` | 768 | 12 | 12 | dinov2_vitb14 |
| `vit_large(patch_size=16)` | 1024 | 24 | 16 | dinov2_vitl14 |
| `vit_giant2(patch_size=16)` | 1536 | 40 | 24 | dinov2_vitg14 |

注意: 工厂函数默认 `patch_size=16`，但 DINOv2 预训练模型使用 `patch_size=14`，由 `hub/backbones.py` 覆盖。

## 关键参数

- `num_register_tokens` — 0 (标准) 或 4 (_reg 变体)
- `channel_adaptive` — Cell-DINO 的 channel-adaptive 模式
- `interpolate_antialias` / `interpolate_offset` — 位置编码插值控制
- `block_chunks` — FSDP 分块参数

## 修改指南

- 添加新 backbone → 新建工厂函数，注册到 `hub/backbones.py`
- 修改前向传播 → 编辑 `forward_features()` 或 `get_intermediate_layers()`
- 改变输入分辨率 → 位置编码自动插值（`interpolate_pos_encoding()`）
