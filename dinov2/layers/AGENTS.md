# dinov2/layers/ — NN 构建模块

## 概述

ViT 的底层构建模块：注意力、Transformer block、FFN、patch embedding 等。

## 文件

| 文件 | 内容 | 关键类/函数 |
|------|------|------------|
| `attention.py` | 多头注意力 | `Attention`, `MemEffAttention` (xFormers 加速版) |
| `block.py` | Transformer block | `NestedTensorBlock` (支持 xFormers nested tensor), `CausalAttentionBlock` |
| `mlp.py` | 标准 MLP FFN | `Mlp` (Linear → GELU → Drop → Linear → Drop) |
| `swiglu_ffn.py` | SwiGLU FFN | `SwiGLUFFN`, `SwiGLUFFNFused` (ViT-g 使用) |
| `patch_embed.py` | Patch embedding | `PatchEmbed` (Conv2d 实现) |
| `dino_head.py` | DINO 投影头 | `DINOHead` (MLP + L2 norm + weight norm) |
| `layer_scale.py` | Layer Scale | `LayerScale` (可学习的逐通道缩放) |
| `drop_path.py` | Stochastic Depth | `DropPath` (训练时随机丢弃整个残差路径) |

## xFormers 依赖

- 训练时 xFormers 提供内存高效注意力 (`MemEffAttention`) 和 nested tensor 支持
- **推理/特征提取不需要 xFormers** — 设置 `XFORMERS_DISABLED=1` 即可
- 当 xFormers 不可用时，自动回退到标准 PyTorch 实现

## 导出 (__init__.py)

```python
from .dino_head import DINOHead
from .layer_scale import LayerScale
from .mlp import Mlp
from .patch_embed import PatchEmbed
from .swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused, SwiGLUFFNAligned
from .block import NestedTensorBlock, CausalAttentionBlock
from .attention import Attention, MemEffAttention
```

## 修改指南

- 修改注意力机制 → `attention.py`
- 修改 FFN 类型 → `mlp.py` 或 `swiglu_ffn.py`
- 修改 block 结构（pre-norm、残差连接） → `block.py`
- 修改图像切 patch 方式 → `patch_embed.py`
