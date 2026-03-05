# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-05
**Commit:** 7b187bd
**Branch:** main

## GUIDELINES 工作总则

- 优先遵循本文件
- 使用UV管理python环境
  - 如,添加python依赖使用`uv add xxx`
- 本机器配置为RTX4060显卡 13th Gen Intel(R) Core(TM) i9-13900H
- 对跨模块变更（环境注册、算法接入、配置项）要保持“代码 + 配置 + 文档”同步。
- 每次工作可以修改的文件**仅限项目目录内**
- 每次工作**基本流程**
  - 阅读该文档
  - 根据AGENTS.md文档确定自己任务下一步需要看哪个部分并仔细阅读
  - 进行修改
  - 执行必要的测试
  - 直到达到目标
  - 将以上过程记录在一个md文件内，命名规则<日期>_<session-id>_<概括任务信息>，放在**report/目录**下
    - 并且生成一份给其他agent读，方便你的其他“同事”了解工作情况的文档，放在**report/coop/目录**下，命名规则同上
- 同一个会话中一般有多次工作，每次工作**如有必要的话**要**更新**对应的report和coop下的文件，如：1. 做了新的修改，在report中的内容补充对当次工作的描述；2. coop下的内容经过当次工作修改后已不适用，需要修改；......

## OVERVIEW

DINOv2 — Meta AI's self-supervised vision foundation model (ViT backbones with optional register tokens). PyTorch research codebase for pretraining (DINO+iBOT+KoLeo SSL), distillation, and downstream evaluation (classification, segmentation, depth). Python 3.9+, OmegaConf config, SLURM/submitit job orchestration. Includes biology extensions (Cell-DINO, X-ray DINO).

## STRUCTURE

```
dinov2/
├── models/          # Backbone architecture (ViT only, vision_transformer.py)
├── layers/          # NN building blocks (attention, block, patch_embed, swiglu_ffn, dino_head, layer_scale, drop_path)
├── loss/            # SSL losses (DINOLoss cls-token, iBOTPatchLoss, KoLeoLoss)
├── train/           # Training loop (train.py), meta-arch (ssl_meta_arch.py SSLMetaArch)
├── eval/            # Downstream evaluation
│   ├── knn.py       # k-NN classification
│   ├── linear.py    # Linear probe classification
│   ├── log_regression.py  # Logistic regression (cuML)
│   ├── depth/       # Monocular depth estimation
│   ├── segmentation/      # Semantic segmentation (linear head)
│   ├── segmentation_m2f/  # Mask2Former segmentation
│   └── cell_dino/   # Biology-specific eval (Cell-DINO)
├── data/            # Data pipeline (datasets, augmentations, samplers, masking, collate)
│   ├── datasets/    # ImageNet, ImageNet22k, Cell-DINO datasets
│   └── cell_dino/   # Cell-DINO specific augmentations
├── configs/         # OmegaConf YAML configs (ssl_default_config.yaml + train/)
├── hub/             # torch.hub entry points (backbones, classifiers, depthers, text/dinotxt, cell_dino, xray_dino)
│   ├── depth/       # Depth head models
│   ├── text/        # dino.txt multimodal (vision/text towers, tokenizer)
│   ├── cell_dino/   # Cell-DINO hub backbones
│   └── xray_dino/   # X-ray DINO hub backbones
├── distributed/     # Distributed training primitives
├── fsdp/            # FSDP wrappers
├── run/             # Job submission (submitit/SLURM)
│   ├── train/       # Training launcher
│   └── eval/        # Eval launchers (knn, linear, log_regression, cell_dino)
├── logging/         # Logging setup
├── utils/           # Utilities (config, cluster, dtype, etc.)
└── thirdparty/      # Vendored CLIP (text encoder)
hubconf.py           # torch.hub.load() surface
notebooks/           # Demo notebooks (depth, segmentation, dinotxt, cell_dino)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add/modify backbone | `dinov2/models/vision_transformer.py` | ViT only (no ConvNeXt in v2) |
| Change attention/layers | `dinov2/layers/` | `attention.py`, `block.py`, `swiglu_ffn.py` |
| Add new loss | `dinov2/loss/` | Follow `DINOLoss`, `iBOTPatchLoss` pattern |
| Modify training loop | `dinov2/train/train.py` | Main training script |
| SSL meta-architecture | `dinov2/train/ssl_meta_arch.py` | `SSLMetaArch` — student/teacher + losses |
| Add eval task | `dinov2/eval/` | knn.py, linear.py, log_regression.py as templates |
| Add dataset | `dinov2/data/datasets/` | Subclass pattern, register in `__init__.py` |
| Change augmentations | `dinov2/data/augmentations.py` | `DataAugmentationDINO` class |
| Expose via torch.hub | `dinov2/hub/` + `hubconf.py` | Add function in hub/, re-export in hubconf.py |
| Change config defaults | `dinov2/configs/ssl_default_config.yaml` | OmegaConf schema |
| Modify job submission | `dinov2/run/submit.py` | Submitit/SLURM launcher |
| Feature extraction API | `dinov2/models/vision_transformer.py` | `forward_features()`, `get_intermediate_layers()` |
| Image preprocessing | `dinov2/data/transforms.py` | `make_classification_eval_transform()` |
| Biology extensions | `dinov2/data/cell_dino/`, `dinov2/eval/cell_dino/`, `dinov2/hub/cell_dino/` | Cell-DINO channel-adaptive |

## CONVENTIONS

- **Config**: OmegaConf throughout. CLI overrides via `key=value` args, merged with YAML configs. Default config in `ssl_default_config.yaml`. Config setup in `dinov2/utils/config.py`.
- **Entry pattern**: All runnable scripts expose `main()` + `if __name__ == "__main__"`. Invoked via `python dinov2/run/train/train.py` or `python dinov2/run/eval/knn.py` for SLURM, or directly with `python`/`torchrun`.
- **Imports**: `PYTHONPATH=.` required at invocation (not installed as editable by default).
- **Logging**: Single logger `logging.getLogger("dinov2")` used everywhere.
- **Line length**: 120 chars (Black formatter). Flake8 ignores `E203,E501,W503`.
- **Type checking**: Type annotations used extensively with `typing` module, but no mypy config found.
- **Formatting**: Black (120 char), pylint (disabled all except misc/similarities), flake8.
- **`__init__.py` exports**: Some barrel-export (layers, loss, data), some empty.
- **Patch size**: 14 for all DINOv2 ViT models (unlike DINOv3 which uses 16).
- **Hub models**: `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14` + `_reg` variants with register tokens.
- **Package version**: `0.0.1` in `dinov2/__init__.py`.

## ANTI-PATTERNS

- `dinov2/layers/block.py:110` — FIXME: `drop_path2` issue.
- `dinov2/loss/dino_clstoken_loss.py:67` — TODO: "Use cross_entropy_distribution here".
- `dinov2/data/collate.py:11` — TODO: "dtype = torch.half — Remove".
- `dinov2/data/loaders.py:135` — TODO: "Remove support for old shuffling".
- `dinov2/eval/depth/models/losses/sigloss.py:37` — HACK: warmup sigloss implementation.
- `dinov2/hub/text/dinotxt_model.py:17` — XXX: Untested replacement for mmcv.imdenormalize().
- `dinov2/hub/depth/decode_heads.py:17` — XXX: Untested replacement for mmcv.imdenormalize().
- Segmentation eval code (segmentation_m2f) has significant hack/TODO debt.
- No `type: ignore` or `noqa` suppressions in new code.

## COMMANDS

```bash
# ====== 环境设置（使用 uv） ======
# 虚拟环境: /home/m1zu/ws/dinov2/.venv/ (Python 3.12.12)
# PyTorch: 2.10.0+cu128
# GPU: RTX 4060 Laptop GPU, 8GB VRAM

# 安装/同步依赖
uv sync

# ====== 特征提取（日常使用） ======
# 图像特征提取 - demo
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_image_features.py --demo

# 图像特征提取 - 指定目录
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_image_features.py --image-dir <DIR> --output features.pt

# 视频特征提取 - demo
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_video_features.py --demo --max-frames 20

# 视频特征提取 - 指定视频
XFORMERS_DISABLED=1 PYTHONPATH=/home/m1zu/ws/dinov2 uv run python extract_feature/extract_video_features.py --video <FILE> --fps 2 --output vid_feats.pt

# ====== 训练（SLURM，需要 xFormers） ======
PYTHONPATH=. python dinov2/run/train/train.py \
  --nodes 4 --config-file dinov2/configs/train/vitl16_short.yaml \
  --output-dir <OUT> \
  train.dataset_path=ImageNet:split=TRAIN:root=<DATA>:extra=<DATA>

# ====== 评估 ======
PYTHONPATH=. python dinov2/run/eval/knn.py \
  --config-file <OUT>/config.yaml \
  --pretrained-weights <OUT>/eval/training_24999/teacher_checkpoint.pth \
  --output-dir <OUT>/eval/training_24999/knn \
  --train-dataset ImageNet:split=TRAIN:root=<DATA>:extra=<DATA> \
  --val-dataset ImageNet:split=VAL:root=<DATA>:extra=<DATA>

PYTHONPATH=. python dinov2/run/eval/linear.py ...
PYTHONPATH=. python dinov2/run/eval/log_regression.py ...
```

## NOTES

- **No test suite**: No pytest/unittest framework.
- **xFormers dependency**: `ssl_meta_arch.py` asserts xFormers is available for training. Not needed for inference/feature extraction.
- **DINOv2 uses patch_size=14** (not 16 like DINOv3).
- **Register token models** (`*_reg`) add 4 register tokens; accessed via `x_norm_regtokens` in forward_features output.
- **Public pretrained weights**: Available at `https://dl.fbaipublicfiles.com/dinov2/` (no gated access required, unlike DINOv3).
- **Weight URLs pattern**: `{BASE_URL}/dinov2_{arch}/dinov2_{arch}_pretrain.pth`
- **SLURM-oriented**: Hardcoded cluster defaults in `dinov2/utils/cluster.py`. Override via config for other environments.
- **CI**: GitHub Actions lint-only (Ubuntu, Python 3.9).
- **Dense eval extras**: depth/segmentation require `conda-extras.yaml` (adds mmcv, mmseg).
- **Biology extensions**: Cell-DINO and X-ray DINO have separate data/eval/hub paths.
- **forward_features() returns dict**: `x_norm_clstoken` [B,D], `x_norm_regtokens` [B,R,D], `x_norm_patchtokens` [B,N,D], `x_prenorm`, `masks`.
- **get_intermediate_layers(x, n=k, reshape=False, return_class_token=False, norm=True)** returns per-layer patch tokens (optionally with cls token).
- **Canonical eval transform**: Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet mean/std).
