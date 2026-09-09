# WOW-Seg-J：WOW-Seg 的 Jittor 推理实现

## 目录

```text
jittor_wowseg/
├── demo/                         # Jittor Web Demo
├── weights/                      # 本地模型文件
├── config.py                     # 配置读取
├── model.py                      # Jittor 模型
├── predictor.py                  # 区域推理封装
├── preprocess.py                 # 图像和 mask 预处理
├── tokenizer.py                  # tokenizer 适配器
├── convert_weights.py            # 权重转换
├── infer.py                      # 单图推理
└── visualize_samples.py          # 批量推理和可视化
```

## 环境

在已配置好的 Jittor/CUDA 环境中，运行前设置路径：

```bash
export PATH=/usr/bin:/bin:/usr/local/cuda/bin
export CUDA_HOME=/usr/local/cuda
export use_cutlass=0
```

## 模型权重

转换后的模型：

```text
https://huggingface.co/AAwcAA/WOW-Seg-J
```

将模型文件放在：

```text
jittor_wowseg/weights/wow_seg/
```

目录至少需要包含 `wow_seg.pkl`、`config.json` 和 tokenizer 文件。

## 权重转换

从原始 WOW-Seg Safetensors 转换：

```bash
python jittor_wowseg/convert_weights.py \
  --input /path/to/model.safetensors \
  --output /path/to/wow_seg.pkl
```

转换会保留参数名，将 BF16 转为 FP16，并生成校验 manifest。

## 推理

使用二值 mask 对区域分类：

```bash
python -m jittor_wowseg.infer \
  --model-dir jittor_wowseg/weights/wow_seg \
  --image /path/to/image.jpg \
  --mask /path/to/mask.png \
  --max-dynamic-patch 12
```

省略 `--mask` 时使用整张图片。分类输出在首个逗号 token 处停止，避免重复类别。

## Web Demo

计图 Demo 位于 `jittor_wowseg/demo/`，支持 Auto、Point Prompt 和 BBox Prompt：

```bash
python -m jittor_wowseg.demo.app_web \
  --wow_model jittor_wowseg/weights/wow_seg \
  --sam_ckpt demo/weights/sam/sam_vit_b_01ec64.pth \
  --sam_model_type vit_b \
  --jittor_max_dynamic_patch 12
```

SAM 只用于 Demo 的 mask 生成，需要单独准备 checkpoint，不随代码或模型仓库上传。

## 动态 Tile

默认 `max_dynamic_patch=12`。显存不足时可通过参数降低 tile 数。
