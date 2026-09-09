# WOW-Seg-J Demo

## 目录

```text
jittor_wowseg/demo/
├── app_web.py
├── batch_pipeline.py
└── sam_helpers.py
```

## 环境

在已配置好的 Jittor/CUDA 环境中，运行前设置：

```bash
export PATH=/usr/bin:/bin:/usr/local/cuda/bin
export CUDA_HOME=/usr/local/cuda
export use_cutlass=0
```

## 启动

从仓库根目录运行：

```bash
python -m jittor_wowseg.demo.app_web \
  --wow_model jittor_wowseg/weights/wow_seg \
  --sam_ckpt demo/weights/sam/sam_vit_b_01ec64.pth \
  --sam_model_type vit_b \
  --jittor_max_dynamic_patch 12
```

访问 `http://127.0.0.1:7861`。

支持 Auto、Point Prompt 和 BBox Prompt。SAM checkpoint 只用于生成 mask，需要单独准备。
