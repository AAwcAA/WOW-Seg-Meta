# WOW-Seg-J Demo：WOW-Seg 的 Jittor Web 推理

## 目录

```text
jittor_wowseg/demo/
├── app_web.py       # FastAPI Web UI
├── batch_pipeline.py
├── sam_helpers.py
└── README.md
```

原项目的 `demo/` 目录保持不变；本目录只包含 Jittor Demo 的适配代码。

## 环境

先安装原项目 Demo 依赖和 Jittor 依赖：

```bash
pip install -r requirements.txt
pip install -r requirements/jittor_wowseg.txt
```

## 模型与 SAM

将 [AAwcAA/WOW-Seg-J](https://huggingface.co/AAwcAA/WOW-Seg-J) 下载到本地，
并准备一个 SAM checkpoint。SAM 只用于 Auto、Point Prompt、BBox Prompt 的
mask 生成，不参与 WOW-Seg-J 权重。

## 启动

从仓库根目录启动：

```bash
python -m jittor_wowseg.demo.app_web \
  --wow_model jittor_wowseg/weights/wow_seg \
  --sam_ckpt demo/weights/sam/sam_vit_b_01ec64.pth \
  --sam_model_type vit_b \
  --jittor_max_dynamic_patch 12
```

访问 `http://127.0.0.1:7861`。

`--jittor_max_dynamic_patch` 默认是 12。显存不足时可降低为 1 或 2；部署配置
不应为了本地低显存测试永久修改默认值。
