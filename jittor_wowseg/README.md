# WOW-Seg-J：WOW-Seg 的 Jittor 推理实现

本目录提供 WOW-Seg 的 Jittor 推理后端，参考 `SLIP-RS-J` 的组织方式实现：
保留原始模型参数名，离线转换权重，并在加载前校验 tensor 数量和形状。
原项目根目录下的 PyTorch 推理代码和 `demo/` 未作修改。

## 目录

```text
jittor_wowseg/
├── demo/                         # Jittor Web Demo（Auto/Point/BBox）
├── weights/                      # 本地模型文件（不提交到 GitHub）
├── config.py                     # WOW-Seg 配置读取
├── model.py                      # InternViT + Qwen2 Jittor 模型
├── predictor.py                  # mask-to-token 区域推理封装
├── preprocess.py                 # 动态切图、crop、mask 预处理
├── tokenizer.py                  # tokenizer.json 适配器
├── convert_weights.py            # Safetensors -> Jittor pickle
├── infer.py                     # 单图/单 mask 推理
└── visualize_samples.py          # 批量推理与可视化
```

## 环境

建议使用 Python 3.10+、CUDA 和 Jittor。以 WSL 的 `my_internvl` 环境为例：

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate my_internvl
export PATH=/usr/bin:/bin:/usr/local/cuda/bin
export CUDA_HOME=/usr/local/cuda
export use_cutlass=0

cd /mnt/e/jitu/WOW-Seg-Meta-main
pip install -r requirements/jittor_wowseg.txt
```

若 CUDA 使用 GCC 版本不匹配，请确保 Jittor 使用 CUDA 支持的 GCC（例如
CUDA 11.x 使用 GCC 9/10/11）。

## 模型权重

转换后的 Jittor 模型已发布到：

```text
https://huggingface.co/AAwcAA/WOW-Seg-J
```

下载该仓库文件到任意本地目录，例如：

```text
jittor_wowseg/weights/wow_seg/
├── wow_seg.pkl
├── wow_seg.manifest.json
├── config.json
├── tokenizer.json
└── ...
```

`wow_seg.pkl` 是由原始 WOW-Seg BF16 `model.safetensors` 转换得到的 FP16
Jittor 权重。原始 PyTorch 权重不包含在本仓库中。

## 权重转换

如果需要从原始 WOW-Seg Safetensors 重新转换，使用一个能够读取 PyTorch
Safetensors 的环境：

```bash
python jittor_wowseg/convert_weights.py \
  --input /path/to/model.safetensors \
  --output /path/to/wow_seg.pkl
```

转换器会保留全部参数键，BF16 转为 FP16，并写出包含 tensor 数量、参数数量和
源文件 SHA-256 的 manifest。

## 推理

使用二值 mask 对一个区域进行分类：

```bash
python -m jittor_wowseg.infer \
  --model-dir jittor_wowseg/weights/wow_seg \
  --image /path/to/image.jpg \
  --mask /path/to/mask.png \
  --max-dynamic-patch 12
```

省略 `--mask` 时使用整张图片作为目标区域。分类任务默认在生成首个类别分隔符
后停止，以避免 `train, train, train` 形式的重复输出。

## Web Demo

原始 PyTorch Demo 保持不变。Jittor Demo 位于 `jittor_wowseg/demo/`：

```bash
python -m jittor_wowseg.demo.app_web \
  --wow_model jittor_wowseg/weights/wow_seg \
  --sam_ckpt demo/weights/sam/sam_vit_b_01ec64.pth \
  --sam_model_type vit_b \
  --jittor_max_dynamic_patch 12
```

支持：

- `Auto`：由 SAM 自动生成 mask，再逐个使用 WOW-Seg 分类
- `Point Prompt`：点击一点后由 SAM 生成 mask
- `BBox Prompt`：框选区域后由 SAM 生成 mask

SAM 只负责 Demo 的 mask 生成，不属于 WOW-Seg-J 语言视觉模型。SAM checkpoint
需要单独下载，且不会上传到 GitHub 或 WOW-Seg-J 仓库。

## 显存与动态 Tile

默认 `max_dynamic_patch=12`，与原 WOW-Seg 配置保持一致。显存有限时可以降低：

```bash
--max-dynamic-patch 1
```

较大的 tile 数会提供更多整图视觉信息，但会显著增加显存占用。

## 验证

当前实现已在 WSL `my_internvl` 环境验证：

- Jittor CUDA 编译和导入成功
- 637/637 参数键匹配
- 0 个缺失参数、0 个多余参数、0 个形状不匹配
- 单 tile 端到端推理成功
