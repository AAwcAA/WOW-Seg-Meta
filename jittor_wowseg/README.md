# WOW-Seg Jittor Inference

This directory adds an inference-only Jittor backend for the released
`AAwcAA/WOW-Seg` checkpoint. It preserves the released `model.safetensors`
parameter names and validates that every parameter is present before loading.

## Checkpoint

The checkpoint has been downloaded to:

```text
jittor_wowseg/weights/wow_seg/
```

The converted checkpoint is also published at
[`AAwcAA/WOW-Seg-J`](https://huggingface.co/AAwcAA/WOW-Seg-J). Download that
repository into the path above, or place its files in any local model
directory passed to the inference commands.

`model.safetensors` is a hard link to the downloaded file in `weights/`, so it
does not consume a second 1.88 GB copy.

## Install

Use Python 3.10+ and install Jittor appropriate for the target CUDA runtime,
then install the remaining dependencies:

```powershell
pip install -r requirements/jittor_wowseg.txt
```

## Convert the weights

Run this with a Python environment that has PyTorch and `safetensors`; PyTorch
is used only for reading BF16 safetensors during conversion, never at runtime.

```powershell
python jittor_wowseg/convert_weights.py
```

This writes `jittor_wowseg/weights/wow_seg/wow_seg.pkl` and its manifest.
The converter uses FP16 for BF16 tensors because that is the portable Jittor
CUDA representation.

## One region

```powershell
python -m jittor_wowseg.infer `
  --model-dir jittor_wowseg/weights/wow_seg `
  --image path/to/image.jpg `
  --mask path/to/mask.png
```

The mask is a binary target-region mask. Omitting `--mask` runs whole-image
classification.

## Jittor Web demo

The original `demo/` directory is unchanged. The Jittor-specific Web UI is
contained in `jittor_wowseg/demo/` and uses the original SAM source tree:

```powershell
cd WOW-Seg-Meta-main
python -m jittor_wowseg.demo.app_web `
  --wow_model jittor_wowseg/weights/wow_seg `
  --sam_ckpt demo/weights/sam/sam_vit_b_01ec64.pth `
  --sam_model_type vit_b `
  --jittor_max_dynamic_patch 12
```

The Jittor demo preserves Auto, Point Prompt, and BBox Prompt modes. SAM
creates masks; the WOW-Seg encoder, mask-to-token selection, Qwen decoder, and
generation use Jittor. `--jittor_max_dynamic_patch` defaults to 12; reduce it
only for GPUs with limited memory.
