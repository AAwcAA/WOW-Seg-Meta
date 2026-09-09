---
license: apache-2.0
tags:
- jittor
- image-segmentation
- vision-language
- wow-seg
---

# WOW-Seg-J

Jittor inference weights for [WOW-Seg](https://huggingface.co/AAwcAA/WOW-Seg),
converted from the official BF16 `model.safetensors` checkpoint.

## Files

- `wow_seg.pkl`: Jittor checkpoint in FP16.
- `wow_seg.manifest.json`: tensor count and source SHA-256 manifest.
- `config.json`, tokenizer files, and preprocessor configuration: required by
  `jittor_wowseg` inference code.

The checkpoint contains 637 tensors and 938,193,024 parameters. The original
PyTorch/Hugging Face checkpoint is not included.

## Usage

Use the `jittor_wowseg` support code from the
[WOW-Seg-Meta repository](https://github.com/AAwcAA/WOW-Seg-Meta):

```bash
python -m jittor_wowseg.infer \
  --model-dir path/to/WOW-Seg-J \
  --image path/to/image.jpg \
  --mask path/to/mask.png
```

The Jittor implementation defaults to 12 dynamic image tiles to match the
original inference configuration. Reduce this only when GPU memory is limited.

## Conversion

The release was produced with `jittor_wowseg/convert_weights.py`. It converts
BF16 tensors to FP16 because FP16 is the portable Jittor CUDA format, preserves
all original parameter keys, and validates the checkpoint before inference.
