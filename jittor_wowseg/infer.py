#!/usr/bin/env python
"""Run one WOW-Seg region classification entirely through the Jittor model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .predictor import WowSegJittorPredictor


def load_mask(path, image_shape):
    if path is None:
        return np.ones(image_shape[:2], dtype=np.uint8)
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Hugging Face WOW-Seg checkpoint directory.")
    parser.add_argument("--checkpoint", help="Converted wow_seg.pkl path; defaults to MODEL_DIR/wow_seg.pkl.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", help="Optional binary target mask. Omit for whole-image classification.")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-dynamic-patch", type=int, default=12)
    parser.add_argument("--cpu", action="store_true", help="Disable Jittor CUDA execution.")
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    mask = load_mask(args.mask, image.shape)
    predictor = WowSegJittorPredictor(
        args.model_dir,
        converted_checkpoint=args.checkpoint,
        use_cuda=not args.cpu,
        max_dynamic_patch=args.max_dynamic_patch,
    )
    answer, _ = predictor.predict_region(image, mask, prompt=args.prompt, max_new_tokens=args.max_new_tokens)
    print(answer)


if __name__ == "__main__":
    main()
