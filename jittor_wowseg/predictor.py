"""Drop-in region classifier used by the existing WOW-Seg Web UI."""

from __future__ import annotations

from pathlib import Path

import jittor as jt
import numpy as np

from .model import WowSegJittor
from .preprocess import CATEGORY_ONLY_PROMPT, build_region_query, clean_category_response, prepare_region
from .tokenizer import WowTokenizer


class WowSegJittorPredictor:
    """Jittor replacement for ``demo.wow_inference.WowSegPredictor``.

    SAM is intentionally left in the original demo because it owns mask
    proposal generation; all WOW-Seg visual-language encoding and decoding is
    executed by Jittor.
    """
    def __init__(self, model_path, converted_checkpoint=None, max_dynamic_patch=12, scale=2.5, use_cuda=True):
        self.use_cuda = bool(use_cuda)
        self.model_path = Path(model_path).resolve()
        self.converted_checkpoint = Path(converted_checkpoint or self.model_path / "wow_seg.pkl").resolve()
        if not self.converted_checkpoint.is_file():
            raise FileNotFoundError(
                f"Converted Jittor weight is missing: {self.converted_checkpoint}. "
                "Run jittor_wowseg/convert_weights.py first."
            )
        if self.use_cuda:
            jt.flags.use_cuda = 1
        self.model = WowSegJittor.from_checkpoint(self.model_path, self.converted_checkpoint)
        if self.use_cuda:
            # The released checkpoint is BF16. The converter stores it as
            # FP16, so keep every Jittor parameter and input at one dtype for
            # CUDA/cuBLAS compatibility.
            self.model.float16()
        self.tokenizer = WowTokenizer(self.model_path)
        self.max_dynamic_patch = max_dynamic_patch
        self.scale = scale

    def predict_region(self, image_rgb, mask_hw, prompt=None, max_new_tokens=100):
        pixel_values, mask_16, full_tile_count, crop_rgb = prepare_region(
            image_rgb,
            mask_hw,
            max_dynamic_patch=self.max_dynamic_patch,
            image_size=self.model.config.image_size,
            scale=self.scale,
        )
        question = prompt or CATEGORY_ONLY_PROMPT
        # ``pixel_values`` contains all dynamic full-image tiles followed by
        # one crop tile. The Jittor path retains every full-image tile and
        # applies the 16x16 mask only to the final crop row, so the query must
        # contain exactly ``full_tile_count * 256`` full-image contexts.
        query = build_region_query(question, full_tile_count, int(mask_16.sum()))
        input_ids = self.tokenizer.encode(query + ",")
        token_ids = self.model.generate_masked_region(
            input_ids=input_ids,
            pixel_values=jt.array(pixel_values).float16() if self.use_cuda else jt.array(pixel_values),
            pixel_mask_16x16=mask_16,
            valid_region_idx=full_tile_count,
            image_context_token_id=self.tokenizer.image_context_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            stop_token_ids={self.tokenizer.comma_token_id} if self.tokenizer.comma_token_id is not None else None,
        )
        return self.tokenizer.decode(token_ids).strip(), crop_rgb

    def classify_region(self, image_rgb, mask_hw, prompt=None, max_new_tokens=100):
        text, _ = self.predict_region(image_rgb, mask_hw, prompt=prompt, max_new_tokens=max_new_tokens)
        return clean_category_response(text)
