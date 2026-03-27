"""WOW-Seg region inference helpers for the Gradio demo."""

from __future__ import annotations

import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
WOW_EVAL_DIR = REPO_ROOT / "wow_eval"

for _path in (REPO_ROOT, WOW_EVAL_DIR):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from internvl.train.dataset import build_transform, dynamic_preprocess_with_target_size  # noqa: E402

DEFAULT_REGION_PROMPT = (
    "This is the original image: <image>. Please classify the specified target area. "
    "Specified target area:<image>"
)
CATEGORY_ONLY_SINGLE_IMAGE_PROMPT = (
    "This is the original image: <image>. Please classify the specified target area indicated by the mask. "
    "Reply with only the category name of the target area, without explanation."
)
CATEGORY_ONLY_PROMPT = (
    "This is the original image: <image>. Please classify the specified target area. "
    "Reply with only the category name of the target area, without explanation. "
    "Specified target area:<image>"
)
SHORT_REGION_PROMPT = (
    "This is the original image: <image>. Please describe the specified target area in a short phrase. "
    "Specified target area:<image>"
)
DETAILED_REGION_PROMPT = (
    "This is the original image: <image>. Please describe the specified target area in detail. "
    "Specified target area:<image>"
)


@dataclass
class PreparedRegionInputs:
    full_pixel_values: torch.Tensor
    combined_pixel_values: torch.Tensor
    foreground_pixel_values: torch.Tensor
    pixel_masks: torch.Tensor
    num_patches_list: list[int]
    crop_rgb: np.ndarray


def normalize_region_prompt(prompt: str | None):
    prompt_text = (prompt or "").strip()
    if not prompt_text:
        return DEFAULT_REGION_PROMPT

    image_count = prompt_text.count("<image>")
    if image_count >= 2:
        return prompt_text
    if image_count == 1:
        return f"{prompt_text.rstrip()} Specified target area:<image>"
    return f"This is the original image: <image>. {prompt_text.rstrip()} Specified target area:<image>"


def clean_category_response(text: str):
    cleaned = (text or "").replace("\r", "\n").strip()
    if not cleaned:
        return cleaned

    first_line = cleaned.split("\n")[0].strip()
    lowered = first_line.lower()
    for prefix in ("category:", "label:", "class:", "answer:"):
        if lowered.startswith(prefix):
            first_line = first_line[len(prefix) :].strip()
            break

    if ". " in first_line:
        first_line = first_line.split(". ", 1)[0].strip()
    return first_line.strip(" .")


def _prepare_mask(mask_hw: np.ndarray, image_shape_hw: tuple[int, int]):
    height, width = image_shape_hw
    mask = np.asarray(mask_hw)
    if mask.shape[:2] != (height, width):
        raise ValueError(f"Mask shape {mask.shape} does not match image size {(height, width)}.")

    if mask.dtype not in (np.float32, np.float64):
        mask = mask.astype(np.float32)
    if mask.max() <= 1.0:
        mask = (mask > 0.5).astype(np.float32)
    else:
        mask = (mask > 127).astype(np.float32)
    return mask


def _compute_crop_box(mask_tensor: torch.Tensor, resized_size_wh: tuple[int, int], scale: float):
    ones_coords = torch.nonzero(mask_tensor)
    if ones_coords.numel() == 0:
        return None

    min_row = torch.min(ones_coords[:, 0])
    max_row = torch.max(ones_coords[:, 0])
    min_col = torch.min(ones_coords[:, 1])
    max_col = torch.max(ones_coords[:, 1])
    center_row = (min_row + max_row) / 2
    center_col = (min_col + max_col) / 2
    box_width = max_col - min_col + 1
    box_height = max_row - min_row + 1

    resized_width, resized_height = resized_size_wh
    new_width = min(max(box_width, box_height) * scale, resized_width)
    new_height = min(max(box_width, box_height) * scale, resized_height)
    new_min_col = max(0, int(center_col - new_width / 2))
    new_max_col = min(resized_width - 1, int(center_col + new_width / 2))
    new_min_row = max(0, int(center_row - new_height / 2))
    new_max_row = min(resized_height - 1, int(center_row + new_height / 2))

    if new_max_col - new_min_col + 1 < new_width:
        if new_min_col == 0:
            new_max_col = min(resized_width - 1, int(new_min_col + new_width))
        else:
            new_min_col = max(0, int(new_max_col - new_width))
    if new_max_row - new_min_row + 1 < new_height:
        if new_min_row == 0:
            new_max_row = min(resized_height - 1, int(new_min_row + new_height))
        else:
            new_min_row = max(0, int(new_max_row - new_height))

    return new_min_col, new_min_row, new_max_col + 1, new_max_row + 1


class WowSegPredictor:
    def __init__(
        self,
        model_path: str,
        device: str | torch.device = "cuda",
        force_image_size: int = 448,
        max_dynamic_patch: int = 12,
        scale: float = 2.5,
        use_thumbnail: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.force_image_size = force_image_size
        self.max_dynamic_patch = max_dynamic_patch
        self.scale = scale
        self.use_thumbnail = use_thumbnail
        self.transform = build_transform(is_train=False, input_size=force_image_size)

        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        config = self._load_with_local_fallback(
            AutoConfig.from_pretrained,
            model_path,
            trust_remote_code=True,
        )
        setattr(config, "_attn_implementation", "eager")
        if hasattr(config, "llm_config"):
            setattr(config.llm_config, "_attn_implementation", "eager")
            setattr(config.llm_config, "attn_implementation", "eager")
            if hasattr(config.llm_config, "_attn_implementation_internal"):
                setattr(config.llm_config, "_attn_implementation_internal", "eager")
            if hasattr(config.llm_config, "use_flash_attn"):
                setattr(config.llm_config, "use_flash_attn", False)
            if hasattr(config.llm_config, "use_flash_attention_2"):
                setattr(config.llm_config, "use_flash_attention_2", False)

        self.model = self._load_model_compat(model_path, config, dtype).to(self.device)
        self.model.eval()
        self.tokenizer = self._load_with_local_fallback(
            AutoTokenizer.from_pretrained,
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )

    def _load_with_local_fallback(self, loader, model_path: str, **kwargs):
        if os.path.isdir(model_path):
            try:
                return loader(model_path, local_files_only=True, **kwargs)
            except Exception:
                return loader(model_path, **kwargs)
        return loader(model_path, **kwargs)

    def _load_model_compat(self, model_path: str, config, dtype: torch.dtype):
        base_kwargs = dict(
            model_path=model_path,
            config=config,
            torch_dtype=dtype,
            attn_implementation="eager",
        )

        def _load():
            return self._load_with_local_fallback(
                AutoModelForCausalLM.from_pretrained,
                base_kwargs["model_path"],
                config=base_kwargs["config"],
                torch_dtype=base_kwargs["torch_dtype"],
                trust_remote_code=True,
                attn_implementation=base_kwargs["attn_implementation"],
            )

        try:
            return _load()
        except ImportError as exc:
            if "PyTorch SDPA requirements in Transformers are not met" not in str(exc):
                raise

            original = PreTrainedModel._check_and_enable_sdpa

            def _no_sdpa_check(cls, cfg, *args, **kwargs):
                return cfg

            PreTrainedModel._check_and_enable_sdpa = classmethod(_no_sdpa_check)
            try:
                return _load()
            finally:
                PreTrainedModel._check_and_enable_sdpa = original

    def _prepare_region_inputs(self, image_rgb: np.ndarray, mask_hw: np.ndarray):
        height, width = image_rgb.shape[:2]
        mask = _prepare_mask(mask_hw, (height, width))
        if mask.sum() == 0:
            return None

        image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
        images, resized_img = dynamic_preprocess_with_target_size(
            image,
            min_num=1,
            max_num=self.max_dynamic_patch,
            image_size=self.force_image_size,
            use_thumbnail=self.use_thumbnail,
        )

        pixel_values = torch.stack([self.transform(img) for img in images]).to(
            dtype=self.model.dtype,
            device=self.device,
        )

        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        mask_tensor = F.interpolate(
            mask_tensor,
            size=(resized_img.size[1], resized_img.size[0]),
            mode="nearest",
        ).squeeze(0).squeeze(0)

        crop_box = _compute_crop_box(mask_tensor, resized_img.size, self.scale)
        if crop_box is None:
            return None

        min_col, min_row, max_col, max_row = crop_box
        cropped_img = resized_img.crop((min_col, min_row, max_col, max_row))
        cropped_mask = mask_tensor.unsqueeze(0)[:, min_row:max_row, min_col:max_col]
        resized_cropped_mask = F.interpolate(
            cropped_mask.unsqueeze(0),
            size=(16, 16),
            mode="nearest",
        ).squeeze(0)
        foreground_pixel_values = self.transform(cropped_img).unsqueeze(0).to(
            dtype=self.model.dtype,
            device=self.device,
        )

        combined_pixel_values = torch.cat([pixel_values, foreground_pixel_values], dim=0)
        return PreparedRegionInputs(
            full_pixel_values=pixel_values,
            combined_pixel_values=combined_pixel_values,
            foreground_pixel_values=foreground_pixel_values,
            pixel_masks=resized_cropped_mask.to(dtype=self.model.dtype, device=self.device),
            num_patches_list=[pixel_values.shape[0], 1],
            crop_rgb=np.array(cropped_img, dtype=np.uint8),
        )

    def _build_chat_kwargs(
        self,
        prompt: str,
        prepared: PreparedRegionInputs,
        max_new_tokens: int,
        mode: str,
    ):
        parameters = inspect.signature(self.model.chat).parameters
        generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}

        chat_kwargs = {
            "tokenizer": self.tokenizer,
            "question": prompt,
            "generation_config": generation_config,
        }

        if mode == "official_combined":
            chat_kwargs["pixel_values"] = prepared.combined_pixel_values
            if "pixel_masks" in parameters:
                chat_kwargs["pixel_masks"] = prepared.pixel_masks
            if "vaild_region_idx" in parameters:
                chat_kwargs["vaild_region_idx"] = torch.tensor(
                    prepared.num_patches_list[0],
                    dtype=torch.long,
                    device=self.device,
                )
            if "valid_region_idx" in parameters:
                chat_kwargs["valid_region_idx"] = torch.tensor(
                    prepared.num_patches_list[0],
                    dtype=torch.long,
                    device=self.device,
                )
            if "thumbnail_values" in parameters:
                chat_kwargs["thumbnail_values"] = prepared.foreground_pixel_values
        elif mode == "combined_with_patch_list":
            chat_kwargs["pixel_values"] = prepared.combined_pixel_values
            if "pixel_masks" in parameters:
                chat_kwargs["pixel_masks"] = prepared.pixel_masks
            if "num_patches_list" in parameters:
                chat_kwargs["num_patches_list"] = prepared.num_patches_list
            if "vaild_region_idx" in parameters:
                chat_kwargs["vaild_region_idx"] = torch.tensor(
                    prepared.num_patches_list[0],
                    dtype=torch.long,
                    device=self.device,
                )
            if "valid_region_idx" in parameters:
                chat_kwargs["valid_region_idx"] = torch.tensor(
                    prepared.num_patches_list[0],
                    dtype=torch.long,
                    device=self.device,
                )
        elif mode == "full_image_mask_only":
            chat_kwargs["pixel_values"] = prepared.full_pixel_values
            if "pixel_masks" in parameters:
                chat_kwargs["pixel_masks"] = prepared.pixel_masks
            if "num_patches_list" in parameters:
                chat_kwargs["num_patches_list"] = [prepared.num_patches_list[0]]
        else:
            raise ValueError(f"Unknown chat mode: {mode}")

        if "pixel_masks" not in parameters and "vaild_region_idx" not in parameters and "valid_region_idx" not in parameters:
            raise RuntimeError(
                "The loaded model.chat interface does not expose mask-conditioned inputs. "
                "Please make sure you are loading WOW-Seg weights/code rather than a plain InternVL checkpoint."
            )

        return chat_kwargs

    def _chat_with_fallbacks(
        self,
        prompt: str,
        prepared: PreparedRegionInputs,
        max_new_tokens: int,
    ):
        attempts = [
            ("official_combined", prompt),
            ("combined_with_patch_list", prompt),
            ("full_image_mask_only", CATEGORY_ONLY_SINGLE_IMAGE_PROMPT),
        ]
        errors = []

        for mode, mode_prompt in attempts:
            try:
                return self.model.chat(
                    **self._build_chat_kwargs(
                        prompt=mode_prompt,
                        prepared=prepared,
                        max_new_tokens=max_new_tokens,
                        mode=mode,
                    )
                )
            except Exception as exc:
                errors.append(f"{mode}: {type(exc).__name__}: {exc}")

        raise RuntimeError(" | ".join(errors))

    @torch.inference_mode()
    def predict_region(
        self,
        image_rgb: np.ndarray,
        mask_hw: np.ndarray,
        prompt: str | None = None,
        max_new_tokens: int = 100,
    ):
        if image_rgb is None or mask_hw is None:
            return "No image or mask.", None

        prepared = self._prepare_region_inputs(image_rgb, mask_hw)
        if prepared is None:
            return "Empty mask; select a visible region first.", None

        normalized_prompt = normalize_region_prompt(prompt)
        response = self._chat_with_fallbacks(
            prompt=normalized_prompt,
            prepared=prepared,
            max_new_tokens=max_new_tokens,
        )
        return str(response).strip(), prepared.crop_rgb

    @torch.inference_mode()
    def classify_region(
        self,
        image_rgb: np.ndarray,
        mask_hw: np.ndarray,
        prompt: str | None = None,
        max_new_tokens: int = 100,
    ):
        text, _crop = self.predict_region(
            image_rgb=image_rgb,
            mask_hw=mask_hw,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        return clean_category_response(text)
