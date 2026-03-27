"""Shared batch inference helpers for WOW-Seg demo and app_test."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from wow_inference import CATEGORY_ONLY_PROMPT

DEFAULT_MAX_NEW_TOKENS = 24


def mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return x0, y0, x1, y1


def sanitize_filename(text: str):
    cleaned = (text or "").strip()
    if not cleaned:
        cleaned = "unknown"
    cleaned = cleaned.replace("/", "_")
    cleaned = cleaned.replace("\\", "_")
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]", "", cleaned)
    return cleaned[:120] or "unknown"


def mask_to_rgba_crop(image_rgb: np.ndarray, mask: np.ndarray):
    bbox = mask_bbox(mask)
    if bbox is None:
        return None, None

    x0, y0, x1, y1 = bbox
    crop_rgb = image_rgb[y0:y1, x0:x1]
    crop_mask = (mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255

    rgba = np.zeros((crop_rgb.shape[0], crop_rgb.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = crop_rgb
    rgba[:, :, 3] = crop_mask
    return rgba, bbox


def save_mask_instance(image_rgb: np.ndarray, mask: np.ndarray, output_path: Path):
    rgba, bbox = mask_to_rgba_crop(image_rgb, mask)
    if rgba is None:
        return None
    Image.fromarray(rgba).save(output_path)
    return bbox


def classify_single_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    wow,
    *,
    index: int = 1,
    category_counter: defaultdict | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
):
    if category_counter is None:
        category_counter = defaultdict(int)

    area = int(mask.sum())
    try:
        category = wow.classify_region(
            image_rgb=image_rgb,
            mask_hw=mask,
            prompt=CATEGORY_ONLY_PROMPT,
            max_new_tokens=max_new_tokens,
        )
    except Exception as exc:
        category = f"error_{type(exc).__name__}"

    safe_category = sanitize_filename(category)
    category_counter[safe_category] += 1
    rgba, bbox = mask_to_rgba_crop(image_rgb, mask)
    if rgba is None or bbox is None:
        return None

    return {
        "index": index,
        "category": category,
        "safe_category": safe_category,
        "sequence_id": category_counter[safe_category],
        "area": area,
        "bbox_xyxy": list(bbox),
        "mask": mask,
        "crop_rgba": rgba,
    }


def _palette(index: int):
    colors = [
        (255, 99, 71),
        (65, 105, 225),
        (50, 205, 50),
        (255, 165, 0),
        (186, 85, 211),
        (0, 191, 255),
        (255, 215, 0),
        (220, 20, 60),
        (72, 209, 204),
        (199, 21, 133),
    ]
    return colors[index % len(colors)]


def render_mask_overlay(image_rgb: np.ndarray, results: list[dict]):
    canvas = image_rgb.copy()

    for idx, result in enumerate(sorted(results, key=lambda item: item["area"])):
        mask = result["mask"].astype(np.uint8)
        color = np.array(_palette(idx), dtype=np.uint8)

        colored = np.zeros_like(canvas, dtype=np.uint8)
        colored[mask > 0] = color
        canvas = np.where(mask[..., None] > 0, (0.55 * canvas + 0.45 * colored).astype(np.uint8), canvas)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (255, 255, 255), 2)

        bbox = result["bbox_xyxy"]
        x0, y0 = int(bbox[0]), int(bbox[1])
        label = result["category"]
        cv2.putText(
            canvas,
            label[:48],
            (x0 + 4, max(20, y0 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label[:48],
            (x0 + 4, max(20, y0 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            tuple(int(v) for v in color.tolist()),
            1,
            cv2.LINE_AA,
        )

    return canvas


def classify_masks_in_image(
    image_rgb: np.ndarray,
    mask_generator,
    wow,
    *,
    min_mask_area: int = 128,
    max_masks: int = 0,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
):
    masks = mask_generator.generate(image_rgb)
    masks = [entry["segmentation"].astype(np.uint8) for entry in masks]
    masks = [mask for mask in masks if int(mask.sum()) >= min_mask_area]
    masks.sort(key=lambda mask: int(mask.sum()), reverse=True)
    if max_masks > 0:
        masks = masks[:max_masks]

    results = []
    category_counter = defaultdict(int)
    for index, mask in enumerate(masks, start=1):
        result = classify_single_mask(
            image_rgb=image_rgb,
            mask=mask,
            wow=wow,
            index=index,
            category_counter=category_counter,
            max_new_tokens=max_new_tokens,
        )
        if result is None:
            continue
        results.append(result)

    return results
