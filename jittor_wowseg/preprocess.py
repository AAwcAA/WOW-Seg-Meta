"""Image, mask, and conversation preparation shared by Jittor CLI and Web UI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)
SYSTEM_MESSAGE = "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"
CATEGORY_ONLY_PROMPT = (
    "This is the original image: <image>. Please classify the specified target area. "
    "Reply with only the category name of the target area, without explanation. "
    "Specified target area:<image>"
)


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio, best_diff = (1, 1), float("inf")
    area = width * height
    for ratio in target_ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_diff:
            best_ratio, best_diff = ratio, diff
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    """Exact tile selection used by the original InternVL dataset helper."""
    image = image.convert("RGB")
    width, height = image.size
    aspect_ratio = width / height
    ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    ratio = find_closest_aspect_ratio(aspect_ratio, sorted(ratios, key=lambda item: item[0] * item[1]), width, height, image_size)
    target_width, target_height = image_size * ratio[0], image_size * ratio[1]
    resized = image.resize((target_width, target_height), Image.BICUBIC)
    tiles = []
    for index in range(ratio[0] * ratio[1]):
        x = (index % ratio[0]) * image_size
        y = (index // ratio[0]) * image_size
        tiles.append(resized.crop((x, y, x + image_size, y + image_size)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size), Image.BICUBIC))
    return tiles, resized


def image_to_tensor(image, image_size=448):
    image = image.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def normalize_mask(mask, image_shape):
    array = np.asarray(mask)
    if array.shape[:2] != tuple(image_shape[:2]):
        raise ValueError(f"Mask shape {array.shape} does not match image shape {image_shape[:2]}")
    return array > (0.5 if array.max() <= 1 else 127)


def compute_crop_box(mask, resized_size, scale=2.5):
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    width, height = resized_size
    min_x, max_x, min_y, max_y = xs.min(), xs.max(), ys.min(), ys.max()
    side = min(max(max_x - min_x + 1, max_y - min_y + 1) * scale, width, height)
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    x0, y0 = max(0, int(center_x - side / 2)), max(0, int(center_y - side / 2))
    x1, y1 = min(width, int(center_x + side / 2)), min(height, int(center_y + side / 2))
    if x1 - x0 < side:
        x0, x1 = (0, min(width, int(side))) if x0 == 0 else (max(0, int(width - side)), width)
    if y1 - y0 < side:
        y0, y1 = (0, min(height, int(side))) if y0 == 0 else (max(0, int(height - side)), height)
    return x0, y0, x1, y1


def prepare_region(image_rgb, mask, max_dynamic_patch=12, image_size=448, scale=2.5):
    image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB")
    binary_mask = normalize_mask(mask, image_rgb.shape)
    if not binary_mask.any():
        raise ValueError("Empty mask; select a visible region first.")
    tiles, resized = dynamic_preprocess(image, max_num=max_dynamic_patch, image_size=image_size, use_thumbnail=True)
    resized_mask = np.asarray(
        Image.fromarray(binary_mask.astype(np.uint8) * 255).resize(resized.size, Image.NEAREST), dtype=np.uint8
    ) > 0
    crop_box = compute_crop_box(resized_mask, resized.size, scale)
    if crop_box is None:
        raise ValueError("Empty mask after resize.")
    crop = resized.crop(crop_box)
    x0, y0, x1, y1 = crop_box
    crop_mask = resized_mask[y0:y1, x0:x1]
    mask_16 = np.asarray(
        Image.fromarray(crop_mask.astype(np.uint8) * 255).resize((16, 16), Image.NEAREST), dtype=np.uint8
    ) > 0
    pixel_values = np.stack([image_to_tensor(tile, image_size) for tile in tiles + [crop]], axis=0)
    return pixel_values, mask_16, len(tiles), np.asarray(crop, dtype=np.uint8)


def build_region_query(question, full_tile_count, target_token_count):
    question = question.strip()
    if question.count("<image>") < 2:
        question = CATEGORY_ONLY_PROMPT if not question else f"This is the original image: <image>. {question} Specified target area:<image>"
    image_context = "<IMG_CONTEXT>"
    first_image = "<img>" + image_context * (256 * full_tile_count) + "</img>"
    target_image = "<img>" + image_context * int(target_token_count) + "</img>"
    question = question.replace("<image>", first_image, 1).replace("<image>", target_image, 1)
    return (
        "<|im_start|>system\n" + SYSTEM_MESSAGE + "<|im_end|>\n"
        + "<|im_start|>user\n" + question + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )


def clean_category_response(text):
    text = (text or "").replace("\r", "\n").strip().split("\n", 1)[0].strip()
    for prefix in ("category:", "label:", "class:", "answer:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
            break
    # Greedy decoding can continue a category with comma-separated repeats
    # (for example, ``"train, train, train"``). The first segment is the
    # model's category answer and keeps UI labels concise.
    text = text.split(". ", 1)[0].split(",", 1)[0]
    return text.strip(" .,")
