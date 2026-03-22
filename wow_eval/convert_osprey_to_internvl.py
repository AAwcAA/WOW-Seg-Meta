#!/usr/bin/env python
"""
Convert Osprey-style JSON annotations to InternVL JSONL format.

How to configure paths:
1) COCO_ROOT_DIR
   - Set this to the directory that contains `train2017/` and `val2017/`.
   - Example:
       COCO_ROOT_DIR = "path/to/coco"
   - If left as empty string "", auto split detection is disabled and `image_prefix`
     from each task is used directly.

2) BATCH_TASKS[*]["input"]
   - Absolute or relative path to the source Osprey JSON file.
   - The input JSON must be a list of items.

3) BATCH_TASKS[*]["output"]
   - Absolute or relative path to the output JSONL file.
   - Parent directories are created automatically.

4) BATCH_TASKS[*]["image_prefix"]
   - Used when COCO_ROOT_DIR is empty or image file is not found.
   - Typical value: "val2017" or "train2017".

Run:
    python convert_osprey_to_internvl.py
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np
from pycocotools import mask as mask_utils


# If set, script auto-detects image split by checking:
#   {COCO_ROOT_DIR}/train2017/{file_name}
#   {COCO_ROOT_DIR}/val2017/{file_name}
# If empty (""), script uses each task's `image_prefix` directly.
# Example: "path/to/coco"
COCO_ROOT_DIR = "path/to/coco"


# Batch conversion tasks.
# - name: only for logging
# - input: source Osprey JSON file path
# - output: target InternVL JSONL file path
# - image_prefix: fallback prefix in `image` field (e.g., "val2017")
BATCH_TASKS = [
    {
        "name": "lvis",
        "input": "path/to/osprey_lvis_val.json",
        "output": "path/to/osprey2internvl_lvis_val.jsonl",
        "image_prefix": "val2017",
    },
    {
        "name": "paco",
        "input": "path/to/osprey_paco_val.json",
        "output": "path/to/osprey2internvl_paco_val.jsonl",
        "image_prefix": "val2017",
    },
]


def _normalize_category_name(category_name: str) -> str:
    # Keep behavior aligned with build_lvis.py::process_category_name.
    while "(" in category_name and ")" in category_name:
        start = category_name.find("(")
        end = category_name.find(")", start)
        if end == -1:
            break
        if start > 0 and category_name[start - 1] == "_":
            start -= 1
        category_name = category_name[:start].strip() + category_name[end + 1 :]

    category_name = category_name.replace("_", " ")
    category_name = category_name.replace("-", " ")

    parts = category_name.split(":")
    if len(parts) > 1:
        parts = [p.strip() for p in parts]
        category_name = ": ".join(parts)

    category_name = " ".join(category_name.split())
    return category_name.strip()


def _polygon_to_rle_cv2(polygon_segmentation: Any, height: int, width: int) -> Dict[str, Any]:
    mask = np.zeros((height, width), dtype=np.uint8)
    if isinstance(polygon_segmentation, list) and polygon_segmentation:
        if isinstance(polygon_segmentation[0], list):
            for poly in polygon_segmentation:
                if len(poly) >= 6:
                    pts = np.array(poly).reshape(-1, 2)
                    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
        else:
            if len(polygon_segmentation) >= 6:
                pts = np.array(polygon_segmentation).reshape(-1, 2)
                cv2.fillPoly(mask, [pts.astype(np.int32)], 1)

    rle = mask_utils.encode(np.asfortranarray(mask))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return {"size": rle["size"], "counts": rle["counts"]}


def _segmentation_to_rle(segmentation: Any, height: int, width: int) -> Dict[str, Any]:
    """Convert segmentation to COCO RLE, aligned with build_lvis.py behavior."""
    if isinstance(segmentation, dict) and "counts" in segmentation and "size" in segmentation:
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
    elif isinstance(segmentation, list) and segmentation:
        # Follow build_lvis.py:
        # - list[list[float]] => polygon(s), rasterize then encode
        # - list[int]         => already uncompressed RLE counts
        if isinstance(segmentation[0], list):
            return _polygon_to_rle_cv2(segmentation, height, width)
        rle = {"size": [height, width], "counts": segmentation}
        rle = mask_utils.frPyObjects(rle, height, width)
    else:
        raise ValueError(f"Unsupported segmentation type: {type(segmentation)}")

    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return {"size": rle["size"], "counts": rle["counts"]}


def _resolve_image_path(file_name: str, image_prefix: str, coco_root: Optional[Path]) -> str:
    """
    Resolve image relative path.

    Priority:
    1) train2017 if file exists
    2) val2017 if file exists
    3) fallback to {image_prefix}/{file_name}
    """
    if coco_root is None:
        return f"{image_prefix.rstrip('/')}/{file_name}"

    train_path = coco_root / "train2017" / file_name
    if train_path.exists():
        return f"train2017/{file_name}"

    val_path = coco_root / "val2017" / file_name
    if val_path.exists():
        return f"val2017/{file_name}"

    # Fallback to configured prefix if file not found in either split.
    return f"{image_prefix.rstrip('/')}/{file_name}"


def _convert_item(item: Dict[str, Any], image_prefix: str, coco_root: Optional[Path]) -> Dict[str, Any]:
    image_id = str(item["id"])
    height = int(item["height"])
    width = int(item["width"])
    file_name = item["file_name"]
    labels = [_normalize_category_name(x) for x in item.get("categories", [])]
    annotations = item.get("annotations", [])

    masks = []
    for ann in annotations:
        if "segmentation" not in ann:
            continue
        masks.append(_segmentation_to_rle(ann["segmentation"], height, width))

    # Keep label and mask lengths aligned.
    if len(labels) != len(masks):
        n = min(len(labels), len(masks))
        labels = labels[:n]
        masks = masks[:n]

    return {
        "id": image_id,
        "image": _resolve_image_path(file_name=file_name, image_prefix=image_prefix, coco_root=coco_root),
        "width": width,
        "height": height,
        "conversations": [
            {"from": "human", "value": ""},
            {"from": "gpt", "value": " , ".join(labels)},
        ],
        "masks": masks,
        "label": labels,
    }


def convert_json_to_jsonl(
    input_path: Path, output_path: Path, image_prefix: str, coco_root: Optional[Path]
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Input must be a JSON array: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in data:
            converted = _convert_item(item, image_prefix=image_prefix, coco_root=coco_root)
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")


def _batch_tasks_from_config() -> Iterable[Dict[str, Any]]:
    for task in BATCH_TASKS:
        yield {
            "name": task["name"],
            "input": Path(task["input"]),
            "output": Path(task["output"]),
            "image_prefix": task.get("image_prefix", "val2017"),
        }


def main() -> None:
    coco_root_text = COCO_ROOT_DIR.strip()
    coco_root = Path(coco_root_text) if coco_root_text else None
    if coco_root is not None and not coco_root.exists():
        raise FileNotFoundError(f"COCO_ROOT_DIR does not exist: {coco_root}")

    for task in _batch_tasks_from_config():
        convert_json_to_jsonl(
            task["input"],
            task["output"],
            image_prefix=task["image_prefix"],
            coco_root=coco_root,
        )
        print(f"[{task['name']}] {task['input'].name} -> {task['output'].name}")


if __name__ == "__main__":
    main()
