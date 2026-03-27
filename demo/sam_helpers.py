"""SAM (segment-anything) image segmentation helpers (point / box / everything)."""

from __future__ import annotations

import gc
import importlib
import os
import sys
from typing import List, Tuple

import numpy as np
import torch


def build_sam_bundle(
    checkpoint_path: str,
    model_type: str = "vit_h",
    device: str | torch.device = "cuda",
):
    def _import_sam_symbols():
        # Compatibility for launching from demo/ when a local source tree exists:
        # demo/segment_anything/segment_anything/__init__.py
        local_sa_repo = os.path.join(os.path.dirname(__file__), "segment_anything")
        local_sa_pkg_init = os.path.join(local_sa_repo, "segment_anything", "__init__.py")
        if os.path.isfile(local_sa_pkg_init):
            if local_sa_repo not in sys.path:
                sys.path.insert(0, local_sa_repo)
            importlib.invalidate_caches()
            # If a namespace module was already created from demo/segment_anything,
            # remove it so Python can re-import from local_sa_repo/segment_anything.
            if "segment_anything" in sys.modules:
                del sys.modules["segment_anything"]

        try:
            from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
            from segment_anything.predictor import SamPredictor
            from segment_anything.build_sam import sam_model_registry
            return SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
        except Exception as e:
            raise ImportError(
                "Failed to import segment_anything submodules. "
                "Install official package with: "
                "pip install \"git+https://github.com/facebookresearch/segment-anything.git\" "
                "or keep local source tree at demo/segment_anything/segment_anything/."
            ) from e

    SamAutomaticMaskGenerator, SamPredictor, sam_model_registry = _import_sam_symbols()

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam = sam.to(dev)
    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(model=sam)
    return predictor, mask_generator


def predict_from_points(
    predictor,
    image_rgb: np.ndarray,
    points: List[Tuple[Tuple[int, int], int]],
    multimask_output: bool = False,
) -> np.ndarray:
    """points: [((x,y), label), ...] label 1=fg, 0=bg."""
    if not points:
        raise ValueError("no points")
    predictor.set_image(image_rgb)
    pts = torch.tensor([p for p, _ in points], device=predictor.device).unsqueeze(0)
    lbs = torch.tensor([int(l) for _, l in points], device=predictor.device).unsqueeze(0)
    transformed_points = predictor.transform.apply_coords_torch(pts, image_rgb.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=transformed_points,
        point_labels=lbs,
        multimask_output=multimask_output,
    )
    masks = masks.cpu().detach().numpy()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return masks[0][0].astype(np.uint8)


def predict_from_box(
    predictor,
    image_rgb: np.ndarray,
    box_xyxy: np.ndarray,
    multimask_output: bool = False,
) -> np.ndarray:
    predictor.set_image(image_rgb)
    input_boxes = torch.tensor(box_xyxy[None, :], device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image_rgb.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=multimask_output,
    )
    masks = masks.cpu().detach().numpy()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return masks[0][0].astype(np.uint8)


def generate_all_masks(mask_generator, image_rgb: np.ndarray) -> List[np.ndarray]:
    anns = mask_generator.generate(image_rgb)
    return [m["segmentation"].astype(np.uint8) for m in anns]

