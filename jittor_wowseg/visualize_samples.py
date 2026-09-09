#!/usr/bin/env python
"""Run WOW-Seg Jittor on five sample images and render mask visualizations."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DEMO))


def color_for(index):
    palette = [
        (255, 99, 71), (65, 105, 225), (50, 205, 50), (255, 165, 0),
        (186, 85, 211), (0, 191, 255), (255, 215, 0), (220, 20, 60),
        (72, 209, 204), (199, 21, 133),
    ]
    return palette[index % len(palette)]


def render(image_rgb, results):
    canvas = image_rgb.astype(np.float32).copy()
    for idx, result in enumerate(results):
        mask = result["mask"] > 0
        color = np.asarray(color_for(idx), dtype=np.float32)
        canvas[mask] = canvas[mask] * 0.52 + color * 0.48
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (255, 255, 255), 2)
        x0, y0, x1, y1 = result["bbox"]
        label = f'{idx + 1}: {result["category"][:36]}'
        cv2.putText(canvas, label, (x0 + 4, max(22, y0 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (x0 + 4, max(22, y0 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color_for(idx), 1, cv2.LINE_AA)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=r"E:\111\rr_7k_test\sa_000000")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "jittor_samples"))
    parser.add_argument("--model-dir", default=str(ROOT / "jittor_wowseg" / "weights" / "wow_seg"))
    parser.add_argument("--sam-checkpoint", default=str(DEMO / "weights" / "sam" / "sam_vit_b_01ec64.pth"))
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-masks", type=int, default=10)
    parser.add_argument("--max-dynamic-patch", type=int, default=12)
    parser.add_argument("--min-mask-area", type=int, default=512)
    parser.add_argument("--cpu-sam", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(Path(args.input_dir).glob("*.jpg")) + sorted(Path(args.input_dir).glob("*.png"))
    if len(image_paths) < args.samples:
        raise RuntimeError(f"Only found {len(image_paths)} images in {args.input_dir}")
    rng = random.Random(20260909)
    selected_paths = rng.sample(image_paths, args.samples)

    import torch
    from sam_helpers import build_sam_bundle
    from jittor_wowseg.predictor import WowSegJittorPredictor

    device = "cpu" if args.cpu_sam or not torch.cuda.is_available() else "cuda"
    print(f"Loading SAM on {device}...", flush=True)
    predictor, _ = build_sam_bundle(args.sam_checkpoint, model_type="vit_b", device=device)
    # Fewer point samples keeps interactive/demo-sized runs practical.
    from segment_anything import SamAutomaticMaskGenerator
    mask_generator = SamAutomaticMaskGenerator(
        model=predictor.model,
        points_per_side=16,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.80,
        min_mask_region_area=args.min_mask_area,
    )
    print("Loading WOW-Seg Jittor...", flush=True)
    wow = WowSegJittorPredictor(args.model_dir, max_dynamic_patch=args.max_dynamic_patch, use_cuda=device != "cpu")

    all_records = []
    rendered = []
    for image_index, image_path in enumerate(selected_paths, 1):
        print(f"[{image_index}/{len(selected_paths)}] {image_path.name}", flush=True)
        image_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        anns = mask_generator.generate(image_rgb)
        anns = [a for a in anns if int(a["segmentation"].sum()) >= args.min_mask_area]
        anns.sort(key=lambda item: int(item["area"]), reverse=True)
        anns = anns[: args.max_masks]
        results = []
        for mask_index, ann in enumerate(anns, 1):
            mask = ann["segmentation"].astype(np.uint8)
            category = wow.classify_region(image_rgb, mask, max_new_tokens=24)
            ys, xs = np.where(mask > 0)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            results.append({"index": mask_index, "category": category, "area": int(mask.sum()), "bbox": bbox, "mask": mask})
            print(f"  mask {mask_index}/{len(anns)}: {category}", flush=True)
        overlay = render(image_rgb, results)
        output_path = output_dir / f"{image_path.stem}_wowseg_jittor.png"
        Image.fromarray(overlay).save(output_path)
        rendered.append((image_path.name, overlay, results))
        all_records.append({"image": str(image_path), "output": str(output_path), "masks": [{k: v for k, v in r.items() if k != "mask"} for r in results]})
        try:
            import jittor as jt
            jt.gc()
        except Exception:
            pass

    # Contact sheet: resize each result to a common width and annotate the source name.
    thumb_w = 900
    rows = []
    for name, image, results in rendered:
        scale = thumb_w / image.shape[1]
        thumb = Image.fromarray(image).resize((thumb_w, max(1, int(image.shape[0] * scale))), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (thumb.width, thumb.height + 42), "white")
        panel.paste(thumb, (0, 42))
        draw = ImageDraw.Draw(panel)
        draw.text((12, 12), f"{name} | {len(results)} masks", fill="black")
        rows.append(panel)
    sheet = Image.new("RGB", (thumb_w, sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet_path = output_dir / "contact_sheet.png"
    sheet.save(sheet_path)
    (output_dir / "results.json").write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {sheet_path}", flush=True)
    print(f"Wrote {output_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
