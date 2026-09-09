# WOW-Seg Jittor Demo

This is the Jittor-only demo entry point. The original project demo under
`../../demo/` is intentionally unchanged.

Run from the repository root:

```bash
python -m jittor_wowseg.demo.app_web \
  --wow_model jittor_wowseg/weights/wow_seg \
  --sam_ckpt demo/weights/sam/sam_vit_b_01ec64.pth \
  --sam_model_type vit_b \
  --jittor_max_dynamic_patch 12
```

The demo supports Auto, Point Prompt, and BBox Prompt. Install the original
demo/SAM dependencies in addition to `requirements/jittor_wowseg.txt`.
