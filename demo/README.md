# WOW-Seg Demo

## Install Dependencies

First install the base dependencies of the repository:

```bash
pip install -r ../requirements.txt
```


If you do not want to use the local `demo/segment_anything/` source tree, you can also install the official Segment Anything package:

```bash
pip install "git+https://github.com/facebookresearch/segment-anything.git"
```

## Prepare Checkpoints

This demo needs two checkpoints:

### 1. WOW-Seg

Download from:

```text
https://huggingface.co/AAwcAA/WOW-Seg
```

Place the model under:

```text
demo/weights/wow_seg/
```

The folder should be a Hugging Face style checkpoint that can be loaded with `from_pretrained`.

### 2. SAM

Download Segment Anything checkpoints from:

```text
https://github.com/facebookresearch/segment-anything
```

For example, if you use `ViT-B`, place it at:

```text
demo/weights/sam/sam_vit_b_01ec64.pth
```

Example layout:

```text
demo/
├── app_web.py
├── batch_pipeline.py
├── wow_inference.py
├── sam_helpers.py
├── weights
│   ├── wow_seg
│   │   ├── config.json
│   │   ├── *.safetensors / pytorch_model*.bin
│   │   └── ...
│   └── sam
│       └── sam_vit_b_01ec64.pth
└── segment_anything
```

## Run

Run the web demo from the `demo/` directory:

```bash
python app_web.py --wow_model ./weights/wow_seg --sam_ckpt ./weights/sam/sam_vit_b_01ec64.pth --sam_model_type vit_b
```

Then open:

```text
http://127.0.0.1:7861
```

The current web UI supports:

- `Auto`
- `Point Prompt`
- `BBox Prompt`
