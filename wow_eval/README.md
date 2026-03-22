# WOW-Seg Evaluation (`wow_eval`)

Open-vocabulary classification on region masks: convert annotations to InternVL JSONL, then run `single_mask_infer.py` for inference.

Run all Python commands from **`wow_eval/`** so the local `internvl` package can be imported.

```bash
cd wow_eval
```

---

## Before you start

Download or prepare the following **data** and **models** in advance:

| Item | Description |
|------|-------------|
| **COCO images** | Images that match your evaluation split (typically `train2017/`, `val2017/`, etc.). Used with `--image_root` and relative paths from the JSONL. |
| **Osprey evaluation JSON** | Evaluation annotations from [Osprey](https://github.com/CircleRadon/Osprey) (exact files follow their release notes). |
| **WOW-Seg weights** | InternVL-format checkpoint for inference, e.g. [AAwcAA/WOW-Seg](https://huggingface.co/AAwcAA/WOW-Seg), passed as `--model_path`. |
| **SentenceBERT** | For example [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), passed as `--bert_path`. |

---

## Osprey data and conversion

Fetch the original evaluation JSON according to Osprey; this repo does not ship it. Then convert it to InternVL **JSONL** with **`convert_osprey_to_internvl.py`** (set `COCO_ROOT_DIR` and `BATCH_TASKS` inputs/outputs in the script):

```bash
python convert_osprey_to_internvl.py
```

---

## Inference

```bash
python single_mask_infer.py \
  --dataset_path path/to/osprey2internvl_xxx.jsonl \
  --image_root path/to/coco-all-imgs \
  --model_path path/to/WOW-Seg \
  --bert_path path/to/all-MiniLM-L6-v2 \
  --output_path path/to/results \
  --subset_num 1 \
  --subset_idx 0
```

For multiple GPUs, shard with `--subset_num` / `--subset_idx`; outputs are usually `subset_*.json`. See `single_mask_infer.sh` as a template (edit script paths inside if needed).

---

## Files

| File | Role |
|------|------|
| `convert_osprey_to_internvl.py` | Osprey-style JSON → InternVL JSONL |
| `single_mask_infer.py` | Mask-conditioned inference |
| `single_mask_infer.sh` | Multi-GPU launcher example |
| `internvl/` | Model code used by inference |
