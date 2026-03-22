SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/single_mask_infer.py"
EVAL_SCRIPT="${SCRIPT_DIR}/semantic_eval.py"


SUBSET_NUM=8
MAX_DYNAMIC_PATCH=12

SCALE=2

PROMPT="You are an expert in image region classification. I will provide you with a complete image and a part of this image. You need to classify the region part based on the full image. Complete image: <image>. Region image:<image>"

IMAGE_ROOT="path/to/coco"
DATASET_PATH="path/to/osprey2internvl_lvis_val.jsonl"

BERT_PATH="path/to/all-MiniLM-L6-v2"
MODEL_PATH="path/to/wow-seg-internvl"
OUTPUT_PATH="path/to/results"


# Create output path so writes do not fail on missing dirs
if [[ "$OUTPUT_PATH" == *.json ]]; then
  mkdir -p "$(dirname "$OUTPUT_PATH")"
else
  mkdir -p "$OUTPUT_PATH"
fi


# Step-1: run inference (one GPU job per subset index, SUBSET_NUM parallel jobs)
for i in $(seq 0 $((SUBSET_NUM - 1))); do
  CUDA_VISIBLE_DEVICES=$i python "$SCRIPT" \
    --subset_idx=$i \
    --subset_num=$SUBSET_NUM \
    --output_path=$OUTPUT_PATH \
    --model_path=$MODEL_PATH \
    --bert_path=$BERT_PATH \
    --dataset_path=$DATASET_PATH \
    --image_root=$IMAGE_ROOT \
    --prompt="$PROMPT" \
    --scale=$SCALE \
    --max_dynamic_patch=$MAX_DYNAMIC_PATCH &
done

wait

# Step-2: semantic eval (keep num_processes / gpus aligned with SUBSET_NUM)
# Build GPU list, e.g. SUBSET_NUM=4 -> gpus="0,1,2,3"
GPU_LIST=$(seq -s "," 0 $((SUBSET_NUM - 1)))

python "$EVAL_SCRIPT" \
  --results_path=$OUTPUT_PATH \
  --bert_path=$BERT_PATH \
  --num_processes=$SUBSET_NUM \
  --gpus="$GPU_LIST"
