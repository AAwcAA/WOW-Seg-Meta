import os
import json
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import sys
from multiprocessing import Pool
from multiprocessing import get_context
import torch
import re
import pdb

def parse_args(args):
    parser = argparse.ArgumentParser(description="WOW-Seg Evaluation")
    parser.add_argument("--bert_path", default="path/to/all-MiniLM-L6-v2")
    parser.add_argument("--num_processes", default=1, type=int, help="Number of processes to use")
    parser.add_argument("--gpus", default="0", help="Comma-separated list of GPU indices to use")
    parser.add_argument("--results_path", default="path/to/results")
    return parser.parse_args(args)

def split_into_subsets(lst, subset_num):
    length = len(lst)
    base = length // subset_num
    remainder = length % subset_num
    result = []
    start = 0
    for i in range(subset_num):
        end = start + base + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end
    return result

def semantic_iou(value: str, target: str) -> float:
    intersection = len(set(value.split()) & set(target.split()))
    union = len(set(value.split()) | set(target.split()))
    return intersection / union if union > 0 else 0

def process_subset(args):
    subset, bert_path, results_path, gpu_id, subset_idx = args
    print(f"Worker {subset_idx}: using GPU {gpu_id}")
    device = f"cuda:{gpu_id}"
    bert_model = SentenceTransformer(bert_path).to(device)
    per_image_results = []
    for sub_json in tqdm(subset):
        if sub_json.endswith(".json"):
            with open(os.path.join(results_path, sub_json), "r") as f:
                data = json.load(f)
                for item in data:
                    per_image_results.append(item)
    sem_sim = 0
    sem_iou = 0
    num_ins = 0
    for i in tqdm(per_image_results):
        for pred_label, gt_label in zip(i["pred_labels"], i["gt_labels"]):
            if "_(" in gt_label and ")" in gt_label:
                gt_label = re.sub(r"_\(.*?\)", "", gt_label)
            pred_label = pred_label.replace('_', ' ').replace(":", " ")
            gt_label = gt_label.replace('_', ' ').replace(":", " ")
            pred_embeddings = bert_model.encode(pred_label, convert_to_tensor=True)
            gt_embeddings = bert_model.encode(gt_label, convert_to_tensor=True)
            cosine_scores = util.cos_sim(pred_embeddings, gt_embeddings)
            semantic_iou_score = semantic_iou(pred_label.lower(), gt_label.lower())
            sem_sim += float(cosine_scores[0][0])
            sem_iou += semantic_iou_score
            num_ins += 1
    return sem_sim, sem_iou, num_ins

def main(args):
    args = parse_args(args)
    bert_path = args.bert_path
    results_path = args.results_path
    num_processes = args.num_processes
    json_list = os.listdir(results_path)
    gpus = args.gpus.split(',')

    if len(gpus) < num_processes:
        print(f"Warning: fewer GPUs ({len(gpus)}) than processes ({num_processes}); reusing GPUs.")
        # Cycle through the GPU list
        gpus = [gpus[i % len(gpus)] for i in range(num_processes)]

    job_list = split_into_subsets(json_list, num_processes)

    process_args = [(job_list[subset_idx], bert_path, results_path, gpus[subset_idx], subset_idx) for subset_idx in range(num_processes)]

    with get_context('spawn').Pool(processes=num_processes) as pool:
        # tqdm over pool workers
        results = list(tqdm(pool.imap(process_subset, process_args), 
                           total=num_processes, 
                           desc="Processing progress"))
    
    total_sem_sim = 0.0
    total_sem_iou = 0.0
    total_num_ins = 0
    for sem_sim, sem_iou, num_ins in results:
        total_sem_sim += sem_sim
        total_sem_iou += sem_iou
        total_num_ins += num_ins
    print(
        f"Mean semantic similarity (per instance): {total_sem_sim/total_num_ins:.4f} | "
        f"Mean SemanticIOU: {total_sem_iou/total_num_ins:.4f} | "
        f"Total instances: {total_num_ins}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])