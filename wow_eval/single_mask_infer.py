import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from pycocotools import mask as maskUtils
import json
import os
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from torch.utils.data import Dataset
import argparse
import sys
sys.path.append(".")

from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
from internvl.train.dataset import build_transform, dynamic_preprocess, dynamic_preprocess_with_target_size
from internvl.conversation import get_conv_template
import pdb

import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, Subset

from pycocotools import mask as mask_utils

from PIL import ImageFilter
from PIL import Image

def parse_args(args):
    parser = argparse.ArgumentParser(description="WOW-Seg Evaluation")
    parser.add_argument("--model_path", default="path/to/wow-seg-internvl")
    parser.add_argument("--bert_path", default="path/to/all-MiniLM-L6-v2")
    parser.add_argument("--dataset_path", default="path/to/osprey2internvl_lvis_val.jsonl")
    parser.add_argument("--image_root", default="path/to/coco")
    parser.add_argument("--output_path", default="path/to/results")
    parser.add_argument("--subset_num", default=4, type=int)
    parser.add_argument("--subset_idx", default=0, type=int)
    parser.add_argument("--force_image_size", default=448, type=int, help="force_image_size")
    parser.add_argument("--dynamic_image_size", default=True, type=bool, help="dynamic_image_size")
    parser.add_argument("--use_thumbnail", default=True, type=bool, help="use_thumbnail")
    parser.add_argument("--max_dynamic_patch", default=12, type=int, help="max_dynamic_patch")
    parser.add_argument("--scale", default=2.5, type=float, help="scale")
    parser.add_argument("--prompt", default="This is the original image: <image>. Please classify the specified target area. Specified target area:<image>", type=str)
    return parser.parse_args(args)


def decode_rle(rle, height, width):
    if isinstance(rle, dict) and 'counts' in rle:
        return maskUtils.decode(rle)
    elif isinstance(rle, list):
        rles = maskUtils.frPyObjects(rle, height, width)
        rle = maskUtils.merge(rles)
        return maskUtils.decode(rle)
    elif isinstance(rle, str):
        rle_dict = {
            'counts': rle,
            'size': [height, width]
        }
        return maskUtils.decode(rle_dict)
    else:
        raise ValueError('Invalid RLE format')

def get_prompt(data, img_full_path):
    conversations = data.get('conversations', [])
    for conv in conversations:
        if conv.get('from') == 'human':
            return conv.get('value')
    raise ValueError(f"No valid question (conversations entry with from==human): {img_full_path}")

def semantic_iou(value: str, target: str) -> float:
    intersection = len(set(value.split()) & set(target.split()))
    union = len(set(value.split()) | set(target.split()))
    return intersection / union if union > 0 else 0

class OWSMaskDataset(Dataset):
    def __init__(self, jsonl_path, image_root, force_image_size=448, dynamic_image_size=True, use_thumbnail=True, max_dynamic_patch=12, scale=None):
        self.samples = []
        self.image_root = image_root
        self.force_image_size = force_image_size
        self.transform = build_transform(is_train=False, input_size=force_image_size)
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_dynamic_patch = max_dynamic_patch
        self.scale = scale

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line.strip())
                image_path = data['image']
                folder_name = image_path.split('/')[0]
                self.samples.append(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        mask_id = data['id']
        image_path = data['image']
        img_full_path = os.path.join(self.image_root, image_path)
        image = Image.open(img_full_path).convert('RGB')
        height = data.get('height', image.height)
        width = data.get('width', image.width)
        mask_data = data['masks']
        gt_labels = data['label']
        
        if self.dynamic_image_size:
            images, resized_img = dynamic_preprocess_with_target_size(image, image_size=self.force_image_size,
                                     use_thumbnail=self.use_thumbnail,
                                     max_num=self.max_dynamic_patch)
        else:
            images = [image]
            resized_img = image
        
        pixel_values_list = []
        for img in images:
            pixel_values = self.transform(img)
            pixel_values_list.append(pixel_values)

        pixel_values = torch.stack(pixel_values_list)  # [num_patches, 3, H, W]
        
        # Process all masks; each mask refers to the full image
        n_prompts = len(mask_data)
        processed_masks_list = []
        gt_labels_list = []
        
        for i in range(n_prompts):
            rle = mask_data[i]
            gt_label = gt_labels[i]
            
            # Decode mask
            mask = decode_rle(rle, height, width)
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            mask_tensor = torch.nn.functional.interpolate(mask_tensor, size=(resized_img.size[1], resized_img.size[0]), mode='nearest')
            mask_tensor = mask_tensor.squeeze(0).squeeze(0)  # Drop batch/channel; shape [H, W]

            # Skip empty masks
            if mask_tensor.sum() == 0:
                print(f"Empty mask (all zeros): {image_path}")
                continue
            
            # Bounding box of mask
            ones_coords = torch.nonzero(mask_tensor)
            min_row = torch.min(ones_coords[:, 0])
            max_row = torch.max(ones_coords[:, 0])
            min_col = torch.min(ones_coords[:, 1])
            max_col = torch.max(ones_coords[:, 1])

            # Bbox center
            center_row = (min_row + max_row) / 2
            center_col = (min_col + max_col) / 2

            # Original width and height
            width = max_col - min_col + 1
            height = max_row - min_row + 1

            # Scale bbox size by `scale`
            new_width = min(max(width, height) * self.scale, resized_img.size[0])
            new_height = min(max(width, height) * self.scale, resized_img.size[1])

            # New min/max coords
            new_min_col = max(0, int(center_col - new_width / 2))
            new_max_col = min(resized_img.size[0] - 1, int(center_col + new_width / 2))
            new_min_row = max(0, int(center_row - new_height / 2))
            new_max_row = min(resized_img.size[1] - 1, int(center_row + new_height / 2))

            # If crop is smaller than requested, shift from the clamped edge
            if new_max_col - new_min_col + 1 < new_width:
                if new_min_col == 0:
                    new_max_col = min(resized_img.size[0] - 1, int(new_min_col + new_width))
                else:
                    new_min_col = max(0, int(new_max_col - new_width))

            if new_max_row - new_min_row + 1 < new_height:
                if new_min_row == 0:
                    new_max_row = min(resized_img.size[1] - 1, int(new_min_row + new_height))
                else:
                    new_min_row = max(0, int(new_max_row - new_height))

            # Crop from resized image
            cropped_img = resized_img.crop((new_min_col, new_min_row, new_max_col + 1, new_max_row + 1))
            
            # Crop mask to match
            cropped_mask = mask_tensor.unsqueeze(0)[:, new_min_row:new_max_row + 1, new_min_col:new_max_col + 1]
            resize_cropped_mask = F.interpolate(cropped_mask.unsqueeze(0), size=(16,16), mode='nearest').squeeze(0)
            
            # wow infer
            foreground_pixel_values = self.transform(cropped_img)
            
            processed_masks_list.append({
                'original_mask': mask_tensor,
                'cropped_mask': resize_cropped_mask,
                'foreground_pixel_values': foreground_pixel_values,
                'crop_coords': (new_min_col, new_min_row, new_max_col + 1, new_max_row + 1)
            })
            gt_labels_list.append(gt_label)
        
        return {
            'pixel_values': pixel_values,  # [num_patches, 3, H, W]
            'processed_masks': processed_masks_list,  # list of per-mask dicts
            'gt_labels': gt_labels_list,
            'image_path': image_path,
            'num_patches': len(images),
            'num_masks': len(processed_masks_list),
            'resized_img': np.array(resized_img),
            'mask_id': mask_id,
        }

def main(args):
    # ---------------------------------------- config ---------------------------------------
    args = parse_args(args)
    model_path = args.model_path
    bert_path = args.bert_path
    dataset_path = args.dataset_path
    image_root = args.image_root
    subset_num = args.subset_num
    subset_idx = args.subset_idx
    output_path = args.output_path.split('.json')[0] + '/subset_' + str(subset_idx) + '.json'
    force_image_size = args.force_image_size
    dynamic_image_size = args.dynamic_image_size
    use_thumbnail = args.use_thumbnail
    max_dynamic_patch = args.max_dynamic_patch
    prompt = args.prompt
    scale = args.scale

    # Guard subset_idx so it stays in range (must match launcher SUBSET_NUM / GPU jobs)
    if subset_idx >= subset_num:
        raise ValueError(
            f"subset_idx={subset_idx} must be < subset_num={subset_num}; "
            "check SUBSET_NUM vs. number of parallel GPU jobs in the shell script."
        )

    print(args)

    # ------------------------------------ prepare model ------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    bert_model = SentenceTransformer(bert_path)

    # ------------------------------------ prepare metric -----------------------------------
    per_mask_sim = 0
    per_mask_iou = 0
    mask_count = 0
    image_count = 0

    # ------------------------------------ prepare data -------------------------------------
    dataset = OWSMaskDataset(
        jsonl_path=dataset_path,
        image_root=image_root,
        force_image_size=force_image_size,
        dynamic_image_size=dynamic_image_size, 
        use_thumbnail=use_thumbnail,       
        max_dynamic_patch=max_dynamic_patch,
        scale=scale                
    )

    # ----------------------------- split into subsets for each GPU -------------------------
    def split_into_subsets(lst, subset_num):
        length = len(lst)
        base = length // subset_num
        remainder = length % subset_num
        result = []
        start = 0
        for i in range(subset_num):
            # First `remainder` chunks get one extra sample
            end = start + base + (1 if i < remainder else 0)
            result.append(range(start, end))
            start = end
        return result

    job_list = split_into_subsets(dataset, subset_num)[subset_idx]
    sub_dataset = Subset(dataset, list(job_list))
    dataloader = DataLoader(sub_dataset, batch_size=1, shuffle=False, num_workers=8)

    # ------------------------------------ start inference -----------------------------------
    results_log = []
    for idx, sample in enumerate(tqdm(dataloader, desc='Processing {}'.format(subset_idx))):
        pixel_values = sample['pixel_values'].to(torch.bfloat16).cuda().squeeze(0)

        processed_masks = sample['processed_masks']
        gt_labels = [item[0] for item in sample['gt_labels']]
        image_path = sample['image_path']
        num_patches = sample['num_patches']
        num_masks = sample['num_masks']
        mask_id = sample['mask_id']

        pred_labels = []
        per_image_sim = 0
        per_image_iou = 0
        
        for mask_idx in range(num_masks):
            # Current mask entry
            current_mask_info = processed_masks[mask_idx]
            original_mask = current_mask_info['original_mask']
            cropped_mask = current_mask_info['cropped_mask']
            foreground_pixel_values = current_mask_info['foreground_pixel_values'].cuda()
            crop_coords = current_mask_info['crop_coords']

            template = get_conv_template(model.config.template)
            template.system_message = getattr(model, "system_message", "")
            template.append_message(template.roles[0], prompt)
            template.append_message(template.roles[1], None)
            eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
            generation_config = dict(max_new_tokens=100, do_sample=False, eos_token_id=eos_token_id)

            fb_pixel_values = torch.cat([pixel_values, foreground_pixel_values], dim=0).to(torch.bfloat16).cuda()
            vaild_region_idx = len(fb_pixel_values) - 1
            
            with torch.inference_mode():
                response = model.chat(
                    tokenizer=tokenizer,
                    question=prompt,
                    generation_config=generation_config,
                    pixel_values=fb_pixel_values,
                    vaild_region_idx=torch.tensor(vaild_region_idx, dtype=torch.long),
                    pixel_masks=cropped_mask,
                )
            
            pred = response
            pred_labels.append(pred)
            
            per_image_sim += 0
            per_image_iou += 0
            per_mask_sim += 0
            per_mask_iou += 0
            mask_count += 1
        
        results_log.append({'id': mask_id[0], 'image': image_path, 'pred_labels': pred_labels, 'gt_labels': gt_labels, 'num_patches': num_patches.item(), 'num_masks': num_masks.item(), 'per_image_sim': (per_image_sim/num_masks).item(), 'per_image_iou': (per_image_iou/num_masks).item()})
        
        image_count += 1

    # Create parent dir of output_path (path may end with .json)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(results_log, fout, ensure_ascii=False)

if __name__ == '__main__':
    main(sys.argv[1:])
