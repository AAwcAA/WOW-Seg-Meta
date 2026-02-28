import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
# : 25 <img> 151665 </img> 151666 <IMG_CONTEXT> 151667

# input_text = "<img><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT></img><img><IMG_CONTEXT></img><img><IMG_CONTEXT></img><img><IMG_CONTEXT></img>In the front are the original image and the masks that need to be classified in sequence. Please classify each mask in sequence. The category of each mask is: people, car"

# tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/wth/ALL_MODEL/lvis_paco_all_mask_1epoch_focus_mask_fore_token", trust_remote_code=True, use_fast=False)

# input_ids = tokenizer(input_text, return_tensors="pt").input_ids[0]

import torch
import numpy as np

def parse_input_sequence_v10(input_ids_list: list[int], config: dict):
    """
    解析输入的token ID序列（版本10）。
    - 新增功能：识别并分离出位于序列开头的前缀文本（Prefix Prompt）。
    """
    segments = {
        "prefix_prompt": [],
        "original_image": [],
        "fgits": [],
        "infix_prompt": [], # "prompt" key is renamed for clarity
        "outputs": []
    }
    
    img_start_id = config['img_start']
    img_end_id = config['img_end']
    prompt_delimiter_id = config['prompt_delimiter']
    output_separator_id = config['output_separator']
    pad_token_id = config.get('pad_token')
    
    def find_first_occurrence(lst, value):
        try: return lst.index(value)
        except ValueError: return -1
        
    def find_last_occurrence(lst, value):
        try: return len(lst) - 1 - lst[::-1].index(value)
        except ValueError: return -1

    def find_second_last_occurrence_slice(lst: list, value) -> int:
        """
        通过切片和复用 find_last_occurrence 函数来查找倒数第二次出现的索引。

        Args:
            lst: 要搜索的列表。
            value: 要查找的值。

        Returns:
            如果值出现至少两次，则返回其倒数第二次出现的索引；否则返回 -1。
        """
        # 1. 找到最后一次出现的位置
        last_index = find_last_occurrence(lst, value)

        # 2. 如果连一次都找不到，直接返回-1
        if last_index == -1:
            return -1

        # 3. 在最后一个位置之前的部分，再次寻找最后一次出现的位置
        # 这就是原列表中的倒数第二次出现的位置
        return find_last_occurrence(lst[:last_index], value)

    # 1. 识别前缀文本
    first_img_start_idx = find_first_occurrence(input_ids_list, img_start_id)
    if first_img_start_idx == -1: # 如果没有图像，则整个序列都是prompt
        segments["prefix_prompt"] = list(range(len(input_ids_list)))
        return segments
    
    if first_img_start_idx > 0:
        segments["prefix_prompt"] = list(range(first_img_start_idx))

    # 2. 识别图像部分 (从第一个图像开始)
    is_first_image = True
    processed_indices = set(segments["prefix_prompt"])
    for i in range(first_img_start_idx, len(input_ids_list)):
        if i in processed_indices or input_ids_list[i] == pad_token_id: continue
        if input_ids_list[i] == img_start_id:
            start_index = i
            try:
                end_index = input_ids_list.index(img_end_id, start_index)
                indices = list(range(start_index, end_index + 1))
                if is_first_image: segments["original_image"] = indices; is_first_image = False
                else: segments["fgits"].append(indices)
                processed_indices.update(indices)
            except ValueError: pass

    # 3. 识别中缀文本和输出
    try:
        last_img_end_idx = find_last_occurrence(input_ids_list, img_end_id)
        if last_img_end_idx == -1: return segments

        last__n_idx = find_second_last_occurrence_slice(input_ids_list, prompt_delimiter_id)
        # 20250903 modify
        # prompt_delimiter_idx = input_ids_list.index(prompt_delimiter_id, last_img_end_idx)
        prompt_delimiter_idx = last__n_idx
        
        segments["infix_prompt"] = list(range(last_img_end_idx + 1, prompt_delimiter_idx + 1))
        
        output_start_idx = prompt_delimiter_idx + 1
        separator_indices = [i for i, token in enumerate(input_ids_list) if i >= output_start_idx and token == output_separator_id]
        
        current_chunk_start = output_start_idx
        for sep_idx in separator_indices:
            chunk = list(range(current_chunk_start, sep_idx + 1))
            if chunk: segments["outputs"].append(chunk)
            current_chunk_start = sep_idx + 1
        
        last_chunk = [i for i in range(current_chunk_start, len(input_ids_list)) if input_ids_list[i] != pad_token_id]
        if last_chunk: segments["outputs"].append(last_chunk)
    except (ValueError, IndexError): pass
    return segments

# 内部函数，生成0/1掩码
def _create_binary_mask_torch_v10(
    input_ids: torch.Tensor,
    config: dict,
    apply_fgit_masking: bool = True,
    apply_output_masking: bool = True) -> torch.Tensor:
    assert input_ids.dim() == 1, "Input must be a 1D tensor"
    seq_len = input_ids.size(0)
    device = input_ids.device
    # 基础因果掩码，前缀文本将自动遵循此规则
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.long))
    
    segments = parse_input_sequence_v10(input_ids.tolist(), config)
    
    fgits_indices = segments["fgits"]
    output_chunks = segments["outputs"]
    
    # --- 规则应用部分 (与V9完全相同) ---
    if apply_fgit_masking:
        for i in range(len(fgits_indices)):
            for query_idx in fgits_indices[i]:
                for j in range(len(fgits_indices)):
                    if i == j: continue
                    for key_idx in fgits_indices[j]: mask[query_idx, key_idx] = 0
    
    if apply_output_masking:
        if output_chunks and len(output_chunks) != len(fgits_indices):
            print(f"Warning: Number of output chunks ({len(output_chunks)}) doesn't match FGITs ({len(fgits_indices)}).")
            
        for i in range(len(output_chunks)):
            current_chunk = output_chunks[i]
            for query_idx in current_chunk:
                if i < len(fgits_indices):
                    for j in range(len(fgits_indices)):
                        if i == j: continue
                        for key_idx in fgits_indices[j]: mask[query_idx, key_idx] = 0
                for j in range(i):
                    for key_idx in output_chunks[j]: mask[query_idx, key_idx] = 0

    # --- Padding部分 (与V9完全相同) ---
    padding_indices = (input_ids == config['pad_token']).nonzero(as_tuple=True)[0]
    mask[:, padding_indices] = 0
    return mask

def create_custom_attention_mask_v10(input_ids: torch.Tensor, config: dict) -> torch.Tensor:
    """
    最终版本 (V10):
    - 支持位于序列开头的前缀文本。
    - 保持所有现有自定义规则和主流的padding处理。
    - 输出生产可用的加性掩码 (0.0 / -inf 格式)。
    """
    binary_mask = _create_binary_mask_torch_v10(input_ids, config)
    additive_mask = torch.full(binary_mask.shape, -torch.inf, dtype=torch.float32, device=binary_mask.device)
    additive_mask.masked_fill_(binary_mask.bool(), 0.0)
    return additive_mask

if __name__ == "__main__":

    # # V8配置与V7完全相同
    # token_config_v8 = {
    #     'img_start': 1,
    #     'img_end': 3,
    #     'prompt_delimiter': 7,  # ':'
    #     'output_separator': 8,  # ','
    #     'pad_token': 0          # Padding token ID
    # }
    # # 示例输入：<org> <fgit1> <fgit2> : cat , dog
    # # 对应的ID序列
    # v3_input_ids = [
    #     151665,2,2,2,2,2,151666,
    #     151665,2,2,151666,
    #     151665,2,151666,
    #     4,4,
    #     25,
    #     20,21,
    #     11,
    #     30,31,32,
    #     11,
    #     33,34,
    #     0,0,0,0
    # ]

    # # 生成掩码
    # custom_mask_v3 = create_custom_attention_mask_torch(v3_input_ids, token_config_v8)

    # print(custom_mask_v3)

    # # 可视化函数 (与V2相同)
    # def visualize_mask_v3(mask: np.ndarray, input_ids: list[int]):
    #     """可视化函数，🟥=不可见, 🟩=可见"""
    #     print(" " * 6 + " ".join([f"{i:<2}" for i in range(len(input_ids))]))
    #     print(" " * 6 + " ".join([f"{token:<2}" for token in input_ids]))
    #     print(" " * 5 + "-" * (len(input_ids) * 3))
    #     for i, row in enumerate(mask):
    #         print(f"{i:<2} | {input_ids[i]:<2}", end=" ")
    #         for val in row:
    #             print("🟩" if val == 1 else "🟥", end=" ")
    #         print()

    # print("生成的自定义注意力掩码 V3 (🟥: 不可见, 🟩: 可见):")
    # visualize_mask_v3(custom_mask_v3, v3_input_ids)

    # # 验证解析结果
    # print("\n使用V3解析函数解析出的Token结构:")
    # parsed_segments_v3 = parse_input_sequence_for_torch(v3_input_ids, token_config_v8)
    # print(parsed_segments_v3)

    # 检查是否有可用的GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前使用的设备是: {device}")
    # V10配置与之前版本相同
    token_config_v10 = {
            'img_start': 151665,
            'img_end': 151666,
            'prompt_delimiter': 25,  # ':'
            'output_separator': 11,  # ','
            'pad_token': 151643
        }
    # 将输入序列创建为PyTorch张量
    # v9_input_ids = torch.tensor(
    #     [8,8,8,
    #     151665,2,2,2,2,2,151666,
    #     151665,2,2,151666,
    #     151665,2,151666,
    #     4,4,
    #     25,
    #     20,21,
    #     11,
    #     30,31,32,
    #     11,
    #     151643,151643
    # ],
    #     dtype=torch.long,
    #     device=device
    # )
    v9_input_ids = torch.tensor([151644,   8948,    198, 105043,  90286,  21287,  13935, 116669,   3837,
        105205,  13072,  20412,  67916,  30698,   3837, 104625, 100633, 104455,
        104800,   5373, 109065,  81217, 104581,  99721,  75317, 101101, 100013,
          9370,  42140,  53772,  35243,  26288, 102064, 104949,   1773, 151645,
           198, 151644,    872,    198,   7039,   1052,    525,   1378,   5335,
            13,    576,   1156,   2168,    374,    279,   4024,   2168, 151665,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151666,    198,     11,   8482,    678,
           279,   1995,     13,    576,   2086,   2168,    374,    264,    949,
           315,    419,   2168, 151665, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667, 151667,
        151667, 151667, 151667, 151667, 151667, 151667, 151667, 151666,    198,
            13,   1446,   1184,    311,   6782,  26887,    678,    279,   1995,
           323,   8253,   1128,   5582,    279,   2086,   2168,  17180,    311,
            13, 151645,    198, 151644,  77091,    198,  14877,  59703, 151645,
           198],
        dtype=torch.long,
        device=device
    )

    # 生成V9的加性掩码张量
    additive_mask_v9_tensor = create_custom_attention_mask_v10(v9_input_ids, token_config_v10)
    from PIL import Image
    Image.fromarray(np.array(additive_mask_v9_tensor.cpu()==0, dtype=np.uint8)*255).save("/home/ubuntu/InternVL/internvl_chat/cascade_attention_mask_vis/additive_mask_v9_tensor.png")

    # 新的可视化函数，用于打印数值
    def visualize_additive_mask(mask: torch.Tensor):
        mask_np = mask.cpu().numpy()
        print("      " + " ".join([f"{i:<5}" for i in range(mask_np.shape[1])]))
        print("    " + "-" * (mask_np.shape[1] * 6))
        for i, row in enumerate(mask_np):
            print(f"{i:<2} |", end=" ")
            for val in row:
                # 用 '0.0' 和 '-inf' 来表示，更清晰
                print(f"{val:<5.1f}", end=" ")
            print()

    print("\n生成的 PyTorch 加性注意力掩码 V9 (0.0 / -inf 格式):")
    visualize_additive_mask(additive_mask_v9_tensor)

    print(f"\n生成的掩码类型: {type(additive_mask_v9_tensor)}")
    print(f"掩码的数据类型: {additive_mask_v9_tensor.dtype}")
    print(f"掩码所在的设备: {additive_mask_v9_tensor.device}")