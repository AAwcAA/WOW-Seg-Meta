import torch

def parse_input_sequence(input_ids_list: list[int], config: dict) -> dict:
    """
    Parses the input token ID sequence to categorize different segments.
    
    Identifies and separates the prefix prompt, original images, foreground 
    regions, infix prompts, and output sequences based on provided token configurations.
    
    Args:
        input_ids_list (list[int]): The list of input token IDs.
        config (dict): Dictionary containing token ID configurations 
                       (img_start, img_end, prompt_delimiter, output_separator, pad_token).
                       
    Returns:
        dict: A dictionary containing lists of indices for each segment.
    """
    segments = {
        "prefix_prompt": [],
        "original_image": [],
        "foreground_regions": [],
        "infix_prompt": [], 
        "outputs": []
    }
    
    img_start_id = config['img_start']
    img_end_id = config['img_end']
    prompt_delimiter_id = config['prompt_delimiter']
    output_separator_id = config['output_separator']
    pad_token_id = config.get('pad_token')
    
    def find_first_occurrence(lst: list, value: int) -> int:
        try: 
            return lst.index(value)
        except ValueError: 
            return -1
        
    def find_last_occurrence(lst: list, value: int) -> int:
        try: 
            return len(lst) - 1 - lst[::-1].index(value)
        except ValueError: 
            return -1

    def find_second_last_occurrence_slice(lst: list, value: int) -> int:
        last_index = find_last_occurrence(lst, value)
        if last_index == -1:
            return -1
        return find_last_occurrence(lst[:last_index], value)

    # 1. Identify Prefix Prompt
    first_img_start_idx = find_first_occurrence(input_ids_list, img_start_id)
    if first_img_start_idx == -1: 
        # If no image is found, the entire sequence is treated as a prompt
        segments["prefix_prompt"] = list(range(len(input_ids_list)))
        return segments
    
    if first_img_start_idx > 0:
        segments["prefix_prompt"] = list(range(first_img_start_idx))

    # 2. Identify Image Segments (starting from the first image)
    is_first_image = True
    processed_indices = set(segments["prefix_prompt"])
    
    for i in range(first_img_start_idx, len(input_ids_list)):
        if i in processed_indices or input_ids_list[i] == pad_token_id: 
            continue
            
        if input_ids_list[i] == img_start_id:
            start_index = i
            try:
                end_index = input_ids_list.index(img_end_id, start_index)
                indices = list(range(start_index, end_index + 1))
                
                if is_first_image: 
                    segments["original_image"] = indices
                    is_first_image = False
                else: 
                    segments["foreground_regions"].append(indices)
                    
                processed_indices.update(indices)
            except ValueError: 
                pass

    # 3. Identify Infix Prompt and Outputs
    try:
        last_img_end_idx = find_last_occurrence(input_ids_list, img_end_id)
        if last_img_end_idx == -1: 
            return segments

        prompt_delimiter_idx = find_second_last_occurrence_slice(input_ids_list, prompt_delimiter_id)
        
        segments["infix_prompt"] = list(range(last_img_end_idx + 1, prompt_delimiter_idx + 1))
        
        output_start_idx = prompt_delimiter_idx + 1
        separator_indices = [
            i for i, token in enumerate(input_ids_list) 
            if i >= output_start_idx and token == output_separator_id
        ]
        
        current_chunk_start = output_start_idx
        for sep_idx in separator_indices:
            chunk = list(range(current_chunk_start, sep_idx + 1))
            if chunk: 
                segments["outputs"].append(chunk)
            current_chunk_start = sep_idx + 1
        
        last_chunk = [
            i for i in range(current_chunk_start, len(input_ids_list)) 
            if input_ids_list[i] != pad_token_id
        ]
        if last_chunk: 
            segments["outputs"].append(last_chunk)
            
    except (ValueError, IndexError): 
        pass
        
    return segments

def _create_binary_mask(
    input_ids: torch.Tensor,
    config: dict,
    apply_region_masking: bool = True,
    apply_output_masking: bool = True
) -> torch.Tensor:
    """
    Internal function to generate a 0/1 binary attention mask.
    """
    assert input_ids.dim() == 1, "Input must be a 1D tensor"
    seq_len = input_ids.size(0)
    device = input_ids.device
    
    # Base causal mask (lower triangular)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.long))
    
    segments = parse_input_sequence(input_ids.tolist(), config)
    
    regions_indices = segments["foreground_regions"]
    output_chunks = segments["outputs"]
    
    # Apply masking rules for foreground regions
    if apply_region_masking:
        for i in range(len(regions_indices)):
            for query_idx in regions_indices[i]:
                for j in range(len(regions_indices)):
                    if i == j: 
                        continue
                    for key_idx in regions_indices[j]: 
                        mask[query_idx, key_idx] = 0
    
    # Apply masking rules for output chunks
    if apply_output_masking:
        if output_chunks and len(output_chunks) != len(regions_indices):
            print(f"Warning: Number of output chunks ({len(output_chunks)}) doesn't match foreground regions ({len(regions_indices)}).")
            
        for i in range(len(output_chunks)):
            current_chunk = output_chunks[i]
            for query_idx in current_chunk:
                # Output chunk shouldn't attend to unaligned regions
                if i < len(regions_indices):
                    for j in range(len(regions_indices)):
                        if i == j: 
                            continue
                        for key_idx in regions_indices[j]: 
                            mask[query_idx, key_idx] = 0
                
                # Output chunk shouldn't attend to previous output chunks
                for j in range(i):
                    for key_idx in output_chunks[j]: 
                        mask[query_idx, key_idx] = 0

    # Apply padding mask
    if 'pad_token' in config:
        padding_indices = (input_ids == config['pad_token']).nonzero(as_tuple=True)[0]
        mask[:, padding_indices] = 0
        
    return mask

def create_cascade_attention_mask(input_ids: torch.Tensor, config: dict) -> torch.Tensor:
    """
    Creates a production-ready additive attention mask (0.0 / -inf format) 
    for Transformer architectures.
    
    Args:
        input_ids (torch.Tensor): 1D tensor of input token IDs.
        config (dict): Dictionary mapping special structural tokens.
        
    Returns:
        torch.Tensor: Float tensor of shape (seq_len, seq_len) mapped to 0.0 and -inf.
    """
    binary_mask = _create_binary_mask(input_ids, config)
    additive_mask = torch.full(
        binary_mask.shape, 
        -torch.inf, 
        dtype=torch.float32, 
        device=binary_mask.device
    )
    additive_mask.masked_fill_(binary_mask.bool(), 0.0)
    return additive_mask


if __name__ == "__main__":
    # --- Minimal Reproducible Example ---
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    token_config = {
        'img_start': 151665,
        'img_end': 151666,
        'prompt_delimiter': 25,  # ':'
        'output_separator': 11,  # ','
        'pad_token': 151643
    }
    
    # Sample token sequence
    sample_input_ids = torch.tensor([
        151644, 8948, 198, 105043, 90286, 21287, 13935, 116669, 3837,
        151665, 151667, 151667, 151666, 198, 11, 8482, 678,
        151665, 151667, 151667, 151666, 198, 13, 1446, 1184, 
        25, 151643, 151643
    ], dtype=torch.long, device=device)

    # Generate the additive mask
    attention_mask = create_cascade_attention_mask(sample_input_ids, token_config)
    
    def visualize_additive_mask(mask: torch.Tensor):
        """Helper to print a small mask to the console for verification."""
        mask_np = mask.cpu().numpy()
        print("      " + " ".join([f"{i:<5}" for i in range(mask_np.shape[1])]))
        print("    " + "-" * (mask_np.shape[1] * 6))
        for i, row in enumerate(mask_np):
            print(f"{i:<2} |", end=" ")
            for val in row:
                print(f"{val:<5.1f}", end=" ")
            print()

    print("Generated Cascade Attention Mask (0.0 / -inf format):")
    visualize_additive_mask(attention_mask)
    
    print(f"\nMask Type: {type(attention_mask)}")
    print(f"Mask Dtype: {attention_mask.dtype}")
    print(f"Mask Device: {attention_mask.device}")