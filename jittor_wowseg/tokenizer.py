"""Tokenizer adapter backed by the checkpoint's tokenizer.json (no PyTorch)."""

from pathlib import Path

from tokenizers import Tokenizer


class WowTokenizer:
    def __init__(self, checkpoint_dir):
        self.tokenizer = Tokenizer.from_file(str(Path(checkpoint_dir) / "tokenizer.json"))
        self.image_context_token_id = self.tokenizer.token_to_id("<IMG_CONTEXT>")
        self.eos_token_id = self.tokenizer.token_to_id("<|im_end|>")
        self.comma_token_id = self.tokenizer.token_to_id(",")
        if self.image_context_token_id is None or self.eos_token_id is None:
            raise ValueError("Checkpoint tokenizer is missing WOW-Seg special tokens.")

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
