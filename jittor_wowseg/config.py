"""Configuration reader for the released WOW-Seg checkpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WowSegConfig:
    image_size: int
    patch_size: int
    vision_hidden_size: int
    vision_intermediate_size: int
    vision_num_heads: int
    vision_num_layers: int
    vision_norm_eps: float
    llm_vocab_size: int
    llm_hidden_size: int
    llm_intermediate_size: int
    llm_num_heads: int
    llm_num_kv_heads: int
    llm_num_layers: int
    llm_rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    downsample_ratio: float
    template: str
    eos_token_id: int

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path) -> "WowSegConfig":
        path = Path(checkpoint) / "config.json"
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        vision = raw["vision_config"]
        llm = raw["llm_config"]
        return cls(
            image_size=int(raw.get("force_image_size") or vision["image_size"]),
            patch_size=int(vision["patch_size"]),
            vision_hidden_size=int(vision["hidden_size"]),
            vision_intermediate_size=int(vision["intermediate_size"]),
            vision_num_heads=int(vision["num_attention_heads"]),
            vision_num_layers=int(vision["num_hidden_layers"]),
            vision_norm_eps=float(vision.get("layer_norm_eps", 1e-6)),
            llm_vocab_size=int(llm["vocab_size"]),
            llm_hidden_size=int(llm["hidden_size"]),
            llm_intermediate_size=int(llm["intermediate_size"]),
            llm_num_heads=int(llm["num_attention_heads"]),
            llm_num_kv_heads=int(llm["num_key_value_heads"]),
            llm_num_layers=int(llm["num_hidden_layers"]),
            llm_rms_norm_eps=float(llm.get("rms_norm_eps", 1e-6)),
            rope_theta=float(llm.get("rope_theta", 1000000.0)),
            max_position_embeddings=int(llm.get("max_position_embeddings", 32768)),
            downsample_ratio=float(raw.get("downsample_ratio", 0.5)),
            template=str(raw.get("template", "internvl2_5")),
            eos_token_id=int(llm.get("eos_token_id", 151643)),
        )
