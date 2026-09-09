"""Inference-only Jittor implementation of the released WOW-Seg model."""

from __future__ import annotations

import gc
import math
from pathlib import Path

import jittor as jt
import numpy as np
from jittor import nn

from .config import WowSegConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        self.weight = jt.ones((hidden_size,))
        self.eps = eps

    def execute(self, x):
        variance = (x * x).mean(dim=-1, keepdims=True)
        return x * jt.rsqrt(variance + self.eps) * self.weight


class VisionEmbeddings(nn.Module):
    def __init__(self, config):
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.class_embedding = jt.zeros((1, 1, config.vision_hidden_size))
        self.patch_embedding = nn.Conv2d(3, config.vision_hidden_size, self.patch_size, stride=self.patch_size)
        patch_count = (self.image_size // self.patch_size) ** 2
        self.position_embedding = jt.zeros((1, patch_count + 1, config.vision_hidden_size))

    def execute(self, pixel_values):
        x = self.patch_embedding(pixel_values)
        batch, _, height, width = x.shape
        x = x.reshape(batch, x.shape[1], height * width).permute(0, 2, 1)
        cls = jt.broadcast(self.class_embedding, (batch, 1, self.class_embedding.shape[-1]))
        # WOW-Seg fixes input tiles at 448px, so interpolation is unnecessary here.
        return jt.concat((cls, x), dim=1) + self.position_embedding


class VisionAttention(nn.Module):
    def __init__(self, config):
        self.heads = config.vision_num_heads
        self.head_dim = config.vision_hidden_size // self.heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(config.vision_hidden_size, 3 * config.vision_hidden_size)
        self.proj = nn.Linear(config.vision_hidden_size, config.vision_hidden_size)

    def execute(self, x):
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attention = nn.softmax(jt.matmul(q * self.scale, k.permute(0, 1, 3, 2)), dim=-1).cast(v.dtype)
        x = jt.matmul(attention, v).permute(0, 2, 1, 3).reshape(batch, length, channels)
        return self.proj(x)


class VisionMLP(nn.Module):
    def __init__(self, config):
        self.fc1 = nn.Linear(config.vision_hidden_size, config.vision_intermediate_size)
        self.fc2 = nn.Linear(config.vision_intermediate_size, config.vision_hidden_size)

    def execute(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class VisionLayer(nn.Module):
    def __init__(self, config):
        self.norm1 = nn.LayerNorm(config.vision_hidden_size, eps=config.vision_norm_eps)
        self.attn = VisionAttention(config)
        self.norm2 = nn.LayerNorm(config.vision_hidden_size, eps=config.vision_norm_eps)
        self.mlp = VisionMLP(config)
        self.ls1 = jt.ones((config.vision_hidden_size,)) * 0.1
        self.ls2 = jt.ones((config.vision_hidden_size,)) * 0.1

    def execute(self, x):
        x = x + self.attn(self.norm1(x)) * self.ls1
        return x + self.mlp(self.norm2(x)) * self.ls2


class VisionEncoder(nn.Module):
    def __init__(self, config):
        self.layers = nn.ModuleList([VisionLayer(config) for _ in range(config.vision_num_layers)])

    def execute(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class InternVisionModel(nn.Module):
    def __init__(self, config):
        self.embeddings = VisionEmbeddings(config)
        self.encoder = VisionEncoder(config)

    def execute(self, pixel_values):
        return self.encoder(self.embeddings(pixel_values))


class QwenMLP(nn.Module):
    def __init__(self, config):
        self.gate_proj = nn.Linear(config.llm_hidden_size, config.llm_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.llm_hidden_size, config.llm_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.llm_intermediate_size, config.llm_hidden_size, bias=False)

    def execute(self, x):
        gated = self.gate_proj(x)
        return self.down_proj((gated * jt.sigmoid(gated)) * self.up_proj(x))


class QwenAttention(nn.Module):
    def __init__(self, config):
        self.heads = config.llm_num_heads
        self.kv_heads = config.llm_num_kv_heads
        self.head_dim = config.llm_hidden_size // self.heads
        self.groups = self.heads // self.kv_heads
        self.scale = self.head_dim ** -0.5
        self.rope_theta = config.rope_theta
        self.q_proj = nn.Linear(config.llm_hidden_size, config.llm_hidden_size, bias=True)
        self.k_proj = nn.Linear(config.llm_hidden_size, self.kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.llm_hidden_size, self.kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.llm_hidden_size, config.llm_hidden_size, bias=False)

    def _rotary(self, states, positions):
        inv_freq = 1.0 / (self.rope_theta ** (np.arange(0, self.head_dim, 2, dtype=np.float32) / self.head_dim))
        freqs = np.outer(np.asarray(positions, dtype=np.float32), inv_freq)
        angles = np.concatenate((freqs, freqs), axis=-1)
        cos = jt.array(np.cos(angles)).reshape(1, len(positions), 1, self.head_dim)
        sin = jt.array(np.sin(angles)).reshape(1, len(positions), 1, self.head_dim)
        first, second = states[:, :, :, : self.head_dim // 2], states[:, :, :, self.head_dim // 2 :]
        rotated = jt.concat((-second, first), dim=-1)
        return states * cos + rotated * sin

    def execute(self, x, cache=None, position_offset=0):
        batch, length, channels = x.shape
        q = self.q_proj(x).reshape(batch, length, self.heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, length, self.kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.kv_heads, self.head_dim)
        positions = np.arange(position_offset, position_offset + length, dtype=np.int32)
        q, k = self._rotary(q, positions), self._rotary(k, positions)
        if cache is not None:
            old_k, old_v = cache
            k, v = jt.concat((old_k, k), dim=1), jt.concat((old_v, v), dim=1)
        new_cache = (k, v)
        # Qwen GQA repeats each KV head contiguously (k0,k0,...,k1,k1,...),
        # rather than appending a second copy of the complete KV tensor.
        keys = jt.concat(
            [k[:, :, head : head + 1, :] for head in range(self.kv_heads) for _ in range(self.groups)], dim=2
        ).permute(0, 2, 1, 3)
        values = jt.concat(
            [v[:, :, head : head + 1, :] for head in range(self.kv_heads) for _ in range(self.groups)], dim=2
        ).permute(0, 2, 1, 3)
        queries = q.permute(0, 2, 1, 3)
        scores = jt.matmul(queries * self.scale, keys.permute(0, 1, 3, 2))
        if length > 1:
            key_length = int(keys.shape[2])
            mask = np.full((length, key_length), -1e30, dtype=np.float32)
            for row in range(length):
                mask[row, : key_length - length + row + 1] = 0.0
            scores = scores + jt.array(mask).cast(scores.dtype).reshape(1, 1, length, key_length)
        attention = nn.softmax(scores, dim=-1).cast(values.dtype)
        x = jt.matmul(attention, values).permute(0, 2, 1, 3).reshape(batch, length, channels)
        return self.o_proj(x), new_cache


class QwenDecoderLayer(nn.Module):
    def __init__(self, config):
        self.input_layernorm = RMSNorm(config.llm_hidden_size, config.llm_rms_norm_eps)
        self.self_attn = QwenAttention(config)
        self.post_attention_layernorm = RMSNorm(config.llm_hidden_size, config.llm_rms_norm_eps)
        self.mlp = QwenMLP(config)

    def execute(self, x, cache=None, position_offset=0):
        residual = x
        x, cache = self.self_attn(self.input_layernorm(x), cache, position_offset)
        x = residual + x
        return x + self.mlp(self.post_attention_layernorm(x)), cache


class QwenModel(nn.Module):
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(config.llm_vocab_size, config.llm_hidden_size)
        self.layers = nn.ModuleList([QwenDecoderLayer(config) for _ in range(config.llm_num_layers)])
        self.norm = RMSNorm(config.llm_hidden_size, config.llm_rms_norm_eps)

    def execute(self, inputs_embeds, cache=None, position_offset=0):
        next_cache = []
        for index, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache[index]
            inputs_embeds, layer_cache = layer(inputs_embeds, layer_cache, position_offset)
            next_cache.append(layer_cache)
        return self.norm(inputs_embeds), next_cache


class QwenForCausalLM(nn.Module):
    def __init__(self, config):
        self.model = QwenModel(config)
        self.lm_head = nn.Linear(config.llm_hidden_size, config.llm_vocab_size, bias=False)

    def execute(self, inputs_embeds, cache=None, position_offset=0):
        x, cache = self.model(inputs_embeds, cache, position_offset)
        return self.lm_head(x), cache


class WowSegJittor(nn.Module):
    """WOW-Seg VLM with greedy, cache-backed Jittor decoding."""
    def __init__(self, config: WowSegConfig):
        self.config = config
        self.vision_model = InternVisionModel(config)
        self.language_model = QwenForCausalLM(config)
        projector_width = config.vision_hidden_size * 4
        # Sequential preserves the released checkpoint keys: mlp1.0 through mlp1.3.
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(projector_width, eps=config.vision_norm_eps),
            nn.Linear(projector_width, config.llm_hidden_size),
            nn.GELU(),
            nn.Linear(config.llm_hidden_size, config.llm_hidden_size),
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path, converted_checkpoint: str | Path):
        model = cls(WowSegConfig.from_checkpoint(checkpoint_dir))
        model.load_converted_checkpoint(converted_checkpoint)
        model.eval()
        return model

    def load_converted_checkpoint(self, checkpoint: str | Path):
        payload = jt.load(str(checkpoint))
        if not isinstance(payload, dict) or payload.get("format") != "wowseg-jittor-v1":
            raise ValueError(f"Not a converted WOW-Seg checkpoint: {checkpoint}")
        parameters = payload["state_dict"]
        expected = self.state_dict()
        missing = sorted(set(expected) - set(parameters))
        unexpected = sorted(set(parameters) - set(expected))
        mismatched = sorted(key for key in set(expected) & set(parameters) if tuple(expected[key].shape) != tuple(parameters[key].shape))
        if missing or unexpected or mismatched:
            raise RuntimeError(
                "Invalid converted checkpoint: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}, shape_mismatch={mismatched[:5]}"
            )
        self.load_parameters(parameters)
        del payload, parameters, expected
        gc.collect()

    def extract_feature(self, pixel_values):
        x = self.vision_model(pixel_values)[:, 1:, :]
        side = int(math.sqrt(x.shape[1]))
        x = x.reshape(x.shape[0], side, side, x.shape[-1])
        # Pixel shuffle v2, identical to InternVL's reference implementation.
        x = x.reshape(x.shape[0], side, side // 2, x.shape[-1] * 2).permute(0, 2, 1, 3)
        # The first reshape has already doubled the channel width; double it
        # once more here (rather than multiplying the intermediate width by 4).
        x = x.reshape(x.shape[0], side // 2, side // 2, x.shape[-1] * 2).permute(0, 2, 1, 3)
        return self.mlp1(x.reshape(x.shape[0], -1, x.shape[-1]))

    def generate(
        self,
        input_ids,
        visual_features,
        image_context_token_id,
        eos_token_id,
        max_new_tokens=24,
        stop_token_ids=None,
    ):
        ids = np.asarray(input_ids, dtype=np.int32).reshape(1, -1)
        embeds = self.language_model.model.embed_tokens(jt.array(ids))
        positions = np.flatnonzero(ids[0] == int(image_context_token_id))
        if len(positions) != int(visual_features.shape[0] * visual_features.shape[1]):
            raise ValueError(f"Image-token count {len(positions)} does not match visual features {visual_features.shape}")
        replacement = visual_features.reshape(-1, visual_features.shape[-1])
        for index, position in enumerate(positions.tolist()):
            embeds[0, int(position), :] = replacement[index]
        logits, cache = self.language_model(embeds)
        generated = []
        next_id = int(logits[:, -1, :].argmax(dim=-1)[0].numpy()[0])
        for _ in range(max_new_tokens):
            if next_id == int(eos_token_id) or next_id in (stop_token_ids or set()):
                break
            generated.append(next_id)
            token_embed = self.language_model.model.embed_tokens(jt.array([[next_id]], dtype="int32"))
            logits, cache = self.language_model(token_embed, cache=cache, position_offset=ids.shape[1] + len(generated) - 1)
            next_id = int(logits[:, -1, :].argmax(dim=-1)[0].numpy()[0])
        return generated

    def generate_masked_region(
        self,
        input_ids,
        pixel_values,
        pixel_mask_16x16,
        valid_region_idx,
        image_context_token_id,
        eos_token_id,
        max_new_tokens=24,
        stop_token_ids=None,
    ):
        """Encode image tiles and retain only the target-region visual tokens.

        This mirrors the released `modeling_internvl_chat.py`: all tokens from
        the dynamic full-image tiles are retained, while the final crop tile is
        filtered by its 16x16 target mask.
        """
        visual = self.extract_feature(pixel_values)
        token_mask = np.ones((int(visual.shape[0]), int(visual.shape[1])), dtype=np.bool_)
        token_mask[int(valid_region_idx)] = np.asarray(pixel_mask_16x16, dtype=np.bool_).reshape(-1)
        selected_indices = np.flatnonzero(token_mask.reshape(-1)).astype(np.int32)
        selected = visual.reshape(-1, visual.shape[-1])[jt.array(selected_indices)]
        return self.generate(
            input_ids,
            selected.reshape(1, selected.shape[0], selected.shape[1]),
            image_context_token_id,
            eos_token_id,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
        )
