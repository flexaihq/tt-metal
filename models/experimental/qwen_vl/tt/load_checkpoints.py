# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file as safetensors_load_file
from tqdm import tqdm


# TODO Update function for large models: For 1 layer tests we only want to load 1 checkpoint file, instead of all.
def load_hf_state_dict(ckpt_dir):
    # First check if index file exists
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        # Multi-file case: Read the index file and load all referenced safetensor files
        with open(index_path, "r") as f:
            index_data = json.load(f)

        # Retrieve the weight file names from the index JSON
        weight_map = index_data["weight_map"]
        safetensor_files = set(weight_map.values())

        # Read each safetensors file mentioned in the index
        loaded_weights = {}
        for file in safetensor_files:
            safetensor_path = os.path.join(ckpt_dir, file)
            weights = safetensors_load_file(safetensor_path)
            loaded_weights.update(weights)  # Merge weights into a single dictionary
    else:
        # Single-file case: Load the single model.safetensors file
        safetensor_path = os.path.join(ckpt_dir, "model.safetensors")
        if not os.path.exists(safetensor_path):
            raise FileNotFoundError(f"Neither model.safetensors.index.json nor model.safetensors found in {ckpt_dir}")
        loaded_weights = safetensors_load_file(safetensor_path)

    return loaded_weights


def standardize_hf_keys(state_dict):
    if not "lm_head.weight" in state_dict:
        # Assume tied to the embeddings if not present
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
    return state_dict


def convert_hf_to_meta(state_dict, head_dim):
    state_dict = split_hf_keys(state_dict)
    state_dict = convert_hf_qkv_to_meta_format(state_dict, head_dim)
    state_dict = map_hf_to_meta_keys(state_dict)
    return state_dict


def convert_vision_hf_to_meta(state_dict, head_dim):
    # state_dict = split_hf_keys(state_dict)
    # state_dict = convert_hf_qkv_to_meta_format(state_dict, head_dim)
    state_dict = map_vision_hf_to_meta_keys(state_dict)
    return state_dict


def map_hf_to_meta_keys(loaded_weights):
    hf_to_meta = {
        # Top level mappings
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
        # Layer level mappings
        "input_layernorm.weight": "attention_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        # Attention module mappings
        "self_attn.q_proj.weight": "attention.wq.weight",
        "self_attn.k_proj.weight": "attention.wk.weight",
        "self_attn.v_proj.weight": "attention.wv.weight",
        "self_attn.o_proj.weight": "attention.wo.weight",
        "self_attn.q_proj.bias": "attention.wq.bias",
        "self_attn.k_proj.bias": "attention.wk.bias",
        "self_attn.v_proj.bias": "attention.wv.bias",
        "self_attn.q_norm.weight": "attention.q_norm.weight",
        "self_attn.k_norm.weight": "attention.k_norm.weight",
        # Feed forward module mappings
        "mlp.gate_proj.weight": "feed_forward.w1.weight",
        "mlp.up_proj.weight": "feed_forward.w3.weight",
        "mlp.down_proj.weight": "feed_forward.w2.weight",
        # Direct module mappings
        "gate_proj.weight": "w1.weight",
        "down_proj.weight": "w2.weight",
        "up_proj.weight": "w3.weight",
        "q_proj.weight": "wq.weight",
        "k_proj.weight": "wk.weight",
        "v_proj.weight": "wv.weight",
        "o_proj.weight": "wo.weight",
        "q_proj.bias": "wq.bias",
        "k_proj.bias": "wk.bias",
        "v_proj.bias": "wv.bias",
        "q_norm.weight": "q_norm.weight",
        "k_norm.weight": "k_norm.weight",
        "weight": "emb.weight",  # For host embeddings
        # Full path layer mappings
        "model.layers.{layer}.input_layernorm.weight": "layers.{layer}.attention_norm.weight",
        "model.layers.{layer}.post_attention_layernorm.weight": "layers.{layer}.ffn_norm.weight",
        "model.layers.{layer}.self_attn.q_proj.weight": "layers.{layer}.attention.wq.weight",
        "model.layers.{layer}.self_attn.k_proj.weight": "layers.{layer}.attention.wk.weight",
        "model.layers.{layer}.self_attn.v_proj.weight": "layers.{layer}.attention.wv.weight",
        "model.layers.{layer}.self_attn.o_proj.weight": "layers.{layer}.attention.wo.weight",
        "model.layers.{layer}.self_attn.q_proj.bias": "layers.{layer}.attention.wq.bias",
        "model.layers.{layer}.self_attn.k_proj.bias": "layers.{layer}.attention.wk.bias",
        "model.layers.{layer}.self_attn.v_proj.bias": "layers.{layer}.attention.wv.bias",
        "model.layers.{layer}.self_attn.q_norm.weight": "layers.{layer}.attention.q_norm.weight",
        "model.layers.{layer}.self_attn.k_norm.weight": "layers.{layer}.attention.k_norm.weight",
        "model.layers.{layer}.mlp.gate_proj.weight": "layers.{layer}.feed_forward.w1.weight",
        "model.layers.{layer}.mlp.up_proj.weight": "layers.{layer}.feed_forward.w3.weight",
        "model.layers.{layer}.mlp.down_proj.weight": "layers.{layer}.feed_forward.w2.weight",
    }

    meta_state_dict = {}
    for key, tensor in loaded_weights.items():
        if key in hf_to_meta:
            # Direct match for top-level keys
            meta_state_dict[hf_to_meta[key]] = tensor
        elif "model.layers." in key:
            # Extract layer number and form a template key
            parts = key.split(".")
            layer_num = parts[2]  # e.g. "0" in "model.layers.0.input_layernorm.weight"
            template_key = "model.layers.{layer}." + ".".join(parts[3:])
            if template_key in hf_to_meta:
                meta_state_dict[hf_to_meta[template_key].format(layer=layer_num)] = tensor
        else:
            meta_state_dict[key] = tensor  # Keep original key if no mapping found

    return meta_state_dict


def map_vision_meta_to_hf_keys(loaded_weights):
    meta_to_hf_mappings = {
        # vision MLP
        "c_fc.weight": "fc1.weight",
        "c_fc.bias": "fc1.bias",
        "c_proj.weight": "fc2.weight",
        "c_proj.bias": "fc2.bias",
        # vision attention
        # "wq.weight": "q_proj.weight",
        # "wk.weight": "k_proj.weight",
        # "wv.weight": "v_proj.weight",
        # "wo.weight": "out_proj.weight",
        # "wq.bias": "q_proj.bias",
        # "wk.bias": "k_proj.bias",
        # "wv.bias": "v_proj.bias",
        # "wo.bias": "out_proj.bias",
        "qkv.weight": "qkv.weight",
        "qkv.bias": "qkv.bias",
        "wo.weight": "proj.weight",
        "wo.bias": "proj.bias",
        # vision encoder block
        "attn.wq.weight": "self_attn.q_proj.weight",
        "attn.wk.weight": "self_attn.k_proj.weight",
        "attn.wv.weight": "self_attn.v_proj.weight",
        "attn.wo.weight": "self_attn.out_proj.weight",
        "attn.wq.bias": "self_attn.q_proj.bias",
        "attn.wk.bias": "self_attn.k_proj.bias",
        "attn.wv.bias": "self_attn.v_proj.bias",
        "attn.wo.bias": "self_attn.out_proj.bias",
        "ln_1.weight": "layer_norm1.weight",
        "ln_1.bias": "layer_norm1.bias",
        "ln_2.weight": "layer_norm2.weight",
        "ln_2.bias": "layer_norm2.bias",
        "mlp.c_fc.weight": "mlp.fc1.weight",
        "mlp.c_fc.bias": "mlp.fc1.bias",
        "mlp.c_proj.weight": "mlp.fc2.weight",
        "mlp.c_proj.bias": "mlp.fc2.bias",
        # vision encoder
        "layers.{layer}.attn.wq.weight": "layers.{layer}.self_attn.q_proj.weight",
        "layers.{layer}.attn.wk.weight": "layers.{layer}.self_attn.k_proj.weight",
        "layers.{layer}.attn.wv.weight": "layers.{layer}.self_attn.v_proj.weight",
        "layers.{layer}.attn.wo.weight": "layers.{layer}.self_attn.out_proj.weight",
        "layers.{layer}.attn.wq.bias": "layers.{layer}.self_attn.q_proj.bias",
        "layers.{layer}.attn.wk.bias": "layers.{layer}.self_attn.k_proj.bias",
        "layers.{layer}.attn.wv.bias": "layers.{layer}.self_attn.v_proj.bias",
        "layers.{layer}.attn.wo.bias": "layers.{layer}.self_attn.out_proj.bias",
        "layers.{layer}.ln_1.weight": "layers.{layer}.layer_norm1.weight",
        "layers.{layer}.ln_1.bias": "layers.{layer}.layer_norm1.bias",
        "layers.{layer}.ln_2.weight": "layers.{layer}.layer_norm2.weight",
        "layers.{layer}.ln_2.bias": "layers.{layer}.layer_norm2.bias",
        "layers.{layer}.mlp.c_fc.weight": "layers.{layer}.mlp.fc1.weight",
        "layers.{layer}.mlp.c_fc.bias": "layers.{layer}.mlp.fc1.bias",
        "layers.{layer}.mlp.c_proj.weight": "layers.{layer}.mlp.fc2.weight",
        "layers.{layer}.mlp.c_proj.bias": "layers.{layer}.mlp.fc2.bias",
        # vision transformer
        "encoder.layers.{layer}.attn.wq.weight": "encoder.layers.{layer}.self_attn.q_proj.weight",
        "encoder.layers.{layer}.attn.wk.weight": "encoder.layers.{layer}.self_attn.k_proj.weight",
        "encoder.layers.{layer}.attn.wv.weight": "encoder.layers.{layer}.self_attn.v_proj.weight",
        "encoder.layers.{layer}.attn.wo.weight": "encoder.layers.{layer}.self_attn.out_proj.weight",
        "encoder.layers.{layer}.attn.wq.bias": "encoder.layers.{layer}.self_attn.q_proj.bias",
        "encoder.layers.{layer}.attn.wk.bias": "encoder.layers.{layer}.self_attn.k_proj.bias",
        "encoder.layers.{layer}.attn.wv.bias": "encoder.layers.{layer}.self_attn.v_proj.bias",
        "encoder.layers.{layer}.attn.wo.bias": "encoder.layers.{layer}.self_attn.out_proj.bias",
        "ln_post.weight": "post_layernorm.weight",
        "ln_post.bias": "post_layernorm.bias",
        # Top level
        "_linear.weight": "weight",  # patch_embedding
        "_linear.bias": "bias",  # patch_embedding
        "positional_embedding": "weight",  # pos_emb
        "vision_tower.vision_model.embeddings.patch_embedding._linear.weight": "vision_tower.vision_model.embeddings.patch_embedding.weight",
        "vision_tower.vision_model.embeddings.patch_embedding._linear.bias": "vision_tower.vision_model.embeddings.patch_embedding.bias",
        "vision_tower.vision_model.embeddings.position_embedding.positional_embedding": "vision_tower.vision_model.embeddings.position_embedding.weight",
        "vision_tower.vision_model.encoder.layers.{layer}.attn.wq.weight": "vision_tower.vision_model.encoder.layers.{layer}.self_attn.q_proj.weight",
        "vision_tower.vision_model.encoder.layers.{layer}.attn.wk.weight": "vision_tower.vision_model.encoder.layers.{layer}.self_attn.k_proj.weight",
        "vision_tower.vision_model.encoder.layers.{layer}.attn.wv.weight": "vision_tower.vision_model.encoder.layers.{layer}.self_attn.v_proj.weight",
        "vision_tower.vision_model.encoder.layers.{layer}.attn.wo.weight": "vision_tower.vision_model.encoder.layers.{layer}.self_attn.out_proj.weight",
        "vision_tower.vision_model.encoder.layers.{layer}.attn.wq.bias": "vision_tower.vision_model.encoder.layers.{layer}.self_attn.q_proj.bias",
        "vision_tower.vision_model.encoder.layers.{layer}.attn.wk.bias": "vision_tower.vision_model.encoder.layers.{layer}.self_attn.k_proj.bias",
        "vision_tower.vision_model.encoder.layers.{layer}.attn.wv.bias": "vision_tower.vision_model.encoder.layers.{layer}.self_attn.v_proj.bias",
        "vision_tower.vision_model.encoder.layers.{layer}.attn.wo.bias": "vision_tower.vision_model.encoder.layers.{layer}.self_attn.out_proj.bias",
        "vision_tower.vision_model.encoder.layers.{layer}.ln_1.weight": "vision_tower.vision_model.encoder.layers.{layer}.layer_norm1.weight",
        "vision_tower.vision_model.encoder.layers.{layer}.ln_1.bias": "vision_tower.vision_model.encoder.layers.{layer}.layer_norm1.bias",
        "vision_tower.vision_model.encoder.layers.{layer}.ln_2.weight": "vision_tower.vision_model.encoder.layers.{layer}.layer_norm2.weight",
        "vision_tower.vision_model.encoder.layers.{layer}.ln_2.bias": "vision_tower.vision_model.encoder.layers.{layer}.layer_norm2.bias",
        "vision_tower.vision_model.encoder.layers.{layer}.mlp.c_fc.weight": "vision_tower.vision_model.encoder.layers.{layer}.mlp.fc1.weight",
        "vision_tower.vision_model.encoder.layers.{layer}.mlp.c_fc.bias": "vision_tower.vision_model.encoder.layers.{layer}.mlp.fc1.bias",
        "vision_tower.vision_model.encoder.layers.{layer}.mlp.c_proj.weight": "vision_tower.vision_model.encoder.layers.{layer}.mlp.fc2.weight",
        "vision_tower.vision_model.encoder.layers.{layer}.mlp.c_proj.bias": "vision_tower.vision_model.encoder.layers.{layer}.mlp.fc2.bias",
        "vision_tower.vision_model.ln_post.weight": "vision_tower.vision_model.post_layernorm.weight",
        "vision_tower.vision_model.ln_post.bias": "vision_tower.vision_model.post_layernorm.bias",
        # Qwen2.5 VL mapping
        # "visual.blocks.{layer}.attn.q_proj.weight": "visual.blocks.{layer}.attn.wq.weight",
        # "visual.blocks.{layer}.attn.k_proj.weight": "visual.blocks.{layer}.attn.wk.weight",
        # "visual.blocks.{layer}.attn.v_proj.weight": "visual.blocks.{layer}.attn.wv.weight",
        # "visual.blocks.{layer}.attn.proj.weight": "visual.blocks.{layer}.attn.wo.weight",
        # "visual.blocks.{layer}.attn.q_proj.bias": "visual.blocks.{layer}.attn.wq.bias",
        # "visual.blocks.{layer}.attn.k_proj.bias": "visual.blocks.{layer}.attn.wk.bias",
        # "visual.blocks.{layer}.attn.v_proj.bias": "visual.blocks.{layer}.attn.wv.bias",
        # "visual.blocks.{layer}.attn.proj.bias": "visual.blocks.{layer}.attn.wo.bias",
    }

    hf_state_dict = {}
    for key, tensor in loaded_weights.items():
        # Handle full model paths with layer numbers
        if "vision_tower.vision_model.encoder.layers." in key:
            print(f"Processing key: {key}")
            parts = key.split(".")
            layer_num = parts[4]
            remainder = ".".join(parts[5:])
            if remainder in meta_to_hf_mappings:
                new_key = f"vision_tower.vision_model.encoder.layers.{layer_num}.{meta_to_hf_mappings[remainder]}"
                hf_state_dict[new_key] = tensor
            continue

        # Handle full vision encoder paths with layer numbers
        if "layers." in key:
            parts = key.split(".")
            layer_num = parts[1]  # e.g. "0" in "model.layers.0.input_layernorm.weight"
            template_key = "layers.{layer}." + ".".join(parts[2:])
            if template_key in meta_to_hf_mappings:
                hf_state_dict[meta_to_hf_mappings[template_key].format(layer=layer_num)] = tensor
                continue

        # Try exact matches first
        if key in meta_to_hf_mappings:
            hf_state_dict[meta_to_hf_mappings[key]] = tensor
            continue

        # For submodule state dicts, try matching the end of the key
        matched = False
        for meta_pattern, hf_pattern in meta_to_hf_mappings.items():
            if key.endswith("." + meta_pattern):
                # Replace only the matching part at the end
                prefix = key[: -len(meta_pattern)]
                new_key = prefix + hf_pattern
                hf_state_dict[new_key] = tensor
                matched = True
                break

        # If no mapping found, keep the original key
        if not matched:
            hf_state_dict[key] = tensor

    return hf_state_dict


def map_vision_hf_to_meta_keys(loaded_weights):
    hf_to_meta = {
        # vision MLP
        "fc1.weight": "c_fc.weight",
        "fc1.bias": "c_fc.bias",
        "fc2.weight": "c_proj.weight",
        "fc2.bias": "c_proj.bias",
        # vision attention
        "q_proj.weight": "wq.weight",
        "k_proj.weight": "wk.weight",
        "v_proj.weight": "wv.weight",
        "out_proj.weight": "wo.weight",
        "proj.weight": "wo.weight",
        "q_proj.bias": "wq.bias",
        "k_proj.bias": "wk.bias",
        "v_proj.bias": "wv.bias",
        "out_proj.bias": "wo.bias",
        "proj.bias": "wo.bias",
        # vision encoder
        "self_attn.q_proj.weight": "attn.wq.weight",
        "self_attn.k_proj.weight": "attn.wk.weight",
        "self_attn.v_proj.weight": "attn.wv.weight",
        "self_attn.out_proj.weight": "attn.wo.weight",
        "self_attn.q_proj.bias": "attn.wq.bias",
        "self_attn.k_proj.bias": "attn.wk.bias",
        "self_attn.v_proj.bias": "attn.wv.bias",
        "self_attn.out_proj.bias": "attn.wo.bias",
        "layer_norm1.weight": "ln_1.weight",
        "layer_norm1.bias": "ln_1.bias",
        "layer_norm2.weight": "ln_2.weight",
        "layer_norm2.bias": "ln_2.bias",
        "mlp.fc1.weight": "mlp.c_fc.weight",
        "mlp.fc1.bias": "mlp.c_fc.bias",
        "mlp.fc2.weight": "mlp.c_proj.weight",
        "mlp.fc2.bias": "mlp.c_proj.bias",
        # Top level
        # vision transformer
        "encoder.layers.{layer}.self_attn.q_proj.weight": "encoder.layers.{layer}.attn.wq.weight",
        "encoder.layers.{layer}.self_attn.k_proj.weight": "encoder.layers.{layer}.attn.wk.weight",
        "encoder.layers.{layer}.self_attn.v_proj.weight": "encoder.layers.{layer}.attn.wv.weight",
        "encoder.layers.{layer}.self_attn.out_proj.weight": "encoder.layers.{layer}.attn.wo.weight",
        "encoder.layers.{layer}.self_attn.q_proj.bias": "encoder.layers.{layer}.attn.wq.bias",
        "encoder.layers.{layer}.self_attn.k_proj.bias": "encoder.layers.{layer}.attn.wk.bias",
        "encoder.layers.{layer}.self_attn.v_proj.bias": "encoder.layers.{layer}.attn.wv.bias",
        "encoder.layers.{layer}.self_attn.out_proj.bias": "encoder.layers.{layer}.attn.wo.bias",
        "post_layernorm.weight": "ln_post.weight",
        "post_layernorm.bias": "ln_post.bias",
        "weight": "_linear.weight",
        "bias": "_linear.bias",
        "weight": "positional_embedding",  # pos_emb
        "vision_tower.vision_model.embeddings.patch_embedding.weight": "vision_tower.vision_model.embeddings.patch_embedding._linear.weight",
        "vision_tower.vision_model.embeddings.patch_embedding.bias": "vision_tower.vision_model.embeddings.patch_embedding._linear.bias",
        "vision_tower.vision_model.embeddings.position_embedding.weight": "vision_tower.vision_model.embeddings.position_embedding.positional_embedding",
        "vision_tower.vision_model.encoder.layers.{layer}.self_attn.q_proj.weight": "vision_tower.vision_model.encoder.layers.{layer}.attn.wq.weight",
        "vision_tower.vision_model.encoder.layers.{layer}.self_attn.k_proj.weight": "vision_tower.vision_model.encoder.layers.{layer}.attn.wk.weight",
        "vision_tower.vision_model.encoder.layers.{layer}.self_attn.v_proj.weight": "vision_tower.vision_model.encoder.layers.{layer}.attn.wv.weight",
        "vision_tower.vision_model.encoder.layers.{layer}.self_attn.out_proj.weight": "vision_tower.vision_model.encoder.layers.{layer}.attn.wo.weight",
        "vision_tower.vision_model.encoder.layers.{layer}.self_attn.q_proj.bias": "vision_tower.vision_model.encoder.layers.{layer}.attn.wq.bias",
        "vision_tower.vision_model.encoder.layers.{layer}.self_attn.k_proj.bias": "vision_tower.vision_model.encoder.layers.{layer}.attn.wk.bias",
        "vision_tower.vision_model.encoder.layers.{layer}.self_attn.v_proj.bias": "vision_tower.vision_model.encoder.layers.{layer}.attn.wv.bias",
        "vision_tower.vision_model.encoder.layers.{layer}.self_attn.out_proj.bias": "vision_tower.vision_model.encoder.layers.{layer}.attn.wo.bias",
        "vision_tower.vision_model.encoder.layers.{layer}.layer_norm1.weight": "vision_tower.vision_model.encoder.layers.{layer}.ln_1.weight",
        "vision_tower.vision_model.encoder.layers.{layer}.layer_norm1.bias": "vision_tower.vision_model.encoder.layers.{layer}.ln_1.bias",
        "vision_tower.vision_model.encoder.layers.{layer}.layer_norm2.weight": "vision_tower.vision_model.encoder.layers.{layer}.ln_2.weight",
        "vision_tower.vision_model.encoder.layers.{layer}.layer_norm2.bias": "vision_tower.vision_model.encoder.layers.{layer}.ln_2.bias",
        "vision_tower.vision_model.encoder.layers.{layer}.mlp.fc1.weight": "vision_tower.vision_model.encoder.layers.{layer}.mlp.c_fc.weight",
        "vision_tower.vision_model.encoder.layers.{layer}.mlp.fc1.bias": "vision_tower.vision_model.encoder.layers.{layer}.mlp.c_fc.bias",
        "vision_tower.vision_model.encoder.layers.{layer}.mlp.fc2.weight": "vision_tower.vision_model.encoder.layers.{layer}.mlp.c_proj.weight",
        "vision_tower.vision_model.encoder.layers.{layer}.mlp.fc2.bias": "vision_tower.vision_model.encoder.layers.{layer}.mlp.c_proj.bias",
        "vision_tower.vision_model.post_layernorm.weight": "vision_tower.vision_model.ln_post.weight",
        "vision_tower.vision_model.post_layernorm.bias": "vision_tower.vision_model.ln_post.bias",
        # Qwen2.5 VL mapping
        "visual.blocks.{layer}.norm1.weight": "visual.blocks.{layer}.norm1.weight",
        "visual.blocks.{layer}.norm1.bias": "visual.blocks.{layer}.norm1.bias",
        "visual.blocks.{layer}.norm2.weight": "visual.blocks.{layer}.norm2.weight",
        "visual.blocks.{layer}.norm1.bias": "visual.blocks.{layer}.norm1.bias",
        "visual.blocks.{layer}.mlp.gate_proj.weight": "visual.blocks.{layer}.mlp.gate_proj.weight",
        "visual.blocks.{layer}.mlp.gate_proj.bias": "visual.blocks.{layer}.mlp.gate_proj.bias",
        "visual.blocks.{layer}.mlp.up_proj.weight": "visual.blocks.{layer}.mlp.up_proj.weight",
        "visual.blocks.{layer}.mlp.up_proj.bias": "visual.blocks.{layer}.mlp.up_proj.bias",
        "visual.blocks.{layer}.mlp.down_proj.weight": "visual.blocks.{layer}.mlp.down_proj.weight",
        "visual.blocks.{layer}.mlp.down_proj.bias": "visual.blocks.{layer}.mlp.down_proj.bias",
        "visual.blocks.{layer}.attn.qkv.weight": "visual.blocks.{layer}.attn.qkv.weight",
        "visual.blocks.{layer}.attn.proj.weight": "visual.blocks.{layer}.attn.proj.weight",
        "visual.blocks.{layer}.attn.qkv.bias": "visual.blocks.{layer}.attn.qkv.bias",
        "visual.blocks.{layer}.attn.proj.bias": "visual.blocks.{layer}.attn.proj.bias",
    }

    remapped = {}
    for key, tensor in loaded_weights.items():
        if key in hf_to_meta:
            remapped[hf_to_meta[key]] = tensor
        elif "vision_tower.vision_model.encoder.layers." in key:
            parts = key.split(".")
            layer_num = parts[4]  # e.g. "0" in "model.layers.0.input_layernorm.weight"
            template_key = "vision_tower.vision_model.encoder.layers.{layer}." + ".".join(parts[5:])
            if template_key in hf_to_meta:
                remapped[hf_to_meta[template_key].format(layer=layer_num)] = tensor
        elif "visual.blocks." in key:
            parts = key.split(".")
            layer_num = parts[2]  # e.g. "0" in "model.layers.0.input_layernorm.weight"
            template_key = "visual.blocks.{layer}." + ".".join(parts[3:])
            if template_key in hf_to_meta:
                remapped[hf_to_meta[template_key].format(layer=layer_num)] = tensor
        else:
            remapped[key] = tensor

    # Remove language_model keys
    non_text_weights = {k: v for k, v in remapped.items() if not k.startswith("language_model.")}
    text_weights = {k: v for k, v in loaded_weights.items() if k.startswith("language_model.")}
    remapped_text = map_hf_to_meta_keys(text_weights, prefix="language_model.")
    return {**non_text_weights, **remapped_text}


def load_meta_state_dict(ckpt_dir, n_layers=None, start_layer_idx=0):
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
    is_chunked = "layers_" in str(checkpoints[0])
    if is_chunked:
        checkpoint = load_chunked_checkpoints(checkpoints, n_layers, start_layer_idx)
    else:
        checkpoint = load_sharded_checkpoints(checkpoints, n_layers)

    return checkpoint


def load_chunked_checkpoints(checkpoints, n_layers, start_layer_idx):
    checkpoint = {}

    (f"Loading {len(checkpoints)} checkpoint files")
    for ckpt in tqdm(checkpoints):
        if n_layers:
            # Layer range is in the file name, like layers_start-end.pth
            layer_range = ckpt.stem.split("_")[1]
            start_layer, end_layer = map(int, layer_range.split("-"))
            if start_layer > n_layers + start_layer_idx:
                continue
            if end_layer < start_layer_idx:
                continue

        loaded_ckpt = torch.load(ckpt, map_location="cpu")
        checkpoint.update(loaded_ckpt)
    return checkpoint


def is_param_replicated_across_shards(key: str) -> bool:
    """
    Return `True` if the parameter is replicated (i.e., not sharded)
    across checkpoint files and should not be concatenated.
    """
    if key.startswith("vision_model."):
        return any(keyword in key for keyword in ("ln", "gate", "embed", "c_proj.bias"))
    else:
        # for Meta checkpoint keys, key either starts with "text_model." or contains no such prefix; both cases are handled here
        return any(keyword in key for keyword in ("norm", "gate"))


def load_sharded_checkpoints(checkpoints, n_layers):
    checkpoint = {}
    logger.info(f"Loading {len(checkpoints)} checkpoint files")
    for ckpt in tqdm(checkpoints):
        loaded_ckpt = torch.load(ckpt, map_location="cpu")
        for (
            key,
            value,
        ) in loaded_ckpt.items():
            if "layers." in key:
                layer_num = int(key.split("layers.")[1].split(".")[0])
                if n_layers and layer_num >= n_layers:
                    continue
            if key in checkpoint:
                checkpoint[key] += [value]
            else:
                checkpoint[key] = [value]
        del loaded_ckpt

    # concat checkpoint values
    for key, value in checkpoint.items():
        if len(value) == 1 or "norm" in key:
            checkpoint[key] = value[0]
        else:
            if key == "tok_embeddings.weight" or key == "output.weight":
                assert value[0].shape[1] == 8192  # FIXME: do we need this hardcoded shape?
                # Concatenate along dimension 0 for llama3 token embeddings weight and lm head
                checkpoint[key] = torch.cat(value, dim=0)
            else:
                # cat_dim is index of the smallest dimension in value[0].shape
                cat_dim = torch.argmin(torch.tensor(value[0].shape))
                checkpoint[key] = torch.cat(value, dim=cat_dim)

    return checkpoint


def split_hf_keys(loaded_weights):
    converted_weights = {}
    for key, tensor in loaded_weights.items():
        if "self_attn.qkv_proj" in key:
            # split Q, K and V
            q_key = key.replace("self_attn.qkv_proj", "self_attn.q_proj")
            k_key = key.replace("self_attn.qkv_proj", "self_attn.k_proj")
            v_key = key.replace("self_attn.qkv_proj", "self_attn.v_proj")
            q_tensor, k_tensor, v_tensor = torch.split(tensor, tensor.shape[0] // 3, dim=0)
            converted_weights[q_key] = q_tensor
            converted_weights[k_key] = k_tensor
            converted_weights[v_key] = v_tensor
        elif "mlp.gate_up_proj" in key:
            # Split Gate and Up
            gate_key = key.replace("mlp.gate_up_proj", "mlp.gate_proj")
            up_key = key.replace("mlp.gate_up_proj", "mlp.up_proj")
            gate_tensor, up_tensor = torch.split(tensor, tensor.shape[0] // 2, dim=0)
            converted_weights[gate_key] = gate_tensor
            converted_weights[up_key] = up_tensor
        else:
            # Keep all other weights unchanged
            converted_weights[key] = tensor
    return converted_weights


def convert_hf_qkv_to_meta_format(loaded_weights, head_dim):
    """Convert HuggingFace QKV weights to Meta format for RoPE compatibility."""
    converted_weights = {}
    for key, tensor in loaded_weights.items():
        if "q_proj.weight" in key or "k_proj.weight" in key:
            # For weights: n_heads = tensor.shape[0] // head_dim
            n_heads = tensor.shape[0] // head_dim
            converted_weights[key] = reverse_permute(tensor, n_heads, tensor.shape[0], tensor.shape[1])
        elif "q_proj.bias" in key or "k_proj.bias" in key:
            # For biases: n_heads = tensor.shape[0] // head_dim
            n_heads = tensor.shape[0] // head_dim
            converted_weights[key] = reverse_permute(tensor, n_heads, tensor.shape[0], 1).squeeze(-1)
        elif "q_norm.weight" in key or "k_norm.weight" in key:
            converted_weights[key] = reverse_permute_1d(tensor)
        else:
            # Keep all other weights unchanged
            converted_weights[key] = tensor
    return converted_weights


def convert_meta_to_hf(state_dict, head_dim):
    state_dict = convert_meta_qkv_to_hf_format(state_dict, head_dim)
    state_dict = map_meta_to_hf_keys(state_dict)
    return state_dict


def convert_vision_meta_to_hf(state_dict, head_dim):
    # state_dict = convert_meta_qkv_to_hf_format(state_dict, head_dim)
    state_dict = map_vision_meta_to_hf_keys(state_dict)
    return state_dict


def map_meta_to_hf_keys(loaded_weights):
    # Define mappings at each level of the hierarchy
    meta_to_hf_mappings = {
        # Top level
        "tok_embeddings.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
        # Layer level
        "attention_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        # Attention module
        "attention.wq.weight": "self_attn.q_proj.weight",
        "attention.wk.weight": "self_attn.k_proj.weight",
        "attention.wv.weight": "self_attn.v_proj.weight",
        "attention.wo.weight": "self_attn.o_proj.weight",
        "attention.wq.bias": "self_attn.q_proj.bias",
        "attention.wk.bias": "self_attn.k_proj.bias",
        "attention.wv.bias": "self_attn.v_proj.bias",
        "attention.q_norm.weight": "self_attn.q_norm.weight",
        "attention.k_norm.weight": "self_attn.k_norm.weight",
        # Feed forward module
        "feed_forward.w1.weight": "mlp.gate_proj.weight",
        "feed_forward.w3.weight": "mlp.up_proj.weight",
        "feed_forward.w2.weight": "mlp.down_proj.weight",
        # Direct mappings for when we get just the final components
        "w1.weight": "gate_proj.weight",
        "w2.weight": "down_proj.weight",
        "w3.weight": "up_proj.weight",
        "wq.weight": "q_proj.weight",
        "wk.weight": "k_proj.weight",
        "wv.weight": "v_proj.weight",
        "wo.weight": "o_proj.weight",
        "wq.bias": "q_proj.bias",
        "wk.bias": "k_proj.bias",
        "wv.bias": "v_proj.bias",
        # Host embeddings
        "emb.weight": "weight",
    }

    hf_state_dict = {}
    for key, tensor in loaded_weights.items():
        # Handle full model paths with layer numbers
        if "layers." in key:
            parts = key.split(".")
            layer_num = parts[1]
            remainder = ".".join(parts[2:])
            if remainder in meta_to_hf_mappings:
                new_key = f"model.layers.{layer_num}.{meta_to_hf_mappings[remainder]}"
                hf_state_dict[new_key] = tensor
            continue

        # Try exact matches first
        if key in meta_to_hf_mappings:
            hf_state_dict[meta_to_hf_mappings[key]] = tensor
            continue

        # For submodule state dicts, try matching the end of the key
        matched = False
        for meta_pattern, hf_pattern in meta_to_hf_mappings.items():
            if key.endswith(meta_pattern) and key[-len(meta_pattern) :] != meta_pattern:
                # Replace only the matching part at the end
                prefix = key[: -len(meta_pattern)]
                new_key = prefix + hf_pattern
                hf_state_dict[new_key] = tensor
                matched = True
                break

        # If no mapping found, keep the original key
        if not matched:
            hf_state_dict[key] = tensor

    return hf_state_dict


def convert_meta_qkv_to_hf_format(loaded_weights, head_dim):
    """Convert Meta QKV weights back to HuggingFace format."""
    converted_weights = {}
    for key, tensor in loaded_weights.items():
        if "wq.weight" in key or "wk.weight" in key:
            # For weights: n_heads = tensor.shape[0] // head_dim
            n_heads = tensor.shape[0] // head_dim
            converted_weights[key] = permute(tensor, n_heads, tensor.shape[0], tensor.shape[1])
        elif "wq.bias" in key or "wk.bias" in key:
            # For biases: n_heads = tensor.shape[0] // head_dim
            n_heads = tensor.shape[0] // head_dim
            converted_weights[key] = permute(tensor.unsqueeze(-1), n_heads, tensor.shape[0], 1).squeeze(-1)
        elif "q_norm.weight" in key or "k_norm.weight" in key:
            converted_weights[key] = permute_1d(tensor)
        else:
            # Keep all other weights unchanged
            converted_weights[key] = tensor
    return converted_weights


def reverse_permute(tensor, n_heads, dim1, dim2):
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def permute(tensor, n_heads, dim1, dim2):
    return tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def reverse_permute_1d(tensor):
    """Convert the last dim of a tensor from separate real and imaginary parts (r1, r2, i1, i2, ...) to interleaved rope format (r1, i1, r2, i2, ...)"""
    shape = tensor.shape
    dim = shape[-1]
    assert dim % 2 == 0, "Last dimension must be even"
    reals = tensor[..., : dim // 2]
    imags = tensor[..., dim // 2 :]
    interleaved = torch.stack((reals, imags), dim=-1).flatten(start_dim=len(shape) - 1)
    return interleaved


def permute_1d(tensor):
    """Convert the last dim of a tensor from interleaved rope format (r1, i1, r2, i2, ...) to separate real and imaginary parts (r1, r2, i1, i2, ...)"""
    shape = tensor.shape
    dim = shape[-1]
    assert dim % 2 == 0, "Last dimension must be even"
    reshaped = tensor.reshape(*shape[:-1], dim // 2, 2)
    reals = reshaped[..., 0]
    imags = reshaped[..., 1]
    return torch.cat((reals, imags), dim=-1)
