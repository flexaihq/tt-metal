# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "tt_layer_name, torch_layer_name, dim, mlp_dim",
    (
        ("language_model.layers.0.attention.wq", "layers.0.self_attn.q_proj", 2560, None),
        ("language_model.layers.0.attention.wk", "layers.0.self_attn.k_proj", 2560, None),
        ("language_model.layers.0.attention.wv", "layers.0.self_attn.v_proj", 2560, None),
        # ("language_model.layers.0.attention.wo", "layers.0.self_attn.o_proj", 2048, None),
        # ("language_model.layers.0.feed_forward.w1", "layers.0.mlp.gate_proj", 2560, (-2, -1)),
        # ("language_model.layers.0.feed_forward.w2", "layers.0.mlp.down_proj", 10240, (-1, -2)),
        # ("language_model.layers.0.feed_forward.w3", "layers.0.mlp.up_proj", 2560, (-2, -1)),
    ),
)
@pytest.mark.parametrize(
    "seq_len",
    (32,),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_lm_head_inference(
    seq_len, batch_size, mesh_device, reset_seeds, tt_layer_name, torch_layer_name, dim, mlp_dim
):
    dtype = ttnn.bfloat16

    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=seq_len)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()
    print("state_dict keys() ", state_dict.keys())
    # state_dict_prefix =
    # print("state_dict_prefix ",state_dict_prefix)
    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {
        "weight": state_dict[f"{tt_layer_name}.weight"],
    }
    print("partial_state_dict.keys() ", partial_state_dict.keys())
    model_args.WEIGHTS_DTYPE = dtype
    reference_model = model_args.reference_transformer(wrap=False)  # Gemma3 Entire Model
    # print(f"Reference model: {reference_model}")
    reference_model = reference_model.model.get_submodule(torch_layer_name)
    print(f"Reference model: {reference_model}")

    reference_model.load_state_dict(partial_state_dict)

    torch_input = torch.randn(1, 1, seq_len, dim)
    reference_output = reference_model(torch_input)
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, dims=(None, 3) if model_args.is_galaxy else (None, None), mesh_shape=model_args.cluster_shape
        ),
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Run MLP Linear test")
    # tt_output = tt_model(tt_input)

    # pad_hidden_dim = lambda tensor, dim: pad_to_size(tensor, dim=dim, size=args.hidden_dim)

    weights = torch.transpose(partial_state_dict["weight"], -2, -1)
    weights = ttnn.as_tensor(
        # pad_hidden_dim(
        # ),  # Grab only the wX part of the name
        weights,
        dtype=ttnn.bfloat8_b,
        device=mesh_device,
        # mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dim=mlp_dim, mesh_shape=model_args.cluster_shape),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG
        # cache_file_name=cache_name(name),
    )
    tt_output = ttnn.linear(tt_input, weights)
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, model_args.cluster_shape, dims=(3, 1) if model_args.is_galaxy else (1, 3)
        ),
    )
    tt_output_torch = tt_output_torch[:, 0:1, :, : model_args.vocab_size]
    non_zero_indices = tt_output_torch.ne(0).nonzero(as_tuple=True)
    tt_output_torch = tt_output_torch[non_zero_indices]
    reference_output = reference_output[non_zero_indices]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {torch_layer_name} {pcc_message}")
    if passing:
        logger.info("MLP Linear Passed!")
    else:
        logger.warning("MLP Linear Failed!")

    assert passing, f"MLP Linear output does not meet PCC requirement {pcc_required}: {pcc_message}."
