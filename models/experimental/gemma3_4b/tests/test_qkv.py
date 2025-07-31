# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.rope import RotarySetup

from models.tt_transformers.tt.common import PagedAttentionConfig

from models.experimental.gemma3_4b.tt.attention import Attention
from models.tt_transformers.tt.model_config import ModelArgs
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize("tt_layer_name, torch_layer_name, dim, mlp_dim", (("", "", 2560, ""),))
@pytest.mark.parametrize(
    "seq_len",
    (32,),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
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
@pytest.mark.parametrize(
    "paged_attention",
    (
        True,
        # False,
    ),
    ids=(
        "paged_attention",
        # "default_attention",
    ),
)
def test_lm_head_inference(
    seq_len,
    batch_size,
    mesh_device,
    reset_seeds,
    paged_attention,
    page_params,
    tt_layer_name,
    torch_layer_name,
    dim,
    mlp_dim,
):
    dtype = ttnn.bfloat16

    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=1)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()
    print("state_dict keys() ", state_dict.keys())
    # state_dict_prefix =
    # print("state_dict_prefix ",state_dict_prefix)
    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict_q = {
        "weight": state_dict[f"language_model.layers.0.attention.wq.weight"],
    }
    partial_state_dict_k = {
        "weight": state_dict[f"language_model.layers.0.attention.wk.weight"],
    }
    partial_state_dict_v = {
        "weight": state_dict[f"language_model.layers.0.attention.wv.weight"],
    }

    # print("partial_state_dict.keys() ",partial_state_dict.keys())
    model_args.WEIGHTS_DTYPE = dtype
    reference_model = model_args.reference_transformer(wrap=False)  # Gemma3 Entire Model
    # print(f"Reference model: {reference_model}")
    reference_q = reference_model.model.get_submodule("layers.0.self_attn.q_proj")
    reference_k = reference_model.model.get_submodule("layers.0.self_attn.k_proj")
    reference_v = reference_model.model.get_submodule("layers.0.self_attn.v_proj")

    reference_q.load_state_dict(partial_state_dict_q)
    reference_k.load_state_dict(partial_state_dict_k)
    reference_v.load_state_dict(partial_state_dict_v)

    torch_input = torch.randn(batch_size, 1, model_args.dim)

    reference_output_q = reference_q(torch_input)
    reference_output_k = reference_k(torch_input)
    reference_output_v = reference_v(torch_input)

    reference_output = torch.cat((reference_output_q, reference_output_k, reference_output_v), dim=-1)

    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )

    # Implied shuffling of blocks
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    # Page table which maps virtual blocks to physical
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(
        model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
    )
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, -2) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )
    rope_setup = RotarySetup(
        mesh_device,
        batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling_factor,
        model_args.orig_context_len,
    )

    transformation_mats = rope_setup.get_both_trans_mats()

    tt_model = Attention(
        mesh_device,
        state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_idx=0,
        dtype=dtype,
        layer_num=0,
        transformation_mats=transformation_mats,
        configuration=model_args,
        paged_attention_config=paged_attention_config,
    )
    logger.info("Run QKV Linear test")
    attention_input = model_args.prepare_residual_tensor_decode(
        torch_input,
        model_args.model_config["SHARDED_ATTN_INPUT_MEMCFG"],
        force_replicated=False if model_args.is_galaxy else True,
    )
    generation_start_pos = 0
    current_pos = torch.tensor([generation_start_pos for _ in range(batch_size)])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    rot_mats = rope_setup.get_rot_mats(current_pos)

    tt_out = tt_model(
        attention_input,
        current_pos_tensor,
        rot_mats=rot_mats,
        mode="decode",
        page_table=page_table_tt,
    )

    tt_output_torch = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, model_args.cluster_shape, dims=(3, 1) if model_args.is_galaxy else (1, 3)
        ),
    )

    print("tt_output_torch shape:", tt_output_torch)
    print("reference_output shape:", reference_output)
    tt_output_torch = tt_output_torch[:, :, 0, :]  # → [1, 1, 4096]
    print("tt_output_torch shape:", tt_output_torch)

    # non_zero_indices = tt_output_torch.ne(0).nonzero(as_tuple=True)
    # tt_output_torch = tt_output_torch[non_zero_indices]
    # reference_output = reference_output[non_zero_indices]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {torch_layer_name} {pcc_message}")
    if passing:
        logger.info("QKV Linear Passed!")
    else:
        logger.warning("QKV Linear Failed!")

    assert passing, f"QKV Linear output does not meet PCC requirement {pcc_required}: {pcc_message}."
