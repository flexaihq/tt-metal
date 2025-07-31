# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.utility_functions import comp_allclose, comp_pcc
from loguru import logger


def test_unary_gelu_ttnn(device):
    input_torch = torch.load("input_act_gelu.pt")
    reference_output = torch.load("output_act_gelu.pt")
    input_tt = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    output_tt = ttnn.gelu(input_tt, fast_and_approximate_mode=False)

    tt_output_torch = ttnn.to_torch(output_tt)
    non_zero_indices = tt_output_torch.ne(0).nonzero(as_tuple=True)
    tt_output_torch = tt_output_torch[non_zero_indices]
    reference_output = reference_output[non_zero_indices]

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("GELU Passed!")
    else:
        logger.warning("GELU Failed!")

    assert passing, f"GELU output does not meet PCC requirement {0.99}."
