import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import comp_allclose, comp_pcc


def custom_ttnn_rms(
    inp, epsilon=None, weight=None, program_config=None, memory_config=None, compute_kernel_config=None
):
    inp = ttnn.sharded_to_interleaved(inp, ttnn.DRAM_MEMORY_CONFIG)
    xnorm = ttnn.pow(inp, 2)
    xnorm = ttnn.mean(xnorm, dim=-1, keepdim=True)
    xnorm = ttnn.rsqrt(xnorm + epsilon)
    xnorm = ttnn.multiply(inp, xnorm)
    weight = ttnn.reshape(weight, [1, 1, -1])
    output = ttnn.multiply(xnorm, (weight), use_legacy=False)

    if memory_config is not None:
        output = ttnn.to_memory_config(output, memory_config)

    ttnn.deallocate(xnorm)
    ttnn.deallocate(weight)

    return output


@pytest.mark.parametrize("rms_func", ["ttnn", "custom"])
def test_rms_comp(device, rms_func):
    input = torch.load("models/common/rms_inputs/rms_input.pt").float()
    weight = torch.load("models/common/rms_inputs/weight.pt").float().view(1, 1, 3584)

    torch_rms_norm = torch.nn.functional.rms_norm(input, normalized_shape=(3584,), weight=weight.view(-1), eps=1e-6)

    tt_input = ttnn.from_torch(input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_weight = ttnn.from_torch(weight, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    if rms_func == "ttnn":
        compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        tt_rms_norm = ttnn.rms_norm(
            tt_input,
            epsilon=1e-6,
            weight=ttnn_weight,
            memory_config=None,
            program_config=None,
            compute_kernel_config=compute_kernel_config_hifi2,
        )
    elif rms_func == "custom":
        tt_rms_norm = custom_ttnn_rms(tt_input, 1e-6, ttnn_weight, None, None)
    else:
        raise ValueError(f"Unknown RMSNorm function: {rms_func}")

    tt_output_torch = ttnn.to_torch(tt_rms_norm)

    passing, pcc_message = comp_pcc(torch_rms_norm, tt_output_torch)

    logger.info(comp_allclose(torch_rms_norm, tt_output_torch))
    logger.info(pcc_message)

    assert passing, f"RMSNorm implementation `{rms_func}` failed PCC check (expected >= 0.99)"
