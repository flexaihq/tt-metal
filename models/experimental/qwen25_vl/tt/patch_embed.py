"""
This is the patch embedding implementation for Qwen-VL-7B.

The existing TtLlamaConv2dPatch from tt_transformers uses Conv2d, but Qwen needs Conv3d instead.
Since ttnn.experimental.conv3d currently only supports Conv3d with stride (1, 1, 1)
(see: https://github.com/tenstorrent/tt-metal/issues/24634),
we're using PyTorch's Conv3d here instead.
"""

import ttnn
import torch.nn.functional as F


class TTQwen2_5_VisionPatchEmbed:
    def __init__(
        self,
        device,
        patch_size,
        temporal_patch_size,
        in_channels,
        embed_dim,
        state_dict,
        weight_key,
        layer_num=None,
        state_dict_prefix="",
        weight_cache_path=None,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_dtype=ttnn.bfloat16,
        mode="decode",
    ):
        super().__init__()
        self.mode = mode
        self.device = device

        weight_name_1 = f"{state_dict_prefix}{weight_key}proj.weight"
        self.weight_1 = state_dict[weight_name_1]

        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]

    def __call__(self, x):
        x = ttnn.to_torch(
            x,
        )
        weight = self.weight_1.to(dtype=x.dtype)
        x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = F.conv3d(input=x, weight=weight, stride=(2, 14, 14), padding=0, dilation=1, groups=1).view(
            -1, self.embed_dim
        )

        x = ttnn.from_torch(
            x,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return x
