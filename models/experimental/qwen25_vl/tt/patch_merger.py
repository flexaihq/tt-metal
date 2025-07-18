import ttnn
from models.experimental.qwen25_vl.tt.rmsnorm import RMSNorm
from models.tt_transformers.tt.model_config import ModelArgs


class TtLinear:
    def __init__(self, device, parameters, hidden_size, intermediate_size, bias, memory_config=None):
        self.weight = parameters
        self.hidden_size = hidden_size
        self.bias = bias
        self.intermediate_size = intermediate_size

    def __call__(self, x):
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        self.weight = ttnn.to_layout(self.weight, ttnn.TILE_LAYOUT)
        self.bias = ttnn.to_layout(self.bias, ttnn.TILE_LAYOUT)
        return ttnn.linear(x, self.weight, bias=self.bias, memory_config=ttnn.L1_MEMORY_CONFIG)


class TTQwen2_5_VLPatchMerger:
    def __init__(
        self,
        device,
        dim,
        state_dict,
        weight_key,
        layer_num=None,
        state_dict_prefix="",
        weight_cache_path=None,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_dtype=ttnn.bfloat16,
        is_distributed=None,
        eps: float = 1e-06,
        dims=3584,
        context_dim=1280,
        spatial_merge_size=2,
        mode="decode",
    ):
        super().__init__()
        self.eps = eps
        self.mode = mode

        tt_model_args = ModelArgs(
            device,
            max_batch_size=1,
            max_seq_len=128,
        )

        weight_name_1 = f"{state_dict_prefix}{weight_key}ln_q.weight"
        weight_name_2 = f"{state_dict_prefix}{weight_key}mlp.0.weight"
        weight_name_3 = f"{state_dict_prefix}{weight_key}mlp.2.weight"

        bias_name_2 = f"{state_dict_prefix}{weight_key}mlp.0.bias"
        bias_name_3 = f"{state_dict_prefix}{weight_key}mlp.2.bias"

        self.weight_1 = ttnn.as_tensor(
            state_dict[weight_name_1],
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=weight_memory_config,
        )

        self.weight_2 = ttnn.as_tensor(
            state_dict[weight_name_2],
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=weight_memory_config,
        )

        self.weight_3 = ttnn.as_tensor(
            state_dict[weight_name_3],
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=weight_memory_config,
        )

        self.bias_2 = ttnn.as_tensor(
            state_dict[bias_name_2],
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=weight_memory_config,
        )

        self.bias_3 = ttnn.as_tensor(
            state_dict[bias_name_3],
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=weight_memory_config,
        )

        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = RMSNorm(
            device=device,
            dim=1280,
            state_dict=state_dict,
            state_dict_prefix="",
            weight_key="visual.merger.ln_q",
            weight_dtype=ttnn.bfloat16,
            is_distributed=False,
            sharded_program_config=tt_model_args.get_model_config()["SHARDED_NORM_ATTN_PRGM_CFG"],
            sharded_output_config=False,
        )

        self.weight_3 = ttnn.transpose(self.weight_3, 0, 1)

        self.weight_2 = ttnn.transpose(self.weight_2, 0, 1)

        self.mlp = [
            TtLinear(device, self.weight_2, self.hidden_size, self.hidden_size, self.bias_2, None),
            ttnn.gelu,
            TtLinear(device, self.weight_3, self.hidden_size, dims, self.bias_3, None),
        ]

    def __call__(self, x):
        x = self.ln_q(x, mode=self.mode)

        x = ttnn.reshape(x, (-1, self.hidden_size))

        for layer in self.mlp:
            x = layer(x)

        return x
