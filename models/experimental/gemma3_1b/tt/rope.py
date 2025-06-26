import ttnn
import torch


def rotate_half(x):
    x1 = ttnn.slice(x, (0, 0, 0, 0), (x.shape[0], x.shape[1], x.shape[2], x.shape[-1] // 2))
    x2 = ttnn.slice(x, (0, 0, 0, x.shape[-1] // 2), (x.shape[0], x.shape[1], x.shape[2], x.shape[-1]))
    return ttnn.concat([ttnn.multiply(x2, -1), x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = ttnn.unsqueeze(cos, 1)
    sin = ttnn.unsqueeze(sin, 1)
    q_embed = ttnn.add(ttnn.mul(q, cos), ttnn.mul(rotate_half(q), sin))
    k_embed = ttnn.add(ttnn.mul(k, cos), ttnn.mul(rotate_half(k), sin))
    return q_embed, k_embed


def _compute_default_rope_parameters(model_args):
    base = model_args.rope_theta
    partial_rotary_factor = model_args.partial_rotary_factor if hasattr(model_args, "partial_rotary_factor") else 1.0
    head_dim = model_args.head_dim
    dim = int(head_dim * partial_rotary_factor)

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    return inv_freq


def _compute_linear_scaling_rope_parameters(model_args):
    factor = model_args.rope_scaling_factor

    inv_freq = _compute_default_rope_parameters(model_args)
    inv_freq = inv_freq / factor

    return inv_freq


class TtGemma3RotaryEmbedding:
    def __init__(self, model_args, batch_size, device):
        self.device = device

        self.rope_type = model_args.rope_type

        if self.rope_type == "default":
            self.rope_init_fn = _compute_default_rope_parameters
        else:
            self.rope_init_fn = _compute_linear_scaling_rope_parameters

        self.attention_scaling = 1.0

        self.dim = model_args.dim
        self.batch_size = batch_size

        self.inv_freq = self.rope_init_fn(model_args)
        self.inv_freq = self.inv_freq.reshape((1, self.inv_freq.shape[-1]))
        self.core_grid = device.compute_with_storage_grid_size()
        self.batch_grid = (
            ttnn.CoreGrid(y=4, x=8)
            if ttnn.get_arch_name() == "blackhole"
            else ttnn.num_cores_to_corerangeset(batch_size, self.core_grid, row_wise=True)
        )

    def __call__(self, x=None, position_ids=None):
        print("position_ids", position_ids.shape)

        inv_freq_expanded = self.inv_freq.expand([position_ids.shape[0], -1])
        inv_freq_expanded = inv_freq_expanded.reshape(inv_freq_expanded.shape[0], self.inv_freq.shape[1], 1)

        # inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # position_ids_expanded = position_ids[:, None, :].float()
        position_ids = position_ids.reshape(position_ids.shape[0], -1)  # making it 2D for below expadning
        position_ids_expanded = position_ids.reshape(position_ids.shape[0], 1, position_ids.shape[1])

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        # print("sin", sin)
        cos = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        sin = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        # mem_config = ttnn.create_sharded_memory_config(
        #     shape=(ttnn.TILE_SIZE, 256),
        #     core_grid=self.batch_grid,
        #     strategy=ttnn.ShardStrategy.HEIGHT,
        #     orientation=ttnn.ShardOrientation.ROW_MAJOR,
        #     use_height_and_width_as_shard_shape=True,
        # )

        # cos = ttnn.interleaved_to_sharded(cos, mem_config)  # [1, 1 (= batch / shard_num_cores), 1[32], self.head_dim]
        # sin = ttnn.interleaved_to_sharded(sin, mem_config)  # [1, 1 (= batch / shard_num_cores), 1[32], self.head_dim]

        return [cos, sin]
