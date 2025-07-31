"""
This is the Entire Model implementation for Gemma-3-4b-it
"""

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import collections
import logging
from typing import List, Tuple

import torch
from PIL import Image as PIL_Image

import ttnn
from models.tt_transformers.tt.common import copy_host_to_device

from transformers.cache_utils import (
    StaticCache as TtStaticCache,
)
from models.experimental.gemma3_4b.tt.gemma_vision_crossattention import TtGemmaTransformerVision
from models.experimental.gemma3_4b.tt.text_model import Gemma3_4BTransformer
from models.utility_functions import nearest_32

logger = logging.getLogger(__name__)
MP_SCALE = 8


def get_negative_inf_value(dtype):
    return torch.finfo(dtype).min


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


def build_encoder_attention_mask(
    x: torch.Tensor,
    ar: torch.Tensor,
    ntok: int,
    num_chunks: int,
    n_heads: int,
):
    """
    Build vision encoder attention mask that omits padding tokens.
    """
    masks = []
    for arx in ar:
        mask_i = torch.ones((num_chunks, x.shape[2], 1), dtype=x.dtype)
        mask_i[: arx[0] * arx[1], :ntok] = 0
        mask_i = mask_i.view(num_chunks * x.shape[2], -1)
        mask_i = mask_i @ mask_i.T * get_negative_inf_value(x.dtype)
        mask_i = mask_i.unsqueeze(0)
        masks.append(mask_i)
    masks = torch.stack(masks).to(x.device).expand(-1, n_heads, -1, -1)
    return masks


class Gemma3CrossAttentionTransformer(torch.nn.Module):
    def __init__(
        self,
        mesh_device,
        state_dict,
        weight_cache_path,
        dtype,
        configuration,
        use_paged_kv_cache=False,
    ) -> None:
        super().__init__()

        self.model_dim = configuration.dim

        self.mesh_device = mesh_device
        self.state_dict = state_dict
        self.weight_cache_path = weight_cache_path
        self.dtype = dtype
        self.configuration = configuration

        self.vision_model = TtGemmaTransformerVision(
            mesh_device,
            state_dict,
            state_dict_prefix="vision_tower.vision_model.",
            weight_cache_path=configuration.weight_cache_path(dtype),
            dtype=dtype,
            configuration=configuration,
        )

        self.text_model = Gemma3_4BTransformer(
            args=configuration,
            mesh_device=mesh_device,
            state_dict=state_dict,
            # state_dict_prefix="language_model.",
            weight_cache_path=configuration.weight_cache_path(ttnn.bfloat8_b),
            dtype=ttnn.bfloat8_b,
            use_paged_kv_cache=use_paged_kv_cache,
        )
        self.image_res = configuration.vision_chunk_size
        self.max_num_chunks = configuration.vision_max_num_chunks
        self.num_vision_tokens = self.max_num_chunks * nearest_32(self.configuration.vision_chunk_ntok)
        # self.image_transform = partial(
        #     llama_reference_image_transforms.VariableSizeImageTransform(size=configuration.vision_chunk_size),
        #     max_num_chunks=configuration.vision_max_num_chunks,
        # )

    def _update_causal_mask(self, attention_mask, token_type_ids, past_key_values, cache_position, input_tensor):
        if attention_mask is not None and len(attention_mask.shape) == 4:
            return attention_mask

        min_dtype = -3.4028234663852886e38
        inputs_lead_dim, sequence_length = input_tensor.shape[0], input_tensor.shape[1]

        if isinstance(past_key_values, TtStaticCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, ttnn.Tensor)
                else cache_position[0] + sequence_length + 1
            )

        if attention_mask is not None and len(attention_mask.shape) == 4:
            return attention_mask

        target_length = ttnn.to_torch(target_length).item()
        sequence_length = int(sequence_length)
        target_length = int(target_length)
        min_fill_value = float(min_dtype)
        causal_mask = ttnn.full(
            shape=[sequence_length, target_length],
            fill_value=min_fill_value,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )

        if sequence_length != 1:
            causal_mask = ttnn.triu(causal_mask, diagonal=1)

        a = ttnn.arange(start=0, end=target_length, step=1, device=self.mesh_device)
        a = ttnn.to_layout(a, ttnn.TILE_LAYOUT)
        a = ttnn.unsqueeze(a, 0)

        b = ttnn.reshape(cache_position, (-1, 1))
        c = ttnn.gt(a, b, use_legacy=False)

        causal_mask *= c
        causal_mask = ttnn.reshape(causal_mask, (1, 1, sequence_length, target_length))
        causal_mask = ttnn.expand(causal_mask, [inputs_lead_dim, 1, -1, -1])

        # if attention_mask is not None: TODO Lets pass Attention mask None....
        #     causal_mask = causal_mask.clone()
        #     mask_length = attention_mask.shape[-1]

        #     padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
        #     padding_mask = padding_mask == 0
        #     causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)

        return causal_mask

    def forward(
        self,
        input_ids,
        current_pos,
        inputs_embeds=None,
        cache_position=None,
        past_key_values=None,
        pixel_values=None,
        return_dict=None,
        attention_mask=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        page_table=None,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else False  # self.configuration.use_return_dict

        is_training = False

        # Replace image id woth PAD if the image token if OOV, to avoid index-errors
        if input_ids is not None and self.configuration.image_token_index >= self.configuration.vocab_size:
            special_image_mask = input_ids == self.configuration.image_token_index
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.text_model.embd(llm_input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = ttnn.arange(
                start=past_seen_tokens,
                end=past_seen_tokens + inputs_embeds.shape[1],
                step=1,
                device=self.configuration.mesh_device,
            )
            cache_position = ttnn.to_layout(cache_position, ttnn.TILE_LAYOUT)

        if pixel_values is not None:
            image_features = self.vision_model(pixel_values)

            input_ids = ttnn.to_torch(input_ids)
            inputs_embeds = ttnn.to_torch(inputs_embeds)
            image_features = ttnn.to_torch(image_features)
            special_image_mask = (input_ids == self.configuration.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

            inputs_embeds = ttnn.from_torch(
                inputs_embeds,
                device=self.mesh_device,
                dtype=self.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        causal_mask = self._update_causal_mask(
            attention_mask, token_type_ids, past_key_values, cache_position, inputs_embeds
        )

        # Get cos/sin matrices for the current position of each user
        # rot_mats = tt_model.rope_setup.get_rot_mats(current_pos )  # TODO Fix for sliding window attention #TODO Fix for Gemma3 4B
        # rot_mats_local = tt_model.rope_setup_local.get_rot_mats(current_pos) #TODO Fix for sliding window attention #TODO Fix for Gemma3 4B
        rot_mats_global = self.text_model.rope_setup.get_rot_mats(current_pos)  # default
        rot_mats_local = self.text_model.rope_setup_local.get_rot_mats(current_pos)  # default
        rot_mats = [rot_mats_global, rot_mats_local]

        # x tensor to TILE layout
        inputs_embeds = ttnn.to_layout(inputs_embeds, ttnn.TILE_LAYOUT)

        tt_out = self.text_model(
            inputs_embeds,
            current_pos=current_pos,
            rot_mats=rot_mats,  # should contain both for slidig window and without it  #TODO Fix for Gemma3 4B
            mode="prefill",
            page_table=page_table,
        )

        return tt_out

    def setup_cache(self, max_batch_size):
        return self.text_model.setup_cache(max_batch_size)

    def compute_vision_tokens_masks(
        self,
        batch_images: List[List[PIL_Image.Image]],
        batch_masks: List[List[List[int]]],
        total_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        skip_vision_encoder = False

        assert len(batch_images) == len(batch_masks), "Images and masks must have the same length"

        max_num_images = max(len(x) for x in batch_images)
        bsz = len(batch_images)

        if max_num_images == 0:
            num_chunks = [[self.max_num_chunks] for _ in batch_images]
            skip_vision_encoder = True
        else:
            # images_and_aspect_ratios = [[self.image_transform(im) for im in row] for row in batch_images]
            # transformed_images = [[x[0] for x in row] for row in images_and_aspect_ratios]
            transformed_images = batch_images

        if skip_vision_encoder:
            vision_tokens = torch.zeros(
                (
                    bsz,
                    max_num_images,
                    self.max_num_chunks,
                    int((self.vision_model.image_res / self.vision_model.patch_size) ** 2 + 1),
                    self.model_dim,
                ),
            )
        else:
            # TT vision_model
            vision_tokens = self.vision_model(transformed_images[0])
            # Back to torch
            vision_tokens = ttnn.to_torch(ttnn.get_device_tensors(vision_tokens)[0])
            chunk_seq_len = self.configuration.vision_chunk_ntok
            # NOTE: slicing up to chunk_seq_len is necessary because padding information is lost by this point
            vision_tokens = (
                vision_tokens[0, :, :chunk_seq_len]
                .reshape(bsz, max_num_images, self.max_num_chunks, -1, self.model_dim)
                .float()
            )

        bsz, nimg, nchunk, ntok, image_token_dim = tuple(vision_tokens.shape)
        padded_seq_len = self.num_vision_tokens

        # Prepare vision tokens for TT text_model
        vision_tokens_squeeze = vision_tokens.view(1, bsz, -1, image_token_dim)
        vision_tokens_squeeze = torch.nn.functional.pad(
            vision_tokens_squeeze, (0, 0, 0, padded_seq_len - vision_tokens_squeeze.shape[2]), "constant", 0
        )
        vision_tokens_tt = ttnn.from_torch(
            vision_tokens_squeeze,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        return (vision_tokens_tt, None, None)

    def validate_inputs(self, tokens, position_ids):
        batch, seq_len = tokens.shape[:2]
        assert (
            seq_len <= self.configuration.max_seq_len
        ), f"Sequence length {seq_len} exceeds max sequence length {self.configuration.max_seq_len}"
        assert len(position_ids.shape) == 1, f"Position ids must be 1D, got {len(position_ids.shape)}"

    def prepare_inputs_common(self, position_ids, tokens):
        self.validate_inputs(tokens, position_ids)
        h = self.text_model.get_partially_trainable_embedding(tokens)
        return h

    def prepare_inputs_prefill(self, tokens, start_pos=0, page_table=None, chunk_page_table=None):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        TODO: Debate whether this function is responsible for padding
        """

        tokens = tokens.reshape(1, 1, 1, -1)
        S = tokens.shape[-1]
        tokens = ttnn.from_torch(
            tokens,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        tokens_embd = self.embd(tokens)
        tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)

        # Slice the rot mats to the prefill seqlen
        assert (
            self.rope_setup.cos_matrix.shape[2] >= start_pos + S
        ), f"Padded prefill end idx {start_pos + S} exceeds max seq len {self.rope_setup.cos_matrix.shape[2]}"

        tt_rot_mats_prefill_global = [
            self.rope_setup.cos_matrix[:, :, start_pos : start_pos + S, :],
            self.rope_setup.sin_matrix[:, :, start_pos : start_pos + S, :],
        ]

        tt_rot_mats_prefill_local = [
            self.rope_setup_local.cos_matrix[:, :, start_pos : start_pos + S, :],
            self.rope_setup_local.sin_matrix[:, :, start_pos : start_pos + S, :],
        ]

        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_chunk_page_table = None

        return tokens_embd, [tt_rot_mats_prefill_global, tt_rot_mats_prefill_local], tt_page_table, tt_chunk_page_table

    def prepare_inputs_decode(self, *inputs):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        Its implementation can take advantage of a few other functions which the
        model must implement.
        """
        host_inputs = self.prepare_decode_inputs_host(*inputs)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)  # Helper function
        transformed_device_inputs = self.transform_decode_inputs_device(*device_inputs)
        return transformed_device_inputs

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        """
        Inputs are torch tensors or python types. Outputs are ttnn tensors on host.
        NOTE: Tokens and current_pos are padded to batch
        """
        B = tokens.shape[0]
        assert current_pos.shape[0] == B, "Batch size mismatch"
        assert B == self.args.max_batch_size, "Batch size must be equal to max_batch_size"

        # Necessary padding to be full tile sized when on device
        tokens = torch.nn.functional.pad(tokens.view(-1), (0, 32 - len(tokens)), "constant", 0)
        tokens = ttnn.from_torch(
            tokens,
            device=None,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        tokens = ttnn.unsqueeze_to_4D(tokens)

        rot_current_pos = torch.maximum(
            current_pos, torch.tensor(0, dtype=torch.int64)
        )  # Ensure position indices are non-negative
        rope_idxs = self.rope_setup.get_rot_idxs(rot_current_pos, on_host=True)
        current_pos_tt = ttnn.from_torch(
            current_pos,
            device=None,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(None, 0) if (self.args.is_galaxy and B > 1) else (None, None),
                mesh_shape=self.args.cluster_shape,
            ),
        )

        if page_table is not None:
            page_table = ttnn.from_torch(
                page_table,
                device=None,
                dtype=ttnn.int32,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device,
                    dims=(None, -2) if (self.args.is_galaxy and B > 1) else (None, None),
                    mesh_shape=self.args.cluster_shape,
                ),
            )
        return tokens, current_pos_tt, rope_idxs, page_table

    def transform_decode_inputs_device(self, tokens, current_pos, rope_idxs, page_table=None):
        """
        Inputs are ttnn tensors on device. This function applies any on-device
        transformations which should happen before forward decode.
        For example: tilize, reshape, shard.
        Return transformed device tensors

        Get rope sin/cos
        Embed tokens
        """
        tt_rot_mats = self.rope_setup.get_rot_mats(rope_idxs)
        tt_rot_mats_local = self.rope_setup_local.get_rot_mats(rope_idxs)

        tt_tokens = self.embd(tokens)
        tt_tokens = ttnn.unsqueeze_to_4D(tt_tokens)
        tt_tokens = ttnn.to_memory_config(
            tt_tokens,
            self.args.model_config["DECODE_RESIDUAL_MEMCFG"],
        )
        return tt_tokens, current_pos, [tt_rot_mats, tt_rot_mats_local], page_table

    def process_output_prefill(self, tt_out, last_token_idx):
        """
        Input is ttnn device tensor of logits. Output is torch logits tensor.
        NOTE: In this model, prefill always uses get_last_token
        """
        logits = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self.mesh_device, dims=(3, 1) if self.args.is_galaxy else (1, -1), mesh_shape=self.args.cluster_shape
            ),
        )[0, 0, last_token_idx, : self.vocab_size]
        return logits

    def process_output_decode(self, tt_out, B, S=1, is_tokens=False):
        """
        Input is ttnn device tensor of logits if is_tokens=False, otherwise tokens. Output is the corresponding torch tensor.
        """
        if is_tokens:
            tt_out = ttnn.to_torch(
                tt_out,  # tt_out.cpu(blocking=True, cq_id=1),
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device,
                    dims=(3, 1) if self.args.is_galaxy else (1, -1),
                    mesh_shape=self.args.cluster_shape,
                ),
            )[0, 0, 0, :B]
            return tt_out

        if self.args.num_devices > 1:
            tt_out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
        else:
            tt_out = ttnn.to_torch(tt_out).float()
        tt_out = tt_out[:, :, :B, : self.vocab_size].view(B, S, -1)
        return tt_out

    def ttnn_prefill_forward(
        self,
        x,
        rot_mats,
        user_id,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
    ):
        """
        This method will take device tensors and any other args to run forward.
        It returns ttnn device tensors.
        """
        return self.forward(
            x,
            current_pos=None,
            rot_mats=rot_mats,
            user_id=user_id,
            mode="prefill",
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            get_last_token=get_last_token,
            kv_cache=kv_cache,
        )

    def ttnn_decode_forward(
        self,
        x,
        current_pos,
        rot_mats,
        page_table=None,
        kv_cache=None,
        argmax_on_device=False,
    ):
        """
        This method will take device tensors and any other args to run forward.
        It returns ttnn device tensors.
        """
        tt_logits = self.forward(
            x,
            current_pos,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table,
            kv_cache=kv_cache,
        )

        # Gather the output across all devices and untilize the tensor (for argmax)
        if self.args.num_devices > 1:
            if self.args.is_galaxy:
                tt_logits = ttnn.all_gather(
                    tt_logits,
                    dim=3,
                    num_links=2,
                    cluster_axis=0,
                    mesh_device=self.mesh_device,
                    topology=self.args.ccl_topology(),
                )
            else:
                tt_logits = ttnn.all_gather(tt_logits, dim=3, num_links=1, topology=self.args.ccl_topology())
        tt_logits = ttnn.untilize(tt_logits, use_multicore=True)

        if argmax_on_device:
            tt_logits = ttnn.argmax(  # TODO Add multicore support to batch > 1
                tt_logits,
                dim=3,
                keepdim=True,
                use_multicore=False if self.args.max_batch_size > 1 else True,  # ,output_tensor=tokens
            )
        else:
            # Send output logits to DRAM so L1 is not reserved for ttnn tracing and can be used by subsequent operations
            if not self.args.is_galaxy:
                tt_logits = ttnn.to_memory_config(tt_logits, ttnn.DRAM_MEMORY_CONFIG)

        return tt_logits


def _stack_images(
    images: List[List[PIL_Image.Image]],
    max_num_chunks: int,
    image_res: int,
    max_num_images: int,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Takes a list of list of images and stacks them into a tensor.
    This function is needed since images can be of completely
    different resolutions and aspect ratios.
    """
    out_images, out_num_chunks = [], []
    for imgs_sample in images:
        out_images_i = torch.zeros(
            max_num_images,
            max_num_chunks,
            3,
            image_res,
            image_res,
        )
        _num_chunks = []
        for j, chunks_image in enumerate(imgs_sample):
            out_images_i[j, : chunks_image.shape[0]] = chunks_image
            _num_chunks.append(chunks_image.shape[0])
        out_images.append(out_images_i)
        out_num_chunks.append(_num_chunks)
    return torch.stack(out_images), out_num_chunks


def _pad_masks(
    all_masks: List[List[List[int]]],
    all_num_chunks: List[List[int]],
    total_len: int,
    max_num_chunks: int,
) -> torch.Tensor:
    # dtype = torch.bfloat16
    dtype = torch.float32
    inf_value = get_negative_inf_value(dtype)

    bsz = len(all_masks)
    max_num_media = max([len(m) for m in all_masks])

    out_masks = torch.full(
        (bsz, total_len, max_num_media, max_num_chunks),
        inf_value,
        dtype=dtype,
    )

    for idx, (mask, num_chunks) in enumerate(zip(all_masks, all_num_chunks)):
        for mask_idx, (mask_elem, mask_num_chunks) in enumerate(zip(mask, num_chunks)):
            if len(mask_elem) == 2:
                mask_elem[1] = min(mask_elem[1], total_len)
                if mask_elem[1] == -1:
                    mask_elem[1] = total_len
                out_masks[idx, mask_elem[0] : mask_elem[1], mask_idx, :mask_num_chunks].fill_(0.0)

    return out_masks
