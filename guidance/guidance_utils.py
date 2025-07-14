import os
import random
import sys
from dataclasses import dataclass, field
from os.path import isfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMInverseScheduler,
    DDIMScheduler,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.models.attention import *
from diffusers.models.attention_processor import *
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from torch.cuda.amp import custom_bwd, custom_fwd
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging

from .sd_step import *


def concept_lora_cross_attn_key(attn, text_embeddings, adapter_names, lora_scale=1.0):
    attn_dict = {}

    for adapter_name in adapter_names:
        x = text_embeddings[adapter_name]
        result = attn.to_k.base_layer(x)
        torch_result_dtype = result.dtype
        lora_A = attn.to_k.lora_A[adapter_name]
        lora_B = attn.to_k.lora_B[adapter_name]
        dropout = attn.to_k.lora_dropout[adapter_name]
        x = x.to(lora_A.weight.dtype)
        concept_result = result + lora_B(lora_A(dropout(x))) * lora_scale
        concept_result = concept_result.to(torch_result_dtype)
        attn_dict[adapter_name] = concept_result

    return attn_dict


def concept_lora_cross_attn_val(attn, text_embeddings, adapter_names, lora_scale=1.0):
    attn_dict = {}

    for adapter_name in adapter_names:
        x = text_embeddings[adapter_name]
        result = attn.to_v.base_layer(x)
        torch_result_dtype = result.dtype
        lora_A = attn.to_v.lora_A[adapter_name]
        lora_B = attn.to_v.lora_B[adapter_name]
        dropout = attn.to_v.lora_dropout[adapter_name]

        x = x.to(lora_A.weight.dtype)
        concept_result = result + lora_B(lora_A(dropout(x))) * lora_scale
        concept_result = concept_result.to(torch_result_dtype)
        attn_dict[adapter_name] = concept_result

    return attn_dict


def mod_forward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    **cross_attention_kwargs,
) -> torch.Tensor:
    r"""
    The forward method of the `Attention` class.

    Args:
        hidden_states (`torch.Tensor`):
            The hidden states of the query.
        encoder_hidden_states (`torch.Tensor`, *optional*):
            The hidden states of the encoder.
        attention_mask (`torch.Tensor`, *optional*):
            The attention mask to use. If `None`, no mask is applied.
        **cross_attention_kwargs:
            Additional keyword arguments to pass along to the cross attention.

    Returns:
        `torch.Tensor`: The output of the attention layer.
    """
    # The `Attention` class can call different attention processors / attention functions
    # here we simply pass along all tensors to the selected processor class
    # For standard processors that are defined here, `**cross_attention_kwargs` is empty

    # Encoder hidden state None : Self attention layer
    if encoder_hidden_states is None:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
    else:
        if cross_attention_kwargs["RCA"]:
            return forward_rca(
                self,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

        else:
            return forward_base(
                self,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )


def forward_rca(
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    temb: Optional[torch.FloatTensor] = None,
    scale: float = 1.0,
    **kwargs,
) -> torch.FloatTensor:
    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    # print("Concept lora")
    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    # args = () if USE_PEFT_BACKEND else (scale,)
    args = ()

    # Computing base query
    base_query = attn.to_q(hidden_states, *args)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    concept_mask = kwargs["mask"]
    H = int((base_query.shape[1]) ** (1 / 2))

    hidden_state_dict = {}

    # Bg attention map computation
    resized_concept_mask = F.interpolate(concept_mask["bg"].to(base_query.dtype), size=(H, H), mode="nearest")
    binary_concept_mask = resized_concept_mask.view(batch_size, -1, 1)
    binary_concept_mask = binary_concept_mask.to(base_query.dtype)

    # Base Q,K,V
    bg_query = attn.to_q(hidden_states * binary_concept_mask, *args)
    bg_key = attn.to_k.base_layer(kwargs["text_embeddings"]["bg"], *args)
    bg_value = attn.to_v.base_layer(kwargs["text_embeddings"]["bg"], *args)

    inner_dim = bg_key.shape[-1]
    head_dim = inner_dim // attn.heads

    bg_query = bg_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    bg_key = bg_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    bg_value = bg_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    bg_hidden_states = F.scaled_dot_product_attention(
        bg_query, bg_key, bg_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )

    bg_hidden_states = bg_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    bg_hidden_states = bg_hidden_states.to(base_query.dtype)

    hidden_state_dict["bg"] = binary_concept_mask * bg_hidden_states

    # Precompute concept lora key value
    concept_key = concept_lora_cross_attn_key(
        attn, kwargs["text_embeddings"], kwargs["adapter_names"], kwargs["lora_scale"]
    )
    concept_value = concept_lora_cross_attn_val(
        attn, kwargs["text_embeddings"], kwargs["adapter_names"], kwargs["lora_scale"]
    )

    for i, (adapter) in enumerate(kwargs["adapter_names"]):

        resized_concept_mask = F.interpolate(concept_mask[adapter], size=(H, H), mode="nearest")
        binary_concept_mask = resized_concept_mask.view(batch_size, -1, 1)
        binary_concept_mask = binary_concept_mask.to(base_query.dtype)

        concept_query_i = attn.to_q(binary_concept_mask * hidden_states, *args)
        concept_key_i = concept_key[adapter]
        concept_value_i = concept_value[adapter]

        concept_query_i = concept_query_i.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        concept_key_i = concept_key_i.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        concept_value_i = concept_value_i.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        concept_hidden_states = F.scaled_dot_product_attention(
            concept_query_i, concept_key_i, concept_value_i, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        concept_hidden_states = concept_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        concept_hidden_states = concept_hidden_states.to(base_query.dtype)
        hidden_state_dict[adapter] = binary_concept_mask * concept_hidden_states

    # Sum all hidden states
    hidden_states = sum(value for key, value in hidden_state_dict.items())

    hidden_states = hidden_states.to(base_query.dtype)
    # linear proj
    hidden_states = attn.to_out[0](hidden_states, *args)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states


def forward_base(
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    temb: Optional[torch.FloatTensor] = None,
    *args,
    **kwargs,
) -> torch.Tensor:
    if len(args) > 0 or kwargs.get("scale", None) is not None:
        deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        deprecate("scale", "1.0.0", deprecation_message)

    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k.base_layer(encoder_hidden_states)
    value = attn.to_v.base_layer(encoder_hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states
