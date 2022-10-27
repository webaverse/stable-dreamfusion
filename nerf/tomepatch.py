# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import List, Tuple, Union

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention import SpatialTransformer, BasicTransformerBlock, CrossAttention
from diffusers.models.unet_blocks import UNetMidBlock2DCrossAttn, CrossAttnUpBlock2D, CrossAttnDownBlock2D
from diffusers.utils import BaseOutput
from einops import rearrange
import math
from typing import Callable, Tuple
import torch

def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    """
    Process a constant r or r schedule into a list for use internally.

    r can take the following forms:
     - int: A constant number of tokens per layer.
     - Tuple[int, float]: A pair of r, inflection.
       Inflection describes there the the reduction / layer should trend
       upward (+1), downward (-1), or stay constant (0). A value of (r, 0)
       is as providing a constant r. (r, -1) is what we describe in the paper
       as "decreasing schedule". Any value between -1 and +1 is accepted.
     - List[int]: A specific number of tokens per layer. For extreme granularity.
    """
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]

def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge
class ToMeCrossAttention(CrossAttention):
    def forward(self, hidden_states, context=None, mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        km = key
        value = self.to_v(context)
        dim = query.shape[-1]
        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                key,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, key, self._tome_info["source"]
                )
                self._tome_info["source"] = merge_source(
                    merge, value, self._tome_info["source"]
                )
            key, self._tome_info["size"] = merge_wavg(merge, key)
            value, self._tome_info["size"] = merge_wavg(merge, value)

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)


        # TODO(PVP) - mask is currently never used. Remember to re-implement when used

        # attention, what we cannot get enough of

        if self._slice_size is None or query.shape[0] // self._slice_size == 1:
            hidden_states = self._attention(query, key, value)
        else:
            hidden_states = self._sliced_attention(query, key, value, sequence_length, dim)

        hidden_states = self.to_out(hidden_states)
        return hidden_states

    def _attention(self, query, key, value):
        # TODO: use baddbmm for better performance
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention_probs = attention_scores.softmax(dim=-1)
        # compute attention output
        hidden_states = torch.matmul(attention_probs, value)
        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        # Apply proportional attention
        #size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        #if size is not None:
        #    hidden_states = hidden_states + size.log()[:, None, None, :, 0]
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = (
                torch.matmul(query[start_idx:end_idx], key[start_idx:end_idx].transpose(1, 2)) * self.scale
            )  # TODO: use baddbmm for better performance
            attn_slice = attn_slice.softmax(dim=-1)
            attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])
            # Apply proportional attention
            #size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
            #if size is not None:
            #    attn_slice = attn_slice + size.log()[start_idx:end_idx, None, None, :, 0]

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

class ToMeUNet(UNet2DConditionModel):
    """
    Modifications:
     - Initialize r, token size, and token sources.
    """

    def forward(self, *args, **kwargs):
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, channel, height, width) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        self._tome_info["r"] = parse_r(len([m for m in self.modules() if isinstance(m, BasicTransformerBlock)]), self.r)
        self._tome_info["size"] = None
        self._tome_info["source"] = None

        return super().forward(*args, **kwargs)
        

def apply_patch(
    model: UNet2DConditionModel, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """

    model.__class__ = ToMeUNet
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": False,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, BasicTransformerBlock):
            module.attn1.__class__ = ToMeCrossAttention
            module.attn1._tome_info = model._tome_info
