import copy
import math
import warnings
from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F
from apex.normalization import FusedLayerNorm
from numpy.random import uniform
from torch import Tensor
from torch.cuda.amp import custom_fwd
from torch.nn import Module, Linear, Dropout, ModuleList, Embedding
from torch.nn.functional import linear, pad, softmax, dropout
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.init import xavier_uniform_
from torch.nn.parameter import Parameter

from misc_functions import str2bool

def multi_head_attention_bias(
    query: Tensor,
    key: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    add_zero_attn: bool,
    key_padding_mask: Optional[Tensor] = None,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    attn_bias: Optional[Tensor] = None,
    scale_factor = 1,
) -> Tensor:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    # tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    # if torch.overrides.has_torch_function(tens_ops):
    #     return torch.overrides.handle_torch_function(
    #         multi_head_attention_forward,
    #         tens_ops,
    #         query,
    #         key,
    #         value,
    #         embed_dim_to_check,
    #         num_heads,
    #         in_proj_weight,
    #         in_proj_bias,
    #         bias_k,
    #         bias_v,
    #         add_zero_attn,
    #         dropout_p,
    #         out_proj_weight,
    #         out_proj_bias,
    #         training=training,
    #         key_padding_mask=key_padding_mask,
    #         need_weights=need_weights,
    #         attn_mask=attn_mask,
    #         use_separate_proj_weight=use_separate_proj_weight,
    #         q_proj_weight=q_proj_weight,
    #         k_proj_weight=k_proj_weight,
    #         v_proj_weight=v_proj_weight,
    #         static_k=static_k,
    #         static_v=static_v,
    #         attn_bias=attn_bias,
    #     )
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check

    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim * scale_factor) ** -0.5

    if not use_separate_proj_weight:
        # if (query is key or torch.equal(query, key)):
        if (query is key):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert (
            attn_mask.dtype == torch.float32
            or attn_mask.dtype == torch.float64
            or attn_mask.dtype == torch.float16
            or attn_mask.dtype == torch.uint8
            or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None:
        if static_k is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
    else:
        assert bias_k is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_bias is not None:
        assert list(attn_bias.size()) == [bsz * num_heads, tgt_len, src_len], "Mismatching attn_bias: ({},{},{}) instead of needed ({}*{},{},{})".format(attn_bias.size(0), attn_bias.size(1), attn_bias.size(2),
                                                                                                                                                         bsz , num_heads, tgt_len, src_len)
        attn_output_weights += attn_bias

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    return attn_output_weights


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    attn_bias: Optional[Tensor] = None,
    scale_factor = 1,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    # tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    # if torch.overrides.has_torch_function(tens_ops):
    #     return torch.overrides.handle_torch_function(
    #         multi_head_attention_forward,
    #         tens_ops,
    #         query,
    #         key,
    #         value,
    #         embed_dim_to_check,
    #         num_heads,
    #         in_proj_weight,
    #         in_proj_bias,
    #         bias_k,
    #         bias_v,
    #         add_zero_attn,
    #         dropout_p,
    #         out_proj_weight,
    #         out_proj_bias,
    #         training=training,
    #         key_padding_mask=key_padding_mask,
    #         need_weights=need_weights,
    #         attn_mask=attn_mask,
    #         use_separate_proj_weight=use_separate_proj_weight,
    #         q_proj_weight=q_proj_weight,
    #         k_proj_weight=k_proj_weight,
    #         v_proj_weight=v_proj_weight,
    #         static_k=static_k,
    #         static_v=static_v,
    #         attn_bias=attn_bias,
    #     )
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim * scale_factor) ** -0.5

    if not use_separate_proj_weight:
        # if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
        if (query is key) and (key is value):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
        elif key is value or torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert (
            attn_mask.dtype == torch.float32
            or attn_mask.dtype == torch.float64
            or attn_mask.dtype == torch.float16
            or attn_mask.dtype == torch.uint8
            or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz, "Invalid size: bsz: {} ~ key_padding_mask: {}".format(bsz, key_padding_mask.shape)
        assert key_padding_mask.size(1) == src_len, "Invalid size: src_len: {} ~ key_padding_mask: {}".format(src_len, key_padding_mask.shape)

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_bias is not None:
        assert list(attn_bias.size()) == [bsz * num_heads, tgt_len, src_len], "Mismatching attn_bias: ({},{},{}) instead of needed ({}*{},{},{})".format(attn_bias.size(0), attn_bias.size(1), attn_bias.size(2),
                                                                                                                                                         bsz , num_heads, tgt_len, src_len)
        attn_output_weights += attn_bias

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class MultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

    Note that if :attr:`kdim` and :attr:`vdim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.

    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, scale_factor=1, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.scale_factor = scale_factor

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.empty(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.empty(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None, attn_bias: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shapes for inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
          source sequence length.

          If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
          length, S is the source sequence length. ``attn_mask`` ensure that position i is allowed to attend
          the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

    Shapes for outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, attn_bias=attn_bias,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, attn_bias=attn_bias)

class MultiheadAttentionBias(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        bias: add bias as module parameter. Default: True.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.

    Note that if :attr:`kdim` and :attr:`vdim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.

    """
    bias_k: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, bias=True, add_bias_k=False, scale_factor=1, add_zero_attn=False, kdim=None):
        super(MultiheadAttentionBias, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim
        self.scale_factor = scale_factor

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.empty(embed_dim, self.kdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(2 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(2 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_k:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttentionBias, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None, attn_bias: Optional[Tensor] = None) -> Tensor:
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shapes for inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
          source sequence length.

          If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
          length, S is the source sequence length. ``attn_mask`` ensure that position i is allowed to attend
          the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

    Shapes for outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_bias(
                query, key, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.add_zero_attn,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask, attn_bias=attn_bias,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight)
        else:
            return multi_head_attention_bias(
                query, key, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.add_zero_attn,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask, attn_bias=attn_bias)

class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, attn_scale_factor=1, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, scale_factor=attn_scale_factor, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = FusedLayerNorm(d_model)
        self.norm2 = FusedLayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, attn_bias: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # Implemented Pre-LN Architecture for better efficiency

        # Self-Attention
        src1 = self.norm1(src)
        src1 = self.self_attn(src1, src1, src1, attn_mask=src_mask, attn_bias=attn_bias, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src1)

        # FFN
        src2 = self.norm2(src)
        src2 = self.linear1(src2)
        src2 = self.activation(src2)
        src2 = self.dropout(src2)
        src2 = self.linear2(src2)
        src = src + self.dropout2(src2)

        return src

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return torch.nn.SiLU
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, attn_bias: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, attn_bias=attn_bias, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoderLayer(Module):

    def __init__(self, d_model, nhead, tgt_attn_scale_factor=1,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.tgt_attn = MultiheadAttention(d_model, nhead, scale_factor=tgt_attn_scale_factor, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = dropout
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = FusedLayerNorm(d_model)
        self.norm2 = FusedLayerNorm(d_model)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor,
                tgt_mask: Optional[Tensor] = None, tgt_attn_bias: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        # Self-Attention
        tgt1 = self.norm1(tgt)
        tgt1 = self.tgt_attn(tgt1, tgt1, tgt1, attn_mask=tgt_mask, attn_bias=tgt_attn_bias, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt +  F.dropout(tgt1, p=self.dropout, training=self.training)

        # FFN
        tgt1 = self.norm2(tgt)
        tgt1 = self.activation(self.linear1(tgt1))
        tgt1 = F.dropout(tgt1, p=self.dropout, training=self.training)
        tgt1 = self.linear2(tgt1)
        tgt = tgt + F.dropout(tgt1, p=self.dropout, training=self.training)

        return tgt

class TransformerDecoder(Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor,
                tgt_mask: Optional[Tensor] = None, tgt_attn_bias: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(output, tgt_mask=tgt_mask, tgt_attn_bias=tgt_attn_bias, tgt_key_padding_mask=tgt_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        module.in_proj_weight.data.normal_(mean=0.0, std=0.02)

# From: https://github.com/guolinke/TUPE/blob/master/fairseq/modules/transformer_sentence_encoder.py
# this is from T5
def tupe_relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    ret = 0
    n = -relative_position
    if bidirectional:
        num_buckets //= 2
        ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
        n = torch.abs(n)
    else:
        n = torch.max(n, torch.zeros_like(n))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret

class CrossAttentionPositionalEncoder(Module):
    def __init__(self,
                 d_model,
                 num_attn_heads,
                 mask_src_cls_rel_pos = False,
                 max_seq_len = 512,
                 use_tupe_rel_pos_bias = True,
                 tupe_rel_pos_bins: int = 64,
                 tupe_max_rel_pos: int = 256):
        super(CrossAttentionPositionalEncoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.seq_tr_d_model = d_model
        self.seq_tr_nhead = num_attn_heads
        self.attn_scale_factor = 2.0
        self.mask_src_cls_rel_pos = mask_src_cls_rel_pos

        # This is from TUPE
        self.pos_tgt = Embedding(self.max_seq_len + 1, self.seq_tr_d_model)
        self.pos_src = Embedding(self.max_seq_len + 1, self.seq_tr_d_model)
        self.pos_q_linear = Linear(self.seq_tr_d_model, self.seq_tr_d_model)
        self.pos_k_linear = Linear(self.seq_tr_d_model, self.seq_tr_d_model)
        self.pos_scaling = float(self.seq_tr_d_model / self.seq_tr_nhead * self.attn_scale_factor) ** -0.5
        self.pos_tgt_ln = FusedLayerNorm(self.seq_tr_d_model)
        self.pos_src_ln = FusedLayerNorm(self.seq_tr_d_model)

        self.use_tupe_rel_pos_bias = use_tupe_rel_pos_bias
        if self.use_tupe_rel_pos_bias:
            assert tupe_rel_pos_bins % 2 == 0
            self.tupe_rel_pos_bins = tupe_rel_pos_bins
            self.tupe_max_rel_pos = tupe_max_rel_pos
            self.relative_attention_bias = Embedding(self.tupe_rel_pos_bins + 1, self.seq_tr_nhead)
            seq_len = self.max_seq_len
            context_position = torch.arange(seq_len, dtype=torch.long)[:, None]
            memory_position = torch.arange(seq_len, dtype=torch.long)[None, :]
            relative_position = memory_position - context_position
            self.rp_bucket = tupe_relative_position_bucket(
                relative_position,
                num_buckets=self.tupe_rel_pos_bins,
                max_distance=self.tupe_max_rel_pos
            )
            if self.mask_src_cls_rel_pos:
                self.rp_bucket[:, 0] = self.tupe_rel_pos_bins
                self.cls_pos_embed = Embedding(2, self.seq_tr_nhead)

        self.apply(init_bert_params)

    def get_tupe_rel_pos_bias(self, src_len, tgt_len, device):
        # Assume the input is ordered. If your input token is permuted, you may need to update this accordingly
        if self.rp_bucket.device != device:
            self.rp_bucket = self.rp_bucket.to(device)
        # Adjusted because final x's shape is L x B X E
        rp_bucket = self.rp_bucket[:tgt_len, :src_len]
        values = F.embedding(rp_bucket, self.relative_attention_bias.weight)
        values = values.permute([2, 0, 1])
        return values.contiguous() # (nhead, tgt_len, src_len)

    def get_position_attn_bias(self, src_len, tgt_len, batch_size, device):
        tupe_rel_pos_bias = self.get_tupe_rel_pos_bias(src_len, tgt_len, device) if self.use_tupe_rel_pos_bias else None

        weight_q = self.pos_tgt_ln(self.pos_tgt.weight[:tgt_len, :])
        weight_k = self.pos_src_ln(self.pos_src.weight[:src_len, :])
        pos_q = self.pos_q_linear(weight_q).view(tgt_len, self.seq_tr_nhead, -1).transpose(0, 1) * self.pos_scaling
        pos_k = self.pos_k_linear(weight_k).view(src_len, self.seq_tr_nhead, -1).transpose(0, 1)
        abs_pos_bias = torch.bmm(pos_q, pos_k.transpose(1, 2))
        if self.mask_src_cls_rel_pos:
            abs_pos_bias[:, :, 0] = self.cls_pos_embed(torch.tensor([0], device=device)).view(-1, 1)

        if tupe_rel_pos_bias is not None:
            abs_pos_bias += tupe_rel_pos_bias

        abs_pos_bias = abs_pos_bias.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(-1, tgt_len, src_len)

        return abs_pos_bias


    @custom_fwd
    def forward(self, tgt, src): # L,N,E
        src_len = src.size(0)
        tgt_len = tgt.size(0)
        batch_size = src.size(1)
        device = src.device
        return self.get_position_attn_bias(src_len, tgt_len, batch_size, device)

class KINMT_TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", bert=True):
        super(KINMT_TransformerEncoderLayer, self).__init__()
        self.d_model=d_model
        if bert:
            self.self_attn = MultiheadAttention(d_model, nhead, scale_factor=3, dropout=dropout)
            self.bert_attn_bias = MultiheadAttentionBias(d_model, nhead, scale_factor=3)
        else:
            self.self_attn = MultiheadAttention(d_model, nhead, scale_factor=2, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = FusedLayerNorm(d_model)
        self.norm2 = FusedLayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(KINMT_TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_attn_bias: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                src_bert: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # Implemented Pre-LN Architecture for better efficiency

        # Self-Attention
        src1 = self.norm1(src)
        if src_bert is not None:
            if src_attn_bias is None:
                src_attn_bias = self.bert_attn_bias(src1, src_bert, key_padding_mask=src_key_padding_mask, attn_mask = src_mask)
            else:
                src_attn_bias = src_attn_bias + self.bert_attn_bias(src1, src_bert, key_padding_mask=src_key_padding_mask, attn_mask = src_mask)
        src1 = self.self_attn(src1, src1, src1, attn_mask=src_mask, attn_bias=src_attn_bias, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src1)

        # FFN
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src

class KINMT_TransformerDecoderLayer(Module):

    def __init__(self, d_model, nhead,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu",
                 bert=True,
                 gpt=True):
        super(KINMT_TransformerDecoderLayer, self).__init__()
        self.d_model=d_model


        if bert:
            self.src_attn = MultiheadAttention(d_model, nhead, scale_factor=3, dropout=dropout)
            self.bert_attn_bias = MultiheadAttentionBias(d_model, nhead, scale_factor=3)
        else:
            self.src_attn = MultiheadAttention(d_model, nhead, scale_factor=2, dropout=dropout)
        if gpt:
            self.tgt_attn = MultiheadAttention(d_model, nhead, scale_factor=3, dropout=dropout)
            self.gpt_attn_bias = MultiheadAttentionBias(d_model, nhead, scale_factor=3)
        else:
            self.tgt_attn = MultiheadAttention(d_model, nhead, scale_factor=2, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = dropout
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = FusedLayerNorm(d_model)
        self.norm2 = FusedLayerNorm(d_model)
        self.norm3 = FusedLayerNorm(d_model)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(KINMT_TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, tgt: Tensor,
                src_mask: Optional[Tensor] = None,
                src_attn_bias: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                tgt_attn_bias: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_bert: Optional[Tensor] = None, tgt_gpt: Optional[Tensor] = None, decoding=False) -> Tuple[Tensor,Tensor]:

        # Self-Attention
        tgt1 = self.norm1(tgt)
        if tgt_gpt is not None:
            if tgt_attn_bias is None:
                tgt_attn_bias = self.gpt_attn_bias(tgt1, tgt_gpt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
            else:
                tgt_attn_bias = tgt_attn_bias + self.gpt_attn_bias(tgt1, tgt_gpt, attn_mask=tgt_mask, attn_bias=tgt_attn_bias, key_padding_mask=tgt_key_padding_mask)
        tgt1 = self.tgt_attn(tgt1, tgt1, tgt1, attn_mask=tgt_mask, attn_bias=tgt_attn_bias, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt +  F.dropout(tgt1, p=self.dropout, training=self.training)

        # Source Attention (i.e. Encoder-Decoder Attention)
        tgt1 = self.norm2(tgt)
        if src_bert is not None:
            if src_attn_bias is None:
                src_attn_bias = self.bert_attn_bias(tgt1, src_bert, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
            else:
                src_attn_bias = src_attn_bias + self.bert_attn_bias(tgt1, src_bert, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        tgt1, src_attn_weights = self.src_attn(tgt1, src, src, attn_mask=src_mask, attn_bias=src_attn_bias,
                                               key_padding_mask=src_key_padding_mask, need_weights=decoding)
        tgt = tgt +  F.dropout(tgt1, p=self.dropout, training=self.training)

        # FFN
        tgt1 = self.norm3(tgt)
        tgt1 = self.activation(self.linear1(tgt1))
        tgt1 = F.dropout(tgt1, p=self.dropout, training=self.training)
        tgt1 = self.linear2(tgt1)
        tgt = tgt + F.dropout(tgt1, p=self.dropout, training=self.training)

        return tgt, src_attn_weights

class KINMT_TransformerEncoder(Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer: KINMT_TransformerEncoderLayer, num_layers, norm=None):
        super(KINMT_TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_attn_bias: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                src_bert: Optional[Tensor] = None) -> Tensor:

        output = src

        for mod in self.layers:
            output = mod(output, src_mask=src_mask, src_attn_bias=src_attn_bias, src_key_padding_mask=src_key_padding_mask,
                         src_bert=src_bert)

        if self.norm is not None:
            output = self.norm(output)

        return output

class KINMT_TransformerDecoder(Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer: KINMT_TransformerDecoderLayer, num_layers, norm=None):
        super(KINMT_TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, tgt: Tensor,
                src_mask: Optional[Tensor] = None, src_attn_bias: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None, tgt_attn_bias: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                src_bert: Optional[Tensor] = None,
                tgt_gpt: Optional[Tensor] = None,
                decoding=False) -> Tuple[Tensor, Tensor]:
        output = tgt
        src_attn_weights = None
        for mod in self.layers:
            (output, src_attn_weights) = mod(src, output,
                                         src_mask=src_mask, src_attn_bias=src_attn_bias, src_key_padding_mask=src_key_padding_mask,
                                         tgt_mask=tgt_mask, tgt_attn_bias=tgt_attn_bias, tgt_key_padding_mask=tgt_key_padding_mask,
                                         src_bert=src_bert,
                                         tgt_gpt=tgt_gpt,
                                         decoding=decoding)

        if self.norm is not None:
            output = self.norm(output)

        return (output, src_attn_weights)

class KINMT_Transformer(Module):

    def __init__(self, encoder_layer: KINMT_TransformerEncoderLayer, decoder_layer: KINMT_TransformerDecoderLayer, num_encoder_layers, num_decoder_layers,
                 encoder_norm=None, decoder_norm=None,
                 bert=True, gpt=True, use_cross_pos_attn=True):
        super(KINMT_Transformer, self).__init__()
        self.encoder = KINMT_TransformerEncoder(encoder_layer, num_encoder_layers, norm=encoder_norm)
        self.decoder = KINMT_TransformerDecoder(decoder_layer, num_decoder_layers, norm=decoder_norm)
        self.use_cross_pos_attn = use_cross_pos_attn
        if self.use_cross_pos_attn:
            self.tgt_to_src_pos_attn = CrossAttentionPositionalEncoder(decoder_layer.tgt_attn.embed_dim,
                                                                       decoder_layer.tgt_attn.num_heads)
        if bert:
            self.bert_norm = FusedLayerNorm(decoder_layer.d_model)
        if gpt:
            self.gpt_norm = FusedLayerNorm(decoder_layer.d_model)
    def forward(self, src: Tensor, tgt: Tensor,
                src_mask: Optional[Tensor] = None, src_attn_bias: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None, tgt_attn_bias: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                src_bert: Optional[Tensor] = None,
                tgt_gpt: Optional[Tensor] = None) -> Tensor:
        if src_bert is not None:
            src_bert = self.bert_norm(src_bert)
        if tgt_gpt is not None:
            tgt_gpt = self.gpt_norm(tgt_gpt)
        src = self.encoder(src, src_mask=src_mask, src_attn_bias=src_attn_bias, src_key_padding_mask=src_key_padding_mask,
                src_bert=src_bert)
        if self.use_cross_pos_attn:
            src_attn_bias = self.tgt_to_src_pos_attn(tgt, src)
        else:
            src_attn_bias = None
        return self.decoder(src, tgt,
                           src_mask=src_mask, src_attn_bias=src_attn_bias, src_key_padding_mask=src_key_padding_mask,
                           tgt_mask=tgt_mask, tgt_attn_bias=tgt_attn_bias, tgt_key_padding_mask=tgt_key_padding_mask,
                           src_bert=src_bert,
                           tgt_gpt=tgt_gpt, decoding=False)[0]


    def encode(self, src: Tensor,
                src_mask: Optional[Tensor] = None, src_attn_bias: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                src_bert: Optional[Tensor] = None) -> Tensor:
        if src_bert is not None:
            src_bert = self.bert_norm(src_bert)
        return self.encoder(src, src_mask=src_mask, src_attn_bias=src_attn_bias, src_key_padding_mask=src_key_padding_mask, src_bert=src_bert)

    def decode(self, src: Tensor, tgt: Tensor,
                src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None, tgt_attn_bias: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                src_bert: Optional[Tensor] = None,
                tgt_gpt: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if self.use_cross_pos_attn:
            src_attn_bias = self.tgt_to_src_pos_attn(tgt, src)
        else:
            src_attn_bias = None
        if src_bert is not None:
            src_bert = self.bert_norm(src_bert)
        if tgt_gpt is not None:
            tgt_gpt = self.gpt_norm(tgt_gpt)
        return self.decoder(src, tgt,
                            src_mask=src_mask, src_attn_bias=src_attn_bias, src_key_padding_mask=src_key_padding_mask,
                            tgt_mask=tgt_mask, tgt_attn_bias=tgt_attn_bias, tgt_key_padding_mask=tgt_key_padding_mask,
                            src_bert=src_bert,
                            tgt_gpt=tgt_gpt,
                            decoding=True)

class ASR_Transformer(Module):

    def __init__(self, decoder_layer: KINMT_TransformerDecoderLayer, num_decoder_layers, decoder_norm=None,
                 gpt=True, use_cross_pos_attn=True,
                 max_seq_len = 1024,
                 tupe_rel_pos_bins = 256,
                 tupe_max_rel_pos = 256):
        super(ASR_Transformer, self).__init__()
        self.decoder = KINMT_TransformerDecoder(decoder_layer, num_decoder_layers, norm=decoder_norm)
        self.use_cross_pos_attn = use_cross_pos_attn
        if self.use_cross_pos_attn:
            self.tgt_to_src_pos_attn = CrossAttentionPositionalEncoder(decoder_layer.tgt_attn.embed_dim,
                                                                       decoder_layer.tgt_attn.num_heads,
                                                                       max_seq_len = max_seq_len,
                                                                       tupe_rel_pos_bins = tupe_rel_pos_bins,
                                                                       tupe_max_rel_pos = tupe_max_rel_pos)
        if gpt:
            self.gpt_norm = FusedLayerNorm(decoder_layer.d_model)
    def forward(self, src: Tensor, tgt: Tensor,
                src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None, tgt_attn_bias: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_gpt: Optional[Tensor] = None) -> Tensor:
        if tgt_gpt is not None:
            tgt_gpt = self.gpt_norm(tgt_gpt)
        if self.use_cross_pos_attn:
            src_attn_bias = self.tgt_to_src_pos_attn(tgt, src)
        else:
            src_attn_bias = None
        return self.decoder(src, tgt,
                           src_mask=src_mask, src_attn_bias=src_attn_bias, src_key_padding_mask=src_key_padding_mask,
                           tgt_mask=tgt_mask, tgt_attn_bias=tgt_attn_bias, tgt_key_padding_mask=tgt_key_padding_mask,
                           tgt_gpt=tgt_gpt, decoding=False)[0]

    def decode(self, src: Tensor, tgt: Tensor,
                src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None, tgt_attn_bias: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_gpt: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if self.use_cross_pos_attn:
            src_attn_bias = self.tgt_to_src_pos_attn(tgt, src)
        else:
            src_attn_bias = None
        if tgt_gpt is not None:
            tgt_gpt = self.gpt_norm(tgt_gpt)
        return self.decoder(src, tgt,
                            src_mask=src_mask, src_attn_bias=src_attn_bias, src_key_padding_mask=src_key_padding_mask,
                            tgt_mask=tgt_mask, tgt_attn_bias=tgt_attn_bias, tgt_key_padding_mask=tgt_key_padding_mask,
                            tgt_gpt=tgt_gpt,
                            decoding=True)
