# SPDX-License-Identifier: Apache-2.0
"""Compressed-tensors W4A16 (group, symmetric) helper for XPU.

This module converts a vLLM ``CompressedTensorsWNA16`` layer's parameters
from the ``pack_quantized`` storage layout into the layout expected by the
oneDNN INT4 grouped matmul primitive (``torch.ops._xpu_C.onednn_woq_int4_linear``).

It is the XPU counterpart of the CUDA/Marlin path used in vLLM for Kimi-K2.5
and other models quantized with compressed-tensors W4A16 (group, sym, int).

Supported in v1:
  - num_bits = 4
  - strategy = "group" with any group_size that oneDNN supports and that is a
    multiple of 32 (e.g. 32, 64, 128)
  - symmetric = True  → qzeros is None; oneDNN uses scalar zero-point 8
  - actorder = null   → no g_idx shuffling

Out of scope for v1 (raises NotImplementedError / TORCH_CHECK failure):
  - asymmetric (sym=False)
  - actorder != null
  - num_bits != 4

References:
  - ``vllm_xpu_kernels/quantization/_quantize_convert.py`` – GPTQUtils / AWQUtils
  - ``csrc/xpu/onednn/onednn_matmul.cpp`` – C++ kernel implementation
"""

from __future__ import annotations

import torch

from ._quantize_convert import GPTQUtils


def _unpack_compressed_tensors_int4(
    packed: torch.Tensor,
) -> torch.Tensor:
    """Unpack a compressed-tensors ``pack_quantized`` weight tensor.

    The compressed-tensors format packs INT4 nibbles along the K axis:
      packed[n, k//8] = w[n,k*8] | (w[n,k*8+1]<<4) | ... | (w[n,k*8+7]<<28)
    where values are unsigned in [0, 15].

    Args:
        packed: int32 tensor of shape (N, K//8).

    Returns:
        int8 tensor of shape (N, K) with values in [0, 15] (unsigned int4).
    """
    N, packed_k = packed.shape
    K = packed_k * 8
    # shifts: [0, 4, 8, 12, 16, 20, 24, 28]
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    # (N, K//8, 1) >> (1, 1, 8) → (N, K//8, 8)
    unpacked = torch.bitwise_right_shift(
        packed.unsqueeze(-1), shifts.view(1, 1, 8)
    )
    # Mask to 4-bit unsigned, then flatten K//8 * 8 → K
    return (unpacked.to(torch.int8) & 0x0F).view(N, K)


def _pack_int4_along_k(weight_int8: torch.Tensor) -> torch.Tensor:
    """Pack an int8 weight tensor along the K dimension into int32.

    Inverse of GPTQUtils.unpack_weight with the same nibble ordering
    (low nibble = lower K index):
      packed[k//8, n] = w[k*8,n] | (w[k*8+1,n]<<4) | ... | (w[k*8+7,n]<<28)

    Args:
        weight_int8: int8 tensor of shape (K, N), values in [0, 15].

    Returns:
        int32 tensor of shape (K//8, N).
    """
    K, N = weight_int8.shape
    assert K % 8 == 0, f"K ({K}) must be divisible by 8 for INT4 packing"
    # Use GPTQUtils.pack which already implements this packing convention
    gptq = GPTQUtils(bits=4, blocksize=K)
    return gptq.pack(weight_int8)


def repack_compressed_tensors_w4a16_to_onednn(
    layer: object,
    group_size: int,
    sym: bool,
) -> None:
    """Convert a vLLM CompressedTensorsWNA16 layer from ``pack_quantized``
    storage layout to the oneDNN INT4 matmul layout.

    This function is called from ``CompressedTensorsWNA16.process_weights_after_loading``
    on XPU (the companion vLLM-side PR). It modifies ``layer`` in-place by
    adding/replacing the following attributes:

    Input attributes on ``layer`` (set by vLLM's create_weights):
        weight_packed  : int32, shape (N, K//8), row-packed along K
        weight_scale   : shape (N, K//group_size), compute dtype

    Output attributes on ``layer``:
        qweight    : int32, shape (K//8, N), K-dimension contiguous
        scales     : shape (K//group_size, N), contiguous, same dtype as weight_scale
        qzeros     : None when sym=True
        group_size : int
        sym        : bool

    Args:
        layer:      vLLM CompressedTensorsWNA16 layer object.
        group_size: quantization group size (must divide K; must be ≥ 32).
        sym:        True for symmetric quantization (v1 only).

    Raises:
        AssertionError: if sym is False (asymmetric not yet supported in v1).
        NotImplementedError: if ``weight_g_idx`` is present (act-order not
            supported in v1).
    """
    if not sym:
        raise NotImplementedError(
            "repack_compressed_tensors_w4a16_to_onednn: "
            "asymmetric quantization (sym=False) is not yet implemented. "
            "TODO: add asymmetric support in a follow-up PR."
        )

    # Act-order (g_idx) is not supported in v1
    if hasattr(layer, "weight_g_idx") and layer.weight_g_idx is not None:
        raise NotImplementedError(
            "repack_compressed_tensors_w4a16_to_onednn: "
            "act-order (weight_g_idx / actorder != null) is not yet supported "
            "in v1. TODO: add act-order support in a follow-up PR."
        )

    weight_packed: torch.Tensor = layer.weight_packed
    weight_scale: torch.Tensor = layer.weight_scale

    # --- 1. Unpack weight_packed (N, K//8) → uint4 values (N, K) as int8 ---
    # Values are unsigned INT4 in [0, 15]; stored in int8 for computation.
    weight_unpacked = _unpack_compressed_tensors_int4(weight_packed)

    # --- 2. Transpose to (K, N) – switch to "weight is K×N" layout ---
    weight_kn = weight_unpacked.t().contiguous()  # (K, N)

    # --- 3. Re-pack along K → (K//8, N) int32 ---
    qweight = _pack_int4_along_k(weight_kn)  # (K//8, N)

    # --- 4. Make K-dimension contiguous (oneDNN requirement) ---
    # After this operation the shape stays (K//8, N) but strides become
    # (1, K//8), i.e. the K axis is the fast-running dimension.
    qweight_kcontig = qweight.t().contiguous().t()

    # --- 5. Scales: transpose (N, K//group_size) → (K//group_size, N) ---
    scales = weight_scale.t().contiguous()  # (K//group_size, N)

    # --- 6. Store results on layer ---
    layer.qweight = qweight_kcontig
    layer.scales = scales
    layer.qzeros = None  # symmetric: oneDNN uses scalar zero-point 8 internally
    layer.group_size = group_size
    layer.sym = True


def apply_w4a16_linear(
    layer: object,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the oneDNN INT4 W4A16 grouped matmul for a packed layer.

    Call this from ``CompressedTensorsWNA16.apply_weights`` on XPU after
    ``repack_compressed_tensors_w4a16_to_onednn`` has been called in
    ``process_weights_after_loading``.

    Args:
        layer:  vLLM CompressedTensorsWNA16 layer with ``qweight``, ``scales``,
                ``qzeros``, ``group_size``, and ``sym`` set.
        x:      activation tensor (M, K) or (B, M, K), dtype bf16 or fp16.
        bias:   optional bias tensor (N,), same dtype as x.

    Returns:
        Output tensor of shape (M, N) or (B, M, N), same dtype as x.
    """
    return torch.ops._xpu_C.onednn_woq_int4_linear(
        x,
        layer.qweight,
        layer.scales,
        None if layer.sym else layer.qzeros,
        layer.group_size,
        layer.sym,
        bias,
    )
