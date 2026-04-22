# SPDX-License-Identifier: Apache-2.0
"""
Helpers to convert compressed-tensors W4A16 (group, sym) weight layouts to
the format expected by the oneDNN-backed XPU ops:

  - ``torch.ops._xpu_C.int4_gemm_w4a16``     (per-Linear)
  - ``torch.ops._xpu_C.xpu_int4_woq_fused_moe``  (MoE)

Layout produced by repack helpers
----------------------------------
Per-Linear (``repack_compressed_tensors_w4a16_to_xpu``):
  layer.qweight  : int32, shape (K/8, N), strides (1, K/8) — oneDNN NT format
                   (K-major: 8 int4 packed along K per int32, contiguous)
  layer.scales   : same dtype as activation, shape (K/group_size, N)
  layer.qzeros   : int8, scalar [8] — symmetric zero point for oneDNN

MoE (``repack_compressed_tensors_w4a16_moe_to_xpu``):
  layer.w13_qweight : int32, shape (E, 2*I, K/8)
                      Stored exactly as vLLM's w13_weight_packed (K-major
                      per row); the C++ op reads each expert slice as
                      w13_qweight[e].t() → (K/8, 2*I) strides (1, K/8).
  layer.w13_scales  : same dtype as activation, shape (E, K/group_size, 2*I)
                      Transposed from vLLM's (E, 2*I, K/group_size).
  layer.w2_qweight  : int32, shape (E, K, I/8)  (K = hidden, I = inter)
  layer.w2_scales   : same dtype as activation, shape (E, I/group_size, K)
  layer.w13_qzeros  : None  (sym=True)
  layer.w2_qzeros   : None  (sym=True)
  layer.group_size  : int
  layer.sym         : True
"""

import torch


# ---------------------------------------------------------------------------
# Shared per-expert / per-linear weight transform
# ---------------------------------------------------------------------------

def _repack_one_expert_int4(weight_packed_2d: torch.Tensor) -> torch.Tensor:
    """Convert one 2-D packed-INT4 weight from compressed-tensors
    ``pack_quantized`` layout to oneDNN NT format.

    Args:
        weight_packed_2d: int32 tensor, shape ``(N, K/8)``, contiguous
            (K-major: 8 int4 values packed along K per int32 element).

    Returns:
        A non-contiguous *view* with shape ``(K/8, N)`` and strides
        ``(1, K/8)``.  This satisfies the oneDNN ``strides[dim-2] == 1``
        (NT-format) requirement without any data copy.
    """
    # (N, K/8) contiguous → strides (K/8, 1)
    # .t()                → (K/8, N)  strides (1, K/8)  ← NT format ✓
    return weight_packed_2d.t()


# ---------------------------------------------------------------------------
# Per-Linear repack
# ---------------------------------------------------------------------------

def repack_compressed_tensors_w4a16_to_xpu(layer, group_size: int,
                                           sym: bool) -> None:
    """Convert a single Linear layer's compressed-tensors W4A16 params to the
    layout expected by ``torch.ops._xpu_C.int4_gemm_w4a16``.

    Modifies *layer* in-place, setting:
      - ``layer.qweight`` : int32, shape (K/8, N), strides (1, K/8)
      - ``layer.scales``  : contiguous, shape (K/group_size, N)
      - ``layer.qzeros``  : int8 scalar [8] for sym=True

    Args:
        layer: vLLM Linear layer with ``.qweight`` (N, K/8) int32 and
               ``.scales`` (N, K/group_size).
        group_size: weight quantization group size.
        sym: must be True for v1.

    Raises:
        NotImplementedError: if sym is False (asymmetric not supported in v1).
    """
    if not sym:
        raise NotImplementedError(
            "repack_compressed_tensors_w4a16_to_xpu: "
            "asymmetric (sym=False) is not supported in v1")

    # weight_packed: (N, K/8) → oneDNN NT: (K/8, N) strides (1, K/8)
    w_nt = _repack_one_expert_int4(layer.qweight)
    # Make the NT view concrete (copy into correct memory layout).
    w_nt_contig = w_nt.transpose(0, 1).contiguous().transpose(0, 1)
    layer.qweight.as_strided_(w_nt_contig.shape, w_nt_contig.stride())
    layer.qweight.copy_(w_nt_contig)

    # scales: (N, K/group_size) → (K/group_size, N) contiguous
    layer.scales.data = layer.scales.t().contiguous()

    # symmetric zero point: scalar 8
    layer.qzeros = torch.tensor(
        [8], dtype=torch.int8, device=layer.qweight.device)


# ---------------------------------------------------------------------------
# MoE repack
# ---------------------------------------------------------------------------

def repack_compressed_tensors_w4a16_moe_to_xpu(layer, group_size: int,
                                               sym: bool) -> None:
    """Convert vLLM CompressedTensorsWNA16MoE-style layer params from the
    compressed-tensors ``pack_quantized`` MoE layout into the layout expected
    by ``torch.ops._xpu_C.xpu_int4_woq_fused_moe``.

    Inputs on ``layer`` (per non-Marlin WNA16 MoE ``create_weights`` in vLLM):
      - ``w13_weight_packed``: int32, shape ``(E, 2*I, K/8)``
        Row-packed along K (8 int4 per int32).
      - ``w13_weight_scale``:  shape ``(E, 2*I, K/group_size)``
      - ``w2_weight_packed``:  int32, shape ``(E, K_hidden, I/8)``
      - ``w2_weight_scale``:   shape ``(E, K_hidden, I/group_size)``
      - (no zero-points when sym=True)

    Produces on ``layer``:
      - ``w13_qweight``: int32, shape ``(E, 2*I, K/8)`` — kept as-is; the C++
        op reads each expert as ``w13_qweight[e].t()`` to get NT format
        ``(K/8, 2*I)`` strides ``(1, K/8)``.
      - ``w13_scales``:  shape ``(E, K/group_size, 2*I)`` — axes 1 and 2
        transposed from vLLM's ``(E, 2*I, K/group_size)`` and made contiguous.
      - ``w2_qweight``:  int32, shape ``(E, K_hidden, I/8)`` — kept as-is.
      - ``w2_scales``:   shape ``(E, I/group_size, K_hidden)`` — transposed
        from vLLM's ``(E, K_hidden, I/group_size)`` and made contiguous.
      - ``w13_qzeros``:  None
      - ``w2_qzeros``:   None
      - ``group_size``:  int attribute
      - ``sym``:         True

    Args:
        layer: vLLM MoE layer object with the above ``*_weight_packed``/
               ``*_weight_scale`` attributes.
        group_size: quantization group size (must divide K and I).
        sym: must be True for v1.

    Raises:
        AssertionError: if sym is False.
    """
    assert sym, (
        "repack_compressed_tensors_w4a16_moe_to_xpu: "
        "v1 only supports symmetric W4A16 MoE (sym=True)")

    # --- w13 ---
    # w13_weight_packed: (E, 2*I, K/8) — keep shape, copy to contiguous store
    layer.w13_qweight = layer.w13_weight_packed.contiguous()

    # w13_weight_scale: (E, 2*I, K/group_size)
    # → transpose dims 1,2 → (E, K/group_size, 2*I) contiguous
    layer.w13_scales = layer.w13_weight_scale.permute(0, 2, 1).contiguous()

    # --- w2 ---
    # w2_weight_packed: (E, K_hidden, I/8) — keep shape
    layer.w2_qweight = layer.w2_weight_packed.contiguous()

    # w2_weight_scale: (E, K_hidden, I/group_size)
    # → transpose dims 1,2 → (E, I/group_size, K_hidden) contiguous
    layer.w2_scales = layer.w2_weight_scale.permute(0, 2, 1).contiguous()

    # --- zero points and metadata ---
    layer.w13_qzeros = None
    layer.w2_qzeros = None
    layer.group_size = group_size
    layer.sym = sym


# ---------------------------------------------------------------------------
# Apply wrapper
# ---------------------------------------------------------------------------

def apply_w4a16_moe(layer, x: torch.Tensor, topk_ids: torch.Tensor,
                    topk_weights: torch.Tensor) -> torch.Tensor:
    """Run the XPU INT4 fused MoE FFN.

    Thin wrapper around ``torch.ops._xpu_C.xpu_int4_woq_fused_moe``.
    Expects ``layer`` to have been prepared by
    ``repack_compressed_tensors_w4a16_moe_to_xpu``.

    Args:
        layer: layer object with ``w13_qweight``, ``w13_scales``,
               ``w13_qzeros``, ``w2_qweight``, ``w2_scales``,
               ``w2_qzeros``, ``group_size``, ``sym``.
        x: activation tensor ``[M, K]``, bf16 or fp16.
        topk_ids: expert indices ``[M, top_k]``, int32 or int64.
        topk_weights: expert weights ``[M, top_k]``, float32.

    Returns:
        Output tensor ``[M, K]``, same dtype as ``x``.
    """
    return torch.ops._xpu_C.xpu_int4_woq_fused_moe(
        x,
        topk_ids,
        topk_weights,
        layer.w13_qweight,
        layer.w13_scales,
        None if layer.sym else layer.w13_qzeros,
        layer.w2_qweight,
        layer.w2_scales,
        None if layer.sym else layer.w2_qzeros,
        layer.group_size,
        layer.sym,
    )
