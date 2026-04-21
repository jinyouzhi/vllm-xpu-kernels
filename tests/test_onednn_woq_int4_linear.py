# SPDX-License-Identifier: Apache-2.0
"""Tests for the oneDNN INT4 W4A16 grouped matmul op
(torch.ops._xpu_C.onednn_woq_int4_linear) and the compressed-tensors
repacking helper (repack_compressed_tensors_w4a16_to_onednn).

The test builds a reference quantised GEMM via:
  1. Symmetric group quantisation of a random fp weight → int4 (unsigned [0,15])
  2. Pack into compressed-tensors ``pack_quantized`` layout: (N, K//8) int32
  3. Call repack_compressed_tensors_w4a16_to_onednn to convert to oneDNN layout
  4. Run torch.ops._xpu_C.onednn_woq_int4_linear
  5. Compare against x @ dequant(W).T  (loose tolerance due to quantisation error)
"""

import types
from typing import Optional

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="XPU device not available – skipping onednn_woq_int4_linear tests",
)

MINI_PYTEST_PARAMS: dict = {
    "test_onednn_woq_int4_linear": {
        "dtype": [torch.bfloat16],
        "group_size": [128],
        "with_bias": [False],
        "m": [1],
        "nk": [(4096, 4096)],
    },
    "test_repack_and_apply_w4a16": {
        "dtype": [torch.bfloat16],
        "group_size": [128],
        "nk": [(4096, 4096)],
    },
}


# ---------------------------------------------------------------------------
# Quantisation helpers
# ---------------------------------------------------------------------------

def _group_quantize_sym(
    w: torch.Tensor,
    group_size: int,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric group quantisation of weight W (N, K) to INT4 (unsigned).

    Returns:
        qw   : int8, shape (N, K), values in [0, 15] (unsigned int4, ZP=8)
        scales: compute_dtype, shape (N, K // group_size)
    """
    N, K = w.shape
    assert K % group_size == 0
    num_groups = K // group_size

    w_grouped = w.reshape(N, num_groups, group_size).float()
    max_val = w_grouped.abs().amax(dim=-1)  # (N, num_groups)

    # scale so that max |w| maps to 7 (int4 range is [-8, 7]; use [-7, 7])
    scales = (max_val / 7.0).clamp(min=1e-6).to(compute_dtype)

    # Quantise: round(w / scale) in [-7, 7], then shift to unsigned [1, 15]
    # (strictly, clamp to [-8, 7] to use the full int4 range)
    qw = (w_grouped / scales.float().unsqueeze(-1)).round().clamp(-8, 7)
    qw = (qw + 8).to(torch.int8)  # unsigned [0, 16] → clamp to [0, 15]
    qw = qw.view(N, K)
    return qw, scales


def _pack_compressed_tensors_int4(qw_int8: torch.Tensor) -> torch.Tensor:
    """Pack int8 (N, K) → int32 (N, K//8) in compressed-tensors format.

    Packing: packed[n, k//8] = qw[n,k*8] | (qw[n,k*8+1]<<4) | ...
    Low nibble = lower K index (standard INT4 packing).
    """
    N, K = qw_int8.shape
    assert K % 8 == 0
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qw_int8.device)
    packed_view = qw_int8.view(N, K // 8, 8)  # (N, K//8, 8)
    packed = torch.bitwise_left_shift(
        packed_view.to(torch.int32) & 0x0F,
        shifts.view(1, 1, 8),
    ).sum(dim=-1)
    return packed.to(torch.int32)


def _dequantize_sym(
    qw_int8: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dequantise: (qw - 8) * scale, result in scales.dtype, shape (N, K)."""
    N, K = qw_int8.shape
    num_groups = K // group_size
    # scales: (N, num_groups) → broadcast to (N, K)
    scales_expanded = scales.unsqueeze(-1).expand(N, num_groups, group_size)
    scales_expanded = scales_expanded.reshape(N, K)
    return (qw_int8.float() - 8) * scales_expanded.float()


# ---------------------------------------------------------------------------
# Helpers to build oneDNN-layout tensors directly (without the Python helper)
# ---------------------------------------------------------------------------

def _make_onednn_layout(
    qw_int8: torch.Tensor,
    scales: torch.Tensor,
    device: str,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (qweight, scales) in oneDNN K-contiguous layout from (N, K) int8.

    Returns:
        qweight: int32, shape (K//8, N), K-dim contiguous (stride[0]==1)
        scales:  compute_dtype, shape (K//group_size, N)
    """
    from vllm_xpu_kernels.quantization._quantize_convert import GPTQUtils

    N, K = qw_int8.shape
    # Transpose to (K, N), pack along K → (K//8, N)
    qw_kn = qw_int8.t().contiguous()  # (K, N)
    gptq = GPTQUtils(bits=4, blocksize=K)
    qweight = gptq.pack(qw_kn)  # (K//8, N)

    # Make K-dimension contiguous (strides: (1, K//8))
    qweight_kcontig = qweight.t().contiguous().t()

    # Scales: (N, K//group_size) → (K//group_size, N)
    scales_t = scales.t().contiguous()

    return qweight_kcontig.to(device), scales_t.to(device)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("m", [1, 16, 1024])
@pytest.mark.parametrize(
    "nk",
    [(4096, 4096), (11008, 4096)],
)
def test_onednn_woq_int4_linear(dtype, group_size, with_bias, m, nk):
    """Direct test of the C++ kernel with oneDNN-layout inputs."""
    import vllm_xpu_kernels._xpu_C  # noqa: F401 — registers _xpu_C ops

    N, K = nk
    if K % group_size != 0:
        pytest.skip(f"K={K} not divisible by group_size={group_size}")

    torch.manual_seed(42)
    device = "xpu"

    # Reference float weight
    w_fp = torch.randn(N, K, dtype=torch.float32)

    # Quantise
    qw_int8, scales_cpu = _group_quantize_sym(w_fp, group_size, dtype)

    # Reference dequant weight
    w_ref = _dequantize_sym(qw_int8, scales_cpu, group_size).to(dtype)  # (N, K)

    # Build oneDNN layout on XPU
    qweight_xpu, scales_xpu = _make_onednn_layout(
        qw_int8, scales_cpu, device, dtype
    )

    # Input activation
    x = torch.randn(m, K, dtype=dtype, device=device)

    # Optional bias
    bias: Optional[torch.Tensor] = None
    if with_bias:
        bias = torch.randn(N, dtype=dtype, device=device)

    # Run kernel
    out_int4 = torch.ops._xpu_C.onednn_woq_int4_linear(
        x, qweight_xpu, scales_xpu, None, group_size, True, bias
    )

    # Reference: x @ W.T + bias
    out_ref = x.float().cpu() @ w_ref.float().cpu().t()
    if with_bias:
        out_ref = out_ref + bias.float().cpu()

    assert out_int4.shape == (m, N), (
        f"output shape mismatch: {out_int4.shape} vs expected ({m}, {N})"
    )

    max_abs_err = (out_int4.float().cpu() - out_ref.float()).abs().max().item()
    max_rel_err = (
        (out_int4.float().cpu() - out_ref.float()).abs()
        / (out_ref.float().abs() + 1e-6)
    ).max().item()

    torch.testing.assert_close(
        out_int4.float().cpu(),
        out_ref.float(),
        atol=1e-2,
        rtol=1e-2,
        msg=(
            f"dtype={dtype}, group_size={group_size}, m={m}, N={N}, K={K}, "
            f"with_bias={with_bias}: "
            f"max_abs_err={max_abs_err:.4f}, max_rel_err={max_rel_err:.4f}"
        ),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize(
    "nk",
    [(4096, 4096), (11008, 4096)],
)
def test_repack_and_apply_w4a16(dtype, group_size, nk):
    """Test the full Python helper path: pack → repack → apply."""
    import vllm_xpu_kernels._xpu_C  # noqa: F401

    from vllm_xpu_kernels.quantization.compressed_tensors_wna16 import (
        apply_w4a16_linear,
        repack_compressed_tensors_w4a16_to_onednn,
    )

    N, K = nk
    if K % group_size != 0:
        pytest.skip(f"K={K} not divisible by group_size={group_size}")

    torch.manual_seed(7)
    device = "xpu"
    m = 16

    # Reference weight
    w_fp = torch.randn(N, K, dtype=torch.float32)

    # Group-quantise
    qw_int8, scales_cpu = _group_quantize_sym(w_fp, group_size, dtype)
    w_ref = _dequantize_sym(qw_int8, scales_cpu, group_size).to(dtype)

    # Pack to compressed-tensors format on XPU
    weight_packed_xpu = _pack_compressed_tensors_int4(qw_int8).to(device)
    weight_scale_xpu = scales_cpu.to(device)

    # Build a fake layer object
    layer = types.SimpleNamespace(
        weight_packed=weight_packed_xpu,
        weight_scale=weight_scale_xpu,
    )

    # Repack to oneDNN layout
    repack_compressed_tensors_w4a16_to_onednn(layer, group_size, sym=True)

    # Verify output attributes
    assert layer.qweight is not None
    assert layer.scales is not None
    assert layer.qzeros is None
    assert layer.sym is True
    assert layer.group_size == group_size

    # Check shapes
    assert layer.qweight.shape == (K // 8, N), (
        f"qweight shape: {layer.qweight.shape}"
    )
    assert layer.scales.shape == (K // group_size, N), (
        f"scales shape: {layer.scales.shape}"
    )
    # K-contiguous: strides()[0] == 1
    assert layer.qweight.stride(0) == 1, (
        f"qweight not K-contiguous: strides={layer.qweight.stride()}"
    )

    # Run kernel via Python helper
    x = torch.randn(m, K, dtype=dtype, device=device)
    out = apply_w4a16_linear(layer, x, bias=None)

    # Reference
    out_ref = x.float().cpu() @ w_ref.float().cpu().t()

    max_abs_err = (out.float().cpu() - out_ref.float()).abs().max().item()
    max_rel_err = (
        (out.float().cpu() - out_ref.float()).abs()
        / (out_ref.float().abs() + 1e-6)
    ).max().item()

    torch.testing.assert_close(
        out.float().cpu(),
        out_ref.float(),
        atol=1e-2,
        rtol=1e-2,
        msg=(
            f"dtype={dtype}, group_size={group_size}, N={N}, K={K}: "
            f"max_abs_err={max_abs_err:.4f}, max_rel_err={max_rel_err:.4f}"
        ),
    )


def test_unsupported_asymmetric_raises():
    """v1: asymmetric (sym=False) must raise NotImplementedError."""
    from vllm_xpu_kernels.quantization.compressed_tensors_wna16 import (
        repack_compressed_tensors_w4a16_to_onednn,
    )

    layer = types.SimpleNamespace(weight_packed=None, weight_scale=None)
    with pytest.raises(NotImplementedError, match="asymmetric"):
        repack_compressed_tensors_w4a16_to_onednn(layer, group_size=128, sym=False)


def test_unsupported_actorder_raises():
    """v1: act-order (weight_g_idx) must raise NotImplementedError."""
    from vllm_xpu_kernels.quantization.compressed_tensors_wna16 import (
        repack_compressed_tensors_w4a16_to_onednn,
    )

    layer = types.SimpleNamespace(
        weight_packed=None,
        weight_scale=None,
        weight_g_idx=torch.zeros(4, dtype=torch.int32),
    )
    with pytest.raises(NotImplementedError, match="act-order"):
        repack_compressed_tensors_w4a16_to_onednn(layer, group_size=128, sym=True)


@pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="XPU device not available",
)
def test_kernel_sym_false_raises():
    """v1: calling the C++ kernel with sym=False must raise a TORCH_CHECK."""
    import vllm_xpu_kernels._xpu_C  # noqa: F401

    N, K, M, group_size = 64, 128, 4, 32
    x = torch.randn(M, K, dtype=torch.bfloat16, device="xpu")
    qw = torch.zeros(K // 8, N, dtype=torch.int32, device="xpu")
    # make K-contiguous
    qw = qw.t().contiguous().t()
    scales = torch.ones(K // group_size, N, dtype=torch.bfloat16, device="xpu")
    fake_qzeros = torch.zeros(
        K // group_size, N // 8, dtype=torch.int32, device="xpu"
    )

    with pytest.raises(RuntimeError, match="sym=True"):
        torch.ops._xpu_C.onednn_woq_int4_linear(
            x, qw, scales, fake_qzeros, group_size, False, None
        )
