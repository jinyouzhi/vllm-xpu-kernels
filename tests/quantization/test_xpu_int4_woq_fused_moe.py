# SPDX-License-Identifier: Apache-2.0
"""
Tests for xpu_int4_woq_fused_moe — INT4 W4A16 grouped-matmul fused MoE FFN
on Intel XPU via oneDNN.

The test builds a small MoE reference in pure PyTorch (dequant → bf16 GEMM
→ SwiGLU → topk-reduce), packs the weights into the compressed-tensors
pack_quantized MoE layout, repacks via
``repack_compressed_tensors_w4a16_moe_to_xpu``, and compares
``apply_w4a16_moe`` against the reference with loose tolerances.
"""

import pytest
import torch

# Skip all tests when XPU is not available.
XPU_AVAILABLE = torch.xpu.is_available() if hasattr(torch, "xpu") else False
pytestmark = pytest.mark.skipif(
    not XPU_AVAILABLE, reason="Intel XPU device not available")

# Import the module under test only when XPU is available so that import
# errors (missing extension library) are handled gracefully.
if XPU_AVAILABLE:
    import vllm_xpu_kernels._xpu_C  # noqa: F401 — triggers op registration
    from vllm_xpu_kernels.quantization.compressed_tensors_wna16 import (
        apply_w4a16_moe,
        repack_compressed_tensors_w4a16_moe_to_xpu,
    )
    from vllm_xpu_kernels.quantization._quantize_convert import GPTQUtils


# ---------------------------------------------------------------------------
# Quantisation helpers
# ---------------------------------------------------------------------------

def _sym_quantize_int4(w: torch.Tensor, group_size: int):
    """Symmetric group-quantize a float weight to INT4 [0, 15] range.

    Args:
        w: float tensor of shape ``(N, K)`` (rows = output channels).
        group_size: number of K elements per quantization group.

    Returns:
        qweight_packed: int32, shape ``(N, K//8)`` — K-major pack_quantized.
        scales:         float, shape ``(N, K//group_size)``.
    """
    N, K = w.shape
    assert K % group_size == 0
    n_groups = K // group_size

    w_grouped = w.view(N, n_groups, group_size).float()
    abs_max = w_grouped.abs().amax(dim=-1, keepdim=True)  # (N, n_g, 1)
    scales = abs_max.clamp(min=1e-8) / 7.0                # (N, n_g, 1)

    q = (w_grouped / scales).round().clamp(-8, 7)         # signed [-8, 7]
    q_u4 = (q + 8).to(torch.uint8)                        # unsigned [0, 15]
    scales_2d = scales.squeeze(-1)                         # (N, n_g)

    # Pack 8 consecutive K values into one int32 (low-nibble first).
    q_k = q_u4.reshape(N, K)                              # (N, K)
    q_packed = _pack_int4_row(q_k)                        # (N, K//8)

    return q_packed, scales_2d


def _pack_int4_row(q_int8: torch.Tensor) -> torch.Tensor:
    """Pack a ``(N, K)`` uint8 tensor (values 0–15) into ``(N, K//8)`` int32.

    8 consecutive K values are packed into one int32: value at K=0 goes into
    bits [3:0], K=1 into bits [7:4], etc.  This matches the compressed-tensors
    ``pack_quantized`` row-packing convention.
    """
    N, K = q_int8.shape
    assert K % 8 == 0
    q = q_int8.to(torch.int32).reshape(N, K // 8, 8)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=q.device)
    packed = (q << shifts[None, None, :]).sum(dim=-1)  # (N, K//8)
    return packed.to(torch.int32)


def _dequantize_int4_packed(qweight_packed: torch.Tensor,
                             scales: torch.Tensor,
                             group_size: int) -> torch.Tensor:
    """Dequantize from pack_quantized (N, K/8) format back to float (N, K).

    Args:
        qweight_packed: int32, shape (N, K//8).
        scales:         float, shape (N, n_groups).
        group_size:     K elements per group.

    Returns:
        Float tensor (N, K).
    """
    N, Kpack = qweight_packed.shape
    K = Kpack * 8
    n_groups = K // group_size

    # Unpack
    shifts = torch.arange(0, 32, 4, dtype=torch.int32,
                          device=qweight_packed.device)
    q = qweight_packed.unsqueeze(-1) >> shifts[None, None, :]  # (N, K//8, 8)
    q = (q & 0xF).to(torch.int32).reshape(N, K)               # (N, K), [0,15]

    # Dequantize: (q - 8) * scale
    g_idx = torch.arange(K, device=q.device) // group_size     # (K,)
    q_f = (q - 8).float()                                      # signed
    s = scales.float()                                         # (N, n_groups)
    return (q_f * s[:, g_idx]).to(scales.dtype)                # (N, K)


# ---------------------------------------------------------------------------
# Reference MoE
# ---------------------------------------------------------------------------

def _ref_moe(x: torch.Tensor, w13_ref: torch.Tensor, w2_ref: torch.Tensor,
             topk_ids: torch.Tensor,
             topk_weights: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch reference MoE FFN (bf16/fp16 GEMM + SwiGLU + topk reduce).

    Args:
        x:            [M, K], same dtype as w13_ref.
        w13_ref:      [E, 2*I, K] dequantized gate+up weights.
        w2_ref:       [E, K_out, I] dequantized down-projection weights
                      (K_out = K = hidden size).
        topk_ids:     [M, top_k], int64.
        topk_weights: [M, top_k], float32.

    Returns:
        output: [M, K], same dtype as x.
    """
    M, K = x.shape
    E = w13_ref.shape[0]
    I_mid = w13_ref.shape[1] // 2
    top_k = topk_ids.shape[1]

    output = torch.zeros_like(x)
    x_f = x.float()

    for e in range(E):
        mask = (topk_ids == e).any(dim=1)  # [M]
        token_idx = mask.nonzero(as_tuple=True)[0]
        if token_idx.numel() == 0:
            continue

        x_e = x_f[token_idx]             # [Me, K]
        w13_e = w13_ref[e].float()        # [2*I, K]
        w2_e = w2_ref[e].float()          # [K_out, I]

        # GEMM1: x_e @ w13_e.T → [Me, 2*I]
        out13 = x_e @ w13_e.T             # [Me, 2*I]

        # SwiGLU
        gate = torch.nn.functional.silu(out13[:, :I_mid])  # [Me, I]
        up = out13[:, I_mid:]                               # [Me, I]
        silu_mul = gate * up                                # [Me, I]

        # GEMM2: silu_mul @ w2_e.T → [Me, K_out]
        out2 = silu_mul @ w2_e.T          # [Me, K_out]

        # Weighted accumulation
        expert_w_mask = (topk_ids[token_idx] == e).float()  # [Me, top_k]
        tw = topk_weights[token_idx].float()                 # [Me, top_k]
        expert_w = (expert_w_mask * tw).sum(1, keepdim=True)  # [Me, 1]

        output[token_idx] += (expert_w * out2).to(x.dtype)

    return output


# ---------------------------------------------------------------------------
# Fake layer object (mimics vLLM layer attributes)
# ---------------------------------------------------------------------------

class _FakeLayer:
    pass


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

DTYPES = [torch.bfloat16, torch.float16]
GROUP_SIZES = [32, 128]
M_VALUES = [1, 16, 256]

E = 8
TOP_K = 2
K_HIDDEN = 512    # hidden size (kept small for CI speed)
I_INTER = 256     # intermediate size (SwiGLU input)


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M", M_VALUES)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_xpu_int4_woq_fused_moe(M: int, group_size: int, dtype: torch.dtype):
    """Compare xpu_int4_woq_fused_moe against a pure-PyTorch reference.

    The test:
      1. Generates random bf16/fp16 expert weights and group-quantizes them
         symmetrically to INT4.
      2. Packs into compressed-tensors pack_quantized MoE layout.
      3. Calls repack_compressed_tensors_w4a16_moe_to_xpu.
      4. Runs apply_w4a16_moe and a float reference.
      5. Asserts numerical closeness.
    """
    assert K_HIDDEN % group_size == 0, "K_HIDDEN must be divisible by group_size"
    assert I_INTER % group_size == 0, "I_INTER must be divisible by group_size"

    torch.manual_seed(42)
    device = torch.device("xpu")

    # ---- random activations and routing ----
    x = torch.randn(M, K_HIDDEN, dtype=dtype, device=device)
    # Random top-k routing: each token gets top_k distinct experts.
    topk_ids = torch.zeros(M, TOP_K, dtype=torch.long, device=device)
    for m in range(M):
        topk_ids[m] = torch.randperm(E, device=device)[:TOP_K]
    topk_weights_raw = torch.rand(M, TOP_K, dtype=torch.float32, device=device)
    topk_weights = topk_weights_raw / topk_weights_raw.sum(1, keepdim=True)

    # ---- generate and quantize w13 ----
    # w13: (E, 2*I, K)  — gate+up projection weight
    w13_dense = torch.randn(E, 2 * I_INTER, K_HIDDEN, dtype=dtype, device=device)
    w13_packed_list = []
    w13_scale_list = []
    w13_ref_list = []
    for e in range(E):
        qp, sc = _sym_quantize_int4(w13_dense[e], group_size)
        # qp: (2*I, K//8), sc: (2*I, K//group_size)
        w13_packed_list.append(qp)
        w13_scale_list.append(sc)
        w13_ref_list.append(_dequantize_int4_packed(qp, sc.to(dtype), group_size))

    w13_weight_packed = torch.stack(w13_packed_list)   # (E, 2*I, K//8)
    w13_weight_scale = torch.stack(w13_scale_list).to(dtype)  # (E, 2*I, K//gs)
    w13_ref = torch.stack(w13_ref_list)                # (E, 2*I, K)

    # ---- generate and quantize w2 ----
    # w2: (E, K_hidden, I_inter)  — down projection weight
    w2_dense = torch.randn(E, K_HIDDEN, I_INTER, dtype=dtype, device=device)
    w2_packed_list = []
    w2_scale_list = []
    w2_ref_list = []
    for e in range(E):
        qp, sc = _sym_quantize_int4(w2_dense[e], group_size)
        # qp: (K, I//8), sc: (K, I//group_size)
        w2_packed_list.append(qp)
        w2_scale_list.append(sc)
        w2_ref_list.append(_dequantize_int4_packed(qp, sc.to(dtype), group_size))

    w2_weight_packed = torch.stack(w2_packed_list)     # (E, K, I//8)
    w2_weight_scale = torch.stack(w2_scale_list).to(dtype)   # (E, K, I//gs)
    w2_ref = torch.stack(w2_ref_list)                  # (E, K, I)

    # ---- reference computation (CPU float32 for accuracy) ----
    x_cpu = x.cpu()
    w13_ref_cpu = w13_ref.cpu()
    w2_ref_cpu = w2_ref.cpu()
    topk_ids_cpu = topk_ids.cpu()
    topk_weights_cpu = topk_weights.cpu()

    ref_out = _ref_moe(
        x_cpu.to(torch.float32),
        w13_ref_cpu.to(torch.float32),
        w2_ref_cpu.to(torch.float32),
        topk_ids_cpu,
        topk_weights_cpu,
    ).to(dtype)

    # ---- repack into XPU format ----
    layer = _FakeLayer()
    layer.w13_weight_packed = w13_weight_packed
    layer.w13_weight_scale = w13_weight_scale
    layer.w2_weight_packed = w2_weight_packed
    layer.w2_weight_scale = w2_weight_scale

    repack_compressed_tensors_w4a16_moe_to_xpu(layer, group_size, sym=True)

    # ---- run XPU kernel ----
    xpu_out = apply_w4a16_moe(layer, x, topk_ids, topk_weights)

    # ---- compare ----
    xpu_out_cpu = xpu_out.cpu().to(torch.float32)
    ref_out_cpu = ref_out.cpu().to(torch.float32)

    max_abs = (xpu_out_cpu - ref_out_cpu).abs().max().item()
    max_ref = ref_out_cpu.abs().max().item()
    max_rel = max_abs / (max_ref + 1e-6)

    # MoE accumulates error from two GEMMs; tolerances are loose.
    atol = 2e-2
    rtol = 2e-2
    print(
        f"[M={M}, gs={group_size}, dtype={dtype}] "
        f"max_abs={max_abs:.4f}, max_ref={max_ref:.4f}, max_rel={max_rel:.4f}"
    )
    torch.testing.assert_close(
        xpu_out_cpu,
        ref_out_cpu,
        atol=atol,
        rtol=rtol,
        msg=(f"xpu_int4_woq_fused_moe output mismatch: "
             f"max_abs={max_abs:.4f} (tol {atol}), "
             f"max_rel={max_rel:.4f} (tol {rtol})"),
    )


# ---------------------------------------------------------------------------
# Error-path tests (v1 unsupported features)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not XPU_AVAILABLE, reason="XPU not available")
def test_asymmetric_raises():
    """xpu_int4_woq_fused_moe must raise on sym=False."""
    device = torch.device("xpu")
    E_, I_, K_, gs = 2, 32, 64, 32
    x = torch.randn(2, K_, dtype=torch.bfloat16, device=device)
    topk_ids = torch.zeros(2, 1, dtype=torch.long, device=device)
    topk_weights = torch.ones(2, 1, dtype=torch.float32, device=device)
    w13q = torch.zeros(E_, 2 * I_, K_ // 8, dtype=torch.int32, device=device)
    w13s = torch.ones(E_, K_ // gs, 2 * I_, dtype=torch.bfloat16, device=device)
    w2q = torch.zeros(E_, K_, I_ // 8, dtype=torch.int32, device=device)
    w2s = torch.ones(E_, I_ // gs, K_, dtype=torch.bfloat16, device=device)

    with pytest.raises(RuntimeError, match="sym=true"):
        torch.ops._xpu_C.xpu_int4_woq_fused_moe(
            x, topk_ids, topk_weights,
            w13q, w13s, None,
            w2q, w2s, None,
            gs, False)  # sym=False → should raise


@pytest.mark.skipif(not XPU_AVAILABLE, reason="XPU not available")
def test_qzeros_not_none_raises():
    """xpu_int4_woq_fused_moe must raise when qzeros are provided."""
    device = torch.device("xpu")
    E_, I_, K_, gs = 2, 32, 64, 32
    x = torch.randn(2, K_, dtype=torch.bfloat16, device=device)
    topk_ids = torch.zeros(2, 1, dtype=torch.long, device=device)
    topk_weights = torch.ones(2, 1, dtype=torch.float32, device=device)
    w13q = torch.zeros(E_, 2 * I_, K_ // 8, dtype=torch.int32, device=device)
    w13s = torch.ones(E_, K_ // gs, 2 * I_, dtype=torch.bfloat16, device=device)
    w2q = torch.zeros(E_, K_, I_ // 8, dtype=torch.int32, device=device)
    w2s = torch.ones(E_, I_ // gs, K_, dtype=torch.bfloat16, device=device)
    fake_zp = torch.zeros(1, dtype=torch.int8, device=device)

    with pytest.raises(RuntimeError, match="w13_qzeros must be None"):
        torch.ops._xpu_C.xpu_int4_woq_fused_moe(
            x, topk_ids, topk_weights,
            w13q, w13s, fake_zp,
            w2q, w2s, None,
            gs, True)


@pytest.mark.skipif(not XPU_AVAILABLE, reason="XPU not available")
def test_repack_asymmetric_raises():
    """repack_compressed_tensors_w4a16_moe_to_xpu must raise on sym=False."""
    from vllm_xpu_kernels.quantization.compressed_tensors_wna16 import (
        repack_compressed_tensors_w4a16_moe_to_xpu,
    )

    layer = _FakeLayer()
    with pytest.raises((AssertionError, NotImplementedError)):
        repack_compressed_tensors_w4a16_moe_to_xpu(layer, 32, sym=False)
