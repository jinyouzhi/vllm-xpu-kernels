// SPDX-License-Identifier: Apache-2.0
//
// xpu_int4_woq_fused_moe — oneDNN INT4 grouped-matmul fused MoE FFN for XPU.
//
// This op implements the full Mixture-of-Experts FFN with symmetric grouped
// INT4 weight quantization (W4A16) using oneDNN, targeting Intel GPU (XPU).
// It is intended to back vLLM's CompressedTensorsWNA16MoEMethod (or a new
// XPU-specific subclass) when running on Intel GPU instead of the CUDA/Marlin
// path.
//
// Weight layout contract (must match repack_compressed_tensors_w4a16_moe_to_xpu):
//   w13_qweight : [E, 2*I, K/8]   int32, K-major packing (8 int4 per int32
//                                  along K), i.e. compressed-tensors
//                                  pack_quantized MoE layout.
//   w13_scales  : [E, K/group_size, 2*I]  same dtype as x, transposed from
//                                  vLLM's (E, 2*I, K/group_size) format.
//   w2_qweight  : [E, K, I/8]     int32, K-major packing (I is inter size).
//   w2_scales   : [E, I/group_size, K]   same dtype as x.
//
// For each expert e, the per-expert 2-D slices are obtained as:
//   w13_e = w13_qweight[e].t()   -> shape (K/8, 2*I), strides (1, K/8)
//   s13_e = w13_scales[e]        -> shape (K/group_size, 2*I)
//   w2_e  = w2_qweight[e].t()   -> shape (I/8, K),   strides (1, I/8)
//   s2_e  = w2_scales[e]         -> shape (I/group_size, K)
// Both w*_e satisfy the oneDNN NT-format check strides[dim-2] == 1.
//
// v1 limitations (raise TORCH_CHECK on violation):
//   - sym must be true; qzeros must be None
//   - num_bits = 4 only (enforced by the packed int32 shape convention)
//   - no act-order (g_idx not supported)
//   - no expert parallelism beyond single-rank
//   - no bias

#include <ATen/ATen.h>
#include <torch/torch.h>

#include "xpu/onednn/int4_gemm_w4a16.h"

torch::Tensor xpu_int4_woq_fused_moe(
    torch::Tensor x,                               // [M, K]
    torch::Tensor topk_ids,                        // [M, top_k]
    torch::Tensor topk_weights,                    // [M, top_k], float32
    torch::Tensor w13_qweight,                     // [E, 2*I, K/8], int32
    torch::Tensor w13_scales,                      // [E, K/gs, 2*I]
    std::optional<torch::Tensor> w13_qzeros,       // must be None (sym)
    torch::Tensor w2_qweight,                      // [E, K, I/8], int32
    torch::Tensor w2_scales,                       // [E, I/gs, K]
    std::optional<torch::Tensor> w2_qzeros,        // must be None (sym)
    int64_t group_size,
    bool sym) {
  // ---- v1 constraints ----
  TORCH_CHECK(
      sym,
      "xpu_int4_woq_fused_moe: v1 only supports symmetric (sym=true) W4A16 "
      "MoE; asymmetric is not yet implemented");
  TORCH_CHECK(
      !w13_qzeros.has_value(),
      "xpu_int4_woq_fused_moe: w13_qzeros must be None when sym=true");
  TORCH_CHECK(
      !w2_qzeros.has_value(),
      "xpu_int4_woq_fused_moe: w2_qzeros must be None when sym=true");

  // ---- dtype checks ----
  TORCH_CHECK(
      x.scalar_type() == at::ScalarType::BFloat16 ||
          x.scalar_type() == at::ScalarType::Half,
      "xpu_int4_woq_fused_moe: x must be bf16 or fp16, got ",
      x.scalar_type());
  TORCH_CHECK(
      w13_scales.scalar_type() == x.scalar_type(),
      "xpu_int4_woq_fused_moe: w13_scales dtype must match x dtype");
  TORCH_CHECK(
      w2_scales.scalar_type() == x.scalar_type(),
      "xpu_int4_woq_fused_moe: w2_scales dtype must match x dtype");
  TORCH_CHECK(
      w13_qweight.scalar_type() == at::ScalarType::Int,
      "xpu_int4_woq_fused_moe: w13_qweight must be int32");
  TORCH_CHECK(
      w2_qweight.scalar_type() == at::ScalarType::Int,
      "xpu_int4_woq_fused_moe: w2_qweight must be int32");

  // ---- dimension checks ----
  TORCH_CHECK(x.dim() == 2, "xpu_int4_woq_fused_moe: x must be 2D [M, K]");
  TORCH_CHECK(
      topk_ids.dim() == 2,
      "xpu_int4_woq_fused_moe: topk_ids must be 2D [M, top_k]");
  TORCH_CHECK(
      topk_weights.dim() == 2,
      "xpu_int4_woq_fused_moe: topk_weights must be 2D [M, top_k]");
  TORCH_CHECK(
      w13_qweight.dim() == 3,
      "xpu_int4_woq_fused_moe: w13_qweight must be 3D [E, 2*I, K/8]");
  TORCH_CHECK(
      w13_scales.dim() == 3,
      "xpu_int4_woq_fused_moe: w13_scales must be 3D [E, K/gs, 2*I]");
  TORCH_CHECK(
      w2_qweight.dim() == 3,
      "xpu_int4_woq_fused_moe: w2_qweight must be 3D [E, K, I/8]");
  TORCH_CHECK(
      w2_scales.dim() == 3,
      "xpu_int4_woq_fused_moe: w2_scales must be 3D [E, I/gs, K]");

  // ---- shape derivations ----
  const int64_t M = x.size(0);
  const int64_t K_hidden = x.size(1);
  const int64_t E = w13_qweight.size(0);
  const int64_t N13 = w13_qweight.size(1);   // 2*I (gate+up)
  const int64_t Kpack13 = w13_qweight.size(2);  // K/8
  const int64_t I_mid = N13 / 2;             // intermediate size (after SwiGLU)
  const int64_t top_k = topk_ids.size(1);

  TORCH_CHECK(
      N13 % 2 == 0,
      "xpu_int4_woq_fused_moe: w13 N-dim must be even (gate+up concat)");
  TORCH_CHECK(
      K_hidden == 8 * Kpack13,
      "xpu_int4_woq_fused_moe: w13_qweight K/8 dim inconsistent with x K dim");
  TORCH_CHECK(
      group_size > 0 && K_hidden % group_size == 0,
      "xpu_int4_woq_fused_moe: group_size must be positive and divide K");

  // w2 shape checks
  TORCH_CHECK(
      w2_qweight.size(0) == E,
      "xpu_int4_woq_fused_moe: E mismatch between w13 and w2");
  TORCH_CHECK(
      w2_qweight.size(1) == K_hidden,
      "xpu_int4_woq_fused_moe: w2 N-dim must equal K_hidden");
  const int64_t Kpack2 = w2_qweight.size(2);  // I/8
  const int64_t I_in = 8 * Kpack2;            // input K of w2 = I_mid
  TORCH_CHECK(
      I_in == I_mid,
      "xpu_int4_woq_fused_moe: w2 K-dim (intermediate) must match w13 I_mid");
  TORCH_CHECK(
      I_mid % group_size == 0,
      "xpu_int4_woq_fused_moe: group_size must divide intermediate_size");

  // scale shape checks
  TORCH_CHECK(
      w13_scales.size(0) == E && w13_scales.size(1) == K_hidden / group_size &&
          w13_scales.size(2) == N13,
      "xpu_int4_woq_fused_moe: w13_scales shape must be [E, K/gs, 2*I]");
  TORCH_CHECK(
      w2_scales.size(0) == E && w2_scales.size(1) == I_mid / group_size &&
          w2_scales.size(2) == K_hidden,
      "xpu_int4_woq_fused_moe: w2_scales shape must be [E, I/gs, K]");

  TORCH_CHECK(
      topk_ids.size(0) == M,
      "xpu_int4_woq_fused_moe: topk_ids M-dim must match x");
  TORCH_CHECK(
      topk_weights.size(0) == M && topk_weights.size(1) == top_k,
      "xpu_int4_woq_fused_moe: topk_weights shape must be [M, top_k]");

  const at::DeviceGuard device_guard(x.device());

  // Symmetric zero-point: scalar int8 value = 8 (oneDNN convention for sym
  // INT4: unsigned [0,15] values are offset by 8 to signed [-8,7]).
  auto sym_zp = at::tensor(
      {static_cast<int8_t>(8)},
      at::TensorOptions().dtype(at::kChar).device(x.device()));

  // Output accumulator (zero-initialised so we can scatter-add into it).
  auto output = at::zeros({M, K_hidden}, x.options());

  // Cast topk_ids to int64 for XPU indexing ops if needed.
  auto topk_ids_i64 = topk_ids.to(at::kLong);

  // ---- Per-expert loop ----
  for (int64_t e = 0; e < E; ++e) {
    // Find which tokens are routed to expert e.
    // expert_mask: [M, top_k], bool
    auto expert_mask = topk_ids_i64.eq(e);   // [M, top_k]
    // any_mask: [M], true if token m is routed to expert e for any k
    auto any_mask = expert_mask.any(1);       // [M]
    // token_indices: 1-D tensor of token row indices
    auto token_indices = any_mask.nonzero().squeeze(1);  // [Me]
    int64_t Me = token_indices.numel();
    if (Me == 0) continue;

    // Gather input tokens for this expert: [Me, K_hidden]
    auto x_e = x.index_select(0, token_indices);

    // ----- GEMM 1: x_e @ W13[e] -> [Me, 2*I] -----
    // w13_qweight[e]: shape (2*I, K/8), strides (K/8, 1) — K-major.
    // .t() gives (K/8, 2*I), strides (1, K/8) — NT format required by oneDNN.
    auto w13_e = w13_qweight.select(0, e).t();  // [K/8, 2*I]
    // w13_scales[e]: shape (K/gs, 2*I) — already in oneDNN scale format.
    auto s13_e = w13_scales.select(0, e);       // [K/gs, 2*I]

    auto result13 = at::empty({Me, N13}, x.options());
    oneDNN::dnnl_matmul_w4a16_int4(
        result13, x_e, w13_e, std::nullopt, s13_e, sym_zp, group_size);

    // ----- SwiGLU activation -----
    // gate = result13[:, :I], up = result13[:, I:]
    // silu_mul = silu(gate) * up
    auto gate = result13.narrow(1, 0, I_mid);
    auto up = result13.narrow(1, I_mid, I_mid);
    auto silu_mul = at::silu(gate) * up;  // [Me, I_mid]

    // ----- GEMM 2: silu_mul @ W2[e] -> [Me, K_hidden] -----
    // w2_qweight[e]: shape (K_hidden, I/8), strides (I/8, 1) — K-major.
    // .t() gives (I/8, K_hidden), strides (1, I/8) — NT format for oneDNN.
    auto w2_e = w2_qweight.select(0, e).t();   // [I/8, K_hidden]
    // w2_scales[e]: shape (I/gs, K_hidden) — already in oneDNN scale format.
    auto s2_e = w2_scales.select(0, e);         // [I/gs, K_hidden]

    auto result2 = at::empty({Me, K_hidden}, x.options());
    oneDNN::dnnl_matmul_w4a16_int4(
        result2, silu_mul, w2_e, std::nullopt, s2_e, sym_zp, group_size);

    // ----- Weighted accumulation -----
    // For each of the Me tokens, find the weight assigned to expert e and
    // add weight * result2 to the output at the original token position.
    //
    // expert_mask_e: [Me, top_k], float — 1.0 at the k-slot for expert e.
    auto expert_mask_e = expert_mask
                             .index_select(0, token_indices)
                             .to(topk_weights.dtype());  // [Me, top_k], float
    auto tw_e = topk_weights.index_select(0, token_indices);  // [Me, top_k]
    // Sum across top_k dim to get per-token weight for this expert.
    auto expert_w =
        (expert_mask_e * tw_e).sum(1, /*keepdim=*/true);  // [Me, 1], float

    // Scale result2 and scatter-add into output.
    auto weighted = result2 * expert_w.to(x.dtype());  // [Me, K_hidden]
    output.index_add_(0, token_indices, weighted);
  }

  return output;
}
