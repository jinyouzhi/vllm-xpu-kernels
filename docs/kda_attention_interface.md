# KDA Attention Op Interface

Kimi Delta Attention (KDA) is the linear-attention operator used by the
Kimi-Linear family. This document specifies the ops registered by
[kda_attention_interface.cpp](../csrc/xpu/gdn_attn/kda_attention_interface.cpp),
their argument contracts, and how the underlying SYCL kernels are selected.

## Layout

| File | Role |
| --- | --- |
| [kda_attention_interface.cpp](../csrc/xpu/gdn_attn/kda_attention_interface.cpp) | Op entry points: metadata/tensor validation, dtype and shape dispatch, backend selection. |
| [kda_gate.hpp](../csrc/xpu/gdn_attn/kda_gate.hpp) | The two gate parameterisations and the `beta` logit, shared by every backend. |
| [kda_common.hpp](../csrc/xpu/gdn_attn/kda_common.hpp) | Constants shared by the conv and recurrent kernels. |
| [kda_causal_conv1d.hpp](../csrc/xpu/gdn_attn/kda_causal_conv1d.hpp) | Depthwise causal conv1d + SiLU (general, long-prefill tiled, state write-back). |
| [kda_recurrent_opt.hpp](../csrc/xpu/gdn_attn/kda_recurrent_opt.hpp) | `recurrent_kda_opt_kernel`: the sequential gated delta rule (`opt` backend). |
| [xe_2/chunk_kda_xe2.cpp](../csrc/xpu/gdn_attn/xe_2/chunk_kda_xe2.cpp) | `chunk_kda_xe2`: the XMX/DPAS chunked prefill pipeline (`chunk` backend). |

Three ops are registered on `_xpu_C` (see
[xpu/torch_bindings.cpp](../csrc/xpu/torch_bindings.cpp)):

1. `kda_causal_conv1d()` — short convolution + SiLU, returns `{q, k, v}`.
2. `kda_gated_delta_rule()` — the gated delta-rule recurrence.
3. `kda_attention()` — the two chained, for callers that do not need the
   intermediate `{q, k, v}`.

> **Differences from the GDN ops**
>
> - KDA takes **three separate 2D projections** `q_proj` / `k_proj` / `v_proj`
>   rather than GDN's packed `projected_states_qkvz`.
> - The convolution processes the three streams as one `3 * hidden_dim`
>   channel space; the convolution weights are **float32**.
> - The conv stage returns **three** tensors (not GDN's five
>   `{q, k, v, b, a}`); `raw_gate`, `raw_beta`, `A_log` and `dt_bias` go
>   straight into the recurrence.
> - **non-spec and spec tokens may coexist in one call.** The interface does
>   not enforce mutual exclusion; each path runs its own kernel launch. GDN's
>   `causal_conv1d()` does enforce it.
> - There is no `tp_size` / `reorder_input` argument. Head partitioning happens
>   in the caller, which passes the already-sharded `hidden_dim`.

## Gate parameterisation

`raw_gate` and `raw_beta` are **raw projection outputs**. The kernels apply the
activations themselves, which keeps the op self-consistent and saves the caller
an elementwise pass over `[num_tokens, num_heads * head_dim]`. This matches
FLA, vLLM, FlashInfer, SGLang and the Gated DeltaNet kernels in this
repository.

`beta = sigmoid(raw_beta)`. The forget gate has two modes, selected by the
optional `gate_lower_bound` argument, which mirrors
`linear_attn_config.gate_lower_bound` in the HuggingFace config:

| `gate_lower_bound` | Gate | Formula (`x = raw_gate + dt_bias`) | Range |
| --- | --- | --- | --- |
| `None` (unset) | softplus, unbounded | `g = -exp(A_log[h]) * softplus(x)` | `(-inf, 0]` |
| negative, e.g. `-5.0` | sigmoid, bounded | `g = lower_bound * sigmoid(exp(A_log[h]) * x)` | `(lower_bound, 0)` |

A non-negative `gate_lower_bound` is rejected. Both modes produce a
non-positive log-domain decay, which is what lets the chunked backend clamp its
running cumulative sum (see [Backends](#backends)).

## Common scheduling contract

The ops take these scheduling counts:

- `num_prefills` — prefill sequences.
- `num_decodes` — ordinary decode sequences.
- `num_spec_decodes` — speculative decode sequences.
- `num_actual_tokens` — total valid tokens in this call, written `T` in the
  shapes below.

`validate_metadata()` enforces:

- All counts are non-negative and fit in `int` (kernels index with `int`).
- With `non_spec_batch_size = num_prefills + num_decodes > 0`:
  `non_spec_query_start_loc` (length `>= non_spec_batch_size + 1`) and
  `non_spec_state_indices` (length `>= non_spec_batch_size`) are required and
  contiguous int32; `non_spec_token_indx` and `has_initial_state` (bool) are
  optional.
- With `num_spec_decodes > 0`: `spec_query_start_loc`, `spec_token_indx`,
  `spec_state_indices` (**2D**,
  `[num_spec_decodes, num_speculative_tokens + 1]`) and `num_accepted_tokens`
  are all required.
- Token conservation:
  `non_spec_tokens + spec_tokens == num_actual_tokens`, where the speculative
  count follows from the 2D index shape:
  `spec_tokens = num_spec_decodes * (num_speculative_tokens + 1)`.

Inputs may carry CUDA-graph padding beyond `num_actual_tokens`; only the first
`num_actual_tokens` rows participate. A state slot equal to `pad_slot_id`
(`-1`) is skipped entirely.

## `kda_causal_conv1d()`

Runs a width-`width` causal convolution plus SiLU over each of the three
projection streams, updates `conv_state`, and produces the `{q, k, v}` the
recurrence consumes.

Returns `std::vector<torch::Tensor>`: three `[num_actual_tokens, hidden_dim]`
tensors. When `num_actual_tokens == 0` they are returned unwritten and no
kernel is launched.

### Arguments

| Argument | Type | Dir | Contract |
| --- | --- | --- | --- |
| `q_proj` / `k_proj` / `v_proj` | `const Tensor&` | in | 2D `[>=T, hidden_dim]`, identical shape and dtype, contiguous along channels and sharing one row stride. dtype float16, bfloat16 or float32. The row stride may exceed `hidden_dim`, which is how a fused mixed-QKV projection is passed as three views. |
| `conv_state` | `Tensor&` | in/out | 3D convolution history, dtype float16, bfloat16 or float32. Either **DS** `[slots, 3 * hidden_dim, width - 1]` or **SD** `[slots, width - 1, 3 * hidden_dim]`; `validate_conv_inputs` detects which and derives `dim_stride` / `time_stride`. Slots must be contiguous and non-overlapping. |
| `q_conv_weight` / `k_conv_weight` / `v_conv_weight` | `const Tensor&` | in | `[hidden_dim, width]`, **float32**, contiguous. `width` must be 2–5. |
| `num_prefills` / `num_decodes` / `num_spec_decodes` | `int64_t` | in | Scheduling counts. |
| `has_initial_state` | `optional<Tensor>` | in | non-spec, optional. Contiguous 1D bool, length `num_prefills + num_decodes`. |
| `non_spec_query_start_loc` | `optional<Tensor>` | in | non-spec, required. Contiguous 1D int32, length `num_prefills + num_decodes + 1`. |
| `non_spec_token_indx` | `optional<Tensor>` | in | non-spec, optional. Contiguous 1D int32 mapping logical token to physical row; absent means identity. |
| `non_spec_state_indices` | `optional<Tensor>` | in | non-spec, required. Contiguous 1D int32, one cache slot per sequence. |
| `spec_query_start_loc` | `optional<Tensor>` | in | spec, required. Contiguous 1D int32, length `num_spec_decodes + 1`. |
| `spec_token_indx` | `optional<Tensor>` | in | spec, required. Contiguous 1D int32 row map. |
| `spec_state_indices` | `optional<Tensor>` | in | spec, required. Contiguous **2D** int32 `[num_spec_decodes, num_speculative_tokens + 1]`: one slot per speculative token. |
| `num_accepted_tokens` | `optional<Tensor>` | in | spec, required. Contiguous 1D int32, length `num_spec_decodes`. |
| `num_actual_tokens` | `int64_t` | in | `T`. |

### Kernels

`launch_kda_conv` picks between three kernels in
[kda_causal_conv1d.hpp](../csrc/xpu/gdn_attn/kda_causal_conv1d.hpp). `Width` is
a compile-time constant expanded over `{2, 3, 4, 5}`.

**`causal_conv1d_kernel<T, CacheT, Width, IsSpec>`** — the general path, used
for every batch shape including spec decode. 2D `nd_range`: `group(0)` is the
sequence, `global_id(1)` walks `3 * hidden_dim` channels rounded up to
`work_group_size = 256`. Each work-item owns one `(sequence, channel)` pair,
with `stream = channel / hidden_dim` selecting q, k or v.

Each token applies `acc = dot(history, w[0 : Width - 1]) + current * w[Width - 1]`
followed by `out = silu(acc)`, with `silu(x) = x / (1 + exp(-x))`. Per
work-item:

1. Resolve the state slot. For `IsSpec`, take column
   `max(num_accepted_tokens[batch] - 1, 0)` of the 2D `state_indices` and
   always load it; for non-spec, use `state_indices[batch]` and load only if
   `has_initial_state` says so. A `pad_slot_id` slot returns immediately.
2. Load `history[Width - 1]` from `conv_state`, zero-filled when no initial
   state is loaded.
3. Slide over `[seq_start, seq_end)`, emitting `out` per token and then
   shifting `history` left and appending `current`.
4. Write back: `IsSpec` stores per token at `state_indices[batch, t]` so
   speculative verification can fork; non-spec stores the final `history` to
   the sequence's slot once.

**`causal_conv1d_tiled_kernel<T, CacheT, Width, TileT>`** — long-prefill
specialisation. The general kernel launches only `batch * 3 * hidden_dim`
threads, which starves the GPU when a few long sequences carry all the tokens.
The tiled kernel splits each sequence into `TileT`-token tiles and recomputes
the `Width - 1` token halo per tile, trading arithmetic for parallelism. It is
selected only when the batch is pure prefill (`num_decodes == 0` and
`num_spec_decodes == 0`), the batch is small enough
(`kda::conv1d_tiled_max_batch_size`), the token count clears a
batch-size-dependent threshold, and the activation and cache dtypes match.

**`causal_conv1d_update_state_kernel<T, CacheT, Width>`** — the tiled kernel
does not touch the cache, so this writes back the trailing `Width - 1` tokens
of every sequence afterwards.

## `kda_gated_delta_rule()`

Consumes the conv stage's `{q, k, v}` plus the gate inputs, runs the KDA
recurrence, writes `core_attn_out` and updates `recurrent_state`. Returns
`void`; only the first `T` rows participate.

### Arguments

| Argument | Type | Dir | Contract |
| --- | --- | --- | --- |
| `core_attn_out` | `Tensor&` | out | Contiguous 4D `[1, >=T, num_heads, head_dim]`, dtype equal to `q`. |
| `q` / `k` / `v` | `const Tensor&` | in | 2D `[T, num_heads * head_dim]`, same dtype (float16, bfloat16 or float32). Row-strided views are accepted and densified with `.contiguous()` by the op; the kernels themselves index a packed `token * heads * dim` stride. |
| `raw_gate` | `const Tensor&` | in | Contiguous 4D `[1, >=T, num_heads, head_dim]`, dtype equal to `q`. **`num_heads` and `head_dim` are derived from this tensor.** |
| `raw_beta` | `const Tensor&` | in | Contiguous **float32** `[1, >=T, num_heads]`. Raw logits; the kernels apply `sigmoid`. |
| `recurrent_state` | `Tensor&` | in/out | 4D `[slots, num_heads, head_dim, head_dim]`, **float32 or the activation dtype**. Slots contiguous and non-overlapping; the slot stride may be arbitrary (page-strided caches are supported). |
| `A_log` | `const Tensor&` | in | Contiguous float32 with `numel() == num_heads`; any shape, e.g. `[num_heads]` or `[1, 1, num_heads, 1]`. |
| `dt_bias` | `const Tensor&` | in | Contiguous float32 **1D** `[num_heads * head_dim]`. |
| `num_prefills` / `num_decodes` / `num_spec_decodes` | `int64_t` | in | Scheduling counts. |
| non-spec / spec metadata | `optional<Tensor>` | in | Identical semantics to `kda_causal_conv1d()`. |
| `num_actual_tokens` | `int64_t` | in | `T`. |
| `gate_lower_bound` | `optional<double>` | in | Gate mode, see [Gate parameterisation](#gate-parameterisation). Must be negative when set. |

`head_dim` must be one of `{32, 64, 128, 256}`.

### Backends

The recurrence has two backends. `launch_kda_recurrent` tries `chunk` for the
non-spec half of the batch and uses `opt` for whatever is left; spec-decode
batches always go to `opt`.

**`opt` — `recurrent_kda_opt_kernel<T, StateT, KBucketSize, Mode>`**
([kda_recurrent_opt.hpp](../csrc/xpu/gdn_attn/kda_recurrent_opt.hpp)).
Sequential over tokens. `head_dim` is a compile-time constant
`KBucketSize * 32` with `KBucketSize` in `{1, 2, 4, 8}`; the state may be
float32, bfloat16 or float16. `Mode` distinguishes `general` (prefill),
`decode` and `spec` batch shapes — `decode` drops the token loop entirely and
hoists loop invariants such as `dt_bias` and the per-head decay coefficient out
of it. This backend covers every supported shape and dtype, so it is always the
fallback.

**`chunk` — `chunk_kda_xe2()`**
([xe_2/chunk_kda_xe2.cpp](../csrc/xpu/gdn_attn/xe_2/chunk_kda_xe2.cpp)). A
seven-stage XMX/DPAS pipeline that cuts the sequence into `chunk_size = 64`
token blocks, folds the per-chunk decay and `beta` into the GEMM operands, and
leaves only the inter-chunk state carry sequential. Requirements:

- the batch half is non-spec (`num_prefills + num_decodes > 0`);
- the extension was built with `VLLM_XPU_ENABLE_XE2`;
- `head_dim % 64 == 0`, i.e. `head_dim` in `{64, 128, 256}`;
- activations are **bfloat16** and `recurrent_state` is float32 or bfloat16 —
  only these two combinations are instantiated.

When a requirement is not met the batch silently falls through to `opt`,
including when `chunk` was requested explicitly.

**Selection.** `auto` (the default) uses `chunk` when the above hold, the
average non-spec sequence length reaches `VLLM_XPU_KDA_CHUNK_MIN_SEQLEN`, and
the scratch requirement fits `VLLM_XPU_KDA_CHUNK_MAX_WORKSPACE_MB`. `chunk`
skips the sequence-length heuristic; `opt` disables the chunked path outright.

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_XPU_KDA_RECURRENT_MODE` | `auto` | `auto`, `opt` or `chunk`. Any other value is an error. |
| `VLLM_XPU_KDA_CHUNK_MIN_SEQLEN` | `128` | Below this average non-spec sequence length, `auto` stays on `opt` rather than pay the chunked pipeline's fixed cost. |
| `VLLM_XPU_KDA_CHUNK_MAX_WORKSPACE_MB` | `2048` | Scratch budget; `auto` declines `chunk` above it. |
| `VLLM_XPU_KDA_CHUNK_STRICT` | `0` | Raise instead of falling back when the decay range is exceeded (below). |

All four are read once and cached, so they must be set before the first call.

**Decay range.** The chunked pipeline expresses the intra-chunk term
`exp(G[s] - G[r])` as `exp(G[s]) * exp(-G[r])`, so the per-channel cumulative
log-decay within a chunk has to stay inside the float exponent range. It is
clamped at `g_floor = -80`, and past that clamp the result stops matching the
sequential recurrence. The unbounded softplus gate of a trained model never
comes close; the bounded sigmoid gate decays by up to `lower_bound` per token,
so at `lower_bound = -5` and 64-token chunks it takes only an average gate
activation above 0.25 to saturate. That case is therefore guarded: the pipeline
synchronises after its first stage and, if the clamp engaged, abandons the
launch — nothing but scratch has been written — and the batch is re-run on
`opt`. The guard is skipped when `chunk_size * lower_bound > g_floor` makes
saturation arithmetically impossible, which is the only reason it is not always
on. `VLLM_XPU_KDA_CHUNK_STRICT=1` turns the fallback into an error instead.

### `opt` parallel structure

3D `nd_range`: `group(0)` is the sequence, `group(1)` the head, and `group(2)`
covers `value_bucket * work_group_size` with
`head_dim / values_per_work_group` value buckets.

Constants ([kda_common.hpp](../csrc/xpu/gdn_attn/kda_common.hpp)):
`sub_group_size = 32`, `work_group_size = 256`, `values_per_sub_group = 4`,
`values_per_work_group = 256 / 32 * 4 = 32`, `l2norm_eps = 1e-6`.

A head's state is a `head_dim x head_dim` (value x key) matrix. The **key**
dimension is split across the 32 lanes of a sub-group, `KBucketSize` keys per
lane; the **value** dimension gives each sub-group `values_per_sub_group = 4`
rows. Each work-item therefore holds `state[values_per_sub_group][KBucketSize]`
in registers, loaded from `recurrent_state` (zeroed for a non-spec sequence
with no initial state).

Per token:

1. Load this lane's `q_local[key]` and `k_local[key]`, accumulating squares and
   reducing over the sub-group for the L2 norms.
2. Compute the per-key-channel decay `exp(g)` with `g` from
   [Gate parameterisation](#gate-parameterisation).
3. L2-normalise; `q` additionally scales by `q_scale = rsqrt(head_dim)`.
4. **Read**: apply the decay (`state *= decay`), then
   `kv[v] = sum over key of state[v][key] * k[key]` (sub-group reduction).
5. **Update**: `delta = (v - kv) * beta`, then
   `state += delta * k`.
6. **Output**: `out[v] = sum over key of state[v][key] * q[key]` (sub-group
   reduction); lane 0 stores to `core_attn_out`.

State write-back mirrors the conv stage: `spec` mode saves per token at
`state_indices[batch, t]`, non-spec saves once at the end of the sequence. The
`chunk` backend maintains `recurrent_state` with the same semantics.

## `kda_attention()`

The fused wrapper. Returns `void` and simply:

1. calls `kda_causal_conv1d()`, updating `conv_state` and producing
   `{q, k, v}`;
2. calls `kda_gated_delta_rule()`, updating `core_attn_out` and
   `recurrent_state`.

Its arguments are the union of the two stages with identical semantics:

| Group | Arguments |
| --- | --- |
| Output | `core_attn_out` |
| Projections | `q_proj`, `k_proj`, `v_proj` |
| Gate inputs | `raw_gate`, `raw_beta`, `A_log`, `dt_bias`, `gate_lower_bound` |
| Mutable state | `conv_state`, `recurrent_state` |
| Conv weights | `q_conv_weight`, `k_conv_weight`, `v_conv_weight` |
| Scheduling | `num_prefills`, `num_decodes`, `num_spec_decodes`, `num_actual_tokens` |
| non-spec metadata | `has_initial_state`, `non_spec_query_start_loc`, `non_spec_token_indx`, `non_spec_state_indices` |
| spec metadata | `spec_query_start_loc`, `spec_token_indx`, `spec_state_indices`, `num_accepted_tokens` |

Use the split ops instead when the two stages need separate scheduling, or when
the intermediate `{q, k, v}` must be observed or reused. Because
`kda_attention()` hands the recurrence the dense buffers its own convolution
just wrote, it never pays the `.contiguous()` densification that
`kda_gated_delta_rule()` performs for strided callers.
