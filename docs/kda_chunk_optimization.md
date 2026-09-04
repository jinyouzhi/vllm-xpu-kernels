# KDA chunk backend — optimization report

Target: `csrc/xpu/gdn_attn/xe_2/chunk_kda_kernels_xe2.hpp` and its launcher, the
Xe2 (Battlemage) chunked-prefill path of Kimi Delta Attention.

## Summary

| | |
| --- | --- |
| GPU | Intel Graphics `0xe223` (BMG, Xe2-HPG), 2500 MHz pinned |
| Data type | bf16 activations, fp32 accumulation |
| Bottleneck | **memory bandwidth**, uniformly across all five stages |
| Baseline | 22636 us over the 7-shape chunk sweep |
| **Final** | **19739 us (-12.8%)** |
| End-to-end prefill (49 configs) | **-4.04% geomean**, best config -10.19% |
| Trials | 7 (3 accepted, 3 rejected, 1 neutral-but-simpler) |

Measured with `benchmark/benchmark_kda_chunk_stages.py` (chunk backend only,
per-stage attribution under `unitrace`) and `benchmark/benchmark_kda.py` (133
end-to-end configurations).

### Kimi-Linear tp1 prefill, before vs after

`benchmark_kda_gated_delta_rule.py` times `kda_gated_delta_rule` on its own,
which on these shapes is entirely the five chunk kernels, so it isolates this
work from the conv1d that `benchmark_kda.py` also includes:

| workload | delta rule only | full KDA op |
| --- | ---: | ---: |
| `prefill_b1_1k` | 429.4 -> 390.7 us (**-9.02%**) | 731.6 -> 698.1 us (-4.58%) |
| `prefill_b1_4k` | 1892.7 -> 1698.5 (**-10.26%**) | 3203.9 -> 3015.8 (-5.87%) |
| `prefill_b1_8k` | 3870.7 -> 3451.4 (**-10.83%**) | 6577.6 -> 6150.6 (-6.49%) |
| `prefill_b4_2k` | 3934.2 -> 3473.5 (**-11.71%**) | 6646.3 -> 6189.8 (-6.87%) |
| `prefill_b8_1k` | 3892.2 -> 3399.4 (**-12.66%**) | 6632.1 -> 6130.1 (-7.57%) |
| geomean | **-10.91%** | -6.28% |

The gain grows with sequence length, as expected for a change that removes
bytes rather than instructions.

> When A/B-testing two builds by swapping `.so` files, overwrite
> `build/temp/libgdn_attn_kernels_xe_2.so`, not just the copy in
> `vllm_xpu_kernels/`. `_xpu_C.abi3.so` carries a `RUNPATH` that lists
> `build/temp` first, so the loader takes the chunk kernels from there and a
> swap of the package copy alone silently measures the same binary twice.

## Why this pipeline is bandwidth bound

Per-stage measurement on `b1_8k` (8192 tokens, 32 heads, head_dim 128) before
any change:

| stage | us | % | bytes moved | effective BW |
| --- | ---: | ---: | ---: | ---: |
| PrepareVec | 1476.6 | 35.8% | 537 MB | 364 GB/s |
| FwdO | 1123.2 | 27.2% | ~500 MB | - |
| ComputeWU | 823.4 | 20.0% | 301 MB | 366 GB/s |
| ComputeA | 360.5 | 8.7% | 168 MB | 465 GB/s |
| InverseOpt | 142.6 | 3.5% | 67 MB | 470 GB/s |

Total is ~1.6 GB in 3933 us = **~400 GB/s** against a measured streaming ceiling
of ~393 GB/s (456 GB/s is the paper GDDR6 figure; the stages above 393 are being
helped by the 24 MB last-level cache). XMX occupancy over the same window is
only 15-17% of the 102 TFLOP/s bf16 peak.

The byte counts above are the operands each kernel actually reads and writes,
taken from the source and divided by that kernel's `unitrace` time, which is why
they can be compared against the streaming ceiling. They are a different
quantity from `benchmark_kda.py`'s `KDA_memBandwidth(GB/s)` column: that one
divides a fixed analytic model of the whole fused op — including a hard-coded
six activation planes for the conv/recurrence hand-off — by wall time, so for a
given shape it is a constant over latency and moves only when the kernel gets
faster, not when it moves fewer bytes.

So **arithmetic is free here and bytes are not**: only changes that delete
traffic, or move it from DRAM to cache, can help.

### Where the pipeline sits on the bandwidth curve

The same accounting over the whole sweep gives the number every later estimate
is measured against. Per token-head the pipeline moves ~4900 B:

| stage | B / token-head | what |
| --- | ---: | --- |
| `prepare` | ~1540 | reads `q`, `k`, `gate`; writes `Ka`, `Kb`, `Qt` |
| `fwd_o` | ~1400 | `W`, `U`, `Qt`, `Kb`, `O`, and the `S` carry |
| `compute_wu` | 1152 | `A`, `Ka`, `v`; writes `W`, `U` |
| `compute_A` | 640 | `Ka`, `Kb`; writes `A` |
| `inverse` | ~128-500 | `A` in place |

Over the sweep's 46080 tokens x 32 heads that is 7.2 GB in 19878 us =
**363 GB/s, 92% of the achievable streaming rate**; `prepare` alone is at 98%.

Two consequences. There is no efficiency left to recover, only traffic. And a
fusion is worth the share of those 4900 B it deletes, which is **not** the
wall-clock share of the stages being fused — estimating from stage time
overstates every candidate below by 3-4x.

For scale, the floor for this algorithm (read `q`, `k`, `v`, `gate`, write `O`,
nothing else) is 1280 B/token-head, so a perfectly fused implementation would be
~3.8x faster. That is what FlashKDA's two-kernel CHUNK=16 structure reaches, and
it is a ground-up rewrite rather than an optimization of this pipeline.

## Trial 1 (accepted): read `v` in place instead of materializing `Vp`

`prepare` gathered `v` into a chunk-aligned workspace plane `Vp`, and
`compute_wu` was its only consumer. The gather applies **no math at all** — no
decay, no scaling, no normalization — so `Vp` is a byte-for-byte copy of `v`
with a different row pitch.

`compute_wu` already consumed it through a transposed `(head_dim, chunk_size)`
view. Pointing that view at `v` needs only a row pitch of `num_heads * head_dim`
and a rebased pointer. Two conditions must hold, checked by one shared predicate
`chunk_kda_needs_vp()` so producer and consumer cannot disagree:

* `token_indx == nullptr` — a speculative-decode gather is not an affine row
  mapping, so no strided view can express it.
* `valid == chunk_size` — a partial trailing chunk would let the 2D block load
  address rows past the end of `v`.

There is at most one partial chunk per sequence, so the residual `Vp` traffic is
negligible.

This deletes a 67 MB store and a 67 MB load per 8k-token pass, 25% of all bytes
`prepare` moves:

| stage | before | after |
| --- | ---: | ---: |
| PrepareVec | 1476.6 us | **1046.1 us (-29%)** |
| ComputeWU | 823.4 | 826.0 |
| ComputeA | 360.5 | 359.0 |
| InverseOpt | 142.6 | 142.3 |
| FwdO | 1123.2 | 1125.4 |

`compute_wu` is unchanged, confirming the strided read costs nothing versus the
packed one — the 2D block load handles the wider pitch at the same rate.

End-to-end over 133 configs: prefill **-4.04%** geomean, mix -5.67%, best config
-10.19%. Decode and spec run the untouched `opt` backend and sit at the ~25 us
launch floor where noise is +-7%.

## Trial 2 (rejected): `grf_size<128>` on the chunk-parallel stages

The three chunk-parallel DPAS stages request `grf_size<256>`, which halves the
threads an Xe-core can host, and unitrace reported **zero spill** — so the large
budget looked unused, and more in-flight threads should expose more
memory-level parallelism.

Result: **25171 us, +24.8%.** The zero-spill reading was taken *at 256 GRF*; the
CuTe DPAS fragments plus the `gemm_TTS` staging registers do not fit in 128, and
the resulting spill traffic costs far more than the extra occupancy buys. The
large register file is load-bearing, not leftover.

## Trial 3 (accepted): alternate the traversal direction between stages

The chunk-parallel stages share one flattened (chunk, head) grid and each reads
back what the previous stage wrote — but they all walked it in the *same*
direction, so every hand-off started at the coldest end: the tile a consumer
touched first was the one its producer had written longest ago.

`A` is 8 KB per (chunk, head), so a few thousand work-groups' worth is the same
order as the 24 MB LLC — precisely the regime where traversal order decides the
hit rate. Alternating the direction stage to stage is a pure permutation of the
work assignment, so results are bit-identical.

| stage | before | after | |
| --- | ---: | ---: | ---: |
| PrepareVec | 1046.1 | 1041.4 | -0.4% |
| ComputeA | 359.0 | 356.2 | -0.8% |
| **InverseOpt** | **142.3** | **124.6** | **-12.4%** |
| ComputeWU | 826.0 | 822.0 | -0.5% |
| FwdO | 1125.4 | 1098.3 | -2.4% |

Chunk sweep 20167 -> 19884 us (**-1.4%**), landing exactly where the model
predicts: `inverse` consumes `A` immediately after `compute_A` produces it.

Comparing two problem sizes confirms the cache model directly. `Ka`+`Kb` are
16 MB at `b1_1k` (fits the LLC) and 65 MB at `b1_4k` (does not); pure linear
scaling would be 4.00x:

| stage | b1_1k (us) | b1_4k (us) | ratio |
| --- | ---: | ---: | ---: |
| **ComputeA** | 32.9 | 192.6 | **5.86** |
| PrepareVec | 107.6 | 488.0 | 4.53 |
| ComputeWU | 91.3 | 404.5 | 4.43 |
| FwdO | 139.6 | 545.4 | 3.91 |
| InverseOpt | 20.6 | 62.8 | 3.05 |

`compute_A`, the stage that reads `Ka` and `Kb`, pays a **46% superlinear
penalty** exactly as its operands stop fitting, while `fwd_o` and `inverse`
scale at or below linear.

On the full 133-config sweep the change is neutral (-0.02% over the 93 configs
above 100 us, against a same-binary control band of -2.15%..+1.64%). That is
expected: most configs have an `A` working set that already fits in 24 MB. It is
worth ~1.4% on long-context prefill and nothing elsewhere.

## Trial 4 (accepted): run `prepare` at the native sub-group width

`prepare` was launched at `sub_group_size = 32` on a part whose native width is
16, so every sub-group instruction issued as two hardware waves.

The 32 was assumed load-bearing, on the theory that the per-channel decay cumsum
needed the wider group. It does not: each lane owns a **contiguous slice of
`head_dim`** and runs the whole cumsum inside that slice, with no cross-lane
dependency anywhere in the scan. The only cross-lane operation is the L2
normalization's `reduce_over_group`, which just needs the sub-group to span one
row — true at either width.

Dropping to 16 therefore only changes the issue width, plus one extra
instantiation (`V == 16`, i.e. `head_dim == 256`, which previously fell through
to the scalar kernel).

Chunk sweep 19884 -> **19739 us (-0.70%)**, with clean attribution: `PrepareVec`
64811 -> 63483 us (**-2.05%**) across the sweep, every other stage unchanged
within 0.1%. Outputs are bit-identical on every head_dim including the new
`V = 16` path. Full 133-config sweep -0.75% geomean, -0.01% over the >=100 us
subset.

## Trial 5 (rejected): conditionally size the `Vp` workspace plane

Since trial 1, `Vp` is only written when `chunk_kda_needs_vp()` holds. A sequence
has at most **one** tail chunk, so without a gather the plane needs `batch_size`
chunks rather than the whole token space — a ~13% smaller scratch request, which
matters because `VLLM_XPU_KDA_CHUNK_MAX_WORKSPACE_MB` gates admission to this
backend.

It measured **+0.22%** (19782 vs 19739 mean, within-binary spread +-0.03%), and
splitting `Vp` into its own tensor was worse at +0.85%.

The benchmark shapes are chunk-aligned and gather-free, so `Vp` is never written
or read in any of them: the delta is entirely the changed allocation size
shifting the physical placement of the remaining planes. That is a page-mapping
artefact rather than an algorithmic cost, but it is reproducible, so the change
was reverted. Worth revisiting if the workspace ceiling ever becomes binding.

## Trial 6 (accepted for simplicity): read `Tl` into registers in `fwd_o`

`fwd_o` staged the end-of-chunk decay `Tl` into SLM once per chunk, behind a
barrier. `scale_and_store` indexes it by the lane's own output column, so
consecutive lanes read consecutive entries: a direct global read is already
fully coalesced and warm after the first sub-group touches it.

Reading it per lane removes the fill loop, one of the four per-chunk barriers,
and the SLM allocator — `fwd_o` now uses no SLM at all.

Results are bit-identical and the 133-config sweep is neutral (+0.09% overall,
-0.02% over the >=100 us subset, worst outlier +0.86% against a +-1.6% noise
band). Kept for the simpler kernel, not for speed; the barrier was evidently not
on the critical path, consistent with the stage being bandwidth bound rather
than sync bound.

## Trial 7 (rejected): SLM-stage `U` inside `fwd_o`

`U` is produced and consumed entirely within `fwd_o` on the `has_prev_state`
path — `U := U0 - W @ S0^T` is stored to global memory, then read back twice,
once as the B operand of `O += O2 @ U` and once as the A operand of
`S := S0 + U^T @ Kb`. Nothing outside the kernel reads it, so all three global
accesses can in principle be replaced by one 8 KB SLM tile per value block.

The prize was priced before writing any of it, since this machine reports no
hardware metric groups (`unitrace --metric-list` returns `No metrics found`).
A correctness-breaking probe forced `u_offset = 0` so every (head, chunk) shares
one buffer — same instruction count, same access pattern, but guaranteed
L2-resident. That measured **-5.40%** on the five tp1 prefill workloads, which
confirmed the round trip really does reach DRAM.

The implementation needs three new GEMM variants, because the block-2D copy
atoms cannot address SLM (see the blocker list below) and the operand fragments
must therefore be gathered element-wise: `gemm_TSlmS`, `gemm_SlmTS`, and
`gemm_SlmTS_fused_2B`. Measured, GDR-only, against the 12413.5 us baseline:

| variant | total | delta |
| --- | ---: | ---: |
| SLM for `has_prev_state`, global for the first chunk | 12213.3 us | **-1.61%** |
| single path, everything staged through SLM | 12317.2 us | -0.78% |
| probe ceiling (`U` fully L2-resident) | 11743.0 us | -5.40% |

So the staging recovers less than a third of what the round trip costs: the
element-wise gather gives back most of the bytes it saves. Unifying the two
paths — which is the shape the rest of this kernel wants — gives up another
0.85%, because chunks with no carried state then pay a staging round trip they
previously avoided.

Rejected: 1.6% does not pay for three near-duplicate GEMM templates plus a
dual-source `S` update, and the version that keeps the kernel single-path falls
below the 1% bar outright.

One reusable result did come out of it. The staged tile has to be laid out
`(token, channel)`, i.e. `(K, N)` row-major for a `U^T` operand: a DPAS A/B
sub-group fragment walks the **M/N** axis within a lane, not `K`, so this is the
orientation that keeps each lane's slots contiguous. The intuitive
`(N, K)` choice costs **~6%** (12994.8 us), which is larger than the entire
optimization being attempted.

## Remaining opportunities

Everything left is cross-kernel fusion: keeping an intermediate in registers
instead of round-tripping it through DRAM. Sized by the traffic of the operand
each fusion deletes, as a share of the pipeline's 4900 B per token-head:

| fusion | operand deleted | share |
| --- | --- | ---: |
| `prepare` + `compute_A` | `Ka`, `Kb` (write + read) | ~9% |
| `compute_A` + `inverse` + `compute_wu` | `A` (cold read only) | ~2% |

`U`'s round trip is a further ~7% by the same accounting, but trial 7 measured
what is actually reachable and it is not worth the code.

That gap is itself a warning about this table: the shares are source-level
operand bytes, and trial 7 is the one row where the realisable fraction was
measured end to end. It came out at roughly a third. Treat the remaining shares
as upper bounds, and price any fusion with an L2-residency probe before
implementing it.

This mirrors FlashKDA's split (see `docs/kda_attention_design.md`): its K1 kernel
is our `{prepare, compute_A, inverse}` and its K2 is our `fwd_o`. FlashKDA
reports trying a *single* fully-fused kernel first and finding it **worse** —
the two-kernel split gave it >=15% — so `fwd_o` should stay separate.

Known blockers, so they are not re-derived:

* **`prepare` + `compute_A`** is no longer blocked by the sub-group width, which
  trial 4 fixed. What remains is the grouping: `prepare` covers one (chunk, head)
  as 256 work items partitioned along `head_dim`, while `compute_A` partitions
  the 64x64 `A` tile as a 4x2 sub-group grid. One decomposition serving both
  costs occupancy on small shapes (`b1_1k` would drop to ~25%).
* **`compute_A` + `inverse` + `compute_wu`** is the smallest and hardest of the
  three. `A` is only ~11% of `compute_wu`'s traffic and ~20% of `compute_A`'s,
  and `compute_wu`'s read already hits cache, so the fusion converts cold reads
  into warm ones rather than deleting bytes. Structurally,
  `chunk_gemm_policy_inverse` is `16x16x16` with `SGLayout<_1,_1,_1>`, i.e. **16**
  work items per work-group against 128 for the other two;
  `mma.get_slice(local_id)` is invalid past 16 and `gemm_TTS`/`gemm_STS` contain
  work-group-scoped barriers, so a fused 128-wide group can neither run the
  inverse on sub-group 0 alone (deadlock) nor redundantly (costs more than it
  saves).
* **Register-resident `S` in `fwd_o`.** `S` is read and written every chunk,
  ~1 KB per token-head. It cannot stay in the accumulator because it is produced
  as a C fragment indexed `[value, key]` and consumed by `O = Qt @ S^T` as a
  **B** fragment indexed `[key, value]`; the two global tensor views perform that
  transpose for free today. Doing it in registers needs a work-group-wide 64x64
  transpose. Staging it through SLM is possible but not via the block-2D
  helpers — see the next two entries.
  The round trip is also almost certainly LLC-resident, since the same
  work-group re-reads it immediately, so the win is L2 -> SLM rather than
  DRAM -> SLM.
* **Feeding DPAS from SLM through the 2D block-load helpers silently returns
  zeros.** This was measured, not assumed. `get_block_2d_copy_A` accepts a tensor
  built on `make_smem_ptr` and **compiles cleanly**: nothing on the block-2D path
  asserts on the address space, because `Xe2DTraitsBase` stores the operand as
  `base_ptr((uint64_t) &*src.data())` and hands that to
  `__builtin_IB_subgroup_createBlock2DAddressPayload`, which takes a flat 64-bit
  address. At run time the kernel does not fault either — the payload's
  width/height bounds check simply clips every access, and the fragment comes
  back all zeros. A probe over a 64x32 bf16 tile returned 0 of 2048 correct
  values from SLM, while the identical code over a global tensor returned
  2048 of 2048. Treat the block-2D helpers as global-memory-only, and do not
  expect a compile error to catch the mistake.
* **A hand-written 1D vectorized SLM load does feed DPAS correctly.** Verified
  with the same probe: stage the operand in SLM in *fragment order* so each
  lane's slots are contiguous, read them with one `AlignedArray<bf16, 16>` load,
  and the fragment matches the MMA's own `partition_A` coordinates 2048 of 2048,
  with a real `cute::gemm` consuming it. So SLM staging is available if a fusion
  ever needs it — it just has to bypass CuTe's copy layer.
  cutlass-sycl's own `make_slm_copy` is *not* a working shortcut here: it has
  zero call sites in the vendored tree, and the natural
  `make_A_slm_layout` + `partition_S` spelling fails to compile with
  `Copy_Traits: src failed to vectorize into registers`, because that layout is
  strided rather than contiguous per lane.
  Trial 7 is what this costs in practice: staging a real operand recovered only
  about a third of the DRAM traffic it deleted, so budget the gather, not just
  the bytes saved.
* **`chunk_size` 64 -> 32 or 16.** `A` traffic is proportional to chunk size, so
  32 saves ~4% and 16 ~6%, and 16 would additionally let the whole `g_floor` /
  saturation-guard / `opt`-fallback subsystem be deleted (this is why FlashKDA
  picked 16). But the DPAS work-group tile is `Shape<_64,_64,_32>`, the inverse
  hard-codes a 4x4 blocking of 16x16 blocks, and `chunk_size_xe2` is shared with
  the GDN pipeline and the chunked conv1d layout. `fwd_o` — the largest stage,
  and sequential over chunks — would double its iteration count while its tiles
  shrink below their efficient shape. FlashKDA only makes 16 work by processing
  several chunks per work-group, i.e. a different kernel structure.
* **Explicit cache-control hints.** The `CacheControl` enum exists only in
  cutlass-sycl's *legacy* Xe copy path (`arch/copy_xe_legacy_*.hpp`); the modern
  `copy_traits_xe_2d.hpp` that every helper here uses does not plumb the
  parameter through, so this would mean patching vendored cutlass.
* **Narrower intermediates.** No target: `Ka`, `Kb`, `Qt`, `W`, `U` and `A` are
  already bf16, and the only fp32 buffer, `Tl`, is 1/32 of a plane and carries
  the cumulative decay where precision is load-bearing.

Also set aside: eliminating `Qt` (`fwd_o` would have to recompute the per-token
cumsum), triangular packing of `A` (breaks the dense DPAS layout), and a base-2
exponent in `prepare` (ALU savings cannot help a stage at 98% of roofline).
