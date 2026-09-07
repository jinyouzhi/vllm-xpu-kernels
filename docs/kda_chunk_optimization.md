# KDA chunk backend — optimization report

Target: `csrc/xpu/gdn_attn/xe_2/chunk_kda_kernels_xe2.hpp` and its launcher, the
Xe2 (Battlemage) chunked-prefill path of Kimi Delta Attention.

Every number here was measured on Battlemage. For a port to Crescent Island,
`kda_chunk_optimization_cri.md` restates these trials as a re-evaluation
checklist, separating the conclusions that are about the algorithm from the ones
that are about this machine.

## Summary

| | |
| --- | --- |
| GPU | Intel Graphics `0xe223` (BMG, Xe2-HPG), 2500 MHz pinned |
| Data type | bf16 activations, fp32 accumulation |
| Bottleneck | **memory bandwidth**, uniformly across all five stages |
| Baseline | 22636 us over the 7-shape chunk sweep |
| After trials 1-6 | 19782 us (-12.6%) |
| **Final (trial 7)** | **19426 us (-14.2%)** |
| End-to-end prefill, trials 1-6 (49 configs) | **-4.04% geomean**, best config -10.19% |
| End-to-end, trial 7 (133 configs) | -0.50% geomean; **-1.10%** over the 27 configs it applies to |
| Trials | 10 (5 accepted — one only for simplicity — and 5 rejected) |

Measured with `benchmark/benchmark_kda_chunk_stages.py` (chunk backend only,
per-stage attribution under `unitrace`) and `benchmark/benchmark_kda.py` (133
end-to-end configurations).

### Kimi-Linear tp1 prefill, baseline vs trials 1-6

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

Trial 7 lands on top of this and is reported separately in its own section,
against its own freshly measured baseline: the two sessions differ by up to
~1% of build-to-build drift, so its per-workload deltas are not comparable
against the table above.

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

Trial 10 later re-ran this with the spilling kernels excluded and still lost
4.75%, so spill is only part of the story — see that section for the rest.

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
and the SLM allocator — `fwd_o` now uses no SLM at all. (Trial 7 later puts SLM
back, for a much larger tile and a different reason.)

Results are bit-identical and the 133-config sweep is neutral (+0.09% overall,
-0.02% over the >=100 us subset, worst outlier +0.86% against a +-1.6% noise
band). Kept for the simpler kernel, not for speed; the barrier was evidently not
on the critical path, consistent with the stage being bandwidth bound rather
than sync bound.

## Trial 7 (accepted): SLM-stage `U` inside `fwd_o`

`U` is produced and consumed entirely within `fwd_o` on the `has_prev_state`
path — `U := U0 - W @ S0^T` was stored to global memory, then read back twice,
once as the B operand of `O += O2 @ U` and once as the A operand of
`S := S0 + U^T @ Kb`. Nothing outside the kernel reads it, so all three global
accesses collapse into one 8 KB SLM tile per value block, leaving only the `U0`
load that `compute_wu` produced. The barrier that published the store can also
drop from a global fence to a local one.

The prize was priced before writing any of it, since this machine reports no
hardware metric groups (`unitrace --metric-list` returns `No metrics found`).
A deliberately correctness-breaking probe forced `u_offset = 0` so every
(head, chunk) shared one buffer — same instruction count, same access pattern,
but guaranteed L2-resident. That measured **-5.40%** on the five tp1 prefill
workloads, confirming the round trip really does reach DRAM.

The implementation needs three new GEMM variants — `gemm_TSlmS`, `gemm_SlmTS`
and `gemm_SlmTS_fused_2B` — because the block-2D copy atoms cannot address SLM
(see the blocker list below), so the operand fragments have to be gathered from
the tile by hand. Measured GDR-only as an interleaved A/B against the trial-6
build (12463.5 us on the five tp1 prefill workloads, min of three samples,
`.so` swapped through `LD_LIBRARY_PATH` between rounds). This table attributes
the individual techniques and predates the occupancy gate described below,
which changes the shipped totals:

| variant | total | delta |
| --- | ---: | ---: |
| scalar gather for both operands | 12330.7 us | -1.06% |
| + 8-wide vector load for the A operand | 12279.7 us | -1.46% |
| **+ local-space fence for the publish barrier** | **12262.5 us** | **-1.61%** |
| single path, first chunk staged through SLM too | 12317.2 us | -0.78% |
| probe ceiling, not a shippable version | 11743.0 us | -5.40% |

`benchmark_kda_gated_delta_rule.py` agrees independently at -1.59% over the same
five workloads (12443.4 -> 12245.7 us, mean of 30 iterations), and the seven-shape
chunk sweep moves 19787 -> 19364 us (**-2.14%**), the wider shapes gaining most
(`b8_1k` -2.56%, `b16_512` -2.14%, `b32_256` -2.04%).

So the staging recovers under a third of what the round trip costs: the gather
gives back most of the bytes it saves. Keeping the first chunk of each sequence
on the global path is worth another 0.85%, because those chunks do not update
`U` and would otherwise pay for a staging round trip they do not need — hence
the two paths.

### The staging needs an occupancy gate

`b1_1k` regressed by +0.95% on the sweep above, which looked like a single
awkward shape until the full 133-config run was checked. It is not: the delta
tracks the work-group count of the `fwd_o` grid
(`batch x heads x dv_groups`) almost monotonically, and the sign flips.

| `fwd_o` work-groups | delta, ungated |
| ---: | --- |
| 8 | +6.5% .. +11.6% |
| 16 | +5.6% .. +7.0% |
| 32 | +2.8% .. +4.6% |
| 64 | -0.9% .. +1.3% |
| 128 | -0.6% .. +1.1% |
| 256 | -4.1% .. +0.7% |
| 512-2048 | -1.8% .. +0.1% |

Ungated, that is **+0.56% geomean over all 133 configs** and +1.07% over the
70 prefill and mixed ones — a net loss, despite the clean win on the shapes
the trial was developed against.

The mechanism is the trade the staging makes. It removes DRAM bytes, but it
pays for them with a hand-gathered SLM operand that has no prefetch pipeline,
replacing a block-2D load that was prefetched three tiles ahead —
`gemm_TSlmS` alone issues 64 scalar B-operand loads per k-tile. When there are
enough work-groups in flight, each one's gather hides behind the others and the
saved bandwidth wins. When there are not, the kernel is latency-bound, the
gather sits exposed on the critical path, and nothing hides it.

The fix is to stage only when the grid already saturates the machine, which is
the same threshold `chunk_kda_fwd_o_dv_groups` already uses to pick
`dv_groups`, so the gate reuses `fwd_o_target_work_items` rather than
introducing a second constant:

```c++
const int64_t fwd_o_work_items =
    int64_t(batch_size) * num_heads * dv_groups * wg_size;
stage_u_in_slm = local_dv >= 1 && local_dv <= 2 &&
                 fwd_o_work_items >= fwd_o_target_work_items;
```

Gated, over the 133-config sweep (`benchmark_kda.py`, same build A/B-swapped):

| | configs | geomean | net |
| --- | ---: | ---: | ---: |
| chunk backend, staged | 27 | **-1.10%** | -1307 us |
| chunk backend, gated off | 43 | +0.12% | +107 us |
| not the chunk backend | 63 | -0.66% | -8 us |
| all | 133 | **-0.50%** | -1208 us |

The third row matters for reading the other two. Decode and spec workloads do
not run this pipeline at all — they take the recurrent kernel — so those 63
configs are a control group that no change here can move, and they come out at
-8 us total. Only the first two rows are this trial's doing.

### The gated-off path has to be a separate kernel

The gated-off group is the useful diagnostic here, because it is supposed to
be *zero*: the gate sends it down the same global path the trial-6 kernel took.
It was not. With staging as a runtime `if` inside one kernel the 43 gated-off
chunk configs came out at **+128 us**, and the sign was suspiciously consistent
— four interleaved A/B rounds on the tp1 prefill shapes had the gated-off ones
slower every single round, by 0.1-0.4%.

A branch that is never taken still costs, in two ways. The work-group declares
the SLM allocation whether or not it stages, and trial 6 had deliberately left
`fwd_o` using **no** SLM at all, so a shared kernel silently gave that up on
every shape. Register allocation likewise covers both paths, so the staged
path's live values depress occupancy even where it is switched off.

So `stage_u_in_slm` is a template parameter, not a runtime flag, and the
launcher dispatches to one of two kernels with `std::true_type` /
`std::false_type`; the non-staged instantiation allocates a zero-length
`local_accessor`. That helped **both** sides — the staged path also stops
paying for the global path it no longer uses:

| | shared kernel | split kernels |
| --- | ---: | ---: |
| chunk, staged (27) | -0.92% / -1060 us | **-1.10% / -1307 us** |
| chunk, gated off (43) | +0.25% / +128 us | **+0.12% / +107 us** |
| not the chunk backend (63) | -0.29% / -5 us | -0.66% / -8 us |
| all (133) | -0.24% / -936 us | **-0.50% / -1208 us** |

The gated-off residual shrank but did not reach zero: the 63 configs that never
enter this pipeline net -8 us, which is the control group behaving correctly,
while the 43 gated-off *chunk* configs still net +107 us (~+0.25% each).
Something small still differs on that path; it is worth an extra 1207 us
elsewhere, so it ships, but it is not noise and should not be written off as
such.

That control group is also what catches bad measurements. A non-interleaved
run of the split build appeared to show the gated-off group **-2.99%** faster,
which would have been nonsense — the recurrent-path configs moved -13% in it,
and no `fwd_o` change can touch those. Every config under 40 us had simply
dropped ~3 us of launch overhead for that run. All the numbers above are
therefore min-of-two with the builds alternated between runs.

The gated-off configs take the same *branch* as the baseline but no longer the
same *binary*, which is the whole point of the previous section; their residual
is small but not zero. Where the noise floor really can be read off is the
recurrent-path control group, at about +-1% on the ~26 us decode shapes — which
also says the worst staged case, +0.94%, is not distinguishable from noise. The
worst real regression is gone either way: +11.6% before the gate, +0.94% after,
and only three staged configs regress at all.

Restricting the same sweep to the four Kimi-Linear shapes, dropping the
synthetic ones, does not change the picture, but it does show where the gain is
and is not:

| group | configs | geomean | net |
| --- | ---: | ---: | ---: |
| chunk backend (prefill + mix) | 40 | **-0.49%** | -1062 us |
| — of which staged | 21 | **-1.13%** | -1108 us |
| recurrent backend (decode + spec) | 36 | +1.93% | **+25 us** |

The decode and spec workloads never enter this pipeline, so read them in
absolute terms: 36 configs, +25 us total, under a microsecond each. Their
+1.93% geomean is the same artefact the control group exists to expose — most
of them are ~26 us measurements where a fraction of a microsecond of launch
jitter is a percent, so the geomean of that group carries no information and
only the net does.

The gate also interacts with tensor parallelism, which is worth knowing before
reading much into any single TP configuration. `dv_groups` is bounded by
`head_dim / chunk_size`, so the grid scales with the head count, and sharding
heads across ranks shrinks it: at tp1 seven of ten chunk configs stage, at tp8
only three do, and those three are within noise. A tp8 rank needs batch >= 64
before staging engages at all. The optimization is therefore worth most at low
TP and long sequences, and the gate is what keeps it from being actively
harmful at high TP.

The gate costs some of the headline. The seven-shape chunk sweep gives back
0.33pp (19906 -> 19546 us, **-1.81%** instead of -2.14%) and the five tp1
prefill workloads give back 0.4pp, because `b1_1k`, `b1_4k` and `b1_8k` are
64-work-group shapes and now take the global path (min of four interleaved
rounds):

| workload | `fwd_o` WGs | staged | before | after |
| --- | ---: | --- | ---: | ---: |
| `prefill_b1_1k` | 64 | no | 394.4 | 395.6 (+0.31%) |
| `prefill_b1_4k` | 64 | no | 1707.6 | 1707.6 (-0.00%) |
| `prefill_b1_8k` | 64 | no | 3464.8 | 3467.8 (+0.09%) |
| `prefill_b4_2k` | 256 | yes | 3481.5 | 3416.5 (**-1.87%**) |
| `prefill_b8_1k` | 256 | yes | 3409.8 | 3319.6 (**-2.65%**) |
| total | | | 12458.2 | 12307.2 (**-1.21%**) |

That is the right trade: -1.61% on five hand-picked shapes is worth less than
-0.50% across everything, and the three shapes it gives up were within noise of
break-even anyway. Thresholds of 64 and 128 work-groups were also measured and
are indistinguishable overall (-0.19% both), but leave 3-4x more regression on
the table, so the saturation point is both the best and the most principled of
the three.

Three details are load-bearing and easy to get wrong:

* The updated `U` is subtracted in the **MMA accumulator** layout, not the
  block-2D copy fragment layout. The two have different element orders, so the
  hand-written SLM write must use this kernel's usual `sn`/`sm` index formula
  against a `partition_sg_fragment_C` tensor. Indexing the copy fragment by
  `partition_S` coordinates instead compiles and mostly works, but corrupts
  ~0.1% of the recurrent state.
* The tile is staged `(token, channel)`, i.e. `(K, N)` row-major for the `U^T`
  operand both consumers take. A DPAS A/B sub-group fragment walks the **M/N**
  axis within a lane, not `K`, so this is the orientation that keeps each lane's
  slots contiguous. The intuitive `(N, K)` choice measured **~6% slower**
  (12994.8 us) — larger than the entire optimization.
* Because of that layout, an A operand's slots are not just close together but
  *adjacent*: one DPAS atom's M extent of them is a single 16-byte load. Driving
  the gather from `thr_mma.partition_A/B` of an identity tensor is what makes
  this safe to assume — the coordinates come from the same partitioner as the
  fragment, so slot `i` means the same thing in both. Vectorising the A operand
  is worth -0.40% on its own and is bit-identical to the scalar gather; a B
  operand is in VNNI order, its slots are 8 elements apart, and it stays scalar.

The publish barrier changes with it. It used to have to be a full
`sycl::group_barrier`, because what it ordered was a *global* store of `U`;
staged, it only has to order local memory, and
`item.barrier(access::fence_space::local_space)` is worth a further -0.15%. The
non-staged path keeps `sycl::group_barrier`, which still has a global store to
cover. This also puts SLM back in `fwd_o`, which trial 6 had emptied — for a
much larger tile, but one that is read by DPAS rather than by scalar indexing.

## Trial 8 (rejected): pad the SLM tile pitch from 64 to 72

The staged `U` tile is 64x64 bf16 with a pitch of exactly `chunk_size` = 64
elements = 128 bytes. A power-of-two pitch is the classic SLM bank-aliasing
trigger, and a bank model of the two DPAS operand gathers agreed: at pitch 64
the A operand touches only 4 of 16 banks (16-way conflict) and B is 2-way; at
pitch 72 — still 16-byte aligned, so the vectorised A load survives — A drops
to 4-way and B is conflict-free. It held for both 16- and 32-bank models.

Result, bit-identical and **+6.35% overall**, concentrated exactly on the two
staged workloads: `prefill_b4_2k` +11.99%, `prefill_b8_1k` +11.07%, while the
three gated-off `b1_*` shapes moved by <=0.03%.

Splitting the change in two is what explains it. Allocating the padded 64x72
tile but *indexing* it at stride 64 isolates capacity from arithmetic:

| variant | `b4_2k` | `b8_1k` | total |
| --- | ---: | ---: | ---: |
| padded allocation only | +0.15% | +0.23% | **+0.15%** |
| padded allocation + padded indexing | +11.87% | +11.42% | **+6.38%** |

So bank conflicts were never the cost — the pitch arithmetic was. This is worth
recording twice over, because the standard advice is the opposite: padding SLM
to break bank aliasing made this kernel 12% slower.

It also leaves a genuinely useful positive: **the marginal SLM kilobyte is
nearly free here** (8192 -> 9216 bytes per tile cost 0.15%), so these staged
shapes are not SLM-capacity bound. That does not retroactively invalidate the
SLM argument in *The gated-off path has to be a separate kernel* above — that
comparison was 0 -> 8 KB, which can cross an occupancy cliff that 8 -> 9 KB
does not.

## Trial 9 (rejected): give `U` a single source

`fwd_o` carries `has_prev_state = (local_chunk != 0) || initial_state`, and only
the true branch reads staged `U`; chunk 0 falls back to global `U`. Since SLM
turned out to be nearly free, chunk 0 could stage `U0` too, collapsing the
runtime branch into the compile-time `StageU` and deleting the global-`U` GEMMs
from the staged instantiation entirely.

It was bit-identical and **+6.32% overall** — `b4_2k` +11.33%, `b8_1k` +11.96%.
The gated-off `b1_*` shapes actually improved slightly (-0.13% to -0.16%),
which is consistent: they run the *other* instantiation, which did get smaller.

## The staged `fwd_o` kernel sits on the 256-GRF allocation cliff

Two unrelated changes each costing 11.3-12.0% on exactly the two staged shapes
is not a coincidence, it is a quantized resource cliff. IGC shader dumps
(`IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=<dir>` in front of the build; touch
a source first to force a rebuild) name it directly. Compare the `//.RA type`
and `//.spill size` lines in the `.asm` for
`ChunkKdaFwdOKernelIN7cutlass10bfloat16_tEfLb1EEE`:

| | RA type | spill |
| --- | --- | ---: |
| current | `HYBRID_BC_RA` | none |
| trial 9 | `GRAPH_COLORING_SPILL_FF_BC_RA` | **2816 B** |

and the message mix moves with it — the spilling build gains 72 bare
`load/store.ugm.d32x{16,64}t.a32` scratch messages (no cache-control suffix,
unlike real global traffic which is `a64`) and doubles its SLM messages.

Two practical consequences:

* **Screen with the dump before benchmarking.** A build is ~8 minutes and a
  trustworthy interleaved A/B is far longer; grepping two lines out of the
  `.asm` rejects a change for free. `//.RA type` containing `SPILL` is
  disqualifying on its own.
* Do not trust `.zeinfo` for this. Its `spill_size` field reported `0` for the
  spilling build, and its `slm_size` reported `0` for a kernel that allocates
  8 KB. The vISA statistics footer in the `.asm` is the authoritative source.

Anything added to this kernel has to pay for itself *and* stay under the limit.
That is the standing reason the remaining `fwd_o` ideas below are hard.

## Trial 10 (rejected): 128-GRF only where it provably does not spill

Trial 2 lowered the chunk-parallel stages to `grf_size<128>` and lost 24.8%,
attributing it to spill. With the dump-based screen that explanation can be
tested rather than assumed — and it turns out to be incomplete.

Screening every kernel at 128 GRF shows only two that actually spill:
`compute_wu` (4352 B) and the `V=16` (head_dim 256) `prepare` pack (1344 B).
Leaving those two at 256 and dropping the rest — `prepare` V=2/4/8, the scalar
`prepare`, `compute_A`, `inverse` — gives a **provably spill-free** 128-GRF
build covering ~40% of pipeline time, at twice the resident threads per EU.

It is still slower: bit-identical, **+4.75% total**, and this time uniformly
across all five workloads (+3.81% to +5.77%), as expected for a change to
stages every shape runs.

So spill was never the whole story. The pipeline already runs at ~92% of its
achievable streaming bandwidth, and *at that point extra occupancy has no idle
bandwidth left to fill*, while halving the per-thread register budget costs
unrolling and in-flight loads. Total bytes in flight is roughly threads x
registers, so trading one for the other is neutral at best. `grf_size<256>` is
the right setting for this pipeline for a reason that has nothing to do with
spill, and lowering it should not be retried on occupancy grounds.

## Remaining opportunities

Everything left is cross-kernel fusion: keeping an intermediate in registers
instead of round-tripping it through DRAM. Sized by the traffic of the operand
each fusion deletes, as a share of the pipeline's 4900 B per token-head:

| fusion | operand deleted | share |
| --- | --- | ---: |
| `prepare` + `compute_A` | `Ka`, `Kb` (write + read) | ~9% |
| `compute_A` + `inverse` + `compute_wu` | `A` (cold read only) | ~2% |

Trial 7 already took `U`'s round trip, which this table sized at ~7%; it
returned 1.6%. That gap is a warning about the remaining rows: the shares are
source-level operand bytes, and trial 7 is the one case where the realisable
fraction was measured end to end. It came out at roughly a third, because the
gather that replaces a block-2D load is not free. Treat the shares above as
upper bounds, and price any fusion with an L2-residency probe first.

This mirrors FlashKDA's split (see `docs/kda_attention_design.md`): its K1 kernel
is our `{prepare, compute_A, inverse}` and its K2 is our `fwd_o`. FlashKDA
reports trying a *single* fully-fused kernel first and finding it **worse** —
the two-kernel split gave it >=15% — so `fwd_o` should stay separate.

Known blockers, so they are not re-derived:

* **`prepare` + `compute_A`** is no longer blocked by the sub-group width, which
  trial 4 fixed. What remains is the grouping: `prepare` covers one (chunk, head)
  as 256 work items partitioned along `head_dim`, while `compute_A` partitions
  the 64x64 `A` tile as a 4x2 sub-group grid. One decomposition serving both
  costs occupancy on small shapes (`b1_1k` would drop to ~25%). Trial 7's gate
  is a working precedent for dispatching by shape, so the small-shape objection
  is answerable; the decomposition itself is the real work.
* **Recomputing `Ka` or `Kb` instead of storing both is arithmetically dead**,
  so it should not be re-attempted. With `Ka = k_hat * exp(G)` and
  `Kb = k_hat * exp(-G) * beta`, deriving either from the other requires `G`,
  which is head_dim-sized — exactly the bytes the elimination was supposed to
  save. Every rearrangement of the same idea (storing `exp(G)` in place of `Qt`,
  having consumers re-scan the cumulative sum, and so on) nets zero. Separately,
  `prepare` writes a chunk-aligned, zero-padded workspace precisely so no
  downstream GEMM needs predication; reading the ragged `q`/`k` directly would
  reintroduce varlen indexing into every consumer.
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
  with a real `cute::gemm` consuming it.
  cutlass-sycl's own `make_slm_copy` is *not* a working shortcut here: it has
  zero call sites in the vendored tree, and the natural
  `make_A_slm_layout` + `partition_S` spelling fails to compile with
  `Copy_Traits: src failed to vectorize into registers`, because that layout is
  strided rather than contiguous per lane.
  Trial 7 shipped this: `gdn::gather_sg_fragment` in `gemm.hpp` is the reusable
  form, driven by `thr_mma.partition_A/B` of an identity tensor so the slot
  coordinates come from the same partitioner as the fragment, and
  `gemm_TSlmS` / `gemm_SlmTS` / `gemm_SlmTS_fused_2B` are the GEMMs built on it.
  Trial 7 is also what it costs in practice: staging a real operand recovered
  only about a third of the DRAM traffic it deleted, so budget the gather, not
  just the bytes saved.
* **Peel the first chunk out of the sequential loop.** `has_prev_state` is
  `(local_chunk != 0) || initial_state`, so within a launch it can only be false
  on the very first iteration and is true for every one after. It is
  nonetheless a runtime branch, which means the `StageU == true` kernel carries
  *both* `gemm_SlmTS_fused_2B` and `gemm_TTS_fused_2B` and its register
  allocation has to cover both. That is the same defect the split kernels just
  fixed between launches, still present inside one. Peeling iteration zero out
  of the loop would leave a branch-free body. The cost is a duplicated loop
  body, and unlike the gated-off case the second path is not dead code — every
  sequence really does execute it once — so this is worth measuring rather than
  assuming.
* **Vectorizing the B-operand gather is the obvious next step for trial 7.**
  Only the A operand is vector-loaded today; a B operand is in VNNI order, so
  its slots are 8 elements apart and `gemm_TSlmS` falls back to 64 scalar SLM
  loads per k-tile. That unprefetched burst is the most likely cause of the
  low-occupancy penalty the gate now steps around, so removing it could widen
  the gate or retire it. It needs the operand staged in a second, N-strided
  orientation (or a transposed tile), which conflicts with the `(token,
  channel)` layout the A-operand consumer requires — so the two consumers of
  `U` would need different staged copies, and that has to be priced against the
  extra SLM and the extra write.
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
