# Re-evaluating the KDA chunk optimizations on Crescent Island

`docs/kda_chunk_optimization.md` records ten trials against the KDA chunk
backend on Battlemage. This document restates them as a **re-evaluation
checklist for CRI (Xe3P)**: what each one did, what it cost or bought on BMG,
and — the part that matters — which of them were conclusions about *this
algorithm* and which were conclusions about *that machine*.

## Read this first: one number decides almost everything

Every verdict below is downstream of a single measurement:

> The BMG pipeline moves ~4900 B per token-head and ran at **363 GB/s, 92% of
> the ~393 GB/s achievable streaming rate**, while using only 15-17% of the
> 102 TFLOP/s bf16 XMX peak.

That is why arithmetic was free, bytes were not, and why every accepted trial
deletes traffic while every rejected one merely rearranges work. It is also why
occupancy tuning lost: at 92% of the bandwidth roof there is no idle bandwidth
for extra threads to fill.

**On CRI that ratio is the thing to re-measure before anything else.** CRI is
server/HPC-oriented, so if its achievable bandwidth per unit of compute is
higher, the pipeline moves off the roof and the whole ranking changes: the
latency-hiding and occupancy trials become plausible again, and the
traffic-deleting ones keep working but buy less. If the ratio is similar, the
BMG conclusions should mostly stand and this document is mainly a list of
things you can skip.

So step 0 is not a trial. It is:

1. Measure the achievable streaming ceiling on the actual part (a large
   `torch.copy_` sweep is enough; do **not** use the vendor paper figure — on
   BMG the paper number was 456 GB/s and the achievable one 393 GB/s, and
   grading against 456 would have hidden that this pipeline was already at the
   roof).
2. Re-run the per-stage attribution and divide each stage's hand-counted
   operand bytes by its time.
3. Compare against the ceiling. **If the pipeline is at >85% of achievable
   bandwidth, the BMG ranking transfers. If it is below ~70%, re-open the
   rejected trials — several of them lost for reasons that only exist at the
   roof.**

The per-stage byte counts are algorithmic and carry over unchanged:

| stage | B / token-head | what |
| --- | ---: | --- |
| `prepare` | ~1540 | reads `q`, `k`, `gate`; writes `Ka`, `Kb`, `Qt` |
| `fwd_o` | ~1400 | `W`, `U`, `Qt`, `Kb`, `O`, and the `S` carry |
| `compute_wu` | 1152 | `A`, `Ka`, `v`; writes `W`, `U` |
| `compute_A` | 640 | `Ka`, `Kb`; writes `A` |
| `inverse` | ~128-500 | `A` in place |

The algorithmic floor — read `q`, `k`, `v`, `gate`, write `O` and nothing else —
is 1280 B/token-head, so a perfectly fused implementation is ~3.8x away on any
architecture. That bound is arithmetic, not hardware.

## The ten trials, and whether each one is portable

| # | Trial | BMG | Why it went that way | CRI |
| --- | --- | ---: | --- | --- |
| 1 | Read `v` in place instead of materializing `Vp` | **-4.04%** prefill | Deletes 25% of all bytes | **Keep**, re-measure size |
| 2 | `grf_size<128>` on chunk-parallel stages | +24.8% | Spill, *and* no idle BW | **Re-open** |
| 3 | Alternate traversal direction between stages | **-1.4%** | LLC residency between stages | **Re-measure** |
| 4 | Run `prepare` at the native sub-group width | **-0.70%** | Matches HW SIMD width | **Re-check width** |
| 5 | Conditionally size the `Vp` workspace plane | +0.22% | Allocation size is not the cost | Skip |
| 6 | Read `Tl` into registers in `fwd_o` | neutral | Barrier was not on critical path | Keep (simpler) |
| 7 | SLM-stage `U` inside `fwd_o` | **-1.10%** (27 cfg) | Deletes a global round trip | **Re-tune gate** |
| 8 | Pad SLM tile pitch 64 -> 72 | +6.35% | Pitch arithmetic, not banks | **Re-test** |
| 9 | Give `U` a single source | +6.32% | Pushed `fwd_o` past 256 GRF | **Re-test** |
| 10 | 128 GRF only where provably spill-free | +4.75% | No idle BW to fill | **Re-open** |

Cumulative on BMG: 22636 -> 19426 us, **-14.2%**.

### Portable, because they are about the algorithm

* **Trial 1 — do not materialize `Vp`.** `compute_wu` can read `v` through the
  varlen index instead of having `prepare` write a chunk-aligned copy. This is a
  pure deletion of a 67 MB store plus a 67 MB load per 8k-token pass — 25% of
  all bytes in the pipeline. It wins on any bandwidth-sensitive machine and is
  never worse than neutral. The only CRI question is how much it is worth, not
  whether to keep it.
* **Trial 6 — `Tl` in registers.** Kept for a simpler kernel, not for speed
  (bit-identical, neutral). Nothing to re-evaluate; just do not be surprised
  that it shows no gain.
* **Trial 5 — sizing the `Vp` plane** is a closed question in the other
  direction: workspace *allocation size* was not a cost on BMG (+0.22%), and the
  same is very likely true anywhere. Skip it.
* **The `Ka`/`Kb` recompute dead end.** With `Ka = k_hat * exp(G)` and
  `Kb = k_hat * exp(-G) * beta`, deriving either from the other needs `G`, which
  is head_dim-sized — exactly the bytes you were trying to save. This is
  algebra, so it is dead on CRI too. Do not re-derive it.

### Need re-measurement, because they are about the cache and the SIMD width

* **Trial 3 — alternate the traversal direction.** Consecutive stages walk the
  chunk space in opposite order so the tail of stage *n* is still resident when
  stage *n+1* starts. On BMG this was worth -1.4%, concentrated in `InverseOpt`
  (-12.4%), and `compute_A` showed a 46% superlinear penalty when it was
  disabled. The mechanism is entirely last-level-cache residency, so it is
  sensitive to CRI's cache size and hierarchy — and CRI replaces BMG's
  `MemoryProfile` with `L1Profile`, which suggests the memory path is different
  enough to matter. Re-measure; it could be larger or smaller. It is free to
  keep either way, being bit-identical and a pure loop-order change.
* **Trial 4 — `prepare` at the native sub-group width.** This matched the launch
  to the hardware SIMD width instead of splitting a (chunk, head) across
  sub-groups. **Verify `cute::detail::subgroup_size` and the native width on
  CRI before assuming the current value is right** — if CRI's preferred width
  differs, this trial's tuning is stale rather than wrong, and the `V =
  2/4/8/16`
  pack-width dispatch needs re-deriving from `head_dim / sub_group_size`.
* **Trial 7 — SLM-stage `U`.** The staging itself deletes a global round trip
  and should still pay off. What will not transfer is the **occupancy gate**:
  it is a work-group-count threshold tuned to BMG's Xe-core count, and it
  decides which shapes stage and which fall back to the global path. Re-derive
  the threshold on CRI. Note also the general guidance that "SLM is
  counterproductive on BMG" did *not* hold here — this tile is read by DPAS,
  not by scalar indexing — so treat CRI's SLM path on its own evidence.

### Re-open these, because they lost for reasons specific to BMG

These are the interesting ones. All three are **bit-identical** to the current
code, so they are pure performance experiments with no correctness risk.

* **Trials 2 and 10 — GRF sizing.** Trial 2 dropped the chunk-parallel stages to
  `grf_size<128>` and lost 24.8%; trial 10 repeated it with every spilling
  kernel excluded (only `compute_wu` at 4352 B and the `V=16` `prepare` pack at
  1344 B actually spill at 128) and *still* lost 4.75%, uniformly across all
  five workloads. The residual reason was the bandwidth roof: doubling resident
  threads cannot help when there is no idle bandwidth, while halving the
  register budget costs unrolling and in-flight loads.

  **On CRI this is the most likely verdict to flip**, for two independent
  reasons. First, if step 0 shows the pipeline is *not* at the bandwidth roof,
  the entire objection evaporates. Second, CRI exposes
  `XVE_GRFBLOCK_OCCUPANCY_ALL` — direct register-pressure telemetry that BMG
  simply does not have. On BMG this had to be inferred by grepping compiler
  output; on CRI you can measure it. Sweep GRF per stage with real occupancy
  data rather than guessing.

* **Trial 8 — SLM pitch padding.** Padding the 64-element (128 B) staged `U`
  tile pitch to 72 cost 11-12% on the staged shapes. The decisive experiment
  split allocation from indexing: the padded *allocation* alone was +0.15%,
  the padded *indexing* was +6.38%. So bank conflicts were never the cost — the
  power-of-two pitch is what BMG's SLM fragment-load path wants, and padding
  defeated whatever wide message the compiler was emitting.

  This contradicts the standard "pad SLM to break bank aliasing" advice, and it
  is a statement about one compiler backend and one memory path. **CRI's memory
  path is different enough that the standard advice may actually hold there.**
  Re-test — and re-use the split methodology, because "padding is slow" and
  "the extra kilobyte is slow" are different claims with different fixes.

  Corollary worth carrying over as a hypothesis to re-verify: on BMG the
  **marginal SLM kilobyte was nearly free** (8192 -> 9216 B/tile cost 0.15%),
  i.e. these shapes were not SLM-capacity bound. If that also holds on CRI it
  unlocks the same design freedom it did here.

* **Trial 9 — single `U` source.** `fwd_o` carries
  `has_prev_state = (local_chunk != 0) || initial_state` and only the true
  branch reads staged `U`; chunk 0 falls back to global `U`. Staging `U0` too
  collapses that runtime branch into the compile-time `StageU` and removes the
  global-`U` GEMMs from the staged instantiation entirely. It is a genuine
  simplification that lost 6.32% **purely because it pushed `fwd_o` over the
  256-GRF cliff** (see below) — not because the idea is wrong. On CRI, screen it
  for spill first; if it does not spill, it may well win.

## The 256-GRF allocation cliff, and how to detect it

This is the single most useful mechanism found on BMG, and the detection
technique is worth more than the finding.

Trials 8 and 9 both cost 11-12% on *exactly* the two staged shapes. A
reproducible, quantized jump like that is a resource cliff, and IGC shader dumps
named it. For `ChunkKdaFwdOKernelIN7cutlass10bfloat16_tEfLb1EEE`:

| | `//.RA type` | `//.spill size` |
| --- | --- | ---: |
| current | `HYBRID_BC_RA` | none |
| trial 9 | `GRAPH_COLORING_SPILL_FF_BC_RA` | **2816 B** |

The spilling build also gained 72 bare `load/store.ugm.d32x{16,64}t.a32`
scratch messages — distinguishable from real global traffic, which is `a64` and
carries `.ca`/`.cc` cache-control suffixes.

**The staged `fwd_o` instantiation sits right at the 256-register limit.**
Anything added to it has to pay for itself *and* stay under the limit. That is
the standing reason the remaining `fwd_o` ideas are hard, and it is the first
thing to re-establish on CRI, because CRI's register file and allocator
behaviour may put the cliff somewhere else entirely.

Two mechanics to carry over:

```bash
# dump (touch a source first, or the build is a no-op)
IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=<dir> ninja <target>

# screen, per kernel .asm
grep -E '^//\.(RA type|spill size)' <dir>/*entry_*.asm
```

* **`.zeinfo` cannot be trusted for this.** It reported `spill_size: 0` for the
  build that vISA says spills 2816 B, and `slm_size: 0` for a kernel that
  allocates 8 KB. The `.asm` vISA statistics footer is the authoritative source.
  Any `*_SPILL_*` allocator in `//.RA type` is disqualifying on its own.
* **Use it as a free pre-benchmark screen.** A build is minutes; a trustworthy
  interleaved A/B is far longer. Trial 10 was designed entirely from dump data —
  it identified exactly which two kernels spill at 128 GRF with **zero**
  benchmark runs.

On CRI, prefer `XVE_GRFBLOCK_OCCUPANCY_ALL` where you can get it: it measures
what the dump only lets you infer. Keep the dump screen as the cheap
pre-filter, since it needs no GPU time at all.

## Remaining opportunities, unchanged by architecture

Everything left is cross-kernel fusion — keeping an intermediate in registers
rather than round-tripping it through DRAM. Sized by the traffic each fusion
deletes as a share of the 4900 B per token-head:

| fusion | operand deleted | share |
| --- | --- | ---: |
| `prepare` + `compute_A` | `Ka`, `Kb` (write + read) | ~9% |
| `compute_A` + `inverse` + `compute_wu` | `A` (cold read only) | ~2% |

Calibration from BMG: trial 7 was sized at ~7% by this same method and returned
1.6%, so **treat these shares as upper bounds and expect roughly a third.**

The structural blockers are properties of the code, not the hardware, so they
apply on CRI too:

* **`prepare` + `compute_A`** — `prepare` covers one (chunk, head) as 256 work
  items partitioned along `head_dim`, while `compute_A` partitions the 64x64 `A`
  tile as a 4x2 sub-group grid. One decomposition serving both costs occupancy
  on small shapes. Trial 7's gate is a working precedent for dispatching by
  shape, so the small-shape objection is answerable; the decomposition is the
  real work.
* **`compute_A` + `inverse` + `compute_wu`** — `chunk_gemm_policy_inverse` is
  `16x16x16` with `SGLayout<_1,_1,_1>`, i.e. **16** work items per work-group
  against 128 for the other two. `mma.get_slice(local_id)` is invalid past 16
  and the GEMM helpers contain work-group-scoped barriers, so a fused 128-wide
  group can neither run the inverse on sub-group 0 alone (deadlock) nor
  redundantly (costs more than it saves).
* **Register-resident `S`** — produced as a C fragment indexed `[value, key]`
  and consumed by `O = Qt @ S^T` as a **B** fragment indexed `[key, value]`; the
  global tensor views perform that transpose for free today. Doing it in
  registers needs a work-group-wide 64x64 transpose. The round trip is also
  almost certainly LLC-resident, so the win is L2 -> SLM, not DRAM -> SLM.
* **Block-2D copy atoms are global-memory-only.** Given a `make_smem_ptr` they
  compile and do not fault, but bounds-checking clips everything and the
  fragment returns zeros. Any SLM staging must use the hand-written 1D vector
  gathers.
* A single fully-fused kernel is the wrong target: FlashKDA reports trying it
  and finding it **worse**, with its two-kernel split giving >=15%. Its K1 is
  our `{prepare, compute_A, inverse}` and its K2 is our `fwd_o`.

## New on CRI, with no BMG equivalent

Two capabilities exist on CRI that were simply unavailable here, so they were
never trialled and are not in the table above:

* **XMX FP8 / FP4.** CRI adds FP8 and FP4 to the XMX datatype list (and drops
  INT2). This pipeline is bandwidth bound with XMX at 15-17% utilisation, which
  is exactly the profile where narrowing a *stored* operand is attractive: it
  deletes bytes on a machine that has spare arithmetic. The `S` state carry and
  the `A`/`W`/`U` workspaces are the candidates. This needs a numerical study
  first — KDA's decay terms span a wide dynamic range, and the current design
  already parameterises `StateT` over `float`/`bfloat16`, which is the natural
  place to extend.
* **Direct FLOP, lane-utilisation and GRF-occupancy counters.** On BMG, XMX
  utilisation had to be estimated from instruction counts and register pressure
  from compiler dumps. On CRI both are measurable. Re-do the roofline placement
  with real counters rather than the hand-counted byte model — the hand model is
  reliable but it is a model, and step 0 above depends on it.

Also note the 4th ALU pipe: a single pipe is now 25% of peak issue bandwidth
rather than 33%, so any BMG conclusion phrased in terms of ALU pipe saturation
needs re-deriving. None of the ten trials above turned on that, but the
per-stage analysis in the main document occasionally reasons about it.

## Measurement discipline (this part is not negotiable anywhere)

The BMG work produced two false results before this protocol was adopted, so it
is worth restating for the new environment:

* **Interleave the A/B.** Within-run variance is ~0.5% but build-to-build drift
  is ~1.26%, which is larger than most of the effects being measured. Swap the
  `.so` between runs via `LD_LIBRARY_PATH` and take min-of-N. A non-interleaved
  sweep once showed a -2.99% "gain" that was entirely launch-overhead drift.
* **Keep a control group.** The 63 decode/spec configs run the recurrent kernel
  and never enter `fwd_o`, so they must not move. When they moved -13%, that
  identified the measurement as broken rather than the kernel as fast. Any
  config under ~40 us is at the launch-overhead floor and cannot be trusted
  individually.
* **Check bit-identity first.** Every trial here was bit-identical, which
  reduces each one to a pure performance question and removes tolerance
  arguments entirely. If a CRI variant is *not* bit-identical, that is a
  separate claim needing separate evidence.
* **Watch the loader path.** `_xpu_C.abi3.so` carries a `RUNPATH` listing
  `build/temp` first, so overwriting only the copy in `vllm_xpu_kernels/`
  silently measures the same binary twice.
