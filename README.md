# Qwen3.8-Flash-Next @ 128GB GB10 — a speed-focused local deployment

Run **Qwen3.8-Flash-Next** (a large MoE model with a 51B-parameter N-gram embedding table) at full serving quality on a **128GB unified-memory GB10 box** (ASUS Ascent GX10 / NVIDIA DGX Spark class), using vLLM plus seven layers of memory and speculative-decoding engineering.

**Measured on the target box (all numbers reproducible with this repo, see [Benchmarks](#benchmarks)):**

| Metric | Value |
|---|---|
| Single-stream decode (low-thinking, sgbench) | Q&A 49.3 · Code 60.7 · JSON 61.3 · Math 68.6 · LongCode 60.7 tok/s |
| MTP acceptance length | 5.47 (no-thinking) / 4.80 (thinking), K=6 |
| KV cache | 23.1 GiB / ~748K tokens / 2.86× concurrency at 262K ctx |
| Context | 262,144 tokens, prefix caching on |
| Stability | long-term resident serving, zero errors / preemptions / queueing in `/metrics` |

```
┌────────────────────────────────────────────────────────────┐
│  GB10 box (128GB unified memory, weight budget ≈ 95GB)     │
│                                                            │
│  vLLM                                                      │
│  ├── target: Qwen3.8-Flash-Next  NVFP4 + FP8-hybrid        │
│  │     MoE experts = NVFP4 · dense/heads = FP8 · lm_head FP4 │
│  ├── PLE N-gram table: 51B params → 1.5GB via PQ (M=5)     │
│  │     GPU-side gatherer → no splitting ops                │
│  ├── MTP draft head: reduced vocab (dv=147456) + 4-bit      │
│  │     draft-step cost b ≈ 1.2 ms  (≈ T_step / 57)         │
│  ├── speculative decoding: K=6 (the physical ceiling here) │
│  └── CUDA graphs: FULL_DECODE_ONLY, splitting_ops=[]       │
│                                                            │
│  OpenAI-compatible API on :18300                           │
└────────────────────────────────────────────────────────────┘
```

## Why this repo exists

The official Qwen3.8-Flash-Next release targets server-class HBM. On a GB10 box the problem is different: **128GB of unified LPDDR5X is shared between weights, KV cache, activations and the OS page cache**, and the effective weight budget after everything else is ~95GB. Two blocks make this model hostile to a small box:

1. The Transformer stack (a large MoE) — solvable with NVFP4 quantization.
2. A **~51B-parameter N-gram embedding table (PLE, ~320M rows × 16 heads)** — larger in bytes than most deployable dense models, and read on every token.

This repo is the complete, self-verifying deployment that solves both, plus the speculative-decoding retuning that turns the freed memory and bandwidth into ~2× single-stream throughput over the naive path. All optimizations below were adopted or rejected by measurement on the target box; dead ends are documented too (see [Tuning notes](#tuning-notes-hard-won)).

## The seven optimizations

### 1. Base weights: NVFP4 quantized checkpoint

The official BF16 weights do not fit (≈3.5–4× the footprint of a 4-bit variant). We deploy [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) — per-expert NVFP4 MoE with proper `input_scale`/`weight_scale_2` — as the admission ticket. Capacity rule of thumb on this box (util 0.90): weights ≤ ~95GB ≈ NVFP4 up to ~180B / FP8 up to ~90B.

> Lesson learned the hard way on this platform: a 3-bit-quantized + REAP-pruned community MoE looked fine on perplexity but rolled routing errors into garbage over long multi-turn generations. **Never stack aggressive PTQ + pruning into production; acceptance-test with long-text loop tests, not ppl.**

### 2. The 51B N-gram table: product-quantized (PQ, M=5), table in VRAM

`ple_pq/` — the biggest single win.

The PLE table cannot live in VRAM at any usable precision, and host-side mmap access costs page-mapping per lookup. The first attempt (HashK R=4, community) shrank it to 11.9GB but left only ~15GiB for KV. **Product quantization M5** — M=5 subspaces, K=256 centroids, seg_dim=32, 16 heads — keeps **all 320M rows with zero collisions** (every row remains exactly addressable and reconstructable) in **1.5GB** total (codes 1.49GiB + codebooks 2.5MB):

- builds in ~200s (`ple_pq/build_pq_ple.py`);
- runtime is a single-file hot patch (`ple_pq/rt2/vllm_ple_mmap.py`) volume-mounted into the container — no image rebuild, rollback = one reboot;
- frees ~10.4GB **straight into the KV pool**: available KV 14.85 → ~28GiB, KV tokens ~940K, 262K-context concurrency 1.91× → 3.60×.

### 3. Target model, further quantized: FP8-hybrid + FP4 LM head

`src/vllm_fp8_hybrid_modelopt.py`, `tools/fp8_convert.py`, `scripts/apply-lmhead-fp8-hook.py`.

On top of the NVFP4 checkpoint: 300 **dense** tensors (attention, linear attention, shared expert) are cast to **FP8_E4M3** (worst relative error 0.0354, verified per tensor), while MoE experts stay NVFP4 — block-wise pricing of precision vs bandwidth. The output projection (**lm_head**, unquantized in the upstream checkpoint) is additionally run at **FP4/FP8 via a runtime hook**: it is read in full every decode step, so compressing it wins on both footprint and per-step bandwidth. Output sequences are verified token-identical against the pre-quantization path.

### 4. Draft head: reduced-vocabulary MTP + 4-bit quantization

`scripts/patch-draft-head-skip.py`, `scripts/mtp_head_4bit.block.py`, `src/patch_draft_vocab.py`.

The native MTP head ships **unquantized BF16** (512-expert MoE, 4.86GiB) and drafts against the full vocabulary — speculative decoding paying a heavy tax per step. Two fixes:

- **FR-Spec reduced draft vocab** (after tonyd2wild's scheme, **recalibrated to dv=147456 with 100% Chinese coverage**): the draft head projects only onto a compact vocabulary with a local argmax reduction mapped back to the full vocab; target verification remains exact. The single largest win in the whole project.
- **Draft-head weight quantization** in two stages: FP8 experts (+10% single-stream), then **4-bit** with block size 64 — a fast that holds the entire suite record, output sequences token-identical to the FP8 stage (lossless).

Net effect: **draft-step cost b falls from ~7ms to ~1.2ms** (directly measured: b ≈ T_step/57). Drafting goes from expensive guesswork to nearly free.

### 5. Speculative width K: 4 → 6 (and why 7 is impossible)

A draft position is worth paying for iff its acceptance rate p exceeds `p* = A·b/T_step`. With an uncompressed head (b≈7ms) the threshold is ~0.39 and K=4 is the equilibrium (positions 1–4 already accepted at 0.95/0.90/0.80/0.70). **After the head became cheap, the threshold drops to 0.08–0.35** — positions 5–6, still accepted at 0.45–0.62, turned from losses into free tokens.

Adopting K=6 required passing the QSA divisibility assert (`ring capacity = 4·⌈(4+K)/4⌉` must divide the attention block size) — K=5 once died on `1616 % 12 ≠ 0`; a ring-widening patch generalizes it. **K≥7 is physically unbootable on this model**: K=7 forces mamba page geometry → block size 1648 = 2⁴×103, and 12 can never divide it. K=6 is the hard ceiling, confirmed, not a tuning artifact.

Result vs K=4 (low-thinking single stream): **Code +14%, Math +14%**, LongCode +1.5%, suite wall −3.7%; acceptance length 4.3 → 4.8 (thinking) / 5.5 (no-thinking). The only regression is very short Q&A (−4.5%) — six draft paths on a 53-token question is a known, priced-in cost.

### 6. `FULL_DECODE_ONLY`: the whole decode step inside one CUDA graph

`gpu-gatherer/` (patch + integration notes).

A PLE lookup is host-adjacent work, so it originally ran as a **splitting op** — every decode step was chopped into "graph → outside lookup → graph", paying launch and graph-switch overhead and blocking full-graph capture of the speculative path. The fix is a **GPU-side gatherer**: a Triton kernel reading the memory-mapped PQ table directly through UVA. With lookups inside kernels, `splitting_ops=[]` and the entire step — target forward **and** MTP draft — captures as a single `FULL_DECODE_ONLY` CUDA graph. This and optimization #2 are mutually enabling: only a 1.5GB table made device-side gather practical.

### 7. `sgbench`: the measurement harness that adjudicated everything

`bench/` — a faithful port of azampatti's GB10 methodology (Q&A 256 / Code 512 / JSON 1024 / Math 64 / LongCode 2048 max tokens, temperature 0, run twice and report Run 2, timed on-box — cross-LAN streaming has a ~12 tok/s false ceiling), with four local amendments, each learned from a real misjudgment:

1. **Explicit thinking tiers.** The stock harness inherits the model's default max-effort thinking, which can consume the entire token budget and report 0 tok/s for real work. Variants inject `reasoning_effort: low/medium` or disable thinking (`*_nothink`); measured thinking tax: **30–47%**.
2. **Token-composition reporting.** Throughput is always reported as `total = content + thinking`. This yields the hardest reproducibility evidence in the project: **at temperature 0, two runs with identical per-tier compositions are bit-for-bit the same configuration** — throughput deltas are then just thermal noise; a composition change means something (K, quant env, graph width) actually moved.
3. **Per-mode acceptance baselines.** Acceptance length must be quoted per thinking mode (5.47 no-thinking vs 4.80 with thinking on the same engine); mixing windows produced a poisoned baseline once and cost a false alarm.
4. **Retest discipline.** Any single-run slowdown is re-run before a conclusion is drawn (caught a phantom −16% LongCode drop three times over).

The suite's real output was negative results: dynamic K (−4…−8%), phase-wise K (−3.9% weighted, −18% on Math), an external DFlash drafter, and K≥7 — all closed with data, saving months of flailing.

## Benchmarks

Current configuration, low-thinking single stream (`bench/sgbench-low-split.sh 1`, Run 2):

| Tier | tok/s |
|---|---|
| Q&A | 49.3 |
| Code | 60.7 |
| JSON | 61.3 |
| Math | 68.6 |
| LongCode | 60.7 |

Evolution (single-stream, same harness):

| Stage | Change | Effect |
|---|---|---|
| baseline | NVFP4 + native MTP (k=2, BF16 head) | deployable, 28–38 tok/s |
| +FP8-hybrid, FR-Spec, GPU gatherer | #1/#3/#4/#6 | head becomes cheap, graphs go full |
| +PQ-M5 PLE table | #2 | +10.4GB into KV, 1.91×→3.60× concurrency |
| +4-bit draft head | #4 | +10–12% more, suite record |
| +K=6 | #5 | Code/Math +14% |

## Repository layout

```
Dockerfile, scripts/     image build, weight download, serve (serve.sh / serve-heads.sh),
                         runtime-hook appliers (lm_head FP8/FP4, MTP 4-bit, draft-vocab skip)
src/                     vLLM runtime patches: FP8-hybrid modelopt shim, QSA exact top-k,
                         QSA FP8-KV, mamba block-size guard, PLE mmap backend, draft vocab
ple_pq/                  PQ build (build_pq_ple.py), runtime (rt2/), verification, docs
gpu-gatherer/            GPU-side PLE gatherer patch (+ FULL_DECODE_ONLY unlock notes)
bench/                   sgbench suite: default / low / low-split / medium(-split) / nothink
tools/                   fp8_convert, lm_head & MTP-head probes, acceptance_report
tests/                   CPU-side unit tests for the PLE and QSA patches
docs/                    HOW-IT-WORKS (deep dive), HASHK_TABLE (legacy HashK path)
```

## Quick start

```bash
# 1. weights (NVFP4 checkpoint → hybrid FP8 snapshot conversion is done in-image)
scripts/download-weights.sh

# 2. build the serving image (vLLM base + all runtime patches baked)
scripts/build.sh

# 3. serve — dry-run by default on purpose; REAL=1 actually starts the container
REAL=1 scripts/serve-heads.sh

# 4. benchmark (on-box; expect the five-tier table above)
PORT=18300 bench/sgbench-low-split.sh 1
```

`serve-heads.sh` prints the fully resolved configuration and starts nothing unless invoked with `REAL=1` — this guard exists because an earlier version once removed a live production container while being "just looked at". Key resolved values: `FULL_DECODE_ONLY` + `splitting_ops=[]`, `DRAFT_VOCAB=147456`, `PLE_HASHK=…/ple_pq_M5.pt`, `MTP=6`, `max-num-seqs=8`, `gpu-memory-utilization=0.88`, prefix caching on. Every quantization stage **self-verifies at build time** (numerics + kernel microbench at live shapes) and falls back to the previous behavior on failure.

PQ table build (once, ~200s): `python3 ple_pq/build_pq_ple.py`, then verify: `python3 ple_pq/verify_pq_runtime.py`.

## Tuning notes (hard-won)

- **b is the only number that matters for K.** Every speculative-width decision reduces to `p* = A·b/T_step`. Calibrate b per generation (draft-head quantization moved it 7ms → 1.2ms); conclusions re-keyed on stale constants flip sign (we shipped one such reversal).
- **Ring capacity ≠ block size.** K changes mamba page geometry, not just the ring; the K≥7 wall is a prime-factor accident (1648 = 2⁴×103), not a policy limit.
- **Lowering K to "save wasted drafts" is structurally negative** when b ≪ T_step: you save ~1ms/token of draft cost and throw away ~1.5 tokens of acceptance. Dynamic-K, phase-wise-K and their cousins all die here.
- **KV capacity floats ~±0.8GiB across reboots** on unified-memory boxes (host page-cache state feeds the engine's memory probe). Read `Free memory` / `consumed` boot lines before diagnosing "drift".
- **Configuration identity is proven by output composition**, not throughput: identical `content+thinking` splits across runs ⇒ nothing moved; then, and only then, are ±1% deltas thermal.

## Acknowledgements

| Upstream | Contribution | Form here |
|---|---|---|
| [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX) (Apache-2.0) | base deployment framework | Dockerfile, serve/prepare/smoke chain, PLE mmap skeleton |
| [tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark) (Apache-2.0) | reduced-vocab MTP draft head | draft vocab **recalibrated to dv=147456 for full Chinese coverage** |
| azampatti/GB10-3.8-Flash-Next | HashK PLE compression, FP8 dense cast, sgbench methodology | HashK retained as legacy path (`docs/HASHK_TABLE.md`); PQ supersedes; bench ported to vLLM |
| [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | quantized weights | `-fp8hybrid` snapshot basis |
| NVIDIA SparkInfer | official GB10 inference stack | base image, CUTLASS FP8 block-scaled kernels |

Full attribution: [NOTICE](NOTICE).

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Model weights are governed by their own upstream licenses.
