# TODOS

Deferred work from reviews. Grouped by release target.

## v2.3 (next release after v2.2 ships + is stable)

### Correctness

- **Issue 3 — sparse allreduce (DGC-style).** Restores the README's 200x WiFi compression claim. Non-trivial: top-k indices don't align across peers, so ring-reduce needs to merge sparse representations at each hop. Reference: Lin et al. (2018) "Deep Gradient Compression." Also add MPS-vs-CPU correctness verification: 100-step divergence < 1e-3 relative error, else pin compression to CPU-only gradients.

### Pruning (~700 LOC removal)

- **Issues 7-8 — delete v1 compat.** Remove `compression/pipeline.py`, `compression/topk.py`, `compression/quantize.py`, `engines/serialization.py`. Update `compression/__init__.py` exports. Delete the two torch-compat test classes in `test_protocol.py` and `test_compression.py`. Gate: v2.2 stable for 2+ weeks, no bug reports requiring v1 path.
- **Issue 24 — delete dead distributed imports.** `cli/main.py:273-275` imports `DataParallel`, `CollectiveGroup`, `PeerTransport` in `_train_demo` and never uses them. Remove with Issues 7-8.

### Features

- **E1 — cross-engine export.** `pool.export("mlx")` / `pool.export("coreml")`. Scope: LoRA adapters, linear-only models, decoder-only transformers. Non-goals: generic tracer, custom layers. Depends on Issue 1's Pool.train return semantics.
- **E3 — `macfleet doctor` CLI.** Diagnoses slow training: real-bandwidth probe per pair, allreduce probe, thermal state, bottleneck hint. Builds on existing bench.
- **E6 — fleet-wide LR range test.** `pool.find_lr(model, dataset, lr_range=(1e-6, 1))`. Parallel sweep across nodes.
- **Issue 26 — token discovery UX.** `macfleet join --bootstrap-via airdrop|pasteboard|qr`. Breaks the manual-copy friction in the README Quick Start.

### Code quality

- **Issues 10, 11, 12, 14.** Pre-allocated flat gradient buffers (10). MLXEngine forward_backward to avoid 2x forward (11). In-place ring division (12). `async def async_find_peers` (14).
- **mypy cleanup.** CI currently runs mypy with `continue-on-error: true` — 57 pre-existing Optional-unwrap + type narrowing errors in `macfleet/pool/agent.py` and `macfleet/cli/main.py`. Fix by: (a) tightening `self._registry: ClusterRegistry` (non-optional, assigned in `__init__`) instead of defaulting to None, (b) early-return guards for the `str | bytes` formatting warnings, (c) `assert spec is not None` before `module_from_spec(spec)` in cli/main.py. Then remove `continue-on-error: true` from `.github/workflows/test.yml`.

### Performance (after v2.2 real multi-node profiling data)

- **Issues 15-19.** Per-step allocation reduction, ring phase pipelining opportunity analysis (only act if measured bottleneck).

## v3.0 (architectural leap — separate design doc required per item)

- **E5 — hybrid cloud burst.** `Pool(burst_to="modal")`. XL ~3-5 days CC work. Design doc must spec: Modal auth flow, container build, WAN gradient transport (likely always-AGGRESSIVE compression), cost controls, cancellation semantics, security of cloud peer identity.
- **E7 — checkpoint shard streaming.** Crash-safe training. Design doc must enumerate failure model: single-peer loss, coordinator loss, network partition, disk full.
- **Issue 27 second half — gossip-based mesh discovery.** After one --peer bootstrap, any peer discovers the full mesh via gossip. Replaces the "copy token and IP to every Mac" friction.

## Explicitly skipped (not on any release — would need a revised strategic premise)

- **`mlx.distributed` integration or wrapping.** Evaluated at CEO review 2026-04-19. User chose dual-engine cathedral (custom MLX collectives). Revisit if Apple ships heterogeneous-link support in MLX 1.0.
- **Async / staleness-tolerant gradient sync.** `DataParallelConfig.max_staleness` field exists but non-zero is unsupported. Research-grade. Not in any release window.
- **Pipeline parallelism / tensor parallelism.** `mlx.distributed` covers this for MLX users. For PyTorch-MPS users, single-Mac model size limits haven't been the bottleneck reported. Reconsider when reported.
- **NVIDIA/NCCL bridge.** Zero overlap with Apple Silicon focus. Not a MacFleet problem.

## Strategic calendar items

- **2027-06 (WWDC week) — revisit mlx.distributed.** If Apple ships compelling heterogeneous-link support (WiFi/Thunderbolt auto-routing, adaptive compression), MacFleet's MLX engine becomes maintenance tax. Decision point: keep dual-engine or prune MLX path.
- **6 months post-v2.2 ship — evaluate product position.** If adoption is <100 users, reconsider SELECTIVE EXPANSION or SCOPE REDUCTION posture. The outside voice argued this at the CEO review; user held EXPANSION. Data will tell.

---

Review history:
- 2026-04-19 `/plan-eng-review` — 24 issues, 3 critical gaps, 1 strategic question
- 2026-04-19 `/plan-ceo-review` — SCOPE EXPANSION mode, 8 expansions accepted, plan revised after spec review (quality 9/10) + outside voice (6 findings applied)
- CEO plan: `~/.gstack/projects/vikranthreddimasu-MacFleet/ceo-plans/2026-04-19-macfleet-v3-cathedral.md`

## autoplan additions (2026-04-19)

### v2.3 additions (from /autoplan DX + Eng)

- **E9 — HuggingFace Accelerate adapter.** `macfleet.accelerate` module implementing `DistributedType.MacFleet`. `Accelerator(distributed_backend="macfleet")` transparently uses MacFleet for training. ~200 LOC (fallback `macfleet.hf` wrapper if Accelerate doesn't expose plugin hooks). Ships alongside E1a/E3/E6.
- **E10 — CLI shortcut `macfleet train <model> --dataset <name>`.** First-class CLI for the 80% case: `macfleet train llama-3-lora --dataset alpaca --epochs 3`. Built-in model + dataset catalogs. Matches Exo Labs keystroke count.
- **A18 — msgpack fuzz test.** hypothesis-based fuzz for the wire deserialization. Catches malformed-payload DoS.
- **A19 — CI matrix python 3.13 × MLX `continue-on-error: true`.** MLX wheels lag CPython.
- **A20 — `--peer` race with mDNS dedupe test.** Concurrent discovery paths should dedupe by node_id.
- **A21 — `macfleet rotate-token` CLI.** Generate new token, gossip-propagate, deprecate old.
- **A22 — Pool.map distributed exception semantics.** `list[Result | Exception]` in input order.
- **A25 — 10x gradient size OOM protection.** Pre-train memory check against device RAM.

### v3.0 additions (from /autoplan)

- **A23 — Ring allreduce N>8 hierarchical reduce (E9b).** Document ceiling, hard-warn above N=8, ship hierarchical topology for large fleets.
- **A24 — E5 cloud burst trust domain isolation.** Modal peers form separate trust domain. Pre-commit block: `Pool(origin='modal', burst_to='local')` with untrusted local peers in registry = hard error.
- **E5 direction inverted (DC3 accepted).** E5 is now `Pool(origin='modal', burst_to='local')`. Cloud primary, local Macs secondary. Bidirectional deferred to v3.1+ design doc.

### Explicitly skipped (additions)

- **Async / staleness-tolerant gradient sync** (unchanged from prior)
- **Pipeline parallelism / tensor parallelism** (unchanged)
- **Bidirectional E5 (local + cloud symmetric burst)** — deferred to v3.1+ separate design doc
- **VS Code extension for distributed debugging** — too early, not on any release

### Strategic calendar items (additions)

- **2026-06-01 to 2026-06-10 — WWDC '26 hard pause.** No v2.2 release merges during keynote week. Read signals first.
- **2027-06 — WWDC '27 revisit.** Repeat kill-switch eval.

