# Changelog

All notable changes to MacFleet. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (sparse-on-wire gradient compression, N=2)

- **Compression now actually saves bandwidth.** Previously gradients
  were compressed and immediately decompressed locally — the wire
  always carried dense float32. On 2-node fleets (the dominant case)
  the packed TopK+FP16 payload itself now crosses the network:
  `pack_compressed`/`unpack_compressed` in `compression/adaptive.py`
  (fail-closed validation before any allocation: magic, kind, numel/k
  bounds, exact body length, index range, finite positive scale),
  `CollectiveGroup.exchange_bytes`, and
  `DataParallel._sparse_exchange_n2`. WiFi aggressive (TopK 1% + FP16)
  cuts gradient traffic ~60x.
- **Parameter identity preserved**: both ranks average
  `decompress(own) + decompress(remote)` — the same two arrays on each
  side — and the FP16 scale travels as float64, so dequantization is
  bitwise-equal and fleet parameters stay byte-identical (asserted with
  `assert_array_equal` across multi-step error-feedback tests).
- `DataParallel.compression_ratio` and `_bytes_saved` now report real
  wire savings. N≥3 rings still transmit dense chunks (sparse ring
  merge remains TODOS Issue 3 second half); warmup steps stay dense in
  lockstep on both ranks.

### Security (v3 handshake — all nodes in a secure fleet must upgrade together)

- **Closed the unauthenticated HMAC oracle.** A secure server used to
  answer ANY connector with `HMAC(fleet_key, attacker_chosen_challenge)`
  plus its signed hardware profile — free offline brute-force material
  and hardware recon for anyone on the LAN. The client now proves token
  knowledge in its FIRST message; the server stays byte-silent (and
  rate-limits) otherwise.
- **TLS channel binding (MITM relay defeat).** Handshake HMACs now mix
  in the SHA-256 fingerprint of the server's TLS certificate as each
  side sees it. An active attacker terminating TLS on both legs (easy
  against self-signed + CERT_NONE) produces mismatched digests and the
  handshake fails on both sides.
- **Domain-separated handshake digests.** Each handshake step's HMAC
  carries a distinct label — a digest from one step can never be
  replayed as another step's.
- **scrypt fleet-key derivation.** `fleet_key = scrypt(token,
  salt="macfleet-v3:"+fleet_id, n=2^14, r=8, p=1)` replaces the single
  HMAC-SHA256 derivation, making offline dictionary attacks against
  captured handshakes memory-hard. Tokens under 16 chars log a warning.
- **Heartbeat APONG bound to request nonce.** A captured APONG can no
  longer be replayed as the answer to a later APING; stale/relayed
  responses fail verification and the peer is not registered.
- **Compatibility:** pre-v2.3 secure clients are rejected by v2.3
  servers (their hello carries no proof — answering it would reopen
  the oracle). Open fleets (`--open`) are unaffected. Run the same
  MacFleet version on every node of a secure fleet.

### Setup UX

- **First `macfleet join` auto-prints the pairing QR + URL** (and
  copies the URL to the pasteboard) when it generates a fresh fleet
  token — pairing a second Mac needs zero extra flags: Mac #1
  `macfleet join`, Mac #2 `macfleet pair && macfleet join`.
  `--bootstrap` still re-prints pairing info for an existing token.
  Degrades to URL-only (with an install hint) when `qrcode` is absent.
- **`macfleet doctor` gained a Security section**: fleet token
  configured, token length, token file permissions (0600).

### Added

- **Distributed `Pool.train()`** — when the pool is distributed with
  live peers, `pool.train(...)` automatically runs multi-node
  data-parallel training: SPMD (run the same script on every Mac),
  gradient mesh over the data ports, rank-0 weight broadcast,
  allreduce every step. Single-node behavior is unchanged when there
  are no peers; `distributed=True/False` forces either mode.
  Distributed results gain `rank`, `world_size`, `avg_sync_time_sec`,
  and `params_sha256` (identical across ranks iff the fleet stayed in
  sync). Works for both engines (torch + MLX); `compression=` now
  actually applies to training started from the SDK.
- **`macfleet/training/mesh.py`** — SPMD rendezvous: `form_mesh()`
  turns a registry snapshot into a connected `PeerTransport` +
  `CollectiveGroup`. Ranks derive from sorted node ids (immune to
  transient compute-score disagreement between registries). Outbound
  connects retry until `rendezvous_timeout_sec` (Pool kwarg, default
  60s) so Macs don't need a synchronized start; missing peers produce
  a `MeshFormationError` naming exactly who never showed up.
- **`PeerAuthError`** (subclass of `ConnectionError`) — wrong-token
  handshakes now fail fast during mesh formation instead of retrying
  until the deadline and tripping the peer's rate limiter.

### Tests

- `tests/test_comm/test_mesh.py` — 2-node and 3-node (ring) mesh
  formation, staggered starts, token-secured mesh over TLS,
  wrong-token fail-fast, rendezvous-timeout remediation.
  Framework-agnostic.
- `tests/test_sdk/test_distributed_train.py` — 2-rank end-to-end torch
  training over loopback: byte-identical final params across ranks,
  decreasing loss, matching step counts, compression path, routing
  guards. MLX twin auto-skips when MLX is absent.

### Developer tooling

- Added `tools/two_mac_verify.py`, a structured PASS/FAIL verification
  harness for two-Mac setups. `--self-check` validates a single machine
  (import + version, interface detection, deterministic topology
  selection, link serialization, and a single-node `Pool.train` that
  must reduce loss); distributed mode confirms quorum, registry sanity,
  topology peer-address selection, and a distributed `Pool.train` whose
  `params_sha256` must match across both Macs with no degraded steps. The
  process exit code is the number of failed checks, and distributed runs
  write `~/.macfleet/verify-<hostname>.json` for cross-Mac comparison.
- Added `tests/test_tools/test_two_mac_verify.py`, CI-safe coverage that
  runs the harness self-check in-process on a single machine.

## [2.2.1] — 2026-06-28

### Added

- Added a lightweight pool topology model and deterministic peer-address
  selection so distributed training can prefer likely reachable faster links
  while preserving the existing single-IP fallback.
- Added `docs/getting-started/two-mac-testing.md`, a beginner-friendly
  walkthrough for testing MacFleet with a stronger Mac plus an M1 MacBook Air.
- Added `tools/two_mac_pool_smoke.py`, a small high-level `Pool.train` smoke
  test that verifies world size, degraded state, unsynced steps, and matching
  final parameter hashes across two Macs.

### Changed

- Open-fleet mDNS now advertises compact per-interface network facts for
  topology-aware mesh formation. Secure fleets still minimize broadcast
  metadata and do not expose the extra interface list through mDNS.
- Distributed `Pool.train()` now passes registry network-link facts into mesh
  formation when available.

### Tests

- Added topology tests for Thunderbolt preference, off-subnet link avoidance,
  IPv6 link-local filtering, network-link serialization, and registry
  preservation of interface facts.

## [2.2.0] — 2026-05-01

Graduates v2.2.0-rc1 to stable. Same shipped features (Phase B
security + Phase C product) plus the post-rc bug-fix + test-coverage
sweep. Test surface grew from 447 (rc1) to 700+ — fuzz + stress +
failure-injection + multi-node E2E suites added.

### Fixed (post-rc bug sweep)

- **Concurrency:** `ClusterRegistry.update_hardware` now replaces the
  entire `NodeRecord` atomically inside the lock (was field-level
  mutation, exposed torn `(hardware, data_port)` reads to the
  scheduler).
- **Concurrency:** registry mutation methods + election share one
  lock acquisition (was release/re-acquire window).
- **Network:** `DataParallel.sync_gradients` now catches
  `asyncio.IncompleteReadError` + `EOFError`. A peer dropping
  mid-frame previously crashed training instead of falling back to
  local gradients.
- **Network:** `_validate_model_consistency` runs `gather` + `allreduce`
  in the same order on every rank before any `raise` (was a deadlock
  vector — rank 0 could raise while peers waited at the next
  collective). Gather is bounded with a 10s timeout.
- **Resource leaks:** `_ping_peer` and `_add_manual_peer` close
  writers in a `finally` block so a `readline()` timeout no longer
  drops the StreamWriter mid-flight.
- **Resource leaks:** `Pool._start_agent` cleans up the background
  loop + thread when `agent.start()` raises (port conflict, mDNS
  failure). Previously the orphan loop survived and a retry stacked
  a second loop on top.
- **Resource leaks:** `TaskWorker.stop` drains in-flight task wrappers
  before shutting the executor (was `wait=False`, leaked results
  for tasks completing during shutdown).
- **Resource leaks:** `TaskDispatcher` resolves stranded
  `TaskFuture` objects when a worker disconnects, instead of
  letting callers block forever on `future.result()`.
- **Resource leaks:** `_ping_peer`'s explicit close moved to
  `finally`; `_add_manual_peer` declares the writer with `Optional`
  and closes in `finally`.
- **Coordinator election:** secure-mode gossip pings now carry the
  signed HW exchange (APING v2). mDNS-discovered peers in a
  token-protected fleet refresh from a zero-score placeholder to
  their real `compute_score` after one successful round, instead of
  staying zero forever (Issue 2 followup — was only fixed for the
  manual-peer bootstrap path).
- **Wire protocol:** `WireMessage.unpack(bytes)` now enforces
  `MAX_PAYLOAD_SIZE`. The streaming path already capped, the bytes
  path didn't.
- **Compression accounting:** `_bytes_sent` now reflects actual wire
  bytes (dense, until sparse-on-wire ships). The previous value
  under-counted by claiming sparse savings the wire didn't deliver.
- **Heartbeat:** `writer.drain()` in the heartbeat handler is now
  bounded by 1s — slowloris on the receive side could otherwise
  pin the handler indefinitely.
- **Heartbeat:** empty / RST-on-connect noise no longer counts as a
  failed auth attempt (port scanners would have banned themselves
  + collateral users from the same IP).
- **Engines:** `TorchEngine.apply_flat_gradients` reuses
  `param.grad` via `copy_` (cross-device in-place) instead of
  rebinding to a fresh tensor every step. Optimizers caching
  references to `.grad` now stay valid.
- **Engines:** `TorchEngine` and `MLXEngine` `profile()` delegate to
  `pool.agent.profile_hardware()` instead of returning a zeroed
  placeholder.
- **SDK:** `Pool.leave` consolidated on the `_teardown_loop` helper.
- **Discovery:** `_parse_service_info` and `_parse_ifconfig` handle
  IPv6 addresses (was IPv4-only).
- **Pool / network:** `_classify_interface` no longer assumes en0 is
  WiFi when `networksetup` is unavailable (mis-classified Mac mini
  Ethernet → AGGRESSIVE compression).
- **Training guards:** `_dataset_len` accepts both `(X, y)` tuples
  and `[X, y]` lists when both halves expose `.shape`. Bare 2-element
  lists fall through to `len()` instead of being mis-read as
  features+labels.
- **Training:** `WeightedDistributedSampler.drop_last` is now
  honored (was a dead field).
- **Compute:** `@task` re-registration that replaces a different
  callable now logs at `WARNING` (was `INFO` — silently masked typos).
- **Misc:** f-string typo in `_pick_ephemeral_port` error message;
  retry budget bumped 16→32 with last-port diagnostic; `Dashboard.
  update_training` honors zero values via `None` sentinels;
  `Dashboard.start` falls back gracefully on non-TTY environments;
  thermal probe failure logged once on degraded systems.

### Added (test infrastructure)

- `tests/test_fuzz/` — hypothesis-based fuzz for `WireMessage`,
  `TaskSpec`, `pack_array`, and `HardwareExchange` (38 tests,
  ~600 examples per run).
- `tests/test_stress/` — concurrent registry mutation,
  100-round allreduce soak, FD-leak detection, dispatcher
  high-throughput (17 tests).
- `tests/test_failure_injection/` — peer dropout mid-allreduce,
  brute-force ban, replay attack rejection, oversize payload,
  slowloris (12 + 1 xfail documenting transport-level rate-limit gap).
- `tests/test_production/` — multi-node convergence (N=2..4),
  weighted heterogeneous batching, `setup` rebroadcast, all
  compression levels converge, concurrent train+dispatch, thermal
  pause hysteresis (22 tests).
- New deps: `hypothesis>=6` (dev only).

### Removed

- None. All v2.1.x APIs preserved.

### Migration guide (v2.1.x → v2.2)

Same as v2.2.0-rc1. If you're already on rc1, this is a drop-in
replacement.

## [2.2.0-rc1] — 2026-04-19

Release candidate for v2.2. Phase B (security) and Phase C (product)
shipped in full. Final v2.2.0 tag blocks on a 2-Mac physical smoke
test and the WWDC 2026 scheduling gate.

### Breaking changes

- **`pool.submit` / `pool.map` prefer `@macfleet.task` decorated
  callables.** Bare lambdas still work via a legacy ProcessPool +
  cloudpickle fallback, but that path is going away. Migrate by
  decorating the callable:

  ```python
  @macfleet.task
  def my_fn(x):
      return x * 2

  pool.submit(my_fn, 21)   # routes through registry (no cloudpickle)
  ```

- **Task dispatch wire format changed**: `TaskSpec` now carries the
  registered task NAME + msgpack-native args, not a cloudpickled
  callable. Workers look up the name in a local `TaskRegistry`.
  Tasks must be registered before dispatch or a `TaskNotRegisteredError`
  surfaces.

- **`ProcessPoolExecutor` → `ThreadPoolExecutor` in `TaskWorker`.** The
  process-pool isolation was a defense against arbitrary pickled code
  crashing the worker; now that attack surface is gone, threads give
  cheap access to in-process state.

### Added

- **`@macfleet.task` decorator** registers a callable in a local
  `TaskRegistry`. Supports bare (`@task`) and parameterized forms
  (`@task(name="...", schema=PydanticModel)`). Attached
  `fn.task_name` and `fn.schema` for introspection.

- **Pydantic schema validation** on task args. Declared via
  `@task(schema=MyModel)`, validated on both coordinator (pre-
  dispatch) and worker (pre-invocation). Return values of type
  `BaseModel` get `model_dump(mode="json")` for wire transport.

- **`Pool.join` wires a live `PoolAgent`** (flag-gated,
  `enable_pool_distributed=False` default). With the flag on,
  `Pool.join()` starts mDNS discovery + heartbeat and blocks until
  `quorum_size`. `Pool.world_size` and `Pool.nodes` read live data
  from the registry.

- **`Pool.is_distributed`** property returns True iff flag is set
  AND agent is live.

- **`Pool.dashboard_snapshot()`** returns `[NodeHealth, ...]` for
  `Dashboard.update_nodes()` or headless health checks.

- **APING v2 heartbeat** (Issue 6): 5-field variant carries signed
  HW profile so `--peer` manual peers register with real compute
  scores instead of zero-score placeholders. HMAC binds HW payload
  to per-heartbeat nonce (replay-protected).

- **Heartbeat rate limiter** (Issue 22): per-IP exponential backoff
  via `AuthRateLimiter`. 5 failures → 5-minute ban. Read timeout
  tightened from 5s → 1s to defeat slowloris.

- **Signed HW profile exchange in transport handshake** (Issue 2 +
  A5 + A7): wire_version byte + nonce-bound HMAC lets v2.1 and v2.2
  peers coexist; rogue peer can't inflate compute_score via
  broadcast.

- **TLS via `cryptography`** (Issue 9 + A6 + A12): native EC P-256
  cert generation in-memory, no `openssl` subprocess. Temp pem
  files at mode 0o600 with `try/finally` unlink. Token file chmod
  0o600 enforced after `O_CREAT`.

- **`--peer HOST:PORT`** manual peer bootstrap when mDNS is blocked
  (enterprise WiFi, client isolation). Flag repeatable.

- **Data-port / heartbeat-port split** (Issue 5): heartbeat on 50051,
  transport on 50052. Shared-port collisions (transport handshake
  looks like malformed APING) are gone.

- **`macfleet join --bootstrap`**: prints QR code + pairing URL,
  copies URL to pasteboard. iPhone camera scans → tap → paired.

- **`macfleet pair`** subcommand: reads pairing URL from pasteboard
  (default) or stdin (`--stdin`), writes token to
  `~/.macfleet/token` (mode 0o600).

- **`macfleet quickstart`**: scaffolds a 30-line starter training
  script. `pip install macfleet` → `macfleet quickstart` →
  `python my_macfleet_demo.py` in under 60s.

- **`macfleet doctor`**: friendlier alias for `macfleet diagnose`
  (pattern users know from `brew doctor` / `rustup doctor`).

- **`ThermalPauseController`** (E4 + A16): local thermal FSM with
  hysteresis. Pauses training on `SERIOUS`/`CRITICAL`, resumes on
  `FAIR`/`NOMINAL`. Prevents NaN cascades from sustained throttling.

- **A3 atomic checkpoint writes** (`macfleet.utils.atomic_write`):
  temp-file + fsync + `os.replace` pattern. Ctrl-C / OOM / power
  outage during save leaves the previous checkpoint usable.

- **A4 dataset preflight guard**: `check_dataset_sufficient()` fails
  fast with remediation-rich errors (empty dataset, batch <
  world_size, dataset < one batch).

- **`agent_adapter`** module bridges registry records to `NodeHealth`
  for the Dashboard. `snapshot_all(agent)` returns `[self, peer1,
  peer2, ...]`.

- **mkdocs-material docs site** under `docs/`. Install with
  `pip install "macfleet[docs]"`, serve with `mkdocs serve`.

### Changed

- **Framework-agnostic CI matrix** (Issue E8, PR 1): split into
  `lint`, `framework-agnostic` (ubuntu 3.11/3.12/3.13), `torch-mps`
  (macos-arm64 3.11/3.12/3.13), `mlx` (same), `integration`
  (macos-arm64 3.12). mypy marked `continue-on-error: true` pending
  type-fix sweep in v2.3.

- **Release workflow** (`.github/workflows/release.yml`): triggered
  by `v*` tags, verifies `pyproject.toml` version + `__init__.py`
  version + tag match; builds + uploads via PyPI trusted publishing
  (OIDC, no API key).

- **Pool constructor**: new kwargs `enable_pool_distributed`,
  `quorum_size`, `quorum_timeout_sec`, `data_port`, `peers`.

- **`PoolAgent.__init__`**: new `data_port` and `peers` kwargs.
  Rejects `port == data_port` at construction time.

- **`reuse_address=True`** on the heartbeat server's
  `asyncio.start_server` call (fixes CI port-conflict flakes in
  rapid start/stop test cycles).

- **Name override (`name=` kwarg)** on PoolAgent now replaces both
  `hardware.hostname` AND `hardware.node_id` so mDNS service names
  stay under RFC 6763's 63-byte limit on CI boxes with long
  hostnames.

- **README.md refreshed** to match docs site voice: value prop
  first, 5-minute onramp (quickstart → run → pair), security
  section lists v2.2 additions.

### Fixed

- **EventLoopBlocked on mDNS stop** (c83a259 pre-v2.2): use async
  zeroconf in agent start/stop to avoid blocking the event loop
  with sync mDNS calls.

- **SSL errors on `--peer` connections** (c83a259): client TLS
  context creation now matches the server side's TLS requirements.

### Removed

- None. All pre-v2.2 APIs preserved (with deprecation paths for
  the cloudpickle task dispatch).

### Security

- **No cloudpickle over the wire.** Task dispatch is name-based with
  msgpack args; workers can't be tricked into executing arbitrary
  code.
- **Rate-limited heartbeat.** Brute-force attempts get banned after
  5 failures.
- **Signed HW exchange.** A rogue peer can't inflate compute_score
  to win coordinator election.
- **Tightened TLS cert lifecycle.** Temp pem files at 0o600 with
  guaranteed unlink.

### Test surface

- `2.1.1`: 373 tests
- `2.2.0-rc1`: **447 tests** (+74 net, 0 regressions)

### Migration guide (v2.1.x → v2.2)

Most code works unchanged. Three things to know:

1. **If you use `pool.submit`/`pool.map` with bare lambdas/closures**,
   decorate them with `@macfleet.task`. The old path still works via
   a local ProcessPool fallback but is discouraged and will go away
   in v3.0.

2. **If you catch `macfleet.compute.models.TaskResult.value_bytes`**,
   that attribute is gone; use `TaskResult.value` (msgpack-native).

3. **If you called `TaskSpec.from_call(lambda ...)`**, the wire no
   longer carries pickled code. Lambdas raise `ValueError` at
   dispatch time — decorate instead.

## [2.1.1] — 2026-04-15

SSL and peer-connection stability fixes. See commit `c83a259`.

## [2.1.0] — 2026-04-14

Initial v2.x public release. See commit `9f16588`.
