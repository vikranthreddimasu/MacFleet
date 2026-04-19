# Changelog

All notable changes to MacFleet. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
