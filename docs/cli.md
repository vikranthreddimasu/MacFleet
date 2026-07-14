# CLI reference

Run `macfleet --help` or `macfleet COMMAND --help` for the full option list.


## `macfleet join`

Start the agent on this Mac.

```
macfleet join [--name NAME] [--port PORT] [--data-port PORT]
              [--token TOKEN] [--fleet-id FLEET_ID] [--tls]
              [--peer HOST:PORT] [--bootstrap]
              [--enroll-ttl SECONDS] [--enroll-uses N]
              [--show-token]
              [--open --allow-insecure-open]
```

- `--name` — hostname override for mDNS service name (useful on CI
  boxes with 63+ char hostnames that blow past RFC 6763)
- `--port` — heartbeat port (default 50051)
- `--data-port` — transport port (default port + 1, i.e. 50052)
- `--token` — fleet token. If omitted, auto-generated on first run
  and persisted at `~/.macfleet/fleet-token`
- `--fleet-id` — logical fleet id (scope for multiple fleets on one LAN)
- `--tls` — forced true when token is set (redundant, for documentation)
- `--peer HOST:PORT` — manual peer bootstrap when mDNS is blocked
- `--bootstrap` — start a short-lived one-time enrollment endpoint and
  print a `macfleet pair --host ... --code ...` command
- `--enroll-ttl` — seconds before the enrollment code expires
- `--enroll-uses` — number of Macs that may use the code
- `--show-token` — reveal the permanent fleet token in the terminal
  (dangerous; prefer enrollment)
- `--open --allow-insecure-open` — intentionally run without auth or
  TLS. `--allow-insecure-open` is required so unauthenticated LAN
  exposure cannot happen accidentally.

## `macfleet status`

One-shot snapshot of the current fleet.

```
macfleet status [--json] [--open]
```

Outputs a table:

```
Node                    IP                Chip             GPU  Fleet
mac-mini-studio         192.168.1.10      Apple M2 Max     30   (coordinator)
macbook-pro             192.168.1.11      Apple M1 Pro     16
```

Use `--json` for scripts and monitors. It suppresses human banners and
prints stable JSON with `secure`, `fleet_id`, `count`, and `nodes`.
Use `--open` to scan unauthenticated open fleets without reading the
saved local fleet token.

## `macfleet pair`

Pair this Mac with an existing fleet and write the token to
`~/.macfleet/fleet-token`.

```
macfleet pair --host HOST:PORT --code ONE-TIME-CODE [--yes]
macfleet pair --stdin [--yes]
macfleet pair --pasteboard [--yes]
```

Use `--host/--code` with the command printed by
`macfleet join --bootstrap`. `--stdin` is a legacy migration path for
old `macfleet://pair?token=...` URLs. `--pasteboard` is also available
for legacy URL migration, but it must be requested explicitly so the CLI
does not silently trust pasteboard contents. If a saved fleet token
already exists, `pair` asks before replacing it; `--stdin` requires
`--yes` because stdin is already carrying the legacy URL.

## `macfleet train`

Run the built-in demo or a custom training script.

```
macfleet train [SCRIPT] [--engine torch|mlx] [--epochs N]
               [--batch-size N] [--lr FLOAT]
               [--compression none|light|moderate|aggressive|adaptive]
               [--config PATH]
```

`--config` accepts `.json`, `.yaml`, or `.yml`. Config files may set:
`engine`, `epochs`, `batch_size`, `lr`, and `compression`. Hyphenated
aliases like `batch-size` are accepted. Explicit CLI flags override
config values.

For script mode, define `main()` in the target file. MacFleet keeps
old no-argument scripts working, and injects CLI/config values when
`main` names any of these parameters:

```python
def main(engine, epochs, batch_size, lr, compression, config_path):
    ...
```

## `macfleet rotate-token`

Replace the saved local fleet token.

```
macfleet rotate-token [--yes] [--show-token]
```

After rotation, restart running agents and re-pair every Mac with
`macfleet join --bootstrap`. `--show-token` prints the new permanent
token and records an audit event; avoid it unless you are recovering an
old manual setup.

## `macfleet doctor`

Diagnoses common environment issues.

```
macfleet doctor [--json]
```

Checks:

- Python version >= 3.11
- macOS version >= 14
- Apple Silicon (arch == arm64)
- MPS backend available (if torch installed)
- MLX installed (if applicable)
- Thermal state (not throttling at rest)
- mDNS reachability (can we actually broadcast?)
- Token file regular-file shape and permissions

Prints a checklist. Green = OK, yellow = warning, red = blocker.
Use `--json` for CI and local monitors. It prints `status`, `ready`,
summary counts, and a `checks` array with stable ids, sections, names,
statuses, and redacted details.

## Leaving the pool

There is no standalone `macfleet leave` command. Stop a running
`macfleet join` with Ctrl+C (the agent unregisters mDNS and closes
heartbeat listeners on exit). From Python, call `Pool.leave()` or exit
the `Pool` context manager. Peers mark the node failed within the
default heartbeat failure timeout (~10s).
