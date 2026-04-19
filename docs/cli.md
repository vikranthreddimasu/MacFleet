# CLI reference

## `macfleet join`

Start the agent on this Mac.

```
macfleet join [--name NAME] [--port PORT] [--data-port PORT]
              [--token TOKEN] [--fleet FLEET_ID] [--tls]
              [--peer HOST:PORT] [--bootstrap]
```

- `--name` — hostname override for mDNS service name (useful on CI
  boxes with 63+ char hostnames that blow past RFC 6763)
- `--port` — heartbeat port (default 50051)
- `--data-port` — transport port (default port + 1, i.e. 50052)
- `--token` — fleet token. If omitted, auto-generated on first run
  and persisted at `~/.macfleet/token`
- `--fleet` — logical fleet id (scope for multiple fleets on one LAN)
- `--tls` — forced true when token is set (redundant, for documentation)
- `--peer HOST:PORT` — manual peer bootstrap when mDNS is blocked
- `--bootstrap` — print a QR code + pairing URL for the fleet token,
  also copy to pasteboard

## `macfleet status`

One-shot snapshot of the current fleet.

```
macfleet status
```

Outputs a table:

```
Node                    IP                Chip             GPU  Fleet
mac-mini-studio         192.168.1.10      Apple M2 Max     30   (coordinator)
macbook-pro             192.168.1.11      Apple M1 Pro     16
```

## `macfleet pair`

Read a pairing URL from the pasteboard (or stdin) and write the token
to `~/.macfleet/token`.

```
macfleet pair [--stdin]
```

`--stdin` reads the URL from stdin instead of pasteboard — useful for
SSH sessions where pasteboard sync isn't available.

## `macfleet doctor`

Diagnoses common environment issues.

```
macfleet doctor
```

Checks:

- Python version >= 3.11
- macOS version >= 14
- Apple Silicon (arch == arm64)
- MPS backend available (if torch installed)
- MLX installed (if applicable)
- Thermal state (not throttling at rest)
- mDNS reachability (can we actually broadcast?)
- Token file permissions

Prints a checklist. Green = OK, yellow = warning, red = blocker.

## `macfleet leave`

Graceful departure — unregister mDNS, close heartbeat server, leave
the registry cleanly. The next `status` from other Macs will show
this node as failed within 10s (default failure timeout).
