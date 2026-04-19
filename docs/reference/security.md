# Security reference

How MacFleet keeps your fleet safe on hostile networks.

## Threat model

MacFleet is designed to be safe on untrusted WiFi (coffee shop, open
SSID, enterprise network with client isolation broken). The assumption
is that an attacker on the same LAN can:

- **See all mDNS broadcasts** — so we scope the service type by
  fleet hash (see [Fleet isolation](#fleet-isolation))
- **Send arbitrary TCP packets to advertised ports** — so every
  handshake requires HMAC proof of the fleet token (see [Authentication](#authentication))
- **Record TLS sessions and replay them** — so we use mandatory TLS
  (self-signed EC P-256, rotated per session) + nonces on every
  heartbeat + handshake
- **Try to brute-force the fleet token** — so we rate-limit failed
  auth attempts per IP with exponential backoff (see [Rate limiting](#rate-limiting))

The attacker cannot:

- Execute code in the worker process without the fleet token AND a
  registered `@macfleet.task` name match (see [Task dispatch](#task-dispatch))
- Inflate their hardware profile to win coordinator election
  (compute_score is recomputed locally from declared specs, never
  trusted from the wire)

## Fleet isolation

When you set a token (which happens automatically on first
`macfleet join --bootstrap`), mDNS broadcasts use a scoped service type:

```
_mf-<first-8-hex-of-sha256(fleet_key)>._tcp.local.
```

Other fleets on the same LAN can't see your nodes. They can see *that*
a scoped fleet exists (the hash is visible), but can't enumerate its
members without the token.

## Authentication

### HMAC challenge-response (v2.1 baseline)

Peer A connects to peer B:
1. A sends challenge `c_a` (random 32 bytes)
2. B responds with `HMAC(fleet_key, c_a)` + its own challenge `c_b`
3. A verifies B's response, sends `HMAC(fleet_key, c_b)`
4. B verifies A's response

Both sides now have mutual proof of token possession without either
side sending the token itself.

### HW profile exchange (v2.2 PR 4 addition)

Piggy-backed on the handshake: both sides also send a signed hardware
profile (GPU cores, RAM, chip name, MPS/MLX availability, data port).
The signature binds the profile to the peer's challenge, so the payload
can't be replayed from a previous session.

A separate wire version byte lets v2.1 and v2.2 peers coexist — if
a v2.2 server sees a v2.1 client, it falls back to the bare handshake.

### APING v2 heartbeat (v2.2 PR 5 addition)

Heartbeat pings now carry the same signed HW profile (Issue 6):

```
APING v1 (4 fields): APING {node_id} {nonce_hex} {sig_hex}
APING v2 (5 fields): APING {node_id} {nonce_hex} {sig_hex} {hw_json_hex}
```

This is what makes `--peer host:port` work correctly — a manually-added
peer no longer registers with a zero-score placeholder; the APONG v2
response carries the peer's real HW profile.

## TLS

When the token is set (the default whenever you join a secure fleet),
TLS is mandatory. The cert is:

- EC P-256 self-signed
- Generated in-memory via `cryptography` (no `openssl` subprocess)
- Ephemeral temp file written with mode 0o600 + `try/finally` unlink
- CN = `localhost`, SAN = `DNS:localhost`

**No mutual TLS** — cert validation is disabled. The HMAC challenge-
response *is* the authentication; TLS only provides confidentiality.
This is a deliberate choice: self-signed certs with a stable CA would
require a PKI, and pairing UX would be much more complex.

The `--tls` flag on `Pool(token=..., tls=True)` is redundant when
token is set (forced true); it exists only to document intent.

## Rate limiting

Both the heartbeat server (`AuthRateLimiter` in `agent.py`) and the
transport server (same class in `transport.py`) apply per-IP
exponential backoff:

- 5 consecutive auth failures → 5-minute ban
- Each attempt before the ban: 0.5s, 1s, 2s, 4s, 8s delay before read
- Ban state is per-process, not distributed — an attacker can't sneak
  past by hopping IPs if those IPs all look suspicious to the same node

Slowloris (connecting and never sending) is also counted as a failure.
The heartbeat read timeout was tightened from 5s → 1s in v2.2 PR 6 so
slow attackers get dropped quickly.

## Token file permissions

`~/.macfleet/token` is chmod 0o600 after `O_CREAT`. On every read,
`_check_token_file_mode` warns (not fails) if the mode leaks any bits
to group or other — a soft tripwire that catches users who copied the
file around with `cp` on a poorly-configured system.

## Task dispatch

As of v2.2 PR 7, the wire carries task NAMES (strings), not cloudpickled
callables. The worker looks up the name in a local `TaskRegistry`
populated by `@macfleet.task` decorators at import time. An attacker
who has the fleet token can still call registered tasks, but can't
inject arbitrary code.

Pydantic schemas declared on the decorator add a second layer of
validation on the worker side — bad args surface as
`ValidationError`, not as a crash inside your function.

## What's NOT covered

- **Denial of service from a valid fleet member.** If one of your own
  Macs is compromised and starts flooding the coordinator with valid-
  looking tasks, nothing stops it. Rate limiting is per-peer on the
  server side but a trusted peer isn't rate-limited for its own
  requests.
- **Timing attacks on HMAC.** We use `hmac.compare_digest` everywhere,
  which is constant-time, but the surrounding code (e.g. error
  message logging) may leak small amounts of timing info.
- **TLS cert rotation during a long session.** Certs are session-local;
  if a run goes 48+ hours you're still using the same ephemeral cert.
  Fine in practice but worth knowing.
