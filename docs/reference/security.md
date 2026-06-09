# Security reference

How MacFleet keeps your fleet safe on hostile networks.

## Threat model

MacFleet is designed to be safe on untrusted WiFi (coffee shop, open
SSID, enterprise network with client isolation broken). The assumption
is that an attacker on the same LAN can:

- **See all mDNS broadcasts** — so we scope the service type by
  fleet hash (see [Fleet isolation](#fleet-isolation))
- **Send arbitrary TCP packets to advertised ports** — so every
  handshake requires HMAC proof of the fleet token IN THE FIRST
  MESSAGE; a secure server sends nothing — not even an HMAC response —
  to an unauthenticated connector (see [Authentication](#authentication))
- **Record TLS sessions and replay them** — so we use mandatory TLS
  (self-signed EC P-256, rotated per session) + nonces on every
  heartbeat + handshake, and heartbeat responses are bound to the
  request nonce
- **Actively MITM connections (ARP spoofing, TLS termination)** — so
  every handshake HMAC is bound to the server's TLS certificate
  fingerprint; an attacker relaying the handshake through its own TLS
  legs produces mismatched digests (see [TLS](#tls))
- **Try to brute-force the fleet token** — so we rate-limit failed
  auth attempts per IP with exponential backoff (see
  [Rate limiting](#rate-limiting)), and the fleet key derives via
  scrypt (memory-hard), making offline dictionary attacks from a
  captured handshake expensive

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

### Key derivation (v2.3)

The fleet key is `scrypt(token, salt="macfleet-v3:"+fleet_id, n=2^14,
r=8, p=1)` — memory-hard, ~50 ms one-time cost at startup. An attacker
who captures a full handshake can attempt offline guesses against it,
but each guess costs ~16 MB of memory and ~50 ms; GPU farms don't help
the way they do against plain HMAC-SHA256. The auto-generated token
(64 random hex chars) is beyond any offline attack regardless; the KDF
exists to protect human-chosen tokens. Tokens shorter than 16 chars log
a warning.

### Handshake v3 (client proves first, v2.3)

Peer A connects to peer B over TLS:

1. A sends `node_id || c_a || proof` where `proof =
   HMAC(fleet_key, label_C1 || node_id || ':' || c_a || cert_fp)` and
   `cert_fp` is the SHA-256 of B's TLS certificate as A sees it.
2. B verifies `proof` FIRST. On failure B closes the connection
   without sending a byte — no HMAC response, no hardware profile.
   (Before v2.3, B answered any connector with
   `HMAC(fleet_key, attacker_chosen_challenge)` plus its signed HW
   profile: a free offline brute-force oracle and hardware recon.)
3. B responds with `HMAC(fleet_key, label_S || c_a || cert_fp)` + its
   own challenge `c_b` + its signed HW profile.
4. A verifies, sends `HMAC(fleet_key, label_C2 || c_b || cert_fp)` +
   its signed HW profile.
5. B verifies.

Every digest carries a distinct domain-separation label (a step-2
digest can't be replayed as a step-4 digest) and the TLS cert
fingerprint (see [TLS](#tls)). Replaying a captured hello yields only
the byte-identical ACK the attacker already has — completing the
handshake requires answering the fresh `c_b`.

**Version compatibility:** secure fleets require every node on the
same MacFleet version (>= 2.3). Pre-v2.3 secure hellos carry no proof
and are rejected — answering them would reopen the oracle.

### HW profile exchange (v2.2 PR 4 addition)

Piggy-backed on the handshake: both sides also send a signed hardware
profile (GPU cores, RAM, chip name, MPS/MLX availability, data port).
The signature binds the profile to the peer's challenge, so the payload
can't be replayed from a previous session. Since v2.3 it is only ever
sent to peers that have already proven token knowledge.

### Authenticated heartbeat

Heartbeat pings carry a signed HW profile (Issue 6):

```
APING (4 fields): APING {node_id} {nonce_hex} {sig_hex}
APING (5 fields): APING {node_id} {nonce_hex} {sig_hex} {hw_json_hex}
```

The client signs first (the responder never answers unauthenticated
pings), and since v2.3 the APONG response signature is bound to the
REQUEST nonce — a captured APONG cannot be replayed as the answer to
any later APING.

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

**No PKI, but channel-bound** — cert chain validation is disabled
(self-signed certs with a stable CA would require a PKI and a much
more complex pairing UX). Instead, every handshake HMAC mixes in the
SHA-256 fingerprint of the server's certificate as each side sees it
(v2.3). A MITM that terminates TLS on both legs necessarily shows the
client a different certificate than the server's own, so the digests
the two victims compute disagree and the handshake fails on both
sides. The HMAC challenge-response *is* the authentication; TLS
provides confidentiality; the fingerprint binding welds them together.

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
  which is constant-time, but the surrounding code (rate-limiter
  bookkeeping, error message logging, exception type branching) may
  leak small amounts of timing info that an on-LAN attacker could
  amplify across many attempts. The rate limiter caps practical
  exploitability at ~5 attempts per IP per ban window.
- **TLS cert rotation during a long session.** Certs are session-local;
  if a run goes 48+ hours you're still using the same ephemeral cert.
  Fine in practice but worth knowing.
