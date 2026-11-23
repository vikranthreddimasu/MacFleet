# Wire protocol reference

Wire messages use a fixed 24-byte header (see `macfleet/comm/protocol.py`).


MacFleet uses two protocols on distinct ports:

- **Heartbeat** (line-based ASCII, port 50051 by default): discovery
  + liveness probing
- **Transport** (binary framed, port 50052 by default): gradient sync,
  task dispatch, training coordination

The two are split per Issue 5 (v2.2 PR 2) because their framing rules
are incompatible — a transport handshake looks like a malformed APING
and vice versa.

## Heartbeat protocol

### Open (no-auth) mode

```
→ PING <node_id>
← PONG <node_id>
```

Used only for open pools (no token). Terminal newline-delimited.

### Secure heartbeat

Both nodes must run the same protocol. Over TLS, they derive the heartbeat
key from the fleet key and the server certificate fingerprint:

```
→ APING <client_id> <client_nonce> <client_proof>
← APONG <server_id> <server_nonce> <server_proof> [<server_hw_hex>]
→ AHW <client_hw_proof> <client_hw_hex>   (if the client shares hardware)
```

APONG authenticates both nonces and the server's hardware. Only after
verifying it does the client send its hardware, signed against the server
nonce. A TLS relay with a different certificate cannot forward valid proofs.
A manual peer is registered on the receiving side after a fresh reverse probe.

### Bounds

- Read timeout: 1s (was 5s pre-v2.2 PR 6)
- HW payload: capped at 8 KB (`HW_HANDSHAKE_MAX_JSON_BYTES`)
- Rate limit: 5 failures per IP → 5-minute ban, exponential backoff in
  between

## Transport protocol

Binary, framed, CRC32-verified.

### Header (24 bytes)

```
stream_id:    uint32  (multiplexing; 0 = control, 1..N = tensor/task)
msg_type:     uint16  (CONTROL, TENSOR, HEARTBEAT, TASK, RESULT, ...)
flags:        uint16  (bit 0: compressed, bit 1: chunked, bit 2: last_chunk,
                       bit 3: HANDSHAKE_V2)
payload_size: uint32  (bytes)
sequence:     uint32  (ordering within stream)
checksum:     uint32  (CRC32 of payload)
reserved:     uint32  (future use)
```

Struct format: `!IHHIIII` (big-endian).

### Payload bounds

- Max frame payload: 256 MiB (`MAX_PAYLOAD_SIZE`). A larger collective array
  starts with a zero dtype-length marker followed by its regular array header.
  Raw 8 MiB frames follow, with a shorter final frame if needed. Receivers
  validate the shape, dtype and each chunk size before accepting the array.

### Message types

| Value | Name | Purpose |
|-------|------|---------|
| 0x01 | CONTROL | Handshake, barrier, state |
| 0x02 | TENSOR | Raw tensor broadcast |
| 0x03 | HEARTBEAT | Transport-layer keepalive (distinct from mDNS heartbeat) |
| 0x04 | GRADIENT | Full gradient (uncompressed) |
| 0x05 | COMPRESSED_GRADIENT | Gradient + compression metadata |
| 0x06 | BARRIER | Synchronization point |
| 0x07 | STATE | Checkpoint / recovery state |
| 0x08 | TASK | `@macfleet.task` dispatch (msgpack) |
| 0x09 | RESULT | Task return value (msgpack) |

### Handshake v3 (client proves first, v2.3+)

Secure transport always runs over TLS. The HMACs below are
domain-labeled and include the SHA-256 fingerprint of B's TLS
certificate as A sees it.

1. **A → B**: CONTROL message with flag `HANDSHAKE_V2` set. Payload:
   `node_id || challenge_a || hello_proof`, where `hello_proof`
   proves A knows the fleet key before B sends any bytes.
2. **B verifies first.** On failure, B closes silently. This avoids
   turning B into an offline token-brute-force oracle.
3. **B → A**: CONTROL. Payload:
   `server_response_a || challenge_b || hw_suffix_b`.
4. **A verifies B**, peels and verifies `hw_suffix_b`, then sends:
   `client_response_b || hw_suffix_a`.
5. **B verifies A's response** and the signed hardware suffix.

The HW suffix format (right-to-left peelable):

```
wire_version   (1B)
hw_json_len    (2B BE)
hw_json        (variable, <= 8 KB)
hmac           (32B, signed over wire || peer_challenge || local_id || hw_json)
block_size     (2B BE, trailing — tells receiver how much to peel)
```

The trailing `block_size` lets the receiver peel the suffix off a
variable-length base (`node_id + challenge_a`) without reparsing from
the left.

Secure fleets reject handshakes without the client-first proof. Open fleets
(created only with `--open --allow-insecure-open`) use the older
plain `node_id` exchange and do not carry hardware suffixes.

## TASK / RESULT payloads

Both use `msgpack` with `use_bin_type=True`.

### TaskSpec

```python
{
    "task_id": str,          # UUID hex (32 chars)
    "name": str,             # Registered task name (looked up in TaskRegistry)
    "args": list,            # msgpack-native values
    "kwargs": dict,          # msgpack-native values
    "timeout": float,        # Seconds
}
```

Max size: 64 MB (`MAX_ARGS_BYTES`).

### TaskResult

```python
{
    "task_id": str,
    "ok": bool,
    "value": Any,            # msgpack-native (Pydantic dumped via model_dump)
    "error": Optional[str],  # Traceback when ok=False
}
```

Max size: 256 MB (`MAX_RESULT_BYTES`).
