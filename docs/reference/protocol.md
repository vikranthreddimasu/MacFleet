# Wire protocol reference

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

### Secure mode — APING v1 (4 fields, v2.1 legacy)

```
→ APING <node_id> <nonce_hex> <hmac_sig_hex>
← APONG <responder_id> <nonce_hex> <hmac_sig_hex>
```

Where `hmac_sig = HMAC-SHA256(fleet_key, node_id || nonce)`.

### Secure mode — APING v2 (5 fields, v2.2)

```
→ APING <node_id> <nonce_hex> <hmac_sig_hex> <hw_json_hex>
← APONG <responder_id> <nonce_hex> <hmac_sig_hex> <hw_json_hex>
```

Where `hmac_sig = HMAC-SHA256(fleet_key, node_id || nonce || hw_json)`.

The 5-field variant is used by `--peer` manual-peer bootstrap so the
peer registers with real hardware info (GPU cores, RAM, data_port)
instead of a zero-score placeholder. The HMAC covers the HW JSON so
a rogue peer can't lie about hardware to win coordinator election.

A v2.2 server accepts both variants; a v2.2 client pinging a v2.1
server gets a 4-field APONG back and falls through to the legacy
zero-HW path (logged for visibility).

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

- Max payload: 256 MB (`MAX_PAYLOAD_SIZE`). Set conservatively high —
  100M-param float32 gradients are ~400 MB uncompressed, but
  compressed gradients (Top-K, FP8) are much smaller.

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

### Handshake (v2.2)

1. **A → B**: CONTROL message with flag `HANDSHAKE_V2` set. Payload:
   `node_id || challenge_a` + HW suffix.
2. **B → A**: CONTROL. Payload: `response_a (HMAC of challenge_a) ||
   challenge_b || response_b_to_a_ack` + HW suffix.
3. **A → B**: CONTROL. Payload: `response_b` (HMAC of challenge_b).

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
