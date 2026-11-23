# Pairing Macs

Enrollment codes expire after five minutes by default (`--enroll-ttl` on bootstrap).

MacFleet uses a shared fleet token for mutual authentication and TLS.
The token is auto-generated on first `macfleet join` and saved at
`~/.macfleet/fleet-token` with mode 0600.

The preferred pairing flow does **not** print or paste the permanent
token. Instead, the first Mac starts a short-lived TLS enrollment
endpoint and displays a one-time code.

## Recommended Flow

On Mac #1:

```bash
macfleet join --bootstrap
```

This starts the normal agent and prints a command like:

```bash
Pair another Mac with this enrollment command:
  macfleet pair --host 192.168.1.10:61234 --code AB12CD-EF34GH-IJ56KL-MN78OP
Code expires at 14:05:31 and can be used 1 time.
```

On Mac #2:

```bash
macfleet pair --host 192.168.1.10:61234 --code AB12CD-EF34GH-IJ56KL-MN78OP
macfleet join
```

Install the same MacFleet build on both Macs. Enrollment protocol 2 uses the
code to authenticate both sides. The joining Mac verifies a proof covering the
returned credentials, its fresh nonce and the server's TLS certificate before
saving its token. Earlier enrollment endpoints require an upgrade.

If the connection drops while the credentials are being delivered, or the
joining Mac cannot save them, a single-use code may already be spent. Fix
the connection or file permissions, then issue a fresh code on the bootstrap
Mac. For a named fleet, keep using the `--fleet-id` printed after pairing.

Defaults:

- Code lifetime: 5 minutes
- Uses: 1 Mac
- Token destination: `~/.macfleet/fleet-token`

## When automatic discovery is unavailable

After pairing, leave `macfleet join` running on the first Mac and start the
second with the first Mac's reachable IP address and heartbeat port:

```bash
macfleet join --peer 192.168.1.10:50051
```

Updated token-protected peers verify each other's listening endpoints with
fresh authenticated replies, so this single address establishes membership
on both Macs without mDNS. Both Macs need reachable TCP ports; this cannot
bypass network client isolation. Automatic reciprocal discovery requires the
updated secure protocol. Open-mode and older peers retain their existing
manual discovery behavior.

## Pairing More Than One Mac

Use `--enroll-uses` when you intentionally want one command to enroll
multiple Macs:

```bash
macfleet join --bootstrap --enroll-uses 3 --enroll-ttl 600
```

Prefer the default single-use flow for normal setup. If a command was
shown to the wrong person or posted somewhere public, stop the agent and
rotate the token.

## Legacy Token URLs

Older MacFleet builds used URLs like:

```text
macfleet://pair?token=<permanent-token>&fleet=<fleet-id>
```

Current `macfleet pair` still accepts those URLs from explicit stdin or
pasteboard input for migration:

```bash
echo "macfleet://pair?token=...&fleet=default" | macfleet pair --stdin
macfleet pair --pasteboard
```

Treat a legacy URL exactly like a password. Anyone with it can join the
fleet until the token is rotated. MacFleet's legacy URL rendering helper
redacts the permanent token by default; code must explicitly opt into
revealing or copying token-bearing URLs for migration.

If `~/.macfleet/fleet-token` already exists, `macfleet pair` asks before
replacing it. For scripted legacy stdin migration, pass `--yes`
deliberately.

## Rotating The Token

Rotate when a token, legacy URL, screenshot, or enrollment command may
have been exposed:

```bash
macfleet rotate-token
macfleet join --bootstrap
```

Then re-pair every Mac and restart running agents. Rotation replaces the
local saved token; already-running agents keep their old in-memory token
until restarted.
