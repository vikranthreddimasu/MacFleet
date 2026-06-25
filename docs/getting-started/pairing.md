# Pairing Macs

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
Pair another Mac with this one-time command:
  macfleet pair --host 192.168.1.10:61234 --code AB12CD-EF34GH-IJ56KL-MN78OP
Code expires at 14:05:31 and can be used once.
```

On Mac #2:

```bash
macfleet pair --host 192.168.1.10:61234 --code AB12CD-EF34GH-IJ56KL-MN78OP
macfleet join
```

The one-time code proves to the enrollment server that Mac #2 saw the
human-visible code. The credential transfer itself happens over TLS and
uses a channel-bound HMAC proof, so the permanent fleet token is not
placed in terminal history, pasteboards, QR images, or chat logs.

Defaults:

- Code lifetime: 5 minutes
- Uses: 1 Mac
- Token destination: `~/.macfleet/fleet-token`

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

Current `macfleet pair` still accepts those URLs from pasteboard or
stdin for migration:

```bash
echo "macfleet://pair?token=...&fleet=default" | macfleet pair --stdin
```

Treat a legacy URL exactly like a password. Anyone with it can join the
fleet until the token is rotated. MacFleet's legacy URL rendering helper
redacts the permanent token by default; code must explicitly opt into
revealing or copying token-bearing URLs for migration.

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
