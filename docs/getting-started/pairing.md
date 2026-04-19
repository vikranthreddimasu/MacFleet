# Pairing Macs

MacFleet uses a shared fleet token for mutual authentication (HMAC) and
TLS. The token is auto-generated on first `macfleet join`. Getting
that token from Mac #1 to Mac #2 is the pairing problem.

Three flows, one encoding.

## Flow 1: QR code (recommended)

Easiest when you have your phone handy.

```bash
# Mac #1
macfleet join --bootstrap
```

This prints a QR block:

```
Fleet pairing URL: macfleet://pair?token=...&fleet=default
Scan this QR from a second Mac's iPhone camera:
████ ▄▄▄▄▄ █▀█ █▄▀▄▀ ██▄ ▄▄▄▄▄ ████
...
```

On Mac #2, open the iPhone camera and point it at the QR. iOS
surfaces a tappable `macfleet://` link in the camera preview. Tap it,
Handoff sends the URL to your Mac, and `macfleet pair` picks it up
from the pasteboard.

## Flow 2: Handoff pasteboard

When both Macs are signed into the same Apple ID, `--bootstrap` also
copies the URL to your pasteboard. Any Mac you switch to via Handoff
pasteboard sync sees it instantly:

```bash
# Mac #1
macfleet join --bootstrap   # also copies URL to pasteboard

# Mac #2 (a few seconds later, Handoff has synced)
macfleet pair               # reads URL from pasteboard
macfleet join               # joins using the token now on disk
```

No QR scanning needed. Fastest path if your fleet is all yours.

## Flow 3: Raw URL

SSH-only box? No iPhone? Just share the URL:

```bash
# Mac #1
macfleet join --bootstrap | grep "Fleet pairing URL"
# copy that line, paste into a Slack DM, iMessage, whatever

# Mac #2
echo "macfleet://pair?token=...&fleet=default" | macfleet pair --stdin
```

The URL is the same whether it came from QR, pasteboard, or text.

## What's in the URL

```
macfleet://pair?token=<url-encoded-token>&fleet=<url-encoded-fleet-id>
```

- **`token`** — 32+ char random secret. All fleet operations (HMAC,
  TLS cert derivation) derive from this.
- **`fleet`** — optional logical-fleet name. If you're running
  multiple fleets on the same LAN (lab + personal dev, say), this
  prevents cross-contamination via mDNS scoping.

**Treat the URL like a password.** Anyone with it can join the fleet
and run arbitrary registered tasks. Don't paste it in public Slack
channels.

## Rotating the token

Not yet exposed as a CLI (tracked as A21 in the v2.3 roadmap). For
now: delete `~/.macfleet/token` on every Mac and re-run
`macfleet join --bootstrap` on one of them.
