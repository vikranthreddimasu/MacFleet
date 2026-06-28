# Two-Mac Testing Walkthrough

This guide assumes you have one stronger Mac and one M1 MacBook Air, and that
you are new to MacFleet.

The short version: MacFleet does not turn two Macs into one giant GPU. It runs
the same training program on both Macs. Each Mac processes a shard of the data,
then they exchange gradients after every step so the model stays synchronized.
That means the M1 Air is useful for correctness, networking, discovery, and
smaller data-parallel experiments. On large models or slow Wi-Fi, the weaker
Mac can slow the fleet down because every rank waits at each gradient sync.

## What You Are Testing

Work through these levels in order:

1. Local install works on each Mac.
2. The Macs can pair securely and discover each other.
3. The raw transport/data-parallel path can synchronize parameters.
4. The high-level `Pool.train` path works with both Macs.

Do not start with a big model. A tiny smoke test tells you much more clearly
whether MacFleet itself is working.

## Prepare Both Macs

Use the same network first. Same Wi-Fi is fine. If you can connect the Macs
with Thunderbolt Bridge later, test Wi-Fi first, then Thunderbolt.

On both Macs:

```bash
python3 --version
```

You want Python 3.11 or newer.

Install the published package on both Macs:

```bash
mkdir -p ~/macfleet-test
cd ~/macfleet-test
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install 'macfleet[torch]'
macfleet --version
```

The latest PyPI package checked while writing this guide was `macfleet 2.2.1`.
If you only want to test the released package, the commands above are enough.

If you want to test this repository's newest local changes before they are
published to PyPI, install from the working tree on both Macs instead:

```bash
# Stronger Mac, from this repo
cd ~/Desktop/MacFleet
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[torch]'
```

Then copy the same working tree to the Air:

```bash
rsync -a --delete \
  --exclude .venv \
  --exclude .git \
  ~/Desktop/MacFleet/ USER@M1-AIR-HOSTNAME.local:~/MacFleet/
```

On the Air:

```bash
cd ~/MacFleet
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[torch]'
```

Run diagnostics on both:

```bash
macfleet doctor
macfleet info
```

If `MPS Available` is no, that is okay for the smoke tests below because they
default to CPU. The goal here is to test MacFleet's distributed plumbing.

## Verify Each Mac First (single-machine self-check)

Before involving the network at all, confirm MacFleet itself is sane on each
Mac. This needs only one machine and takes a few seconds. Run it on both:

```bash
source .venv/bin/activate
python tools/two_mac_verify.py --self-check
```

It checks import and version, network-interface detection, deterministic
topology address selection, network-link serialization, and a single-node
`Pool.train` that must actually reduce loss. Expect:

```
  5/5 checks passed
```

The process exits `0` only when every check passes, so you can gate a script on
it. If a Mac fails here, fix that Mac before pairing — a broken install will
only look like a confusing distributed bug later.

If you installed from PyPI rather than this repo, copy
`tools/two_mac_verify.py` from this repository onto each Mac and run it the same
way.

## Pair The Macs

On the stronger Mac:

```bash
source .venv/bin/activate
macfleet join --bootstrap --name strong-mac
```

Leave that terminal open. It starts an agent and prints a one-time pairing
command.

On the M1 Air, run the printed command. It will look like:

```bash
source .venv/bin/activate
macfleet pair --host 192.168.1.10:61234 --code ABCDEF-123456-...
macfleet join --name m1-air
```

Leave this terminal open too.

Open a second terminal on either Mac and check discovery:

```bash
source .venv/bin/activate
macfleet status
```

Pass condition:

- `macfleet status` shows 2 nodes.
- Both chips look roughly right.
- Each node has a heartbeat port and a data port.

If status shows no nodes, check that both Macs are on the same network, the
macOS firewall is allowing Python/Terminal, and VPN/client-isolation features
are not blocking Bonjour/mDNS.

## Important: Stop `join` Before SDK Training

`macfleet join` runs a long-lived agent on ports 50051 and 50052.
`Pool(enable_pool_distributed=True)` also starts an agent. If you leave
`macfleet join` running and then start a `Pool.train` script on the same Mac,
you can get a port conflict.

After pairing and status pass, press `Ctrl-C` in both `macfleet join`
terminals. The fleet token remains saved at `~/.macfleet/fleet-token`.

## Smoke Test The Raw Gradient Path

Skip this section if you installed only from PyPI. It uses repo helper scripts
under `tools/`.

This bypasses mDNS and the SDK. It proves the transport and allreduce can keep
parameters identical across the two machines.

On the stronger Mac:

```bash
source .venv/bin/activate
RANK=0 python tools/two_mac_demo.py
```

On the M1 Air, replace the IP with the stronger Mac's LAN IP:

```bash
source .venv/bin/activate
RANK=1 PEER_IP=192.168.1.10 python tools/two_mac_demo.py
```

Pass condition:

- Both Macs print a final SHA1.
- The final SHA1 is identical on both.

If this fails, focus on basic networking before debugging `Pool.train`.

## Verify The Full Fleet (automated, recommended)

The fastest way to confirm the two Macs work together is the distributed mode
of the same verifier you ran per-Mac. Make sure both `macfleet join` terminals
are stopped first (see the port note above), then run this on **both** Macs
within about 45 seconds of each other:

```bash
source .venv/bin/activate
python tools/two_mac_verify.py
```

On each Mac it forms quorum, checks the registry (two distinct nodes, each with
a data port and a chip name), exercises topology peer-address selection, and
runs a distributed `Pool.train`. Each Mac prints a summary and a single machine
-readable line, and writes `~/.macfleet/verify-<hostname>.json`:

```
  4/4 checks passed
VERIFY-RESULT host=strong-mac world_size=2 degraded=False params_sha256=ab12cd...
```

Pass condition (this is the whole point of two Macs):

- Both Macs print `4/4 checks passed`.
- Both `VERIFY-RESULT` lines show `world_size=2` and `degraded=False`.
- The `params_sha256` on the two Macs is **identical** — that proves the
  gradient allreduce kept the model in sync across machines.

If the two hashes differ, that is a real synchronization bug. Keep both
`~/.macfleet/verify-*.json` artifacts and both terminal logs.

Knobs (same on both Macs) match the smoke script: `DEVICE`, `COMPRESSION`,
`EPOCHS`, `BATCH_SIZE`, `QUORUM_TIMEOUT`, and `MACFLEET_PEERS` (comma-separated
`IP:PORT` peers when mDNS is blocked). Set `VERIFY_TRACEBACK=1` to print full
tracebacks for any failing check.

## Smoke Test The Real Pool.train Path

The verifier above is the recommended check. This section keeps the smaller,
hand-rolled smoke script for when you want to read exactly what the high-level
`Pool(enable_pool_distributed=True)` API does, step by step.

If you installed from this repository, run the included smoke script on both
Macs from the repo root:

```bash
source .venv/bin/activate
python tools/two_mac_pool_smoke.py
```

If you installed from PyPI, use the same script body from
`tools/two_mac_pool_smoke.py` in this repository and save it as
`two_mac_pool_smoke.py` on both Macs. Then run:

```bash
source .venv/bin/activate
python two_mac_pool_smoke.py
```

Start one Mac, then start the other within about 45 seconds. They do not need
to start at exactly the same time.

Pass condition:

- Both print `world_size == 2`.
- Both print `degraded == false`.
- Both print `unsynced_steps == 0`.
- Both print the same `params_sha256`.

The loss values do not need to be identical. Each rank sees a different shard
of data, but the final parameters should match.

Useful knobs:

```bash
DEVICE=cpu python tools/two_mac_pool_smoke.py
COMPRESSION=adaptive python tools/two_mac_pool_smoke.py
EPOCHS=5 BATCH_SIZE=64 python tools/two_mac_pool_smoke.py
```

Keep `DEVICE=cpu` while debugging. Once the CPU smoke test passes, try
`DEVICE=auto` to exercise the normal Torch device selection.

## What The M1 Air Is Good For

Use it to test:

- secure pairing and token persistence
- mDNS discovery and `macfleet status`
- heterogeneous hardware in the registry
- training mesh formation
- gradient allreduce correctness
- thermal behavior during longer runs
- whether topology-aware address selection picks the right network

Do not expect every training job to become faster. With data-parallel training,
all ranks synchronize every step, so the slowest Mac and the network can set
the pace. A tiny M1 Air is still a very good development node because it makes
distributed bugs real instead of theoretical.

## Common Failures

### `No quorum within ... saw 1 node`

The local script started, but it did not see the other Mac.

Try:

```bash
macfleet status
```

If status cannot see both Macs while `macfleet join` is running, fix discovery
first. Same network and firewall permissions are the usual culprits.

### `Address already in use`

You probably left `macfleet join` running while starting
the smoke script. Stop `join` with `Ctrl-C` on that Mac and retry.

### `PeerAuthError`

The Macs have different fleet tokens. Re-pair the Air:

```bash
macfleet rotate-token
macfleet join --bootstrap
```

Then run the printed `macfleet pair ...` command on the Air again.

### Final hashes differ

That is a real synchronization bug. Save both terminal logs. The useful lines
are `rank`, `world_size`, `degraded`, `unsynced_steps`, `last_sync_error`, and
`params_sha256`.

## Longer Test

This section also requires a repository checkout because it uses `tools/`.

After the smoke tests pass, try the longer direct training demo:

```bash
# Stronger Mac
RANK=0 python tools/two_mac_real_train.py

# M1 Air
RANK=1 PEER_IP=192.168.1.10 python tools/two_mac_real_train.py
```

Pass condition:

- same final SHA1 on both Macs
- held-out test accuracy above 80%
- bytes sent on one Mac roughly match bytes received on the other

That test is intentionally more visible and slower. Use it after the small
smoke test, not before.
