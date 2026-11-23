# MacFleet real-world audit

**November 22, 2026 · Apple M5 Pro and Apple M5 Max**

The original observations below describe the audited 2.2.1 checkout. See [post-fix verification](#post-fix-verification) for the fixes, [the sustained CPU follow-up](#follow-up-on-the-remaining-tests-and-sustained-training) for the first longer run, [the GPU workload and final test review](#gpu-workload-and-review-of-the-final-16-tests) for the two-Mac results, and [the B18 verification](#b18-fix-and-verification-on-the-m5-pro) for the subsequent rediscovery fix.

## Verdict

**The two Macs can train together.** They paired, discovered each other and completed the same classification workload on Torch CPU, Torch MPS and MLX. Each engine produced matching final weight hashes on the Pro and Max. When the Max exited mid-training, the Pro aborted. A fresh CPU run on the same paired Macs reproduced the original hash.

**Two verified bugs are release priorities.** Distributed Torch silently decayed an inactive parameter that native single-node training preserved (B1). After a worker reconnect, a remote task executed even though its caller received a timeout (B9). Both were reproduced with authenticated transports on the Pro. A separate source review found a missing client-side issuer check in first-time enrollment (S1). Its potential security impact needs protocol review before deployment on an untrusted LAN. The most visible physical usability problem was the 90-second wait for a peer identity that had already left during the CLI-to-SDK handoff (B14).

## Scope and evidence

The audit follows a new user from installation and enrollment through training, monitoring and recovery. Returning-user checks use disposable credentials. GPT-6.1 Terra operated the machines. A separate regression and source review checked numerical correctness and security boundaries. No product fixes were applied.

The target is the user's working directory, including its uncommitted changes. Its package declares version 2.2.1. The checkout was on `master` at `c7c62c0`, with staged, unstaged and untracked work. A clean checkout of that commit does not reproduce the audited source. The source fingerprint was `a9bcb360622c95d80674e72ce110aa8b7844658e3009291989135fcd7dc28c3f`. To reproduce it, sort the 55 `macfleet/**/*.py` paths, and for each feed its UTF-8 repository-relative path, a NUL byte and the 32-byte SHA-256 digest of its contents into one SHA-256 calculation. The install tests used the local package with dependencies resolved separately. Results do not establish compatibility with every wheel labeled 2.2.1.

MacFleet exposes a CLI, Python SDK and Rich terminal dashboard. Account creation and cloud login are outside its normal local workflow. A CLI agent advertises the fleet. Authenticated heartbeats establish membership. Both ranks must run the training program, which starts its own agent, forms a separate data mesh, broadcasts initial state and averages gradients. A training result can report matching weight hashes while both ranks have made the same incorrect update. The independent optimizer comparison below tests that case.

## Physical test environment

| | Primary | Secondary |
|---|---|---|
| Machine | Apple M5 Pro, 15 CPU cores, 16 GPU cores, 24 GB unified memory | MacBook Pro, Apple M5 Max, 18 CPU cores, 40 GPU cores, 48 GB unified memory |
| OS | macOS 27.0.1 | macOS 27.0.1 (26A434) |
| Initial runtime | Python 3.12.13. Regression interpreter had Torch 2.11.0 and NumPy 2.4.4. Fresh venv resolved Torch 2.15.0 with MPS. | Default Python 3.9.6, below MacFleet's 3.11 minimum. No MacFleet checkout, executable or Python distribution found in the locations checked. |
| Network | Wi-Fi, IPv4 on en0 | IPv4 on en0. Standard SSH port refused locally. Both LAN addresses fall in the primary's /22. |

Codex could reach the Max through a dedicated chat even though its Bonjour hostname did not resolve on the Pro. A connection to the Pro's own SSH port also returned refusal. These observations concern the operator access path. MacFleet uses its own listeners and discovery. Initial sandbox runs could not bind multicast sockets or inspect all hardware. Host-access reruns supply the hardware and ordinary-network results below.

The Max operator found a bundled Python 3.12.14 and created a private venv. The Pro built the checkout wheel. The Max verified its SHA-256 as `84d78919b8ceb32d1885f33c9b04fce3248d1d3d2ff46e5b20df0dda3de60b12` before installing it. Transfer used a temporary mutual-TLS endpoint restricted to the Max, with the Pro certificate checked against an independently delivered fingerprint. This provides direct evidence of Pro–Max IPv4 reachability independent of MacFleet. The Pro's isolated fleet directory, token and audit log had modes 0700, 0600 and 0600 respectively.

## Physical fleet results

| Scenario | Observed on the physical machines |
|---|---|
| First-time enrollment | The Pro started a token-protected bootstrap agent, with TLS enabled, heartbeat port 51411 and data port 51412. The Max used the one-time CLI command and saved its new credential in the isolated home. The operator suppressed the Pro helper's automatic system clipboard copy so that test credentials remained in scratch. |
| Bonjour discovery | Both CLI agents joined without manual peer hints. Pro and Max each saw `secure=true`, `count=2`, both hostnames and both port pairs in status. The Pro logged `Discovered audit-max.local`. Both JSON rows reported chip `unknown`, GPU cores 0, RAM 0 and compute score 0.0. The Max's last CLI log still showed world size one when captured. No distinct authenticated readiness signal was established by this status check. |
| Overlapping CLI shutdown and SDK startup | First Torch CPU attempt failed. The Pro included the Max's still-running CLI agent in its training roster. It logged the peer's subsequent removal and the new Max advertisement, then exited 1 with `MeshFormationError: Peers never connected within 90s: audit-max-94c8800c`. The new Max rank connected and entered model-consistency setup. Its gather ended with an EOF waiting for a 24-byte header. Max elapsed time 64.69 s, peak RSS 337,543,168 bytes, exit 1. Neither side produced a training result. |
| Returning user and SDK training | Once both CLI agents exited, the same saved credentials worked on both machines. CPU, MPS and MLX completed with two distinct ranks and matching final weights. In the MLX startup log the Pro authenticated the Max and received its M5 Max, 40-core GPU, 48 GB hardware profile. |
| Remote compute and task failure | The Pro submitted a named square task over a secure mesh. The Max returned 49 for input 7, with its own OS hostname and process ID. An intentionally failing task surfaced its expected error to the Pro. The next square task returned 64 from the same Max worker. This used the explicit dispatcher/worker API. |
| Link and peer interruption | Ten 1 MiB mean allreduces returned exactly 1.5 in every element from ranks supplying 1.0 and 2.0. In a second run the Max exited with the intended code 17 after three exact reductions. The Pro's next reduction raised `IncompleteReadError` in 24.82 ms. Fresh agents reused port 51421 and completed another ten exact reductions. |
| Active training failure and fresh restart | In a separate CPU training run the Max deliberately exited at its fourth model forward (exit 17, 4.06 s process wall time). The Pro exited 1 with `SyncDegradedError: Gradient sync degraded (allreduce_failed:ConnectionResetError)` and returned no success. The Pro process took 23.43 s including staggered startup. Both restarted from initialization on the same ports without re-pairing. Both completed 24 steps, no degradation, and reproduced the full original CPU hash. Pro SDK interval 0.331 s, Max process wall time 11.14 s. These intervals use different timers. |
| Explicit model and optimizer recovery | After one CPU epoch, each rank saved its Torch model and Adam state at optimizer step 8. The uninterrupted second segment reached step 16. Fresh processes loaded each rank's saved state with `weights_only=True` and repeated that segment. In both executions **both ranks** reached step 16 with SHA-256 `937e65ae31dba93a0fb905830f3d6bf3eba7bfd25eb48722ad75701e124d38ee` and zero validation fallbacks. The Pro's local loss was 0.1570277363 on both continuations. The Max's was 0.1030076949 on both. The Max save/control and resume processes took 4.61 s and 14.11 s. This verifies explicit restore at a completed epoch boundary. Both control and resume restarted the sampler at the same point. Automatic mid-epoch resume was not exercised. |

### Training evidence and performance

Each backend trained for three epochs on 512 examples. The Max was rank 0 and the Pro was rank 1. Both completed 24 steps with zero unsynced or validation-fallback steps. Loss is the mean over each rank's own shard. The complete SHA-256 values were collected separately on both machines and compared. The hashes need to match **within** a backend.

| Backend | Loss, Pro / Max (first → last epoch) | SHA-256, identical on the two Macs |
|---|---|---|
| Torch CPU | 0.99586 → 0.04185 / 0.93522 → 0.11030 | `f8e78ec7e28952e986cc37ebaefb09eb3f028a1d5cf95d75649329197b86aa99` |
| Torch MPS | 0.99586 → 0.04185 / 0.93522 → 0.11030 | `1d1ea6d949b2c8afc6d59e6f7007f1c500547422bef6e88c0be642f9a5e6c9de` |
| MLX | 24.7469 → 1.44645 / 24.6702 → 0.79521 | `f23ab6d05cc78a31f9f77ffc6306ef68c391f6dc94f3118fef27f1f72d2a880c` |

| Backend | SDK training s, Pro / Max | Mean sync ms, Pro / Max | Local wall s through result, Pro / Max | RSS high-water at result, MiB, Pro / Max | Effective fleet samples/s |
|---|---:|---:|---:|---:|---:|
| Torch CPU | 0.724 / 0.782 | 20.67 / 30.89 | 17.00 / 5.32 | 261 / 321 | 1,964 |
| Torch MPS | 0.546 / 0.548 | 11.18 / 9.68 | 5.64 / 13.25 | 468 / 469 | 2,802 |
| MLX | 1.854 / 1.853 | 73.64 / 10.03 | 5.07 / 12.37 | 101 / 98 | 829 |

Fleet throughput is 1,536 global examples divided by the longer reported training interval. Each rank processes 32 examples per update. Local wall timing starts before joining the pool and therefore includes waiting for the other machine. It ends when the script builds the result, before pool teardown. Start times were staggered. RSS is the process high-water value at that point. The Max's external process monitor recorded higher lifetime peaks, particularly for MLX (207 MiB). Mean sync includes time waiting for the other rank. These numbers cannot isolate network time, and the different rank measurements should not be added together.

For the 4,866-parameter Torch model, the warm single-Pro CPU baseline finished training in roughly 5 ms, versus roughly 782 ms with two Macs. Warm single-Pro MPS training took roughly 38–41 ms, versus 548 ms distributed. This workload is too small to amortize synchronization. The audit did not measure a model-size break-even point. Separately, ten exact 1 MiB secure allreduces measured on the Pro took 64.35–171.52 ms, median 104.22 ms. After restart the range was 91.3–307.6 ms. Each timing covers the full operation on both ranks, including synchronization.

## Primary onboarding and local workflows

| Executed scenario | Result and evidence |
|---|---|
| Install the checkout in a new venv | Clean install succeeded in 5.91 s wall time, approximately 197 MiB peak RSS. Initial restricted-network build could not fetch build dependencies. The host-access retry succeeded. |
| Install the public release as a new user | In a different clean venv outside the checkout, `python -m pip install --no-cache-dir --index-url https://pypi.org/simple macfleet` took 4.67 s. It installed 2.2.1 with NumPy 2.5.5. Root, join and pair help and the generated starter matched the checkout byte for byte. The wheel SHA-256 was `51d90d15e000efb1230d1359c8864918b965dd904a852fa11a2d36ec1a5a25ce`. No public-wheel fleet interoperability run was made. |
| Add the Torch extra | 12.15 s, including a 138.6 MB Torch wheel. Wall times depend on the resolver, cache and network available during the run. |
| Read README, generate and run starter | Core-only run failed at `import torch`. With the Torch extra, the generated script completed 10 epochs and 160 steps, with epoch loss falling from 0.5753 to 0.0435. Total process wall time 2.00 s, approximately 449 MiB peak RSS, SDK training interval 0.445 s. MPS was available. |
| Local diagnostics and status | On the unrestricted primary, doctor identified the M5 hardware and found 14 interface entries. Empty-fleet status returned no nodes in 3.14 s. A restricted multicast socket exposed an unhandled status error. |
| Built-in demo | `macfleet train --epochs 1` completed its CPU demo. The footer's suggested next command failed when tried. |
| Generated starter in a new home | Single-node training succeeded without creating a fleet token. A local pool does not join the LAN. |
| User chooses an output path | Generating into an existing working directory succeeded. A missing parent directory produced a Python traceback. |

On the **physical M5 Max**, the same generated starter ran with Python 3.12.14, Torch 2.15.0 and MLX 0.32.4 installed. It reported world size one, 10 epochs, 160 steps and loss 0.57526 → 0.04351. Training took 1.14 s, the process took 2.59 s and peak RSS was 474,677,248 bytes (453 MiB). Doctor passed 12/13 checks, including MPS and MLX availability. Its only failed check was the absent fleet token before enrollment.

### Single-Mac baseline for the fleet workload

Separate from the starter, the Pro ran three local repetitions of the fleet workload on each backend: 512 examples with eight float features, two hidden layers of width 64, two output classes, batch 64, three epochs and 24 updates. The Torch loss was cross-entropy and the optimizer was Adam. The script set one Torch CPU thread. The outer timer starts immediately before entering the pool and stops after pool exit, excluding data and model creation. The first repetition includes warm-up effects. These are tiny-model latency measurements and use a different local shuffle from the distributed sampler.

| Pro backend | SDK training seconds, repetitions 1 / 2 / 3 | Outer wall seconds, repetitions 1 / 2 / 3 |
|---|---|---|
| Torch CPU | 0.0330 / 0.00518 / 0.00507 | 0.581 / 0.00538 / 0.00521 |
| Torch MPS | 0.6578 / 0.04114 / 0.03825 | 0.6635 / 0.04241 / 0.03955 |
| MLX | 0.6695 / 0.01667 / 0.01550 | 0.7124 / 0.01710 / 0.01588 |

The baseline processes reported cumulative peak RSS of roughly 305 MB for Torch CPU, 461 MB for Torch MPS and 71 MB for MLX. RSS does not measure all GPU/unified-memory allocations. MLX loss histories varied across the repetitions. Use these intervals as observations of the workload size, without inferring identical optimization trajectories.

### Reproducing the physical Torch workload

Use two isolated operator accounts or scratch homes and the same audited wheel. On the Pro run `macfleet join --bootstrap --fleet-id physical-audit-1122 --port 51411 --data-port 51412`. On the Max run its printed pairing command privately, then run `macfleet join --fleet-id physical-audit-1122 --port 51411 --data-port 51412`. After observing both CLI agents, stop them before launching the SDK scripts. Retain the same isolated home on each machine. Set `PEER` to the other Mac's IPv4 address, `ROLE` to `pro` or `max`, and `DEVICE` to `cpu` or `mps`. The core of the measured script is below. The original transferred Torch script had SHA-256 `390e14182bb991526a305e7dfc6f5c67b507b06bd3899f162bb304429179da3a`.

```python
import os
import numpy as np
import torch
from macfleet import Pool
from torch import nn

role = os.environ["ROLE"]
torch.set_num_threads(1)
x = np.random.default_rng(42).normal(size=(512, 8)).astype(np.float32)
y = (x[:, 0] + x[:, 1] > 0).astype(np.int64)
torch.manual_seed(1 if role == "pro" else 2)
model = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 64),
                      nn.ReLU(), nn.Linear(64, 2))
with Pool(name=f"audit-{role}", engine="torch", fleet_id="physical-audit-1122",
          port=51411, data_port=51412, peers=[os.environ["PEER"] + ":51411"],
          enable_pool_distributed=True, quorum_size=2, quorum_timeout_sec=90,
          rendezvous_timeout_sec=90) as pool:
    result = pool.train(model, (x, y), epochs=3, batch_size=64, lr=0.1,
                        loss_fn=nn.CrossEntropyLoss(), device=os.environ["DEVICE"],
                        compression="none")
    print(result)
```

The positive training gate requires two distinct hosts and ranks, 24 steps per rank, `world_size=2`, `degraded=False`, zero unsynced steps, lower final epoch loss on both ranks and the same full final parameter hash. The explicit peer address is a manual discovery fallback for the training run. A positive run with this argument cannot independently prove Bonjour discovery.

## Verified bugs

Each entry gives the smallest observed reproduction, the expected and actual results, and the code responsible. Findings from multiple processes on one machine are labeled explicitly.

### B1 · High · Distributed Torch changes weights unused by every rank

**Reproduce.** Define a Torch module with two trainable 1×1 parameters, both initialized to 1. Return `x @ used` from `forward`, leaving the second parameter unused. Train on 16 rows of ones against zero targets with MSE, SGD at learning rate 0.1 and weight decay 0.1. Run one epoch at global batch 8 on CPU. First use a local `Pool`, then run the same script in two independent, token-authenticated processes with `enable_pool_distributed=True, quorum_size=2`. Compare both parameters after two steps.

| Run | Used weight | Unused weight | Steps |
|---|---:|---:|---:|
| Local Pool | 0.6241 | 1.0000 | 2 |
| Secure rank 0 on the Pro | 0.6241 | 0.9801 | 2 |
| Secure rank 1 on the Pro | 0.6241 | 0.9801 | 2 |

Both processes exited zero. PyTorch skips a parameter whose gradient is absent. Flattening substitutes zeros and applying the reduced array attaches a gradient to every trainable parameter. Weight decay then changes a branch that received no training signal. An independent one-step engine check with AdamW also moved the unused weight from 1.0 to 0.99000001 and created a second optimizer-state entry. To fix this, synchronize per-parameter participation and retain an absent gradient when all ranks skipped that parameter. Test both globally unused and partially used branches against native Torch. [Torch gradient extraction and application](../macfleet/engines/torch_engine.py#L158) · [synchronization](../macfleet/training/data_parallel.py#L283)

### B2 · Medium · NumPy token IDs lose their integer type

**Reproduce.** In a joined local `Pool`, call `pool.train(torch.nn.Embedding(4, 2), (np.arange(16, dtype=np.int64) % 4, np.zeros((16, 2), dtype=np.float32)), loss_fn=torch.nn.MSELoss(), epochs=1, batch_size=8, device="cpu")`. Repeat with the same input IDs wrapped in `torch.from_numpy`.

The NumPy case should preserve integer indices. It raised `Expected tensor for argument #1 'indices' ... Long, Int. But got torch.FloatTensor instead`. The Torch tensor control completed two steps. The shared dataset adapter forces non-tensor features to float32. To fix this, preserve supported NumPy dtypes, including integers and booleans. Exercise token IDs in both single-node and distributed training. Only the single-node case was directly reproduced. [Dataset adapter](../macfleet/sdk/pool.py#L85)

### B3 · Medium · The first successful CLI demo points to a broken multi-Mac command

**Reproduce.** Run `macfleet train --epochs 1`. Follow the footer by calling `macfleet.Pool().train(model, dataset, epochs=10)` with a valid model and dataset. It raises `RuntimeError: Must join pool before training`. Joining a default pool still selects local execution. The expected next step is a runnable two-Mac example. Both demo implementations print this suggestion. To fix this, show a context-managed pool with distributed intent and quorum two, and say to run the same script on both paired Macs. [Torch demo footer](../macfleet/cli/main.py#L871) · [MLX demo footer](../macfleet/cli/main.py#L953) · [Pool guard](../macfleet/sdk/pool.py#L710)

### B4 · Medium · The README's five-minute path fails after the advertised core install

**Reproduce.** Install the core checkout in a fresh venv, run `macfleet quickstart`, then `python my_macfleet_demo.py` as the README directs. The generated file imports Torch and stops with `ModuleNotFoundError: No module named 'torch'`. Installing the Torch extra made that exact script complete. The scaffold's console output gives the correct install hint, but the README's numbered procedure omits it. Its companion quickstart guide also describes generating a token on the first local run, which did not occur. To fix this, start the training path with the Torch extra and describe credential creation at distributed onboarding. [README](../README.md#L48) · [starter](../macfleet/cli/main.py#L1550) · [quickstart guide](../docs/getting-started/quickstart.md)

### B5 · Medium · Doctor treats an optional missing engine as a readiness failure

**Reproduce.** In a clean Torch-only installation with working MPS, run `macfleet doctor --json`, capture its exit status and then run the starter. The MLX check failed, the report said it was not ready, the command exited zero, and the Torch workload completed. The fresh-home test also correctly identified the missing fleet token. That missing token does not explain the separate MLX failure. The doctor implementation requires every check to pass and returns normally even for a failed report. To fix this, distinguish core, selected-engine and fleet readiness. Put absent optional engines in an informational category and document a testable process-exit contract. [Diagnostic checks](../macfleet/cli/main.py#L493) · [report construction](../macfleet/cli/main.py#L550)

### B6 · Medium · Permission-denied multicast leaves status without an actionable result

**Reproduce.** From a process environment that denies multicast socket binding, run `macfleet status` and `macfleet status --json`. Both exposed a traceback from the discovery call. With normal host networking, the scan completed. A user needs the failed operation, the relevant Local Network permission and a useful exit code. JSON consumers need a parseable error. To fix this, catch socket setup errors at the CLI boundary and preserve text/JSON output contracts. This reproduction used the Codex sandbox. A macOS Privacy & Security denial was not induced. [Status discovery call](../macfleet/cli/main.py#L372)

### B7 · Low · Quickstart prints a traceback for a missing destination directory

**Reproduce.** From a fresh directory run `macfleet quickstart --output missing/demo.py`. Expected a usable file or a concise explanation. Actual result was an uncaught `FileNotFoundError` at `Path.write_text`. To fix this, report the missing parent with a Click error or create it deliberately. [Writer](../macfleet/cli/main.py#L1633)

### B8 · Medium · Package and security guide disagree on the secure protocol version

**Reproduce.** Run `macfleet --version` from the audited package and in a second venv installed from public PyPI. Compare the `macfleet/**/*.py` files and the checkout security reference's compatibility paragraph. Both distributions identify as 2.2.1. They contain the same 55 Python filenames, but 44 files differ by content. The checkout implements the client-first exchange. The guide says all secure peers require version 2.3 or newer. The public dependency metadata also lacks the checkout's NumPy <2.5 ceiling. This gives operators conflicting advice during a failed pairing or mixed-version rollout. A mixed-version interop failure was not observed. To fix this, give the wire handshake an explicit visible version and align package metadata, discovery, release notes and compatibility guidance before publication. [Package version](../pyproject.toml#L7) · [security guide](../docs/reference/security.md#L91)

### B9 · High · A reconnected worker executes a task but its caller loses the result

**Reproduce.** Create authenticated loopback `PeerTransport` endpoints with stable IDs. Start a `TaskDispatcher` and `TaskWorker`, and register an echo task that also appends its argument to a worker-side list. Submit `initial`. Disconnect both endpoints, stop the worker, reconnect the same IDs and start the worker. Submit `new-link` through the existing dispatcher with a 0.7 s timeout. Finally, stop and start the dispatcher and submit `restarted`.

The first and third calls returned their argument. The middle call raised `RemoteTaskError: Task timed out`. The worker recorded exactly `['initial', 'new-link', 'restarted']`. The expected outcome is a result on the live connection or rejection before execution. The dispatcher's receiver is tied to the previous connection while dispatch selection admits the replacement. A caller who retries a non-idempotent task can repeat its side effects. To fix this, establish the result listener for the new connection before admitting work. Preserve ambiguous-outcome reporting for requests already in flight. Test a reconnect with the same worker ID. [Dispatcher selection and receive loop](../macfleet/compute/dispatch.py#L162)

### B10 · Medium · An immediate task exception is reported as a 300-second timeout

**Reproduce.** Register a harmless task whose body raises `TimeoutError("sensor read exceeded its own device deadline")`. In a joined local pool call `p.submit(task_fn, timeout=300)`. The caller immediately received a message claiming the pool task had timed out after 300 s and might still be running. The original exception was present as the cause. The task had already exited. Expected the device error through the task-failure path. The handler mistakes the callable's exception for the executor wait expiring. To fix this, check completion state and preserve the callable's exception. Verify the same distinction on the remote worker, where a similar handler exists but this case was not run. [Local task wait](../macfleet/sdk/pool.py#L1164) · [worker](../macfleet/compute/worker.py#L141)

### B11 · Medium · The dashboard reports 17 GB total on a 24 GB Mac

**Reproduce.** Start an isolated authenticated pool on the M5 Pro. Feed `pool.dashboard_snapshot()` into `Dashboard.update_nodes` and compare its Memory column with `sysctl -n hw.memsize` or System Information. The live row showed `9.5/17GB`. The machine and doctor report 24 GB. The reader estimates total RAM by summing a subset of `vm_stat` page categories. The total varies with the state of memory, and omits categories including compressor occupancy. To fix this, use the OS physical-memory size for the denominator. Define used/available memory separately and validate the health score on machines under memory pressure. [Memory accounting](../macfleet/monitoring/health.py#L132)

### B12 · Medium · Common terminal dimensions hide most of the health table

**Reproduce.** Run a live Dashboard with an authenticated local node and explicit sample training/network updates. Capture a PTY frame before leaving the context at 80 columns × 24 rows, then repeat at 140 × 40. At the smaller size, five of seven column headers were just ellipses, the sync header was truncated, the progress bar wrapped and speed and elapsed time disappeared. At 140 × 40 the headings and values were readable. Both runs exited zero. The layout always splits horizontally and divides the right pane into three sections. To fix this, introduce a compact layout prioritizing node, health, progress and warnings. Test rendered frames at common terminal sizes. [Dashboard layout](../macfleet/monitoring/dashboard.py#L309)

Selected rows from the real captures follow. The operator supplied the example epoch, step and speed. The node health and memory came from the machine.

```text
80×24    Node                ┃ … ┃ … ┃ … ┃ … ┃ S… ┃ … ┃    Epoch: 3/10
         audit-node-1d972799 │ … │ … │ … │ … │ -- │ … │    Step:  42
140×40   Node | Status | Thermal | Memory   | Throughput | Sync | Score
         audit-node-2f636c3b | OK | nominal | 9.5/17GB | -- | -- | 1.00
```

### B13 · Medium · The starter's fleet instructions omit the quorum

**Reproduce.** Generate the starter and read the directions above its pool construction. They tell the user to flip only `enable_pool_distributed`. With the default quorum of one, a rank may leave `Pool.join` as soon as it sees itself. If it reaches `train` before a peer joins, the SDK raises the explicit no-live-peers error. The README's separate instructions correctly specify quorum two. The conflicting instructions and control flow were verified in source. The two-machine timing race was not required for this documentation finding. To fix this, put the complete two-Mac configuration and the instruction to stop CLI agents directly in the generated file. [Starter instructions](../macfleet/cli/main.py#L1557) · [quorum default](../macfleet/sdk/pool.py#L435) · [distributed guard](../macfleet/sdk/pool.py#L727)

### B14 · Medium · A departing CLI agent can occupy a training slot for the full rendezvous timeout

**Reproduce.** Start both paired CLI agents. Stop the Pro's CLI agent and start its two-rank SDK training script while the Max's CLI agent still participates. Then stop the Max CLI and start the same SDK script there. The Pro snapshots the old Max identity as its expected rank and waits for that identity even after the registry reports removal. The Max's SDK identity can authenticate to the Pro but cannot complete model setup. The two exit-1 errors and elapsed times are in the physical results above.

The walkthrough does instruct operators to stop both agents first. The two operators overlapped shutdown and startup on this first attempt. The diagnostics pointed at different layers on the two machines. A job should reject a retiring or non-training participant promptly and name the missing training rank on both sides. To fix this, require a training-session readiness signal before fixing the roster, or abort rendezvous when a selected identity leaves. Include the expected and observed peer identities and a coordinated restart instruction. [Roster snapshot](../macfleet/sdk/pool.py#L805) · [wait for original inbound IDs](../macfleet/training/mesh.py#L221)

### B15 · Low · Multi-use enrollment prints the wrong use limit

**Reproduce.** In an isolated secure fleet, run `macfleet join --bootstrap --enroll-uses 2` on free ports. The pairing instructions should say that two Macs may enroll. The CLI instead printed `Code expires at 11:18:50 and can be used once.` The agent exited cleanly on interrupt. The server receives the requested count, but the rendering helper always prints the singular statement. To fix this, pass the configured use count to the formatter and test both singular and plural output. This was a message test. No enrollments were attempted under that code. [CLI argument](../macfleet/cli/main.py#L122) · [rendered instructions](../macfleet/security/enrollment.py#L397)

## Security review

**S1 · High · Suspected enrollment trust failure.** The enrollment server checks the client's code-bound proof, then returns credentials. The client accepts the response over a TLS connection without checking a trusted certificate or verifying a reciprocal proof that the responding server knows the enrollment code. The client-side trust gap is visible in source. Enrollment into an unintended fleet is a possible consequence under endpoint substitution. No such substitution, credential interception or unauthorized enrollment was performed or observed.

**Review procedure and expected result.** Trace a normal `macfleet pair --host … --code …` from the enrollment request to the credential save. Identify the client check authenticating the issuer and binding the returned fleet identity before persistence. The path validates response fields, while the proof in the request authenticates the enrolling client to the server. To fix this, authenticate the issuer and bind the returned identity and credentials to the fresh enrollment exchange before saving anything, or provision an out-of-band server certificate pin. Version the exchange and require both peers to support it. Treat this as a release gate pending protocol review. [Enrollment server](../macfleet/security/enrollment.py#L251) · [client](../macfleet/security/enrollment.py#L305) · [TLS context](../macfleet/security/auth.py#L849) · [credential save](../macfleet/cli/main.py#L1434)

The established data-plane handshake and first-time enrollment are different trust decisions. Successful authenticated allreduce exercises the established fleet path. Additional automated controls exercised TLS-required configuration, certificate lifecycle, token storage, bootstrap output redaction, worker capacity and authorization before execution. The worker-denial test uses a test transport. The audit did not exercise adversarial network traffic. The product's shared-token model gives an enrolled Mac fleet-level trust. Operator identity, individual-device revocation and cloud accounts were outside the tested product contract.

## Automated and same-host evidence

| Scenario on the M5 Pro | Observed result |
|---|---|
| Secure two-process Torch training over the same host's LAN address | Two independent processes, isolated homes and dynamic ports. 12 CPU steps per rank, no degradation, same final SHA-256 `c326010fd81dc2fb48494ebd7b6b3d6d2aa6b6e6e2a5683f1c3689bd23e7fcb5`. Epoch losses 0.7584 → 0.3307 and 0.6873 → 0.3480. Approximately 4.65 s/process including startup, 15.3 ms training interval, 0.22 ms mean sync. |
| Secure loopback task dispatch, interruption and retry | Registered task returned 49 in another PID. After three exact allreduces, an audit-owned peer exited and its survivor reported a connection reset in 1.7 ms. Fresh processes reused the ports and completed three exact reductions. Task/drop/restart harness wall time 1.19 s, measured RSS approximately 72 MiB, excluding fleet-wide aggregation. This exercises a new run after failure. |
| Selected regression suites | Broad selection reported 434 passed, one MLX skip and six environment failures at socket bind (EPERM) in dashboard tests. Those six used open listeners. A separate positive secure mesh/training selection reported two passed, one MLX skip in 4.76 s. Another 69 selected controls passed in 0.89 s. Selections can overlap. |
| MLX on Metal | A cached Python 3.12 MLX 0.32.4 wheel and matching Metal runtime enabled 26 engine/SDK/secure two-process checks to pass in 6.30 s. This covers local Metal and multi-process MLX on the Pro. |

The final inventory contained **1,199 distinct pytest tests** with both engines available. Across seven selections, **1,094 distinct tests passed, six were attempted and blocked, and 99 were unselected**. The passing counts by invocation were 434, 69, 2, 26, 113, 411 and 39. The three later invocations included production convergence and thermal-pause tests, concurrent workloads, scheduler and transport tests, authenticated three-process discovery, and valid-data property checks. The earlier MLX skips were covered once MLX was available. The remaining 99 include open-listener fixtures, adversarial tests and other unselected variants. No full-suite pass is claimed. Timings were taken on active laptops during the audit.

For the positive automated cases see [secure mesh tests](../tests/test_comm/test_mesh.py) and [separate-process Pool training tests](../tests/test_integration/test_pool_training_processes.py). The loopback and same-host timings in this section measure neither the Wi-Fi link nor the Max.

## Product improvements

| Priority | Operator problem and evidence | Recommended change and acceptance check |
|---|---|---|
| Medium | `status` scans mDNS for three seconds. Its `secure` field reflects the local token configuration. The mesh timeout tells users to use this command to check whether a peer is alive. Secure advertisements intentionally withhold hardware details, yet the walkthrough asks users to check both chips in this output. | Distinguish an advertisement, an authenticated heartbeat and a successful data-port connection. Preserve private hardware details on mDNS. Render unknowns explicitly and show authenticated hardware through a local agent view. A discovered peer with a stopped training process should be diagnosable from the CLI. [Status](../macfleet/cli/main.py#L364) · [advertised fields](../macfleet/pool/discovery.py#L448) · [mesh timeout](../macfleet/training/mesh.py#L213) |
| Medium | Users must stop both long-lived `join` agents and start identical SDK scripts on both machines for each job. Running a local demo or seeing two entries in status does not establish that a job crossed the network. | Give the CLI a paired two-Mac verification path with quorum, ranks, engine, synchronized steps and a cross-rank hash comparison. The repository verifier already covers most of the flow. [Walkthrough](../docs/getting-started/two-mac-testing.md) · [existing verifier](../tools/two_mac_verify.py) |
| Medium | `Pool.map` and `Pool.submit` are single-node conveniences. Remote work requires explicitly arranging a dispatcher, worker, transport and shared task registry. The guide gives an immediately runnable local example. | Add a minimal two-terminal secure example for a registered remote task. Print the executing hostname and demonstrate one failed task followed by a successful task. [Task guide](../docs/guides/tasks.md) |
| Low | The install docs enumerate M1 through M4. The hardware bandwidth estimator assigns every Pro the same value and every Max the same value. These estimates contribute to the compute score. | Document tested M5 configurations, label bandwidth as an estimate, and report unknown hardware values as unknown when system probes fail. Keep estimates distinct from measured transfer rates. [Requirements](../docs/getting-started/installation.md) · [profiling](../macfleet/pool/agent.py#L106) |

Equal shards are the training contract on this build. Each of two nodes receives half of a global batch regardless of its GPU count. The scheduler contains separate weighted-allocation primitives. Likewise, the supported compression options can quantize locally while still sending dense arrays. Neither feature warrants a claim of automatic load balancing or reduced network traffic for these runs. [Training contract](../docs/guides/train.md) · [compression accounting](../macfleet/training/data_parallel.py#L156)

## Coverage limits

| Scenario | Limit and reason |
|---|---|
| Sustained performance | Short deterministic workloads established execution and synchronization. No large-model training, crossover-size sweep, multi-hour thermal soak, energy measurement or isolated performance lab. The two Macs shared their ordinary network and other audit tasks ran on the Pro. |
| Connectivity variants | Physical evidence covers the two Macs on the available IPv4 LAN. No Thunderbolt cable, dedicated Ethernet, third physical rank, IPv6-only fleet, different subnet, sleep/wake or OS reboot was tested. Ring behavior for three or more ranks has automated coverage only. |
| Interruption variants | Controlled exits affected audit-owned processes. The audit did not disconnect the user's network, kill unrelated workloads, change their firewall, or remove power. A reset socket and an unresponsive machine can have different detection times. |
| Trust and permissions | Positive pairing, authenticated transport and isolated file modes were checked. The enrollment issuer concern is source-based. No live impersonation, penetration attempt, personal fleet-token rotation or permanent OS permission change. The multicast denial reproduction used process sandboxing. |
| Other software and UI | The public wheel was installed on the Pro. Physical jobs used identical copies of the checkout wheel. No mixed protocol/OS fleet or cross-engine training was tested. The live terminal dashboard was captured at two sizes on the Pro. No screen-reader or interactive macOS permission-dialog test. |
| Other compute paths | Physical remote compute used named low-level tasks. No claim of exactly-once execution, concurrent multi-job isolation on the LAN, physical compression speedups or automatic rank replacement. The reconnect defect was reproduced with secure loopback. |

**Closeout.** All 55 shipped Python source files in the physical wheel compared byte-for-byte equal to the working source at closeout. The temporary Pro transfer server and both machines' audit agents stopped. The Pro verified that ports 51409, 51411, 51412, 51421 and its enrollment port could be rebound. The Max verified no listeners on 51411, 51412 or 51421. Both operators removed their temporary fleet credentials, private transfer keys and pairing instructions. The Max retained its synthetic checkpoint and logs in its private scratch directory. The primary retained temporary venvs and measurement logs. The first install attempt may have left ignored packaging artifacts in the checkout. The operators used isolated homes for fleet tests. Existing fleet settings, system permissions and product source were preserved.

## Prioritized remediation roadmap

1. **Before trusting results or non-idempotent jobs:** fix unused-gradient participation (B1) and dispatcher reconnect handling (B9). Acceptance requires numerical parity with native Torch and a task result delivered across a worker reconnect, or rejection before the worker executes it. Test that retry advice distinguishes a completed task exception from a caller deadline (B10).
2. **Before distributing a secure fleet build:** resolve the enrollment issuer trust gap (S1), make the wire revision inspectable, and publish an unambiguous version/compatibility matrix (B8). Re-run clean enrollment and established TLS training on two separately installed Macs.
3. **Before describing the setup as a first-time path:** fix NumPy token IDs (B2), demo/starter instructions (B3, B4, B13), and readiness/permissions reporting (B5, B6). Verify a new user can start from a PyPI install, find both Macs, and prove both ranks ran without consulting the source tree.
4. **Make failure diagnosis a product feature:** distinguish discovered, authenticated and job-ready peers, and abort promptly when a selected training participant leaves (B14). Re-run the overlapping shutdown case, a clean retry and a mid-job interruption. Surface the same recovery action on both Macs.
5. **Polish monitoring and the CLI:** correct physical memory accounting, render a compact dashboard, and fix filesystem errors and enrollment copy (B7, B11, B12, B15). After these gates, repeat performance measurements with representative models and longer runs to determine when the second Mac pays for its synchronization cost.

## Post-fix verification

The fixes were applied to the same working tree on November 22. The new local package identifies itself as **2.3.0.dev0**. Existing installations require a coordinated upgrade for enrollment v2 and the revised Torch training layout. The original physical CPU, MPS and MLX hashes above belong to the pre-fix audit.

Three successive local wheels used version 2.3.0.dev0. Both Macs used `82bfc4866d335c128c625fc53315096c262e1f7204d717dbf17ae0b374c7b65e` for enrollment and passed all five local self-checks. The Pro's check ran 32 training steps and loss fell from 0.3954 to 0.0899. In the first workload batch, the Pro used `51ee04bf3efaca310bb456d2ac513c23ef82930229f4cacf2f398f045e29e5ec` and the Max used the preceding wheel. That revision changed the enrollment rejection message and CLI help. The final wheel, `369e0d3d83ce0a020d0c9878c6962dca25e89efdbcae7b6388102979f2cc1c5a`, also corrected the verifier's interface selection. Its 56 Python files matched the final working tree byte for byte.

On the Pro, **1,131 pytest cases passed in one 88.67 s run out of 1,236 collected**, with no failures or skips in the selection and two expected legacy API deprecation warnings. The other 105 cases include adversarial and open-listener fixtures. The five verifier-specific tests were rerun after the last interface correction. Ruff passed, mypy reported no issues across 56 source files, and `git diff --check` passed.

| Finding | Change | Post-fix evidence |
|---|---|---|
| B1 · High | Torch ranks exchange per-parameter participation before applying averaged gradients. A parameter unused everywhere keeps its absent gradient. A parameter used on just one rank still updates on both. The training signature distinguishes the new layout. | Numerical regressions compared clipped SGD and AdamW against native Torch, including weight decay, conditional branches and a globally unused parameter. |
| B9 · High | The dispatcher follows the current worker connection. A disconnected in-flight task finishes with an explicit unknown-outcome error. New submissions listen on the replacement connection. The dispatcher never automatically replays a task. | Secure reconnect regression checked replacement, receipt of a subsequent result and exactly one submission per task. The documented two-process example on the Pro returned the worker's input error and then returned 81 with its hostname. |
| S1 · High concern | The enrolling client verifies a reciprocal issuer proof before accepting credentials. The proof binds the enrollment code, nonce, TLS certificate and returned credentials, including fleet identity. Enrollment v1 is rejected. | Positive issuer/client and isolated negative verification cases passed, including old-version and invalid response checks. No live endpoint substitution was performed. |
| B2, B10 | NumPy integer inputs retain their dtype. The worker and local Pool distinguish a callable's own timeout exception from a caller's deadline. | On the Pro, integer inputs trained through a Torch embedding for two steps. A registered callable reported `TimeoutError: sensor read deadline`. A separate actual deadline warned that work could still be running. |
| B3, B4, B13 | Demo and starter instructions specify a two-Mac quorum and how to hand off from CLI agents to identical SDK scripts. The onboarding path installs the engine used by the example. | The generated starter ran 160 MPS steps on the Pro. Loss fell from 0.575 to 0.0435. The demo, generated script and install text have focused checks. |
| B5, B6, B7 | Doctor checks the selected engine and, when requested, fleet credentials. Status labels discovery-only results and returns a structured permission error. Quickstart reports an inaccessible destination concisely. | Pro operator checks covered Torch with optional MLX absent, requested MLX absent, required token missing and a denied discovery socket. The status failure returned nonzero and parseable JSON. Quickstart on `missing/demo.py` exited 1 with a single missing-directory error. |
| B8, B15 | Package metadata and security guides describe the development version, handshake v3 and enrollment v2. `macfleet info` displays the wire versions. Pairing copy reflects the configured use count. | CLI and enrollment tests passed. A real Pro bootstrap with `--enroll-uses 2` displayed “can be used 2 times.” |
| B11, B12 | The dashboard uses OS physical memory and a compact layout for smaller terminals. | Pro total matched the OS at 24.0 GiB. At 80×24 the operator could read status, thermal state, memory, step, loss, speed, elapsed time and warnings. A 140×40 capture exercised the wide layout. Sample training metrics were supplied to the dashboard for this layout check. |
| B14 | A training rendezvous watches the selected identities for departure. On removal it cancels pending connects, names the missing and current participants, and tells operators how to restart. A closed connection during model setup also produces a peer-specific diagnostic. | Outbound and inbound loopback regressions both aborted when the selected identity left and successfully rebound the data port. |

A separate installed-wheel check used two processes on the Pro. Clipped SGD produced weights `[0.9452786446, 0.9676393270, 1.0]` on both ranks. AdamW produced `[0.8899999857, 0.8899999857, 1.0]`. Each matched its native Torch control within floating-point tolerance. The unused parameter retained an absent gradient and acquired no optimizer state. The same pair of processes returned both the original and post-reconnect task results. A separate real bootstrap and pair on loopback saved the same token in each isolated home with mode 0600.

### Two-Mac retest

The Max enrolled in the Pro's new temporary fleet using the installed CLI. The Pro discovered two 2.3.0.dev0 advertisements. When its SDK authenticated the Max, it obtained the expected 40 GPU cores and 48 GiB. For B14, the Pro captured the still-running Max CLI identity as a training participant and the Max operator then stopped that agent. The Pro exited with status 1, identified the departed node and told both operators how to restart. The **entire run took 27.28 s**, including time spent waiting for the operator's stop signal. The previous trial exhausted a 90 s rendezvous timeout. The elapsed time from departure alone was not separately measured.

| First workload batch | Outcome on the Pro and Max |
|---|---|
| Torch CPU, 24 synchronized steps | Both ranks reported SHA-256 `f8e78ec7e28952e986cc37ebaefb09eb3f028a1d5cf95d75649329197b86aa99`, with zero unsynced steps. Loss fell from 0.99586 to 0.04185 on the Pro and 0.93522 to 0.11030 on the Max. |
| B1 through public `Pool.train`, two AdamW steps | Both ranks matched native Torch and each other. SHA-256 `8695839ad839967d489c9c67a26bfd6f2e66d2ba92163e2ca431948b7c2af670`. A branch used on only one rank updated on both, while the globally unused parameter stayed at exactly 2.0 with no gradient or optimizer state. |
| B1 with clipping | For both SGD and AdamW, each Mac independently compared its three parameters with a native full-batch reference. Both checks passed on both Macs. |
| B9 and B10, registered worker | The same Pro dispatcher obtained `first`, then `new-link` from the Max across a worker disconnect/reconnect. The Max's callable then raised its own device deadline and the Pro received that task failure. |
| Installed verifier, first attempt | Both Macs exited 1 with three of four checks passing. Training completed 24 steps and both reported `8da44e7414ab2165b6852751b46def46c9437075194a37c4d1a220c8fb95f119`. The topology diagnostic picked the first interface returned by macOS. On this network that interface lacked a dialable address. The fix uses the local address already in the registry, just as mesh setup does. A regression exercises a non-dialable first interface. |

Both Macs then installed the identical final wheel. Each successful job below completed **24 steps on each rank**, world size two, with zero unsynced steps and no degradation. The Max was rank 0 and the Pro was rank 1. Full hashes were collected independently. Process wall times include startup, the wait for the other rank and teardown. These short, staggered trials measure functional recovery. They do not establish a speedup.

| Final-wheel scenario | Pro / Max process wall | Result |
|---|---:|---|
| Installed `macfleet verify --fleet-id NAME` | 24.94 / 6.02 s | **4/4 checks passed on each Mac**, exit 0. The Pro selected the Max's LAN address. Both hashes were `8da44e7414ab2165b6852751b46def46c9437075194a37c4d1a220c8fb95f119`. |
| Torch MPS | 5.33 / 13.19 s | Matching `1d1ea6d949b2c8afc6d59e6f7007f1c500547422bef6e88c0be642f9a5e6c9de`. |
| MLX on Metal | 3.98 / 13.53 s | Matching `f23ab6d05cc78a31f9f77ffc6306ef68c391f6dc94f3118fef27f1f72d2a880c`. |
| Max exits at its fourth CPU forward | 5.10 / 15.00 s | The isolated Max process exited 17 as instructed. The Pro exited 1 with `SyncDegradedError` and an incomplete-read cause. It returned no training success. |
| Fresh Torch CPU run after the exit | 4.44 / 10.70 s | Matching `f8e78ec7e28952e986cc37ebaefb09eb3f028a1d5cf95d75649329197b86aa99`, identical to the clean CPU baseline. The same scratch credentials and ports worked again. |

The first post-fix CPU launch exhausted its 90 s quorum deadline while the Max awaited approval to run workloads. Once both operators launched the script, it succeeded. Several subsequent runs logged an initial connection refusal before the other side became ready and completed within the retry window.

### Operator guidance and remaining limits

The installed CLI now includes `macfleet verify --self-check` and `macfleet verify --fleet-id NAME`. Each distributed rank prints its rank, step count, synchronization state and full final parameter hash. Compare both result lines to confirm agreement. The new [two-terminal remote task example](../docs/guides/tasks.md) exercises a registered worker. Hardware reporting identifies the M5 family and leaves unmeasured M5 memory bandwidth unknown. `status` still reads advertisements. The paired verifier provides authenticated registry and data-path checks. A read-only view into an already-running agent would be a separate product improvement.

Enrollment authenticates possession of the short-lived code. A lost delivery or failed local save can spend a single-use code. Issue a fresh one after correcting the problem. The independent source review verified the issuer check precedes persistence, and the client now displays a local error for an unauthenticated rejection. Upgrade secure fleet members together. This local development build has not been published. The earlier limits on long-duration testing, alternate physical networks and adversarial traffic still apply. [Pairing guidance](../docs/getting-started/pairing.md) · [Two-Mac verification](../docs/getting-started/two-mac-testing.md) · [Optimizer regression](../tests/test_training/test_loop.py) · [Reconnect regressions](../tests/test_compute/test_dispatch.py)

**Cleanup.** Both operators stopped their validation processes and removed the temporary fleet tokens, private transfer keys, pairing instructions and secret-bearing logs. The Pro confirmed all dedicated ports could be rebound, including the transfer, enrollment and worker ports. The Max confirmed no remaining validation processes or listeners on its checked ports. Sanitized results and isolated test environments remain in scratch.

## Follow-up on the remaining tests and sustained training

### Regression coverage

Of the previous 105 unexecuted tests, **89 passed and 16 remain unexecuted**. Comparing the JUnit results by exact pytest node ID gives **1,220 distinct passes out of 1,236 collected** across three runs. The two new selections had zero failures, errors, skips or warnings.

| Execution on the M5 Pro | Passed | Duration | Conditions |
|---|---:|---:|---|
| Original post-fix selection | 1,131 | 88.67 s | Both engines available. Two expected legacy API deprecation warnings. |
| Additional functional and defensive cases | 72 | 11.89 s | Existing tests ran unmodified. Includes serialization and protocol validation, authentication, enrollment, and peer dropout. |
| Additional local networking cases | 17 | 9.21 s | A scratch harness confined TCP sockets to loopback and suppressed multicast registration and browsing. Six dashboard checks, six SDK lifecycle/quorum checks and five two-rank Torch/MLX training cases passed. This run exercised the actual backends and sockets. Physical LAN discovery is covered separately. |

The 16 exclusions comprise two live malformed compute-frame cases, six adversarial failure-injection cases, one repeated credential-attempt case, three live pre-authentication handshake probes, three altered or oversized signed hardware-message cases, and one historical response fixture that disables TLS. No pass or failure is assigned to these cases.

<details>
<summary>Accounting for all 105 previously unexecuted tests, by file under tests/</summary>

| Test file | Passed unmodified | Passed with loopback containment | Unexecuted |
|---|---:|---:|---:|
| `test_comm/test_mesh.py` | 1 | — | — |
| `test_comm/test_transport.py` | 2 | — | — |
| `test_compute/test_dispatch.py` | — | — | 2 |
| `test_failure_injection/test_peer_dropout.py` | 1 | — | — |
| `test_failure_injection/test_security_attacks.py` | 4 | — | 6 |
| `test_fuzz/test_fuzz_array_serialization.py` | 3 | — | — |
| `test_fuzz/test_fuzz_hw_exchange.py` | 7 | — | — |
| `test_fuzz/test_fuzz_task_spec.py` | 7 | — | — |
| `test_fuzz/test_fuzz_wire_message.py` | 5 | — | — |
| `test_monitoring/test_agent_adapter.py` | — | 6 | — |
| `test_sdk/test_distributed_train.py` | — | 5 | — |
| `test_sdk/test_pool_distributed.py` | — | 5 | — |
| `test_sdk/test_pool_task_dispatch.py` | — | 1 | — |
| `test_security/test_auth.py` | 5 | — | — |
| `test_security/test_authenticated_heartbeat.py` | 4 | — | — |
| `test_security/test_authenticated_transport.py` | 5 | — | 1 |
| `test_security/test_enrollment.py` | 3 | — | — |
| `test_security/test_handshake_v3.py` | — | — | 2 |
| `test_security/test_heartbeat_rate_limit.py` | 6 | — | 1 |
| `test_security/test_heartbeat_with_hw.py` | 6 | — | 3 |
| `test_security/test_hw_exchange.py` | 11 | — | 1 |
| `test_training/test_data_parallel.py` | 2 | — | — |
| **Total** | **72** | **17** | **16** |

</details>

### Sustained physical workload

The M5 Pro and M5 Max used the same audited 2.3.0.dev0 wheel (`369e0d3d83ce0a020d0c9878c6962dca25e89efdbcae7b6388102979f2cc1c5a`), a new temporary authenticated fleet, and the public Torch `Pool.train` path on the existing Wi-Fi LAN. The workload exercised synchronization over many consecutive updates of a small model. Each process used one CPU training thread. The MLX and MPS results above are separate short physical runs.

| Reproduction setting | Value |
|---|---|
| Data | 32,768 rows × 8 float32 features from `np.random.default_rng(42).normal`. Int64 label is 1 where the first two features sum to a positive value. |
| Model | Linear widths 8 → 64 → 64 → 2, ReLU after each hidden layer. Torch initialization seeds 1 on the Pro and 2 on the Max before the initial parameter broadcast. |
| Optimizer and loss | Adam, learning rate 0.001, cross-entropy, no compression. |
| Allocation | Quorum two. Global batch 64, 32 examples per rank per update and 512 updates per epoch. Fixed 65 epochs, 33,280 optimizer updates per rank, **2,129,920 global sample presentations**. |
| Instrumentation | The operator scripts sampled process RSS and `pmset -g therm` every 30 s on a separate thread, emitted progress every 4,096 forwards, and collected per-rank `Pool.train` results. There were no artificial delays in training. Workload script SHA-256 `bc4ef3dfd495b0dfd8a4af93c0d55cc9bd02aa462dc89c16b9b6e9473d532a61`. |

A separate three-epoch calibration completed 1,536 updates per rank in 27.52 s on the Pro and 27.53 s on the Max. Both returned `9bdb47307798fb9c5ed6cba2116cdb6767d39b513534dd45181b4a9041bce57e` with zero unsynced or validation-fallback steps. This measured rate set the fixed epoch count for a target near ten minutes. The sustained job initialized a fresh model.

**Result.** Both physical ranks completed 65 epochs and returned the same final parameter SHA-256, `cb13b9aebc33f2a4c144b3f1c6c3102cf4de0652ef9a177d1a5ce10037090898`. Wall time inside the training loop was **7 min 22 s**, shorter than the calibration forecast. The fixed workload reached its final step. Using the slower rank's SDK interval, fleet throughput was approximately **4,814 global examples/s** or **75.2 synchronized updates/s**.

| Observed on each physical Mac | M5 Pro, rank 1 | M5 Max, rank 0 |
|---|---:|---:|
| Completed updates | 33,280 | 33,280 |
| SDK training time | 442.39 s | 442.40 s |
| Mean sync / update | 12.43 ms | 12.28 ms |
| Process wall time, including startup and teardown | 456.65 s | 446.64 s |
| Mean training loss on that rank, epochs 1 → 33 → 65 | 0.101107 → 0.004743 → 0.004433 | 0.100311 → 0.004566 → 0.004273 |
| Unsynced / validation-fallback steps | 0 / 0 | 0 / 0 |
| Degraded training / last sync error | False / none | False / none |
| Sampled process RSS, min–max | 188.6–327.5 MiB | 205.5–324.8 MiB |
| Thermal readings | 16 total, 15 after training started, no recorded warning | 15 total, 14 after training started, no recorded warning |

About 93% of the Pro's SDK training interval was spent synchronizing. That includes waiting for the other rank. Isolating Wi-Fi latency or measuring speedup would require separate benchmarks. The Pro built two small diagnostic wheels during the run. The measurements reflect ordinary laptop activity. The thermal tool reports warning status. Multi-hour performance and GPU thermal endurance remain untested.

### New findings in the startup log

| Verified issue | Reproduction and actual evidence | Expected behavior and implemented fix |
|---|---|---|
| B16 · Low · Secure discovery reports a 40-core GPU as having zero cores | Start the paired Pro SDK job first, then the Max job with Bonjour enabled. The Pro printed `Discovered sustained-max.local (unknown, 0 GPU cores)`. A secure advertisement deliberately omits hardware. The rendering code formatted the internal zero placeholder as a measured count. | The discovery message identifies that secure advertisements omit hardware. In open discovery it prints a known count or labels it unknown. The change is confined to rendering. |
| B17 · Low · An early connection refusal gives the wrong recovery action for SDK training | Start the Pro's SDK job with its manual peer address while the Max's job is still stopped. The Pro printed `Make sure the peer is running 'macfleet join' and is reachable`. Starting the same SDK script on the Max allowed the job to proceed. Starting a separate CLI agent on those ports can instead recreate B14. | The connection error preserves the failing address and cause. It now tells operators to check the heartbeat endpoint and start the same training script on every Mac for SDK training. Both the CLI and SDK use the corrected agent diagnostic. |

Both were observed during calibration and sustained startup. Four new regression cases cover secure discovery, open discovery with a known count, open discovery with an unknown count, and the connection-refusal guidance. All 15 tests in the affected module passed after the change. This adds four distinct passes beyond the original inventory, bringing cumulative coverage to **1,224 of 1,240**, with the same 16 exclusions. Ruff, mypy and `git diff --check` passed. Only `macfleet/pool/agent.py` changed among the 56 shipped Python files between the sustained-training wheel and the diagnostic-fix wheel. [Agent diagnostics](../macfleet/pool/agent.py) · [Regression tests](../tests/test_pool/test_port_split.py)

**Post-fix physical check.** Both Macs installed wheel SHA-256 `9ead84523f0d601b39c75884ae6eceb8def4858a73251512638b5452eb17037b`. The Pro started first, hit the expected early refusal and printed `Check the peer's heartbeat address and port. For SDK training, start the same training script on every Mac.` When the Max started, the Pro logged `Discovered sustained-max.local (hardware details omitted from secure discovery)`. Both ranks completed 512 synchronized updates in 6.61 s, with no degradation or fallbacks and matching SHA-256 `9dcb4f79b485616abc9462717b948be71ba70926bcc6287c3b5f2848aaa24fd2`. All 89 newly executed cases also passed again on the corrected source, 72 in 12.07 s and 17 in 9.56 s.

**Closeout.** Both machines retired this run's temporary fleet credentials, transfer private keys and pairing instructions. Both operators verified that their validation processes and listeners had stopped. Sanitized logs were retained in scratch. The corrected wheel remains a local development build.

## GPU workload and review of the final 16 tests

The test review started with the exact 16 pytest IDs excluded above. The physical GPU workload used the local diagnostic-fix wheel (`9ead84523f0d601b39c75884ae6eceb8def4858a73251512638b5452eb17037b`) on the M5 Pro and M5 Max.

### Disposition of the 16 tests

**Seven passed with in-memory I/O; nine remain unexecuted.** The seven original test bodies and assertions completed in 5.61 s with zero failures, errors or skips. The scratch harness rejected creation of IPv4 and IPv6 sockets, used queues for the compute transports, and paired memory streams for the other fixtures. The run covered the production handlers and validators indicated below. It did not exercise their original TCP/TLS connections. The seven IDs were distinct from all prior passes: **1,227 of the original 1,236**, or **1,231 of 1,240** including the four startup-diagnostic cases added later.

In both tables the file is relative to `tests/`, and the second column identifies the original pytest class and method.

| File | Case that passed with adapted I/O | What the run established |
|---|---|---|
| `test_compute/test_dispatch.py` | `TestDispatcherWorkerIntegration::test_dispatcher_ignores_malformed_result_frame` | The dispatcher rejected the invalid result record, kept its listener running, and delivered the next computation. |
| `test_compute/test_dispatch.py` | `TestDispatcherWorkerIntegration::test_worker_ignores_malformed_task_frame` | The worker rejected the invalid task record and processed the subsequent valid task. |
| `test_failure_injection/test_security_attacks.py` | `TestBruteForceBan::test_five_wrong_tokens_triggers_ban` | The original test itself seeds four prior failures. One more failure triggered the real limiter, and a subsequent valid request was denied. Five independent network attempts were never part of this test. |
| `test_failure_injection/test_security_attacks.py` | `TestOversizePayloadDoS::test_oversize_hw_in_aping_v2_rejected` | One fixed field just above the schema limit reached the production size check. The handler rejected it. Resource use under load was outside this check. |
| `test_failure_injection/test_security_attacks.py` | `TestSlowloris::test_handler_drops_silent_connection_within_timeout` | One idle memory reader reached the real timeout. The handler recorded a failure and closed the stream. |
| `test_security/test_heartbeat_with_hw.py` | `TestAPingV2ServerHandler::test_v2_oversize_hw_rejected` | Production parsing rejected the bounded over-limit field and the handler sent no reply to the memory stream. |
| `test_security/test_heartbeat_with_hw.py` | `TestAddManualPeerCapabilityExchange::test_manual_peer_rejects_legacy_unbound_apong` | The real verifier rejected a freshly generated legacy response and the registry stayed empty. The original fixture's cleartext setting was confined to an in-memory peer. |

The following nine original scenarios were withheld from active execution. The blockers concern what each test must actually observe. Replacing the endpoint, authentication exchange or response with a stub would change the claim. Related controls in the last column have separate passing test evidence and are excluded from the count of these nine.

| File | Unexecuted case | Specific blocker and available evidence |
|---|---|---|
| `test_failure_injection/test_security_attacks.py` | `TestOversizePayloadDoS::test_oversize_wire_message_rejected` | Exercises a raw transport resource-abuse scenario. A parser-only fixture omits the listener and handshake. The original also has no assertion that parsing was reached or that memory stayed bounded. The wire unpacker's maximum-size check passed separately. |
| `test_failure_injection/test_security_attacks.py` | `TestReplayAttack::test_captured_aping_replayed_with_changed_nonce_rejected` | The case couples an active replay to both the secure handler's response and its failure accounting. Independent nonce-binding and limiter controls passed. Neither observes that combined sequence. |
| `test_failure_injection/test_security_attacks.py` | `TestReplayAttack::test_captured_hw_payload_with_different_nonce_rejected` | Requires the endpoint's behavior during replay of signed hardware metadata. A verifier-only check confirmed nonce binding but cannot establish the endpoint response. |
| `test_security/test_authenticated_transport.py` | `TestRateLimiterIntegration::test_banned_ip_rejected` | The original repeatedly attempts authentication through the production transport. Preloading the counter removes the repeated-connection behavior. Threshold and already-banned admission checks passed separately. |
| `test_security/test_handshake_v3.py` | `TestOracleClosed::test_server_silent_to_bad_hello_proof` | Requires measuring externally observable silence on pre-authentication TLS and checking accounting during a historical disclosure probe. A simulated close would assume the expected result. Ordinary credential mismatch and channel-binding checks passed. |
| `test_security/test_handshake_v3.py` | `TestPeerIdLengthCap::test_oversized_peer_id_is_rejected_silently` | Combines an out-of-range identity in a pre-authentication handshake with wire silence, registry state and rate limiting. A length-validator check omits those effects. The passing wrong-peer identity test does not cover this length limit. |
| `test_security/test_heartbeat_rate_limit.py` | `TestHeartbeatRateLimiter::test_tampered_hw_v2_records_failure` | Requires a counter transition specifically from active alteration of signed capabilities. Signature validation and wrong-key failure accounting each passed separately. Their composition for this event remains unverified. |
| `test_security/test_heartbeat_with_hw.py` | `TestAPingV2ServerHandler::test_v2_tampered_hw_no_response` | Requires observing the handler's wire response to altered signed peer metadata. The real signature helper rejected modified metadata in isolation. That result alone cannot establish wire silence. |
| `test_security/test_hw_exchange.py` | `TestTransportV2Handshake::test_legacy_v21_client_rejected_by_v3_server` | Probes disclosure during a historical pre-authentication exchange. Valid legacy heartbeat handling, ordinary mismatched credentials and downgrade rejection have passing controls; none measures what this specific handshake reveals. |

### GPU workload design

| Setting | Value |
|---|---|
| Model and data | 330,756-parameter MLP, linear widths 128 → 512 → 512 → 4, with ReLU after each hidden layer. A seeded NumPy generator (42) produced 32,768 float32 feature vectors and a 128 × 4 teacher matrix. The highest teacher score assigned each example's class. |
| Distributed training | Stock `Pool.train` with the PyTorch MPS backend on each Mac, cross-entropy, Adam at 0.001, global batch 128, quorum two, and no compression. Each rank received 64 examples per update. Each epoch contained 256 updates. Ranks used different initial model seeds before broadcast. |
| GPU placement checks | Each operator ran a native MPS matrix multiplication, synchronized the device and checked its result. Inside training, hooks checked all model parameters, the first forward's inputs and output, and the loss function's logits and targets. The process set `PYTORCH_ENABLE_MPS_FALLBACK=0` before importing Torch. |
| Measurement | Training time is end-to-end within the SDK loop, including distributed synchronization. The monitor sampled process RSS, Torch MPS tensor and driver allocations, and `pmset -g therm` every 30 s. The test did not measure GPU occupancy or temperature. Harness SHA-256 `8e6cb7e60a4c2c93e9d31586d24ae30568054afc9fa1e38f99b3bf188b2095b9`. |

**Calibration.** The two-epoch calibration completed 512 updates in 66.89 s per Mac. The Pro and Max both reported `mps:0` for model weights, inputs, output, logits and targets. Both returned `d564407835cb582fa03446e93cf70a0441becf61751f2b38d5031b73feb77f1e`, with no degraded, unsynced or validation-fallback steps. Mean sync time was 125.35 ms on the Pro and 123.57 ms on the Max. This calibration set the sustained job at 32 epochs, 8,192 updates per rank and 1,048,576 global sample presentations.

A separate Pro-only MPS reference completed 512 stock local training steps in 0.95 s of SDK time on the same synthetic task, with a 1.51 s wall interval around the pool and training call. Local loss fell from 0.324285 to 0.102537. The local and distributed paths use different samplers. This single trial documents how inexpensive the small model is on one GPU; it is not a general benchmark of distributed training.

### Sustained GPU result

**Both GPUs completed the job in 18 min 10 s.** Both operators independently recorded final parameter SHA-256 `a24d8818c7d61ef75df95c734699cbd4c3d7b990784e1ae651be4234b5cabefc`. Each rank completed 32 epochs and 8,192 optimizer updates, with matching final weights and zero unsynced or validation-fallback steps. Each rank verified the model parameters, forward input, output, logits and targets on `mps:0`. No product change was needed to finish the run.

| Measured result | M5 Pro | M5 Max |
|---|---:|---:|
| Rank / world size | 1 / 2 | 0 / 2 |
| SDK training interval | 1,089.61 s | 1,089.62 s |
| Process wall time including startup and shutdown | 1,104.99 s | 1,094.94 s |
| Mean sync per update | 127.99 ms | 128.48 ms |
| Mean loss on local shard, epoch 1 → 17 → 32 | 0.327442 → 0.025963 → 0.00002501 | 0.320051 → 0.020027 → 0.00002386 |
| Degraded / unsynced updates / validation fallbacks | False / 0 / 0 | False / 0 / 0 |
| Last sync error | None | None |
| Sampled live process RSS during training | 280.9–529.9 MiB | 88.4–520.9 MiB |
| Sampled live MPS tensor allocation | 5,293,312–5,295,360 bytes | 5,293,312 bytes |
| Sampled MPS driver allocation | 17,514,496–17,776,640 bytes | 17,514,496–18,563,072 bytes |
| Thermal/performance warnings after first forward | None in 36 samples | None in 36 samples |

Fleet throughput was **962 global examples/s**, or 7.52 synchronized updates/s, based on the slower SDK interval. The measured sync intervals account for approximately 96% of each rank's training time, including waiting for its peer. The local calibration and sync timings identify distributed synchronization as the bottleneck for this particular small model on this Wi-Fi link. No packet-level latency or GPU utilization was measured. Training loss was finite and fell on both shards, with increases in some intermediate epochs. Held-out accuracy was outside this run.

### B18 · Medium · A repeated discovery advertisement clears an authenticated peer's state

**Physical observation.** After the 5,120-step progress message, the Pro printed a second discovery of `gpu-max.local`, elected itself coordinator and reported `world_size=1`. The established training connection continued to make progress. The discovery message prints a hostname without the full node ID, so the cause of that particular physical event is unconfirmed.

**Verified reproduction.** In an isolated registry with two nodes, introduce a normal secure discovery record, then process a valid authenticated heartbeat for the peer. Record its hardware, coordinator and measured throughput. Deliver the identical discovery advertisement again, with the same full node ID, hostname, IP address and ports. Observe the registry before and after the next valid heartbeat. This check used real heartbeat authentication with in-memory I/O.

| State in the controlled check | After authenticated heartbeat | After identical advertisement | After next authenticated heartbeat |
|---|---|---|---|
| Peer status / live world size | Alive / 2 | Suspected / 1 | Alive / 2 |
| Peer GPU cores / RAM | 32 / 64 GB | 0 / 0 GB | 32 / 64 GB |
| Elected coordinator | Peer | Local | Peer |
| Recorded throughput, samples/s | 123.5 | 0 | 0 |

**Expected behavior and recommended fix.** A repeat announcement of an unchanged endpoint should preserve its authenticated membership, hardware and measurements. The agent handles service additions and updates through the same callback, and the registry replaces the entire record. Handle an unchanged secure advertisement idempotently. Authenticate endpoint or identity changes before granting them existing peer trust. Add a regression covering both an unchanged advertisement and a genuinely replaced endpoint. The fix and its verification follow below.

**Closeout.** Both operators retired the temporary GPU fleet credentials, transfer private keys and pairing instructions and confirmed the dedicated listeners had stopped. Sanitized test and performance records were retained in scratch. The product source was unchanged during this follow-up.

## B18 fix and verification on the M5 Pro

**Status: fixed and verified locally, November 22, 2026.** The agent previously replaced both the registry record and the heartbeat tracker every time discovery reported a service. Secure discovery now keeps those objects when the same node ID announces the same endpoint. This also preserves the coordinator election, the signed hardware profile, the throughput measurement and any missed heartbeat rounds. If a heartbeat has corrected the advertised data port, repeating the original announcement preserves the verified port. A changed IP address, heartbeat port, data port or a departure followed by rejoin starts a new membership that must authenticate. A reply already in flight for the old membership cannot validate its replacement. [Discovery handling](../macfleet/pool/agent.py) · [Registry update](../macfleet/pool/registry.py) · [Regression tests](../tests/test_pool/test_rediscovery.py)

**Manual reproduction and retest.** Terra ran two independent MacFleet processes on the M5 Pro and established real authenticated TCP/TLS heartbeats over loopback. A controlled service record entered through `PoolServiceListener.update_service`. On the previously audited wheel, an identical announcement dropped the fleet from two live members to one and erased the peer’s hardware and recorded throughput. Terra repeated the sequence on the changes integrated with main at `ac2d1ca`. The peer reported the host’s actual Apple M5 Pro hardware. The operator seeded throughput at 321 samples/s so a reset would be visible.

| Observation on the fixed build | Live members | Peer status | GPU cores / RAM | Recorded samples/s |
|---|---:|---|---|---:|
| Peer authenticated | 2 | Alive | 16 / 24 GB | 321 |
| Identical advertisement, before next heartbeat | 2 | Alive | 16 / 24 GB | 321 |
| Only the data port changed | 1 | Suspected | Unknown | 0 |
| Heartbeat authenticated the updated endpoint | 2 | Alive | 16 / 24 GB | 0 |
| Only the heartbeat port changed | 1 | Suspected | Unknown | 0 |
| Heartbeat authenticated the moved listener | 2 | Alive | 16 / 24 GB | 0 |

The identical announcement also preserved the elected coordinator, the peer’s compute score of 184 and the heartbeat’s exact last-seen timestamp. A changed endpoint started with no throughput history. The zero after authentication is expected until a new measurement arrives. With the peer’s outbound gossip paused, an announcement pointing to an unused port stayed suspected for the 2.5-second observation. Restoring the real port allowed it to authenticate and rejoin. With gossip active, the peer could independently reauthenticate its original reachable endpoint.

**Regression coverage.** Run `python -m pytest -q tests/test_pool/test_rediscovery.py` from the integrated checkout. It covers duplicate announcements, changes to each address or port, signed port corrections, late replies during an endpoint change or rejoin, reverse-probe cancellation and open-fleet behavior. All 16 new rediscovery cases passed. The focused rediscovery and related pool/heartbeat selection passed 63 tests. The broader integrated selection on the Pro passed 1,117 distinct cases in 49.40 seconds, including all 16 new cases, with two expected legacy SDK deprecation warnings. Ruff and mypy passed.

**Verification scope.** The manual run used controlled discovery callbacks on one physical Mac. The IP-change regression passed using a registry fixture. This host had no second loopback address bound, so the manual run exercised port changes only. The earlier two-Mac training results above remain measurements of the previous wheel. An over-the-air Bonjour repeat and a two-Mac training rerun on this revision remain untested.
