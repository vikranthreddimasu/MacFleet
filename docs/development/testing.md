# Testing distributed training

Install `.[dev,all]` and run `make test`. The integration tests use real localhost
sockets, authenticated discovery, TLS, and two or three independent processes.
MLX cases use the GPU; they require Apple Silicon and Metal access.

## Metal cache isolation

Each simulated Mac must have its own Darwin cache directory, set before its
Python process starts. Separate physical Macs already have separate caches.
The test harness uses `DIRHELPER_USER_DIR_SUFFIX` and removes only directories
it created after the workers have exited. This is the same macOS mechanism used
for [Chromium's isolated GPU process directories](https://chromium.googlesource.com/chromium/src/+/c2706c3317723e987f85322a25ed475bdfe3a719%5E%21/).

Without isolation, a three-process MLX test intermittently exceeded its 60-second
result deadline. A [captured CI failure](https://github.com/vikranthreddimasu/MacFleet/actions/runs/39833323935/job/120015431042)
on macOS 15.7.9, Python 3.13.15, and MLX 0.32.4 showed one worker blocked here:

```text
MLXEngine.forward -> mx.eval -> Gather::eval_gpu
  -> Metal library compilation -> MTLCompilerFSCache::openSync
  -> libCoreFSCache open_data_file -> flock
```

The other two workers were waiting for its gradients. This was contention on
Metal's shared on-disk compiler cache, not slow model computation. The capture
does not identify which process owned the conflicting lock. An asyncio timeout
cannot interrupt the synchronous native call, so timeout diagnostics collect
both Python/asyncio stacks and native `sample` output before terminating workers.

A controlled native probe confirmed the mechanism: holding test-owned Metal
cache locks blocked a worker in the same namespace; another namespace completed
identical GPU work. Releasing the locks let the blocked worker complete.

The regression checks native cache paths and keeps GPU execution enabled.
Do not replace this coverage with retries, CPU execution, or a larger deadline.

## Consecutive training sessions

Successful training retires every mesh listener before any rank can start its
next session. Existing streams stay open for a completion barrier, then close.
Without this ordering, a fast rank can authenticate to a slow rank's previous
listener and lose that connection when the old mesh closes.

`test_repeated_training.py` forces this ordering with events for both engines,
then checks matching weights after two calls. Call `Mesh.finish()` after a
successful low-level training session; use `Mesh.close()` after a failure or
cancellation, when another collective could be unsafe. `Pool.train()` handles
both paths automatically.
