# Registered tasks (`@macfleet.task`)

Callables must be registered with `@macfleet.task` before dispatch — names are sent on the wire, not pickles.


For general-purpose compute across the fleet, register callables with
`@macfleet.task`. This is the secure alternative to cloudpickle-over-
the-wire: the wire carries only the task NAME, and workers look that
name up in a local registry.

## Basic usage

```python
import macfleet

@macfleet.task
def resize(image_path: str, target_w: int) -> dict:
    from PIL import Image
    img = Image.open(image_path)
    img.thumbnail((target_w, target_w))
    return {"width": img.width, "height": img.height}

with macfleet.Pool() as pool:
    result = pool.submit(resize, "/tmp/a.jpg", target_w=512)
    results = pool.map(resize, ["a.jpg", "b.jpg", "c.jpg"])
```

`pool.submit` and `pool.map` detect the `@task` decorator and route the
call through the registry. Args/kwargs are serialized via msgpack.

## Timeouts and local parallelism

`timeout` is enforced for registered tasks in both single-Mac and
fleet-backed execution:

```python
with macfleet.Pool() as pool:
    result = pool.submit(resize, "/tmp/a.jpg", target_w=512, timeout=10)
    results = pool.map(resize, image_paths, timeout=10, max_workers=4)
```

If a task misses its timeout, `Pool.submit`/`Pool.map` raise
`TimeoutError` with the task id and timeout value. For a single-Mac
pool, registered `pool.map` calls run in a local thread pool and preserve
input order while honoring `max_workers`.

## Why not just `pool.submit(lambda x: ..., x)`?

Undecorated functions are rejected by default:

```
ValueError: Pool.submit requires a function decorated with @macfleet.task.
```

That is intentional. Python pickle/cloudpickle can execute arbitrary
code during deserialization, so it is not a safe fleet boundary. If you
are migrating old local-only scripts, you can opt in explicitly:

```python
with macfleet.Pool(allow_legacy_pickle=True) as pool:
    result = pool.submit(lambda x: x + 1, 41)
```

Use that only for code you fully trust on a single Mac. It writes a
local audit event (`compute.legacy_pickle_used`) and should not be used
with untrusted inputs or distributed workers.

## Pydantic schemas for structured args

For richer argument types, declare a Pydantic schema on the decorator:

```python
from pydantic import BaseModel

class TrainArgs(BaseModel):
    epochs: int
    lr: float
    model_name: str

@macfleet.task(schema=TrainArgs)
def train(args: TrainArgs) -> dict:
    # args is a validated TrainArgs instance
    ...
    return {"loss": 0.1, "epochs": args.epochs}

with macfleet.Pool() as pool:
    # Wire carries {"epochs": 3, "lr": 0.01, ...} as msgpack,
    # worker rebuilds the TrainArgs before invoking.
    result = pool.submit(train, TrainArgs(epochs=3, lr=0.01, model_name="bert"))
```

The schema gets applied on both sides:

- **Coordinator**: validates the args you pass before dispatch (fails
  fast on the caller's machine).
- **Worker**: validates the args received on the wire before
  invoking the function (defense in depth against a malicious or
  buggy coordinator).

## Gotchas

### Both Macs must import the task module

Workers look up tasks by name in the *local* registry. If Mac #2
never imports `my_app.tasks`, its registry doesn't know about
`my_app.tasks.resize`, and the dispatch returns:

```
TaskNotRegisteredError: Task 'my_app.tasks.resize' not registered
on this worker. Known tasks: ['builtin.ping', 'builtin.info']
```

Fix: make sure both sides import the same module. One pattern:

```bash
# In your project:
pip install -e .

# Worker script that imports before joining:
python -c "import my_app.tasks; import macfleet; ..."
```

### Args must be msgpack-native or Pydantic-wrapped

Msgpack handles: int, float, str, bytes, bool, list, dict, None.
Anything else (numpy arrays, pandas DataFrames, torch Tensors) needs a
Pydantic schema that defines how to serialize it, OR you send the
underlying bytes/list and reconstruct on the worker.

### Return values follow the same rule

`TaskResult.success()` dumps Pydantic models via `model_dump(mode="json")`
so they survive the msgpack round-trip. For raw types, return directly
— msgpack-native roundtrips just work.

### Remote execution policy

Tasks may opt out of remote execution:

```python
@macfleet.task(remote=False)
def local_cleanup(path: str) -> None:
    ...
```

Workers enforce a `TaskAuthorizationPolicy` before invocation. You can
use policy allowlists/denylists and roles to keep risky maintenance
tasks local while still using the same registry for ordinary compute.

## Introspection

```python
# After decoration, the function exposes:
print(resize.task_name)   # "my_app.tasks.resize"
print(resize.schema)      # None (or the Pydantic class if declared)

from macfleet.compute.registry import get_default_registry
print(get_default_registry().names())
# ['my_app.tasks.resize', 'my_app.tasks.train', ...]
```
