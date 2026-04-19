# Registered tasks (`@macfleet.task`)

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

## Why not just `pool.submit(lambda x: ..., x)`?

You *can* still pass lambdas — `pool.submit` falls through to a legacy
ProcessPool + cloudpickle path when the callable isn't decorated. But
that path is:

1. **Unsafe**: cloudpickle deserializes arbitrary code. If your
   coordinator ever accepts tasks from a less-trusted source, you
   just gave them RCE on every worker.
2. **Slower**: pickling full closures is measurably slow for large
   args.
3. **Going away**: the distributed path (coming in a later PR) only
   dispatches registered tasks. Lambdas will keep working locally
   but won't run across the fleet.

Decorate your functions.

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

## Introspection

```python
# After decoration, the function exposes:
print(resize.task_name)   # "my_app.tasks.resize"
print(resize.schema)      # None (or the Pydantic class if declared)

from macfleet.compute.registry import get_default_registry
print(get_default_registry().names())
# ['my_app.tasks.resize', 'my_app.tasks.train', ...]
```
