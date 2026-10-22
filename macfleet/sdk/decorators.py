"""Migration error for the former decorator that did not synchronize training."""

from functools import wraps
from typing import Any, Callable


def distributed(
    engine: str = "torch",
    compression: str = "none",
    **pool_kwargs: Any,
) -> Callable:
    """Removed misleading wrapper; use an explicit distributed Pool.

    The old decorator invoked the function locally without synchronizing any
    gradients. Keeping an explicit migration error prevents false fleet runs.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError(
                "@macfleet.distributed never synchronized the decorated function. "
                "Use with Pool(enable_pool_distributed=True, quorum_size=N) as pool, "
                "then call pool.train(...) in the same script on every Mac."
            )
        return wrapper
    return decorator
