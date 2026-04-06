"""Decorators for distributed training.

    @macfleet.distributed(engine="torch")
    def train():
        ...
"""

from functools import wraps
from typing import Any, Callable


def distributed(
    engine: str = "torch",
    compression: str = "adaptive",
    **pool_kwargs: Any,
) -> Callable:
    """Decorator to run a training function on the MacFleet pool.

    Args:
        engine: Engine type ("torch" or "mlx").
        compression: Compression type.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            from macfleet.sdk.pool import Pool

            with Pool(engine=engine, **pool_kwargs):
                return func(*args, **kwargs)
        return wrapper
    return decorator
