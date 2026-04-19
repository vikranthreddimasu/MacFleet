"""High-level Python SDK for MacFleet.

    import macfleet

    # Context manager
    with macfleet.Pool() as pool:
        pool.train(model=MyModel(), dataset=ds, epochs=10)

    # One-liner
    macfleet.train(model=MyModel(), dataset=ds, epochs=10)

    # Decorator
    @macfleet.distributed(engine="torch")
    def my_training():
        ...
"""

from macfleet.sdk.decorators import distributed
from macfleet.sdk.pool import Pool
from macfleet.sdk.train import train

__all__ = ["Pool", "train", "distributed"]
