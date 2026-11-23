"""Repository entry point for the installed MacFleet verification harness."""

from macfleet.cli.verify import main, run_distributed_check, run_self_check  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(main())
