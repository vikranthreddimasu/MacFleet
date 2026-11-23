"""Run a registered task across two Macs with their existing fleet credentials."""

import argparse
import asyncio
import socket

from macfleet import task
from macfleet.comm.transport import PeerTransport
from macfleet.compute import RemoteTaskError, TaskDispatcher, TaskWorker
from macfleet.security import SecurityConfig, resolve_token_with_file


@task(name="example.square")
def square(value: int) -> dict:
    if value < 0:
        raise ValueError("expected a nonnegative number")
    return {"worker": socket.gethostname(), "square": value**2}


async def main(host: str | None, port: int, fleet_id: str | None) -> None:
    token = resolve_token_with_file()
    if not token:
        raise SystemExit("Pair both Macs with macfleet join --bootstrap and macfleet pair first.")
    security = SecurityConfig(token=token, fleet_id=fleet_id)
    transport = PeerTransport(local_id="caller" if host else "worker", security=security)
    if host:
        dispatcher = TaskDispatcher(transport, ["worker"])
        try:
            await transport.connect("worker", host, port)
            await dispatcher.start()
            try:
                failed = await dispatcher.submit(square, -1, timeout=10)
                await failed.result()
            except RemoteTaskError as error:
                print(error.remote_traceback.splitlines()[-1])
            else:
                raise RuntimeError("Expected the negative input to fail")
            job = await dispatcher.submit(square, 9, timeout=10)
            result = await job.result()
            assert result["square"] == 81
            print(f"Worker {result['worker']} returned {result['square']}")
        finally:
            await dispatcher.stop()
            await transport.disconnect_all()
    else:
        connected = asyncio.Event()
        worker = TaskWorker(transport, "caller")
        def on_connect(peer, connection):
            if peer == "caller":
                connected.set()

        try:
            await transport.start_server("0.0.0.0", port, on_connect=on_connect)
            print(f"Worker ready on port {port}. Press Ctrl+C to stop.", flush=True)
            await connected.wait()
            await worker.start()
            await asyncio.Event().wait()
        finally:
            await worker.stop()
            await transport.disconnect_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", metavar="WORKER_IPV4", help="Connect to the worker and submit the examples")
    parser.add_argument("--port", type=int, default=50053)
    parser.add_argument("--fleet-id", help="Fleet ID used when pairing, if one was supplied")
    args = parser.parse_args()
    try:
        asyncio.run(main(args.connect, args.port, args.fleet_id))
    except KeyboardInterrupt:
        pass
