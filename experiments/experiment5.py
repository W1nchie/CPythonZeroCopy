import multiprocessing as mp
import os
import time
from multiprocessing import shared_memory

from common import MiB, make_bytes, print_table, summarize_latency


MAX_MB = int(os.environ.get("ZC_MP_MAX_MB", "64"))
TRIALS = int(os.environ.get("ZC_MP_TRIALS", "5"))


def queue_worker(q_in: mp.Queue, q_out: mp.Queue):
    while True:
        payload = q_in.get()
        if payload is None:
            break
        q_out.put(len(payload))


def shared_memory_worker(q_in: mp.Queue, q_out: mp.Queue):
    while True:
        token = q_in.get()
        if token is None:
            break

        name, size = token
        shm = shared_memory.SharedMemory(name=name)
        try:
            # The worker reads the shared payload through a memoryview-backed buffer.
            view = shm.buf[:size]
            checksum = int(view[0]) + int(view[-1])
            q_out.put(checksum)
        finally:
            shm.close()


def measure_queue_round_trip(ctx, size: int, trials: int):
    q_in = ctx.Queue()
    q_out = ctx.Queue()
    worker = ctx.Process(target=queue_worker, args=(q_in, q_out))
    worker.start()

    payload = make_bytes(size)
    samples = []
    try:
        for _ in range(trials):
            start = time.perf_counter()
            q_in.put(payload)
            q_out.get()
            samples.append(time.perf_counter() - start)
    finally:
        q_in.put(None)
        worker.join()
    return samples


def measure_shared_memory_round_trip(ctx, size: int, trials: int):
    q_in = ctx.Queue()
    q_out = ctx.Queue()
    worker = ctx.Process(target=shared_memory_worker, args=(q_in, q_out))
    worker.start()

    shm = shared_memory.SharedMemory(create=True, size=size)
    try:
        shm.buf[:size] = make_bytes(size)
        token = (shm.name, size)
        samples = []
        for _ in range(trials):
            start = time.perf_counter()
            q_in.put(token)
            q_out.get()
            samples.append(time.perf_counter() - start)
    finally:
        q_in.put(None)
        worker.join()
        shm.close()
        shm.unlink()
    return samples


def run_multiprocessing_experiment():
    ctx = mp.get_context("spawn")
    sizes_mib = [size for size in [1, 4, 16, 64] if size <= MAX_MB]
    rows = []

    for size_mib in sizes_mib:
        size = size_mib * MiB

        # We compare two process-boundary designs:
        # sending the bytes payload itself, or sending only a shared-memory token.
        queue_samples = measure_queue_round_trip(ctx, size, TRIALS)
        shm_samples = measure_shared_memory_round_trip(ctx, size, TRIALS)

        for summary in [
            summarize_latency("Queue: bytes payload", queue_samples, size_mib),
            summarize_latency("shared_memory: token", shm_samples, size_mib),
        ]:
            summary["throughput_mib_s"] = size_mib / (summary["median_ms"] / 1e3)
            rows.append(summary)

    print_table(rows, ["name", "size_mib", "median_ms", "p95_ms", "throughput_mib_s"])

    print("\nОтношение задержки Queue / shared_memory:")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["size_mib"], {})[row["name"]] = row["median_ms"]

    ratios = []
    for size_mib, data in grouped.items():
        if "Queue: bytes payload" in data and "shared_memory: token" in data:
            ratios.append(
                {
                    "size_mib": size_mib,
                    "queue_to_shm_ratio": data["Queue: bytes payload"] / data["shared_memory: token"],
                }
            )
    print_table(ratios, ["size_mib", "queue_to_shm_ratio"])


if __name__ == "__main__":
    run_multiprocessing_experiment()
