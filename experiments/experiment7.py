import numpy as np

from common import KiB, MiB, make_bytearray, print_table, time_callable


def loops_for_size(size: int) -> int:
    if size <= 1024:
        return 20_000
    if size <= 64 * KiB:
        return 5_000
    if size <= MiB:
        return 200
    return 10


def run_microbenchmarks():
    sizes = [0, 64, 1024, 64 * KiB, 1 * MiB, 16 * MiB]
    rows = []

    for size in sizes:
        size_mib = size / MiB
        payload = make_bytearray(size)
        view = memoryview(payload)
        loops = loops_for_size(size)

        # These operations isolate fixed descriptor overhead from payload-sized copying.
        cases = [
            ("memoryview creation", lambda payload=payload: memoryview(payload)),
            ("memoryview slice", lambda view=view: view[:]),
            ("bytes copy from view", lambda view=view: bytes(view)),
            ("numpy frombuffer view", lambda payload=payload: np.frombuffer(payload, dtype=np.uint8)),
        ]

        for name, fn in cases:
            timing = time_callable(fn, name=name, size_mib=size_mib, loops=loops)
            rows.append(
                {
                    "operation": name,
                    "size_mib": size_mib,
                    "median_ms": timing.median_ms,
                }
            )

    print_table(rows, ["operation", "size_mib", "median_ms"])

    grouped = {}
    for row in rows:
        grouped.setdefault(row["size_mib"], {})[row["operation"]] = row["median_ms"]

    print("\nОтношение copy к memoryview slice:")
    ratios = []
    for size_mib, data in grouped.items():
        if "bytes copy from view" in data and "memoryview slice" in data:
            ratios.append(
                {
                    "size_mib": size_mib,
                    "copy_to_slice_ratio": data["bytes copy from view"] / data["memoryview slice"],
                }
            )
    print_table(ratios, ["size_mib", "copy_to_slice_ratio"])


if __name__ == "__main__":
    run_microbenchmarks()
