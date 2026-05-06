import os

import numpy as np

from common import MiB, print_table, safe_sizes_mib, time_callable, make_bytearray


MAX_MB = int(os.environ.get("ZC_MAX_MB", "500"))


def run_scaling_benchmarks():
    requested_sizes = [1, 4, 16, 64, 128, 256, 500]
    sizes_mib = safe_sizes_mib([size for size in requested_sizes if size <= MAX_MB], multiplier=4)

    rows = []
    for size_mib in sizes_mib:
        payload = make_bytearray(size_mib * MiB)
        view = memoryview(payload)
        array_view = np.frombuffer(payload, dtype=np.uint8)

        # The chosen operations isolate the main scaling patterns:
        # explicit copy, lightweight view, and NumPy view/copy transitions.
        cases = [
            ("bytes(memoryview) copy", lambda view=view: bytes(view)),
            ("bytearray partial slice copy", lambda payload=payload: payload[1:]),
            ("memoryview slice", lambda view=view: view[1:]),
            ("numpy frombuffer view", lambda payload=payload: np.frombuffer(payload, dtype=np.uint8)),
            ("numpy array copy", lambda array_view=array_view: np.array(array_view, copy=True)),
        ]

        for name, fn in cases:
            timing = time_callable(fn, name=name, size_mib=size_mib)
            rows.append(
                {
                    "operation": name,
                    "size_mib": timing.size_mib,
                    "median_ms": timing.median_ms,
                    "throughput_mib_s": timing.throughput_mib_s,
                }
            )

    print_table(rows, ["operation", "size_mib", "median_ms", "throughput_mib_s"])

    # A short derived summary is useful when we only want the scaling trend.
    print("\nКраткая сводка по выигрышу memoryview slice над bytes(memoryview):")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["size_mib"], {})[row["operation"]] = row["median_ms"]

    speedup_rows = []
    for size_mib, data in grouped.items():
        if "bytes(memoryview) copy" in data and "memoryview slice" in data:
            speedup_rows.append(
                {
                    "size_mib": size_mib,
                    "speedup": data["bytes(memoryview) copy"] / data["memoryview slice"],
                }
            )
    print_table(speedup_rows, ["size_mib", "speedup"])


if __name__ == "__main__":
    run_scaling_benchmarks()
