import os

import numpy as np

from common import MiB, make_bytes, print_table, safe_sizes_mib, time_callable


MAX_MB = min(int(os.environ.get("ZC_MAX_MB", "500")), 256)
HEADER_SIZE = 64


def copy_heavy_pipeline(packet: bytes, body_size: int) -> int:
    # This path materializes several intermediate buffers on purpose.
    payload = packet[HEADER_SIZE:HEADER_SIZE + body_size]
    middle = payload[body_size // 8: body_size * 7 // 8]
    mutable_copy = bytearray(middle)
    array_copy = np.frombuffer(mutable_copy, dtype=np.uint8).copy()
    return int(array_copy.sum())


def view_based_pipeline(packet: bytes, body_size: int) -> int:
    # This path keeps views as long as possible and only reads the data.
    packet_view = memoryview(packet)
    payload = packet_view[HEADER_SIZE:HEADER_SIZE + body_size]
    middle = payload[body_size // 8: body_size * 7 // 8]
    array_view = np.frombuffer(middle, dtype=np.uint8)
    return int(array_view.sum())


def estimated_copied_mib(size_mib: float) -> float:
    # payload copy + middle copy + bytearray copy + NumPy-owned copy
    return size_mib * (1.0 + 0.75 + 0.75 + 0.75)


def run_pipeline_benchmarks():
    sizes_mib = safe_sizes_mib([1, 4, 16, 64, 128, 256], multiplier=5)
    rows = []

    for size_mib in sizes_mib:
        body_size = size_mib * MiB
        packet = make_bytes(HEADER_SIZE + body_size + 128)

        for name, fn in [
            ("copy-heavy pipeline", lambda packet=packet, body_size=body_size: copy_heavy_pipeline(packet, body_size)),
            ("view pipeline", lambda packet=packet, body_size=body_size: view_based_pipeline(packet, body_size)),
        ]:
            timing = time_callable(fn, name=name, size_mib=size_mib)
            rows.append(
                {
                    "pipeline": name,
                    "size_mib": size_mib,
                    "median_ms": timing.median_ms,
                    "estimated_copied_mib": estimated_copied_mib(size_mib) if "copy" in name else 0.0,
                }
            )

    print_table(rows, ["pipeline", "size_mib", "median_ms", "estimated_copied_mib"])

    print("\nОтношение copy-heavy pipeline к view pipeline:")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["size_mib"], {})[row["pipeline"]] = row["median_ms"]

    ratios = []
    for size_mib, data in grouped.items():
        if "copy-heavy pipeline" in data and "view pipeline" in data:
            ratios.append(
                {
                    "size_mib": size_mib,
                    "copy_to_view_ratio": data["copy-heavy pipeline"] / data["view pipeline"],
                }
            )
    print_table(ratios, ["size_mib", "copy_to_view_ratio"])


if __name__ == "__main__":
    run_pipeline_benchmarks()
