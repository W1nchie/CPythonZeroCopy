import math
import os

from common import MiB, force_gc, make_bytearray, print_table, rss_mib


BASE_MB = int(os.environ.get("ZC_MEMORY_DEMO_MB", "64"))


def run_memory_experiment(base_mib: int = BASE_MB, retained_count: int = 6):
    if math.isnan(rss_mib()):
        raise RuntimeError("Для этого эксперимента нужен psutil.")

    rows = []

    def record(stage: str):
        force_gc()
        rows.append({"stage": stage, "rss_mib": rss_mib()})

    # One committed base buffer lets us compare retained views vs retained copies.
    base = make_bytearray(base_mib * MiB)
    record("base buffer")

    # Views reuse the same payload and mostly add only small descriptors.
    views = [memoryview(base)[index * 4096:] for index in range(retained_count)]
    record(f"{retained_count} memoryview slices retained")

    # Copies allocate fresh buffers and should noticeably increase RSS.
    copies = [bytes(base) for _ in range(retained_count)]
    record(f"{retained_count} full copies retained")

    del views
    record("after deleting views")

    del copies
    record("after deleting copies")

    del base
    record("after deleting base buffer")

    start_rss = rows[0]["rss_mib"]
    for row in rows:
        row["delta_from_start_mib"] = row["rss_mib"] - start_rss

    print_table(rows, ["stage", "rss_mib", "delta_from_start_mib"])


if __name__ == "__main__":
    run_memory_experiment()
