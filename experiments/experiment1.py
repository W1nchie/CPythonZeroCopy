import numpy as np

from common import MiB, make_bytes, print_table, time_callable


def run_basic_benchmarks(size_mib: int = 64):
    size = size_mib * MiB

    # One shared payload is enough to show the semantic difference
    # between copy-heavy operations and view-based operations.
    immutable_bytes = make_bytes(size)
    mutable_buffer = bytearray(immutable_bytes)
    numpy_view = np.frombuffer(mutable_buffer, dtype=np.uint8)

    print("Проверка семантики объектов:")
    print("bytes[:] is bytes:", immutable_bytes[:] is immutable_bytes)
    print("bytes[1:] is bytes:", immutable_bytes[1:] is immutable_bytes)
    print("bytearray[:] is bytearray:", mutable_buffer[:] is mutable_buffer)
    print("memoryview(bytearray)[1:].obj is bytearray:", memoryview(mutable_buffer)[1:].obj is mutable_buffer)
    print()

    cases = [
        ("bytes partial slice copy", lambda: immutable_bytes[1:]),
        ("bytearray full slice copy", lambda: mutable_buffer[:]),
        ("bytes(memoryview) copy", lambda: bytes(memoryview(immutable_bytes))),
        ("memoryview creation", lambda: memoryview(mutable_buffer)),
        ("memoryview slice", lambda: memoryview(mutable_buffer)[1:]),
        ("numpy frombuffer view", lambda: np.frombuffer(mutable_buffer, dtype=np.uint8)),
        ("numpy array copy", lambda: np.array(numpy_view, copy=True)),
    ]

    # Each callable exercises one memory behavior on the same payload.
    results = [
        time_callable(fn, name=name, size_mib=size_mib)
        for name, fn in cases
    ]

    rows = sorted(
        (
            {
                "operation": result.name,
                "size_mib": result.size_mib,
                "median_ms": result.median_ms,
                "best_ms": result.best_ms,
            }
            for result in results
        ),
        key=lambda row: row["median_ms"],
    )
    print_table(rows, ["operation", "size_mib", "median_ms", "best_ms"])


if __name__ == "__main__":
    run_basic_benchmarks()
