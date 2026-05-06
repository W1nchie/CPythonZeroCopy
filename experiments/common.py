import gc
import math
import os
import random
import statistics
import time
from dataclasses import dataclass

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None


SEED = 20260502
random.seed(SEED)
np.random.seed(SEED)

KiB = 1024
MiB = 1024 ** 2

DEFAULT_RUNS = int(os.environ.get("ZC_RUNS", "5"))
DEFAULT_WARMUPS = int(os.environ.get("ZC_WARMUPS", "1"))

_BLACKHOLE = 0


@dataclass
class Timing:
    name: str
    size_mib: float
    median_s: float
    best_s: float
    runs: int
    loops: int

    @property
    def median_ms(self) -> float:
        return self.median_s * 1e3

    @property
    def best_ms(self) -> float:
        return self.best_s * 1e3

    @property
    def throughput_mib_s(self) -> float:
        if self.median_s <= 0 or self.size_mib <= 0:
            return math.nan
        return self.size_mib / self.median_s


def consume_result(value) -> None:
    """Touch the result so the benchmark body is not trivially unused."""
    global _BLACKHOLE

    if value is None:
        _BLACKHOLE ^= 0x9E3779B9
        return

    try:
        _BLACKHOLE ^= len(value)
        return
    except TypeError:
        pass

    try:
        _BLACKHOLE ^= int(value) & 0xFFFFFFFF
    except Exception:
        _BLACKHOLE ^= id(value) & 0xFFFFFFFF


def force_gc() -> None:
    gc.collect()
    gc.collect()


def touch_pages(buf, step: int = 4096):
    """Commit OS pages so large buffers are measured more consistently."""
    if isinstance(buf, bytes):
        return buf

    for offset in range(0, len(buf), step):
        buf[offset] = (offset // step) & 0xFF
    return buf


def make_bytearray(size: int) -> bytearray:
    return touch_pages(bytearray(size))


def make_bytes(size: int) -> bytes:
    return bytes(make_bytearray(size))


def rss_mib() -> float:
    if psutil is None:
        return math.nan
    return psutil.Process(os.getpid()).memory_info().rss / MiB


def available_mib() -> float:
    if psutil is None:
        return math.inf
    return psutil.virtual_memory().available / MiB


def safe_sizes_mib(requested, multiplier: int = 4):
    """Skip sizes that are likely to exceed available RAM during copy tests."""
    available = available_mib()
    safe = []
    for size_mib in requested:
        if size_mib * multiplier < available * 0.75:
            safe.append(size_mib)
    return safe


def time_callable(
    fn,
    *,
    name: str,
    size_mib: float = 0.0,
    runs: int = DEFAULT_RUNS,
    warmups: int = DEFAULT_WARMUPS,
    loops: int = 1,
) -> Timing:
    for _ in range(warmups):
        for _ in range(loops):
            consume_result(fn())

    samples = []
    for _ in range(runs):
        force_gc()
        start = time.perf_counter()
        value = None
        for _ in range(loops):
            value = fn()
        elapsed = time.perf_counter() - start
        consume_result(value)
        samples.append(elapsed / loops)

    return Timing(
        name=name,
        size_mib=float(size_mib),
        median_s=statistics.median(samples),
        best_s=min(samples),
        runs=runs,
        loops=loops,
    )


def summarize_latency(name: str, values_s, size_mib: float = 0.0):
    values_ms = [value * 1e3 for value in values_s]
    values_ms.sort()
    index_95 = max(0, min(len(values_ms) - 1, math.ceil(0.95 * len(values_ms)) - 1))
    return {
        "name": name,
        "size_mib": float(size_mib),
        "median_ms": statistics.median(values_ms),
        "mean_ms": statistics.fmean(values_ms),
        "p95_ms": values_ms[index_95],
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
        "count": len(values_ms),
    }


def print_table(rows, columns):
    """Print a simple fixed-width table without extra dependencies."""
    string_rows = []
    widths = {column: len(column) for column in columns}

    for row in rows:
        rendered = {}
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                rendered_value = f"{value:.3f}"
            else:
                rendered_value = str(value)
            rendered[column] = rendered_value
            widths[column] = max(widths[column], len(rendered_value))
        string_rows.append(rendered)

    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)
    print(header)
    print(separator)
    for row in string_rows:
        print(" | ".join(row[column].ljust(widths[column]) for column in columns))
