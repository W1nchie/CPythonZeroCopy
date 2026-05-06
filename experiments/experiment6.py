import time

import numpy as np

from common import MiB, make_bytes, print_table, summarize_latency


FRAME_H = 1080
FRAME_W = 1920
FRAME_C = 3
HEADER_SIZE = 64
FRAME_COUNT = 120
FRAME_BODY_SIZE = FRAME_H * FRAME_W * FRAME_C


def process_frame_copy(packet: bytes) -> int:
    # This path materializes a fresh frame array and a fresh ROI copy.
    body = packet[HEADER_SIZE:HEADER_SIZE + FRAME_BODY_SIZE]
    frame = np.frombuffer(body, dtype=np.uint8).reshape(FRAME_H, FRAME_W, FRAME_C).copy()
    roi = frame[FRAME_H // 4: FRAME_H // 2, FRAME_W // 4: FRAME_W // 2, :].copy()
    return int(roi.mean())


def process_frame_view(packet: bytes) -> int:
    # This path keeps the original packet alive and builds only views on top of it.
    packet_view = memoryview(packet)
    body = packet_view[HEADER_SIZE:HEADER_SIZE + FRAME_BODY_SIZE]
    frame = np.frombuffer(body, dtype=np.uint8).reshape(FRAME_H, FRAME_W, FRAME_C)
    roi = frame[FRAME_H // 4: FRAME_H // 2, FRAME_W // 4: FRAME_W // 2, :]
    return int(roi.mean())


def latency_series(fn, packet: bytes, frames: int = FRAME_COUNT):
    samples = []
    for _ in range(frames):
        start = time.perf_counter()
        fn(packet)
        samples.append(time.perf_counter() - start)
    return samples


def run_streaming_experiment():
    packet = make_bytes(HEADER_SIZE + FRAME_BODY_SIZE)

    # The same synthetic frame stream is processed in two different ways.
    copy_samples = latency_series(process_frame_copy, packet)
    view_samples = latency_series(process_frame_view, packet)

    rows = [
        summarize_latency("copy-heavy frame path", copy_samples, FRAME_BODY_SIZE / MiB),
        summarize_latency("view frame path", view_samples, FRAME_BODY_SIZE / MiB),
    ]

    for row in rows:
        values_ms = [sample * 1e3 for sample in (copy_samples if "copy" in row["name"] else view_samples)]
        row["miss_5ms_pct"] = 100.0 * sum(value > 5.0 for value in values_ms) / len(values_ms)
        row["miss_16_7ms_pct"] = 100.0 * sum(value > 16.7 for value in values_ms) / len(values_ms)

    print_table(
        rows,
        ["name", "size_mib", "median_ms", "p95_ms", "miss_5ms_pct", "miss_16_7ms_pct"],
    )

    print("\nПервые 10 задержек для оценки стабильности:")
    preview_rows = []
    for index, (copy_value, view_value) in enumerate(zip(copy_samples[:10], view_samples[:10]), start=1):
        preview_rows.append(
            {
                "frame": index,
                "copy_ms": copy_value * 1e3,
                "view_ms": view_value * 1e3,
            }
        )
    print_table(preview_rows, ["frame", "copy_ms", "view_ms"])


if __name__ == "__main__":
    run_streaming_experiment()
