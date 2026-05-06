import numpy as np


def show_memoryview_aliasing():
    """Demonstrate that a memoryview slice points to the original buffer."""
    buffer = bytearray(b"abcdef")
    view = memoryview(buffer)
    middle = view[2:5]

    print("До изменения:", buffer)
    middle[0] = ord("X")
    print("После изменения через view:", buffer)
    print("view.obj is buffer:", view.obj is buffer)
    print("readonly:", view.readonly)
    print("shape:", view.shape)
    print("strides:", view.strides)
    print()


def show_contiguous_and_strided_layout():
    """Show how NumPy exports contiguous and strided memory differently."""
    matrix = np.arange(12, dtype=np.int32).reshape(3, 4)
    row = matrix[1, :]
    column = matrix[:, 1]

    for name, obj in [("matrix", matrix), ("row", row), ("column", column)]:
        view = memoryview(obj)
        print(
            f"{name:>6}: shape={view.shape}, strides={view.strides}, "
            f"c_contiguous={view.c_contiguous}"
        )

    try:
        np.frombuffer(memoryview(column), dtype=np.int32)
    except BufferError as exc:
        print("\nNumPy ожидаемо отвергает non-contiguous column view:")
        print(type(exc).__name__ + ":", exc)
    print()


def show_lifetime_and_mutability_guards():
    """Demonstrate why active buffer exports constrain the exporter object."""
    resizable = bytearray(b"camera-frame")
    active_view = memoryview(resizable)

    try:
        resizable.extend(b"-more")
    except BufferError as exc:
        print("Нельзя resize-ить bytearray при активном export:")
        print(type(exc).__name__ + ":", exc)

    active_view.release()
    resizable.extend(b"-more")
    print("После release resize снова возможен:", resizable)

    readonly_view = memoryview(b"immutable")
    try:
        readonly_view[0] = ord("X")
    except TypeError as exc:
        print("Нельзя писать в read-only exporter:")
        print(type(exc).__name__ + ":", exc)


if __name__ == "__main__":
    print("=== memoryview и aliasing ===")
    show_memoryview_aliasing()

    print("=== contiguous и strides ===")
    show_contiguous_and_strided_layout()

    print("=== lifetime и mutability ===")
    show_lifetime_and_mutability_guards()
