def import_shape(path: str, k=None):
    """-> kernel shape. Raises ImportFailed for a wrong extension, missing file or bad transfer.

    `k` lets a caller pass the kernel module it is already using (the endpoints inject theirs, so a
    test can exercise the whole path without OCP installed).
    """
    kernel_module = kernel if k is None else k
    ext = os.path.splitext(str(path))[1].lower()
    if ext not in SUPPORTED:
        raise ImportFailed(f"{path!r}: unsupported format {ext or '(none)'}; use {', '.join(SUPPORTED)}")
    if not os.path.exists(path):
        raise ImportFailed(f"{path!r}: file not found")
    try:
        shape = kernel_module.read_step(path)
    except kernel.KernelUnavailable as exc:
        raise ImportFailed(str(exc)) from None
    except Exception as exc:
        raise ImportFailed(f"{path!r}: kernel could not read the file ({exc})") from None
    if shape is None:
        raise ImportFailed(f"{path!r}: kernel returned no shape")
    return shape
