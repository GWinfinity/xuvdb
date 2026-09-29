"""Runtime initialization for the vdb kernels, factored out to avoid import cycles between submodules."""

import quadrants as qd


def init_runtime():
    """Initialize the quadrants runtime for vdb kernels if no runtime is active yet.

    Uses the CPU backend; volume sampling is float32 work. When quadrants is already initialized
    (typically by `gs.init()`), the existing runtime is kept as is.
    """
    try:
        is_initialized = qd.lang.impl.get_runtime().prog is not None
    except qd.lang.exception.QuadrantsRuntimeError:
        is_initialized = False
    if not is_initialized:
        qd.init(arch=qd.cpu, default_fp=qd.f32)
