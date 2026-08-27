"""Durable Windows fix for apache-tvm-ffi's CUDA JIT build command.

apache-tvm-ffi (<=0.1.x) emits a broken nvcc command on Windows:

    cuda_cflags = -Xcompiler /std:c++17 /O2 -std=c++20 ...

The ``/O2`` host-compiler flag is NOT wrapped in its own ``-Xcompiler``, so
nvcc reads it as a second *input file* and dies with:

    nvcc fatal: A single input file is required for a non-link phase
    when an outputfile is specified

It also omits ``cudart.lib`` from the link line, so linking fails with
unresolved ``cuda*`` symbols.

We cannot patch the installed tvm_ffi package: it dies on every ``uv pip
install``. Instead this module monkeypatches ``tvm_ffi.cpp.extension`` at
import time, from inside the FreeToken repo, so the fix survives reinstalls
and applies on any Windows machine. It is a no-op on Linux/macOS.
"""

from __future__ import annotations

import platform

# Only needed on Windows.
if platform.system() == "Windows":
    import tvm_ffi.cpp.extension as _ext

    _orig_generate = _ext._generate_ninja_build

    def _patched_generate_ninja_build(*args, **kwargs):
        ninja = _orig_generate(*args, **kwargs)
        if not isinstance(ninja, str):
            return ninja

        # 1) Fix the host-compiler standard: tvm-ffi hardcodes /std:c++17 on
        #    Windows, but the kernels use C++20 (std::source_location,
        #    std::integral concepts). Pair /O2 with its own -Xcompiler too.
        #    Before: -Xcompiler /std:c++17 /O2 -std=c++20
        #    After:  -Xcompiler /std:c++20 -Xcompiler /O2 -std=c++20
        ninja = ninja.replace(
            "-Xcompiler /std:c++17 /O2",
            "-Xcompiler /std:c++20 -Xcompiler /O2",
        )

        # 2) Link cudart.lib. Append to the `ldflags =` variable line (the
        #    only place safe to inject a library + LIBPATH on Windows). The
        #    CUDA 12.x import lib lives in lib/x64.
        if "cudart.lib" not in ninja:
            cuda_home = getattr(_ext, "_find_cuda_home", lambda: None)()
            if cuda_home:
                for cand in ("lib/x64", "lib"):
                    p = _ext.Path(cuda_home) / cand
                    if p.exists():
                        cudart = '/LIBPATH:"{}" cudart.lib'.format(p)
                        ninja = ninja.replace(
                            "ldflags = ",
                            "ldflags = {} ".format(cudart),
                            1,
                        )
                        break
        return ninja

    _ext._generate_ninja_build = _patched_generate_ninja_build
