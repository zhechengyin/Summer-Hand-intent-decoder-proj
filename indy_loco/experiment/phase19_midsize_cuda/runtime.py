"""NVRTC + CUDA Driver API bridge, using PyTorch allocations/current stream.

No external compiler/dependency installation is required. This experimental
first-order autograd Function is for eager execution only, not torch.compile.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import os
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
_RUNTIMES = {}


def _api(lib, name, args):
    function = getattr(lib, name)
    function.argtypes = args
    function.restype = C.c_int
    return function


def _check(code, operation):
    if code:
        raise RuntimeError(f"{operation} failed with native error {code}")


def compile_ptx(architecture="compute_89"):
    """Offline compilation is possible even when the GPU is unavailable."""
    libdir = Path(torch.__file__).parent / "lib"
    if os.name != "nt":
        raise RuntimeError("This checked-in loader currently targets Windows only")
    dll_directory = os.add_dll_directory(str(libdir))
    nvrtc = C.CDLL(str(libdir / "nvrtc64_120_0.dll"))
    create = _api(
        nvrtc,
        "nvrtcCreateProgram",
        [
            C.POINTER(C.c_void_p),
            C.c_char_p,
            C.c_char_p,
            C.c_int,
            C.c_void_p,
            C.c_void_p,
        ],
    )
    compile_program = _api(
        nvrtc, "nvrtcCompileProgram", [C.c_void_p, C.c_int, C.POINTER(C.c_char_p)]
    )
    log_size = _api(
        nvrtc, "nvrtcGetProgramLogSize", [C.c_void_p, C.POINTER(C.c_size_t)]
    )
    get_log = _api(nvrtc, "nvrtcGetProgramLog", [C.c_void_p, C.c_void_p])
    ptx_size = _api(nvrtc, "nvrtcGetPTXSize", [C.c_void_p, C.POINTER(C.c_size_t)])
    get_ptx = _api(nvrtc, "nvrtcGetPTX", [C.c_void_p, C.c_void_p])
    destroy = _api(nvrtc, "nvrtcDestroyProgram", [C.POINTER(C.c_void_p)])
    program = C.c_void_p()
    source = (HERE / "ops.cu").read_bytes()
    _check(
        create(C.byref(program), source, b"ops.cu", 0, None, None),
        "nvrtcCreateProgram",
    )
    options = [
        f"--gpu-architecture={architecture}",
        "--std=c++14",
        "--fmad=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    ]
    try:
        argv = (C.c_char_p * len(options))(*(x.encode() for x in options))
        status = compile_program(program, len(options), argv)
        size = C.c_size_t()
        _check(log_size(program, C.byref(size)), "nvrtcGetProgramLogSize")
        log = C.create_string_buffer(size.value)
        _check(get_log(program, log), "nvrtcGetProgramLog")
        if status:
            raise RuntimeError(log.value.decode(errors="replace"))
        _check(ptx_size(program, C.byref(size)), "nvrtcGetPTXSize")
        ptx = C.create_string_buffer(size.value)
        _check(get_ptx(program, ptx), "nvrtcGetPTX")
        return ptx.raw, {
            "options": options,
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "log": log.value.decode(errors="replace"),
        }
    finally:
        destroy(C.byref(program))
        dll_directory.close()


class Runtime:
    def __init__(self, device):
        self.device = device
        with torch.cuda.device(device):
            torch.cuda.init()
            # Ensure the primary context exists before using the Driver API.
            torch.empty(1, device=torch.device("cuda", device))
            capability = torch.cuda.get_device_capability(device)
            ptx, self.metadata = compile_ptx(f"compute_{capability[0]}{capability[1]}")
            self.driver = C.WinDLL("nvcuda.dll")
            load = _api(
                self.driver, "cuModuleLoadData", [C.POINTER(C.c_void_p), C.c_void_p]
            )
            get = _api(
                self.driver,
                "cuModuleGetFunction",
                [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
            )
            self.launch = _api(
                self.driver,
                "cuLaunchKernel",
                [
                    C.c_void_p,
                    *([C.c_uint] * 7),
                    C.c_void_p,
                    C.POINTER(C.c_void_p),
                    C.c_void_p,
                ],
            )
            self.module = C.c_void_p()
            blob = C.create_string_buffer(ptx)
            _check(load(C.byref(self.module), blob), "cuModuleLoadData")
            self.functions = {}
            for name in ("residual_forward", "residual_backward", "norm_forward", "norm_backward", "gate_forward", "gate_backward"):
                function = C.c_void_p()
                _check(
                    get(C.byref(function), self.module, name.encode()),
                    "cuModuleGetFunction",
                )
                self.functions[name] = function

    def call(self, name, tensors, integers, lanes):
        with torch.cuda.device(self.device):
            storage = [C.c_void_p(t.data_ptr()) for t in tensors] + [
                C.c_int(v) for v in integers
            ]
            arguments = (C.c_void_p * len(storage))(
                *(C.cast(C.pointer(v), C.c_void_p) for v in storage)
            )
            stream = C.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
            _check(
                self.launch(
                    self.functions[name],
                    (lanes + 127) // 128,
                    1,
                    1,
                    128,
                    1,
                    1,
                    0,
                    stream,
                    arguments,
                    None,
                ),
                name,
            )


def runtime(device):
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    if index not in _RUNTIMES:
        _RUNTIMES[index] = Runtime(index)
    return _RUNTIMES[index]


