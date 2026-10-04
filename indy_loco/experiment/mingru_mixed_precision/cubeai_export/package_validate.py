"""Package Cube.AI-generated code; validate its real C runtime and ARM compilation."""
from __future__ import annotations

import ctypes
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import export as base

np = base.np
PACK = Path.home() / "STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0"
RUN = base.OUTPUT / "runtime_weights_generate"
WORK = RUN / "workspace/inspector_b2_mingru/workspace"
DEST = base.OUTPUT / "stm32_package"
MINGW = PACK / "Utilities/windows/mingw64/bin"
ARM = next(Path("C:/ST/STM32CubeIDE_1.19.0").rglob("arm-none-eabi-gcc.exe"))


def command(args, log):
    result = subprocess.run([str(a) for a in args], capture_output=True, text=True, errors="replace", timeout=120)
    Path(log).write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Compiler exit {result.returncode}; see {log}\n{result.stderr[-3000:]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="runtime_weights_generate")
    args = parser.parse_args()
    global RUN, WORK
    RUN = base.OUTPUT / args.tag
    WORK = RUN / "workspace/inspector_b2_mingru/workspace"
    receipt = json.loads((RUN / "receipt.json").read_text())
    assert receipt["status"] == "complete"
    generated, adapter, inc, libs, validation = [DEST / p for p in ("generated", "adapter", "include", "lib", "validation")]
    for directory in (generated, adapter, inc, libs, validation):
        directory.mkdir(parents=True, exist_ok=True)
    for src in (WORK / "generated").iterdir():
        if src.suffix in (".c", ".h"):
            shutil.copy2(src, generated / src.name)
    for src in (PACK / "Middlewares/ST/AI/Inc").glob("*.h"):
        shutil.copy2(src, inc / src.name)
    for src in (base.HERE / "b2_decoder.c", base.HERE / "b2_decoder.h",
                base.OUTPUT / "adapter/b2_packed_weights.c", base.OUTPUT / "adapter/b2_packed_weights.h"):
        shutil.copy2(src, adapter / src.name)
    shutil.copy2(PACK / "Middlewares/ST/AI/Lib/GCC/STM32H7/NetworkRuntime1020_CM7_GCC.a", libs)
    shutil.copy2(PACK / "Middlewares/ST/AI/LICENSE.txt", DEST / "ST_RUNTIME_LICENSE.txt")
    shutil.copy2(RUN / "output/LICENSE.txt", DEST / "ST_GENERATED_CODE_LICENSE.txt")
    shutil.copy2(base.OUTPUT / "preprocessing.json", DEST)
    shutil.copy2(RUN / "output/b2_mingru_generate_report.txt", DEST)
    shutil.copy2(base.OUTPUT / "runtime_weights_verification.json", DEST)

    dll = validation / "b2_host_adapter.dll"
    host_command = [MINGW / "gcc.exe", "-shared", "-std=c11", "-O2", "-Wl,--export-all-symbols",
                    f"-I{generated}", f"-I{WORK / 'include'}", f"-I{adapter}", adapter / "b2_decoder.c",
                    adapter / "b2_packed_weights.c", f"-L{WORK / 'build'}", "-lai_b2_mingru", "-o", dll]
    command(host_command, validation / "host_compile.log")
    handles = [os.add_dll_directory(str(p)) for p in (WORK / "lib", MINGW)]
    runtime = ctypes.CDLL(str(dll))
    runtime.b2_decoder_activation_bytes.restype = ctypes.c_size_t
    runtime.b2_decoder_init.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    runtime.b2_decoder_init.restype = ctypes.c_int
    array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    runtime.b2_decoder_run.argtypes = [array, array]
    runtime.b2_decoder_run.restype = ctypes.c_int
    runtime.b2_decoder_destroy.argtypes = []
    runtime.b2_decoder_destroy.restype = None
    activation_bytes = runtime.b2_decoder_activation_bytes()
    arena = ctypes.create_string_buffer(activation_bytes)
    assert runtime.b2_decoder_init(arena, activation_bytes) == 0
    inputs = np.load(base.OUTPUT / "parity_inputs.npy")
    reference = np.load(base.OUTPUT / "reference_outputs.npy")
    outputs = np.empty((len(inputs), 1, 2), dtype=np.float32)
    for i, values in enumerate(inputs):
        status = runtime.b2_decoder_run(values, outputs[i])
        if status:
            raise RuntimeError(f"C inference failed at sample {i}: {status}")
    repeat = np.empty((1, 2), dtype=np.float32)
    assert runtime.b2_decoder_run(inputs[0], repeat) == 0
    reset_passed = bool(np.array_equal(repeat, outputs[0]))
    runtime.b2_decoder_destroy()
    np.save(validation / "generated_c_outputs.npy", outputs)
    comparison = base.error(reference, outputs)
    report = {"status": "passed" if comparison["passed"] and reset_passed else "failed",
              "samples": len(inputs), "split": "128 training plus 128 validation; no held-out test selection",
              "comparison": comparison, "independent_window_reset": reset_passed,
              "activation_bytes": activation_bytes, "host_compile_command": [str(a) for a in host_command],
              "generated_model_c_sha256": base.sha(generated / "b2_mingru.c"),
              "packed_weights_c_sha256": base.sha(adapter / "b2_packed_weights.c"),
              "board_flashed": False, "board_latency_measured": False}
    base.save_json(validation / "host_validation.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not comparison["passed"] or not reset_passed:
        raise RuntimeError("Actual generated C failed parity; package is NOT validated")

    arm_dir = validation / "arm_objects"
    arm_dir.mkdir(exist_ok=True)
    commands = []
    sources = sorted(generated.glob("*.c")) + sorted(adapter.glob("*.c"))
    for source in sources:
        args = [ARM, "-mcpu=cortex-m7", "-mthumb", "-mfpu=fpv5-d16", "-mfloat-abi=hard",
                "-std=c11", "-Os", "-Wall", "-Wextra", "-ffunction-sections", "-fdata-sections",
                f"-I{generated}", f"-I{inc}", f"-I{adapter}", "-c", source, "-o", arm_dir / f"{source.stem}.o"]
        command(args, arm_dir / f"{source.stem}.log")
        commands.append([str(a) for a in args])
    base.save_json(validation / "arm_compile.json", {"status": "passed", "commands": commands,
                   "compiler": str(ARM), "scope": "Cortex-M7 hard-float object compilation; no firmware link or flash"})
    print(f"Cortex-M7 compile passed: {len(sources)} translation units", flush=True)
    # Keep loader-directory handles alive until C inference has finished.
    for handle in handles:
        handle.close()


if __name__ == "__main__":
    main()
