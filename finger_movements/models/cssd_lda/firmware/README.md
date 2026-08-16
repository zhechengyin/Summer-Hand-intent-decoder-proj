# FingerMovements CSSD + LDA firmware port

This directory contains a self-contained C99 inference port of the frozen
Phase 2c 400 ms causal checkpoint. It does not train, fit, normalize, or load
files at runtime. The generated constants are compiled into Flash, and all
mutable stream state is supplied by the caller.

## Frozen behavior

- task: left (`0`) versus right (`1`) finger-movement classification;
- input: 28 simultaneous EEG channels at 100 Hz;
- update contract: five new samples per channel every 50 ms;
- causal history: 40 samples / 400 ms;
- cold start: 50 samples / 500 ms, including 100 ms filter pre-roll;
- output: class, LDA decision score, and right-class probability;
- checkpoint SHA-256:
  `87b84cc2c8baf9efdc1ccf37ad28f5f58ad13c4db2a8f8a273fe73fce9956101`.

The LDA scaler and coefficient arrays are algebraically folded into one weight
vector and bias per branch. This is an inference-only simplification; no model
parameter is relearned.

## Files

```text
include/fm_cssd_lda.h             public constants, state, and streaming API
src/fm_cssd_lda.c                 causal SOS, ring buffer, features, and LDAs
src/fm_cssd_lda_params.c          generated float32 checkpoint constants
src/fm_cssd_lda_params.h          generated internal declarations
tools/export_checkpoint.py        deterministic NPZ-to-C exporter
tools/validate_firmware.py        host compiler and Python/C equivalence test
example/example_main.c            minimal acquisition-loop integration example
CMakeLists.txt                    optional host/embedded CMake library target
```

Only the two `.c` files and the public header are needed by firmware. The
generated private header remains beside the source files.

## Input contract

Every sample must contain channels in this exact order:

```text
F3, F1, Fz, F2, F4, FC5, FC3, FC1, FCz, FC2, FC4, FC6,
C5, C3, C1, Cz, C2, C4, C6, CP5, CP3, CP1, CPz, CP2,
CP4, CP6, O1, O2
```

The values must use the same referencing, scale, and upstream acquisition
convention as the official training data. The C port cannot compensate for a
different electrode montage, reference electrode, gain, ADC scale, sample
rate, or channel order.

`fm_cssd_lda_push_sample()` accepts one 28-value sample. The block API accepts
sample-major interleaved values:

```text
samples[time * 28 + channel]
```

Call `fm_cssd_lda_reset()` with the first sample of a new stream. Reset uses it
only to initialize the causal Butterworth states; it does not consume it, so
the same sample must be passed again as the first pushed sample. Do not reset
between steady-state 50 ms updates.

## STM32 integration

1. Add `src/fm_cssd_lda.c` and `src/fm_cssd_lda_params.c` to the STM32 project.
2. Add `firmware/include/` to the compiler include paths.
3. Allocate one `fm_cssd_lda_state_t` statically or globally.
4. Acquire all 28 channels at exactly 100 Hz in the frozen order.
5. Reset once at stream start, then push each sample or a five-sample block.
6. Ignore `FM_CSSD_LDA_WARMING_UP`; consume output only when the return value is
   `FM_CSSD_LDA_PREDICTION_READY`.
7. Link the C math library because the probability output uses `expf()`.

The state occupies 10,312 bytes (10.07 KiB) with a conventional 32-bit C ABI.
The core generated numeric parameters occupy 221 float32 values plus 19 trend
indices: 903 bytes before linker alignment and optional channel-name/checksum
metadata. Prediction uses about 228 bytes of explicit temporary feature arrays
on the call stack; final stack and Flash usage depend on compiler and linker.

## Regeneration and verification

From the repository root:

```bash
python finger_movements/models/cssd_lda/firmware/tools/export_checkpoint.py
python finger_movements/models/cssd_lda/firmware/tools/validate_firmware.py
```

The exporter refuses a checkpoint whose SHA-256 differs from the frozen Phase
2c checkpoint. The validator compiles with C99, warnings-as-errors, and checks
all 316 official TRAIN cases. Current verified result:

- Python/C prediction mismatches: `0 / 316`;
- one 500 ms block versus ten 50 ms C chunks: `0 / 316` mismatches;
- maximum absolute score error: `1.335e-4`;
- maximum absolute probability error: `3.228e-5`.

These checks establish host-side float32 equivalence. They do not yet establish
cycle count, Flash layout, stack high-water mark, continuous-stream behavior,
or electrode/ADC compatibility on a target STM32. Those require the selected
board, compiler flags, acquisition frontend, and real continuous EEG.
