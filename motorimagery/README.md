# Motor-imagery ECoG pipeline

The pipeline supports combination of feature extractors and classifiers.

## Feature extractors

- `mst`: modified S-transform, 1--35 Hz; 35 features/channel.
- `bandpower` (or alias `psd`): Welch PSD integrated into delta, theta, alpha, beta, low-gamma; 5 features/channel.
- `fft`: one-sided Hann-windowed Fourier power spectrum from 1--35 Hz. With 3 s at 100 Hz this gives 103 features/channel.
- `filterbank`: 4th-order Butterworth band-pass filters for the same five bands, followed by mean-square power; 5 features/channel.

Bandpower, FFT and filter-bank features use `10*log10(power)` by default. Pass `--linear-power` to use raw power.

## Classifiers

- `--classifier svm`
- `--classifier ann` for `input -> FC(64) -> ReLU -> FC(2)`
- `--classifier cnn` for a tiny spectral Conv1d network
- `--classifier tree` for a decision tree with `max_depth=4`

Classifier implementations are organized under `models/`:

```text
models/
├── svm.py
├── ann.py
├── cnn.py
└── tree.py
```

## Examples

All 64 channels, band-power features, ANN:

```bash
python3 run_pipeline.py \
  --train data/Competition_train.mat.gz \
  --test data/Competition_test.mat.gz \
  --labels data/true_labels.txt \
  --output runs/bandpower_ann \
  --features bandpower \
  --classifier ann \
  --use-all-channels
```

FFT + SVM + automatic wrapper selection, stopping at 32 channels:

```bash
python3 run_pipeline.py \
  --train data/Competition_train.mat.gz \
  --test data/Competition_test.mat.gz \
  --labels data/true_labels.txt \
  --output runs/fft_svm \
  --features fft \
  --classifier svm \
  --minimum-channels 32
```