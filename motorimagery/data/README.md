# BCI Competition III Dataset I

This directory contains the data used by the `motorimagery` pipeline. The
dataset records two-class motor imagery from a single participant using an
8-by-8 ECoG electrode grid over the right motor cortex. The two tasks are
imagined movement of the left little finger and imagined tongue movement.

## Official files

| File | Contents | Expected shape |
|---|---|---|
| `Competition_train.mat.gz` | Training ECoG (`X`) and labels (`Y`) | `X=(278,64,3000)`, `Y=(278,1)` |
| `Competition_test.mat.gz` | Official unlabeled test ECoG (`X`) | `X=(100,64,3000)` |
| `true_labels.txt` | Labels released after the competition | 100 labels |

The arrays use this order:

```text
trial × channel × sample
```

Each trial contains 64 channels and 3000 samples recorded at 1000 Hz, giving
three seconds of ECoG. Labels are encoded as `-1` and `+1`. TRAIN contains 139
trials from each class; the released TEST labels contain 50 from each class.

`Competition_train.mat` is an optional uncompressed copy of the training
artifact. The pipeline accepts either `.mat` or `.mat.gz`; the compressed file
is used by the documented commands. Files beginning with `._` are macOS
filesystem metadata and are not dataset inputs.

## Download

From `motorimagery/`, run:

```bash
python download_data.py --out data
```

The downloader retrieves the official files from the BCI Competition III
website. These data remain subject to the original dataset terms and should
not be treated as project-generated artifacts.

## Preprocessing contract

The source files remain at 1000 Hz. During a pipeline run,
`data.downsample_ecog()` applies anti-aliased polyphase resampling along the
sample axis:

```text
(trials, 64, 3000) at 1000 Hz
              ↓
(trials, 64, 300) at 100 Hz
```

Feature extraction then operates on the 100 Hz representation. No processed
arrays are written into this directory; feature caches are stored in the
chosen `runs/<name>/` output directory.

## Evaluation boundary

The 278 training trials come from the first recording session. The 100 test
trials were recorded approximately one week later, so substantial
between-session signal and power differences are possible.

Use TRAIN only for:

- feature or preprocessing choices;
- channel ranking and wrapper selection;
- classifier hyperparameter selection;
- ANN/CNN validation and early stopping.

`true_labels.txt` must be used only for final retrospective evaluation. It
must not be used to choose features, channels, normalization, model settings,
or training epochs.

## Example

```bash
python run_pipeline.py \
  --train data/Competition_train.mat.gz \
  --test data/Competition_test.mat.gz \
  --labels data/true_labels.txt \
  --output runs/mst_svm \
  --features mst \
  --classifier svm \
  --use-paper-channels
```
