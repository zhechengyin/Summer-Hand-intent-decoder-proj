# Data directory

The OpenNeuro dataset **ds004022** is *not* committed to this repo (it is large and
CC0-licensed at the source). Download it into `data/ds004022/` so the tree looks
like:

```
data/ds004022/
├── dataset_description.json
├── participants.tsv
├── task_motorimagery_eeg.json
├── task_motorimagery_nirs.json
├── sub-01/
│   ├── eeg/   sub-01_task-motorimagery_run-{1,2,3}_eeg.set  (+ .fdt, .json, electrodes.tsv)
│   └── fnirs/ sub-01_task-motorimagery_run-{1,2,3}_nirs.mat (+ .json, electrodes.tsv)
├── sub-02/ ...
└── sub-07/
```

## How to download

**Option A — openneuro-py (Python):**
```bash
pip install openneuro-py
python -c "import openneuro; openneuro.download(dataset='ds004022', target_dir='data/ds004022')"
```

**Option B — AWS S3 (no credentials needed, bucket is public):**
```bash
aws s3 sync --no-sign-request s3://openneuro.org/ds004022 data/ds004022
```

**Option C — DataLad:**
```bash
datalad clone https://github.com/OpenNeuroDatasets/ds004022.git data/ds004022
datalad get -d data/ds004022 .    # fetch the annexed binary files
```

Helper wrapper: `python tools/download_data.py --target data/ds004022`

## Dataset facts (verified from the BIDS sidecars)

| Property            | EEG                              | fNIRS                                  |
|---------------------|----------------------------------|----------------------------------------|
| Format              | EEGLAB `.set` + `.fdt`           | BBCI toolbox `.mat` (`nirs_data`)      |
| Channels            | 18 (10-10 system, ref FCz)       | 8 sources × 4 detectors, 2 λ (760/850) |
| Sampling rate       | 500 Hz                           | 7.8125 Hz                              |
| Line noise          | 60 Hz                            | —                                      |
| Subjects            | 7 (orthopedic impairment)        | same                                   |
| Runs per subject    | 3                                | 3                                      |
| Trials per run      | 40 (10 per class)                | 40 (10 per class)                      |

**Trial timeline (15 s):** 3 s fixation → 4 s visual cue (reveals the action) →
3 s ready → **5 s motor imagery**.

### fNIRS raw signal note
In this dataset the fNIRS intensity matrix (`nirs_data.cnt.x`) is serialised as a
MATLAB **`table` (MCOS) object**, which `scipy.io.loadmat` / `h5py` cannot
deserialise directly. **The loader handles this automatically**:
`extract_mcos_double_columns` in `src/load_bids.py` recovers the channel columns
straight from the `.mat` file's `__function_workspace__` subsystem in pure Python
— no MATLAB/Octave required — so the real fNIRS and fused branches work out of the
box. `tools/convert_fnirs_octave.m` remains as an optional MATLAB/Octave fallback
(it writes `*_nirs_converted.mat`, which the loader prefers if present).
