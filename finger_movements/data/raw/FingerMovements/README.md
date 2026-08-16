# FingerMovements official raw data

This directory contains the official 100 Hz release of BCI Competition II,
Data Set IV (`self-paced 1s`). The previous UEA `.ts` and `.arff` conversion
files were removed because their channel layout did not match the official
`time x channels x trials` representation.

## Files

- `sp1s_aa.mat`: official 100 Hz MATLAB data containing `clab`, `x_train`,
  `y_train`, and `x_test`.
- `labels_data_set_iv.txt`: official labels for the 100 competition test
  trials. These labels are reserved for final locked-test evaluation and must
  not be used for model or preprocessing selection.

## Official sources

- Dataset description: https://www.bbci.de/competition/ii/berlin_desc.html
- MATLAB data: https://www.bbci.de/competition/download/competition_ii/berlin/sp1s_aa.mat
- Test labels: https://www.bbci.de/competition/ii/results/labels_data_set_iv.txt

## SHA-256

- `sp1s_aa.mat`: `4ecb9f7bce25a67d71ade1bca68a103ba93f4173fc6e8426bc14aa1dade69f5c`
- `labels_data_set_iv.txt`: `9ae9c41f237c9445ad749fc5539c2861c21a14195ecf308c8ccfdeb92f296c65`

Expected MATLAB shapes:

- `x_train`: `(50, 28, 316)` = time x channels x trials
- `y_train`: `(1, 316)`
- `x_test`: `(50, 28, 100)` = time x channels x trials

