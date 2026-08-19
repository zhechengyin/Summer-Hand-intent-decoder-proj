# Active Indy Loco results

`phase9_deployment_policy_replay/` contains the software-only replay of two
causal cold-start policies for the promoted Phase 6 checkpoint. December
validation selected the continuous rolling calibration-seeded past-window
(pooled R² `0.7526`) over block reset (`0.7021`). The frozen winner then reached
pooled January R² `0.7277`. CSV files contain session and aggregate metrics;
JSON records protocol/selection metadata; traces and the NPZ file provide
representative predictions and golden vectors for later firmware parity tests.

`phase8_future_lookahead_fivefold/` contains the completed Indy lookahead
comparison. The 48 ms and 100 ms conditions reached test R²
`0.7576 ± 0.0396` and `0.7554 ± 0.0397`, respectively. The Loco extension will
write separately to `phase8_loco_future_lookahead_fivefold/`.

Completed Phase 6 and Phase 7 outputs are archived under
`../history/results/indy/`. Phase 7's overall test R² was
`0.7056 ± 0.0722`.
