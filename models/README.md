# Models and artifacts

Reusable architecture code now has a stable import surface at
`src/intent_decoder/model/`. This directory stores model-family references and
trained artifacts.

| Directory | Status |
| --- | --- |
| `indy_32ch/` | Reserved candidate location; no promoted checkpoint yet. |

The historical 96-channel and 8-channel model folders were deleted because
their preprocessing was not end-to-end causal. Their recorded state is preserved
in `docs/history/ARCHIVE_RETIREMENT.md`.

A promoted artifact directory must contain:

- checkpoint;
- exact configuration;
- source-session manifest/checksums;
- channel-selection rule and selected channels;
- input and target normalization state;
- validation protocol and small metrics JSON;
- int8 export result and hardware timing.

Do not treat a result from `experiments/` as the model of record until those
artifacts exist.

The only supported architecture implementation for new work is
`src/intent_decoder/model/tcn_gru.py`, which rejects bidirectional configuration.
