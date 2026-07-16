# Models and artifacts

Reusable architecture code now has a stable import surface at
`src/intent_decoder/model/`. This directory stores model-family references and
trained artifacts.

| Directory | Status |
| --- | --- |
| `tcn_gru/` | Readable shared implementation plus historical 96-channel reference. |
| `tcn_gru_8ch/` | Historical 8-channel checkpoint; network is unidirectional, but saved preprocessing used centered Gaussian input smoothing. |
| `indy_32ch/` | Reserved candidate location; no promoted checkpoint yet. |

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
