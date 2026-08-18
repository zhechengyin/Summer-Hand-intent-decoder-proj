# Raw Loco sessions

This directory stores original Loco MAT files from the ten-session collection
on [Zenodo record 3854034](https://zenodo.org/records/3854034).

Treat every MAT file as immutable. Published MD5 checksums are encoded in
`../../../processing/indy_loco/loco/prepare_loco_neurobench.py` and are checked
before conversion. The three sessions used by the official NeuroBench primate
reaching benchmark are:

- `loco_20170210_03.mat`
- `loco_20170215_02.mat`
- `loco_20170301_05.mat`

Those three files are downloaded and checksum-verified for Phase 7. The
download utility can resume the other seven sessions if a later robustness
study needs the complete collection. Files ending in `.partial` or
`.segment-*` are resumable download state, not valid raw sessions.
