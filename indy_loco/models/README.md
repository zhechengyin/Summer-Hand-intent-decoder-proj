# Canonical model packages

[`manifest.json`](manifest.json) is the authoritative index for six sessions
and two tiers. A valid checkout contains exactly:

```text
midsize/<session>/{checkpoint.pt,manifest.json}
large/<session>/{checkpoint.pt,memory.memlib,manifest.json}
```

Midsize is the best-test-fold neural checkpoint evaluated with continuous
rolling preprocessing and no memory correction. Large is the identical
checkpoint plus selected GRU residual memory; it is a system tier, not a larger
neural architecture.

Selection used test R² and biases absolute performance. Do not describe these
scores as an unbiased generalization estimate.

Run `python indy_loco/models/package_tools.py validate` before consuming or
publishing the packages. `package_tools.py build` regenerates manifests from
the frozen archive but does not retrain or change model parameters.
