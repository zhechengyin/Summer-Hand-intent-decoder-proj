"""Predeclared numerical gates, executed only by an explicit user command."""
import torch
import torch.nn.functional as F

from indy_loco.experiment.phase16_parameter_scaling import protocol
from .model import create_model, optimizer_for
from .ops import GRUGates, LayerNormReLU, ResidualReLU

# These are numerical tolerances, not a claim of bitwise equivalence.
TOLERANCES = {
    "output": {"atol": 2e-5, "rtol": 2e-4},
    "gradient": {"atol": 3e-5, "rtol": 3e-4},
    "update": {"atol": 2e-6, "rtol": 2e-5},
}


def compare(reference, candidate, name, kind, rows):
    if reference.shape != candidate.shape:
        raise AssertionError(f"{name}: shapes differ")
    difference = (reference.detach() - candidate.detach()).abs()
    tolerance = TOLERANCES[kind]
    bound = tolerance["atol"] + tolerance["rtol"] * reference.detach().abs()
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    passed = finite and bool((difference <= bound).all())
    rows.append({
        "name": name, "kind": kind, "shape": list(reference.shape),
        "max_absolute_error": float(difference.max()) if finite else None,
        "max_tolerance_ratio": float((difference / bound).max()) if finite else None,
        "passed": passed,
    })
    if not passed:
        raise AssertionError(f"{name}: {rows[-1]}")


def pair_operator(reference_fn, candidate_fn, inputs, label, rows):
    left = [v.detach().clone().requires_grad_(True) for v in inputs]
    right = [v.detach().clone().requires_grad_(True) for v in inputs]
    a, b = reference_fn(*left), candidate_fn(*right)
    compare(a, b, label + "/output", "output", rows)
    upstream = torch.randn_like(a)
    ga = torch.autograd.grad(a, left, upstream)
    gb = torch.autograd.grad(b, right, upstream)
    for i, (x, y) in enumerate(zip(ga, gb, strict=True)):
        compare(x, y, f"{label}/gradient_{i}", "gradient", rows)


def operator_checks(variant, rows):
    protocol.seed_everything(4301, mps=False)
    if variant in ("epilogue", "causal", "combined"):
        for batch, time in ((1, 1), (7, 17), (128, 50)):
            for padding in (0, 2, 4, 8, 16):
                c = torch.randn(batch, 64, time + padding, device="cuda")
                x = torch.randn(batch, 64, time, device="cuda")
                pair_operator(lambda c, x: F.relu(c[:, :, :x.shape[-1]] + x),
                              ResidualReLU.apply, [c, x],
                              f"residual/B{batch}T{time}P{padding}", rows)
        # ReLU derivative at exactly zero must remain zero.
        z = torch.zeros(2, 64, 3, device="cuda")
        pair_operator(lambda c, x: F.relu(c + x), ResidualReLU.apply,
                      [z, z], "residual/zero", rows)
    if variant in ("layernorm", "combined"):
        for batch, time, scale in ((1, 1, 0), (7, 17, 1), (128, 50, 1), (3, 50, 12)):
            x = torch.randn(batch, 64, time, device="cuda") * scale
            weight = torch.randn(64, device="cuda")
            bias = torch.randn(64, device="cuda")
            pair_operator(
                lambda x, w, b: F.relu(F.layer_norm(x.transpose(1, 2), (64,), w, b, 1e-5)).transpose(1, 2),
                LayerNormReLU.apply, [x, weight, bias],
                f"norm/B{batch}T{time}S{scale}", rows)
    if variant in ("gru", "combined"):
        def reference(u, v, previous):
            ur, uz, un = u.chunk(3, -1)
            vr, vz, vn = v.chunk(3, -1)
            r, z = (ur + vr).sigmoid(), (uz + vz).sigmoid()
            n = (un + r * vn).tanh()
            return (1 - z) * n + z * previous
        for batch, scale in ((1, 0), (7, 1), (128, 1), (3, 12)):
            u = torch.randn(batch, 192, device="cuda") * scale
            v = torch.randn_like(u) * scale
            h = torch.randn(batch, 64, device="cuda")
            pair_operator(reference, GRUGates.apply, [u, v, h],
                          f"gates/B{batch}S{scale}", rows)


def model_case(variant, state, inputs, training, label, rows, update=False, target=None):
    baseline = create_model("pytorch", state).cuda().train(training)
    candidate = create_model(variant, state).cuda().train(training)
    left = inputs.detach().clone().requires_grad_(True)
    right = inputs.detach().clone().requires_grad_(True)
    protocol.seed_everything(4311, mps=False)
    a = baseline(left)
    protocol.seed_everything(4311, mps=False)
    b = candidate(right)
    compare(a, b, label + "/full_sequence_output", "output", rows)
    # Native cuDNN GRU does not retain the backward reserve in evaluation mode.
    # Eval checks are forward-only; gradients are checked separately in train mode.
    if not training:
        return
    if update:
        # Match the real last-step MSE objective, including clipping and AdamW.
        if target is None:
            raise ValueError("Real-batch AdamW verification requires real normalized targets")
        la = (a[:, -1] - target).square().mean()
        lb = (b[:, -1] - target).square().mean()
        la.backward()
        lb.backward()
    else:
        upstream = torch.randn_like(a)
        a.backward(upstream)
        b.backward(upstream)
    compare(left.grad, right.grad, label + "/input_gradient", "gradient", rows)
    aa, bb = dict(baseline.named_parameters()), dict(candidate.named_parameters())
    if aa.keys() != bb.keys():
        raise AssertionError("Parameter names changed")
    for name in aa:
        if aa[name].grad is None or bb[name].grad is None:
            raise AssertionError(f"Missing gradient: {name}")
        compare(aa[name].grad, bb[name].grad, label + "/gradient/" + name, "gradient", rows)
    if update:
        oa, ob = optimizer_for(baseline), optimizer_for(candidate)
        for model in (baseline, candidate):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        oa.step()
        ob.step()
        for name in aa:
            compare(aa[name], bb[name], label + "/AdamW/" + name, "update", rows)


def check_baseline(state, real_inputs):
    """Verify the training wrapper's eval path against the packaged Midsize."""
    from indy_loco.models.midsize.model import MidsizeTCNGRU

    standalone = MidsizeTCNGRU().cuda().eval()
    standalone.load_state_dict(state, strict=True)
    training_wrapper = create_model("pytorch", state).cuda().eval()
    rows = []
    with torch.inference_mode():
        compare(standalone(real_inputs), training_wrapper(real_inputs),
                "canonical_deployment_vs_training_wrapper", "output", rows)
    if sum(p.numel() for p in training_wrapper.parameters()) != 86978:
        raise AssertionError("Baseline parameter count changed")
    return {"status": "passed", "checks": rows}


def check_variant(variant, state, real_inputs, real_targets):
    rows = []
    try:
        operator_checks(variant, rows)
        for batch, time, training in ((1, 1, False), (7, 17, False), (128, 50, False),
                                       (1, 1, True), (7, 17, True), (7, 50, True)):
            protocol.seed_everything(4317, mps=False)
            values = torch.randn(batch, 192, time, device="cuda")
            model_case(variant, state, values, training,
                       f"model/B{batch}T{time}/train{training}", rows)
        model_case(variant, state, real_inputs, True, "real_batch", rows,
                   update=True, target=real_targets)
        torch.cuda.synchronize()
        return {"status": "passed", "checks": rows, "tolerances": TOLERANCES}
    except AssertionError as error:
        # Numerical rejection is an experimental result. CUDA/runtime errors are
        # deliberately NOT swallowed: an unhealthy GPU context must stop the run.
        return {"status": "rejected", "error": str(error), "checks": rows,
                "tolerances": TOLERANCES}
