"""Independent algebra checks and mandatory CUDA gates before benchmarking."""

import torch
from torch.nn import functional as F

from .kernel import conservative_scan, install


def reference(projection, last_only=False):
    a, b = projection.chunk(2, dim=-1)
    logg = torch.where(a >= 0, (a.clamp_min(0) + 0.5).log(), -F.softplus(-a))
    prefix = (-F.softplus(b)).cumsum(1)
    terms = -F.softplus(-b) + logg - prefix
    if last_only:
        return (prefix[:, -1:] + torch.logsumexp(terms, 1, keepdim=True)).exp()
    return (prefix + torch.logcumsumexp(terms, 1)).exp()


def serial_log_reference(projection):
    a, b = projection.chunk(2, dim=-1)
    prefix = torch.zeros_like(a[:, 0])
    total, states = None, []
    for t in range(a.shape[1]):
        candidate, gate = a[:, t], b[:, t]
        logg = torch.where(
            candidate >= 0,
            (candidate.clamp_min(0) + 0.5).log(),
            -F.softplus(-candidate),
        )
        prefix = prefix - F.softplus(gate)
        term = -F.softplus(-gate) + logg - prefix
        total = term if total is None else torch.logaddexp(total, term)
        states.append((prefix + total).exp())
    return torch.stack(states, 1)


def manual_gradient(projection, states, gradient, last_only=False):
    a, b = projection.chunk(2, -1)
    ga, gb = torch.empty_like(a), torch.empty_like(b)
    carry = torch.zeros_like(a[:, 0])
    for t in reversed(range(a.shape[1])):
        q = (
            gradient[:, 0]
            if last_only and t == a.shape[1] - 1
            else (torch.zeros_like(carry) if last_only else gradient[:, t])
        )
        r = q + carry
        aa, bb = a[:, t], b[:, t]
        g = torch.where(aa >= 0, aa + 0.5, (-F.softplus(-aa)).exp())

        def dsp(x):
            return torch.where(x > 20, torch.ones_like(x), x.sigmoid())

        gp = torch.where(aa >= 0, torch.ones_like(aa), g * dsp(-aa))
        decay, write = (-F.softplus(bb)).exp(), (-F.softplus(-bb)).exp()
        previous = states[:, t - 1] if t else torch.zeros_like(carry)
        ga[:, t] = r * write * gp
        gb[:, t] = r * (write * g * dsp(-bb) - decay * previous * dsp(bb))
        carry = r * decay
    return torch.cat([ga, gb], -1)


def cpu_checks():
    torch.manual_seed(43)
    checked = 0
    for t in (1, 2, 50):
        for scale in (1.0, 12.0):
            for last in (False, True):
                p = (torch.randn(2, t, 8, dtype=torch.float64) * scale).requires_grad_()
                y = reference(p, last)
                q = torch.randn_like(y)
                actual = torch.autograd.grad(y, p, q)[0]
                states = serial_log_reference(p)
                torch.testing.assert_close(states, reference(p), atol=1e-10, rtol=1e-9)
                expected = manual_gradient(p, states, q, last)
                torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
                checked += 1
    # Finite differences away from g's nondifferentiable zero and softplus threshold.
    p = (torch.rand(1, 3, 4, dtype=torch.float64) + 0.2).requires_grad_()
    torch.autograd.gradcheck(serial_log_reference, (p,), atol=1e-5, rtol=1e-3)
    return {
        "cpu_algebra_cases": checked,
        "double_precision_finite_difference": "passed",
        "cuda_execution_verified": False,
    }


def cuda_checks(create_model):
    torch.manual_seed(43)
    rows = []
    for b, t, d, scale in (
        (1, 1, 128, 1),
        (3, 2, 7, 1),
        (128, 50, 128, 1),
        (2, 50, 128, 12),
    ):
        for last in (False, True):
            p = (torch.randn(b, t, 2 * d, device="cuda") * scale).requires_grad_()
            y, z = reference(p, last), conservative_scan(p, last)
            q = torch.randn_like(y)
            gy = torch.autograd.grad(y, p, q)[0]
            gz = torch.autograd.grad(z, p, q)[0]
            torch.testing.assert_close(z, y, rtol=3e-4, atol=3e-5)
            torch.testing.assert_close(gz, gy, rtol=5e-4, atol=5e-5)
            rows.append(
                {
                    "shape": [b, t, d],
                    "scale": scale,
                    "last_only": last,
                    "output_max_abs": (z - y).abs().max().item(),
                    "gradient_max_abs": (gz - gy).abs().max().item(),
                }
            )
    import copy

    baseline = create_model().cuda()
    custom = install(copy.deepcopy(baseline))
    initial = copy.deepcopy(baseline.state_dict())
    for training in (False, True):
        baseline.load_state_dict(initial)
        custom.load_state_dict(initial)
        baseline.train(training)
        custom.train(training)
        baseline.zero_grad(set_to_none=True)
        custom.zero_grad(set_to_none=True)
        x = torch.randn(4, 192, 50, device="cuda")
        torch.manual_seed(43)
        y = baseline(x)
        y.square().mean().backward()
        torch.manual_seed(43)
        z = custom(x)
        z.square().mean().backward()
        torch.testing.assert_close(z, y, rtol=3e-4, atol=3e-5)
        for (name, a), (name2, b) in zip(
            baseline.named_parameters(), custom.named_parameters(), strict=True
        ):
            assert name == name2
            torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=5e-5)
        # Same first optimizer update, including parameter-group AdamW settings.
        from indy_loco.experiment.phase18_large_mingru.ab_study import train as ab

        trial = ab.plan.TRIALS["t00"]
        for model in (baseline, custom):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            torch.optim.AdamW(
                ab.optimizer_groups(model, "mingru_b", trial),
                weight_decay=trial["weight_decay"],
            ).step()
        for a, b in zip(baseline.parameters(), custom.parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=5e-4, atol=5e-5)
    torch.cuda.synchronize()
    return {
        "status": "passed",
        "kernel_cases": rows,
        "full_model_train_eval_gradients_and_adamw": "passed",
        "higher_order_gradients": "unsupported; not used",
    }
