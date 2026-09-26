"""Phase-13 training core with the user's faster validation stopping criterion.

Optimizer, loss, batching, scheduler and best-checkpoint selection are unchanged.
protocol.py remains the frozen original; tests compare the two ASTs explicitly.
"""

import math
from typing import Any

import numpy as np

from .protocol import FOLD_COUNT, FOLD_SEED, evaluate_last, rolling_batch
from .stopping import ValidationPlateau


def fit_model(
    model, data, fold, fold_data, args, device, learning_rate, encoder_lr_scale
):
    """Phase-13 fitting with only the explicitly authorized stopping predicate replaced."""
    import torch

    seed = FOLD_SEED
    stopping = ValidationPlateau()
    recurrent_parameters = []
    encoder_parameters = []
    for name, parameter in model.named_parameters():
        recurrent = name.startswith("gru.") or name.startswith("head.")
        parameter.requires_grad = args.train_scope == "all" or recurrent
        if not parameter.requires_grad:
            continue
        (recurrent_parameters if recurrent else encoder_parameters).append(parameter)
    parameter_groups = [
        {"params": recurrent_parameters, "lr": learning_rate, "name": "gru_head"}
    ]
    if encoder_parameters:
        parameter_groups.append(
            {
                "params": encoder_parameters,
                "lr": learning_rate * encoder_lr_scale,
                "name": "encoder_tcn",
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    rng = np.random.default_rng(seed)
    normalized_target = (
        (fold_data.velocity - fold_data.target_mean) / fold_data.target_std
    ).astype(np.float32)
    best_state = None
    best_epoch = 0
    best_validation_loss = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    print(
        f"\n=== {data.spec.name} | fold {fold + 1}/{FOLD_COUNT} ===\n"
        f"bins train/validation/test={len(fold_data.train_bins):,}/"
        f"{len(fold_data.validation_bins):,}/{len(fold_data.test_bins):,} | "
        f"init={args.init} | scope={args.train_scope}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(fold_data.train_bins)
        error_sum = 0.0
        value_count = 0
        gradient_sum = 0.0
        gradient_max = 0.0
        batch_count = 0
        for left in range(0, len(order), args.batch_size):
            bins = order[left : left + args.batch_size]
            inputs = torch.from_numpy(
                rolling_batch(fold_data.normalized_features, bins)
            ).to(device)
            targets = torch.from_numpy(normalized_target[bins]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)[:, -1]
            loss = torch.mean((prediction - targets) ** 2)
            loss.backward()
            gradient = float(
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for group in parameter_groups
                        for parameter in group["params"]
                    ],
                    args.gradient_clip,
                )
            )
            optimizer.step()
            error_sum += float(loss.detach()) * len(bins)
            value_count += len(bins)
            gradient_sum += gradient
            gradient_max = max(gradient_max, gradient)
            batch_count += 1

        validation = evaluate_last(
            model,
            fold_data.normalized_features,
            fold_data.velocity,
            fold_data.validation_bins,
            fold_data.target_mean,
            fold_data.target_std,
            device,
            args.batch_size,
        )
        improved = float(validation["normalized_loss"]) < best_validation_loss
        if improved:
            best_validation_loss = float(validation["normalized_loss"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        row = {
            "session": data.spec.name,
            "subject": data.spec.subject,
            "fold": fold + 1,
            "epoch": epoch,
            "gru_head_lr": float(optimizer.param_groups[0]["lr"]),
            "encoder_tcn_lr": (
                float(optimizer.param_groups[1]["lr"])
                if len(optimizer.param_groups) > 1
                else 0.0
            ),
            "optimization_loss": error_sum / max(value_count, 1),
            "gradient_mean_before_clip": gradient_sum / max(batch_count, 1),
            "gradient_max_before_clip": gradient_max,
            "validation_loss": validation["normalized_loss"],
            "validation_r2": validation["r2_mean"],
            "best": improved,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{args.epochs} | train={row['optimization_loss']:.5f} | "
            f"val={row['validation_loss']:.5f} | val R2={row['validation_r2']:+.4f} | "
            f"grad={row['gradient_mean_before_clip']:.3f}/"
            f"{row['gradient_max_before_clip']:.3f}" + (" *best*" if improved else ""),
            flush=True,
        )
        scheduler.step()
        if stopping.update(epoch, float(validation["normalized_loss"])):
            print(
                f"fast stop at epoch {epoch}: validation improvement below 0.5% for 3 epochs",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    return best_state, best_epoch, history, best_validation_loss
