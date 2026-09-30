"""Phase-13 sampling/loss/selection, with explicit Phase-17 LR groups."""

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .data_contract import protocol
from .models import parameter_groups


@dataclass(frozen=True)
class Training:
    initialization: str = "scratch"
    train_scope: str = "all"
    epochs: int = 20
    patience: int = 6
    batch_size: int = 128
    weight_decay: float = 0.025
    gradient_clip: float = 1.0
    seed: int = 43
    learning_rate_gru_head: float = 3e-4
    encoder_lr_scale: float = 0.25


TRAINING = Training()


def verify_recipe(lock):
    actual = asdict(TRAINING)
    for key, value in lock["training"].items():
        if key not in ("initialization", "device") and actual[key] != value:
            raise ValueError(f"Frozen training recipe changed: {key}")


def fit_model(model, name, session, fold, fold_data, device, epoch_path):
    """No test evaluation. Restore minimum-validation-loss weights before return."""
    groups = parameter_groups(model, name)
    optimizer = torch.optim.AdamW(groups, weight_decay=TRAINING.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, TRAINING.epochs)
    rng = np.random.default_rng(TRAINING.seed)
    targets = (
        (fold_data.velocity - fold_data.target_mean) / fold_data.target_std
    ).astype(np.float32)
    best_state, best_epoch, best_loss, stale = None, 0, math.inf, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, TRAINING.epochs + 1):
        model.train()
        order = rng.permutation(fold_data.train_bins)
        error_sum, value_count, gradient_sum, gradient_max, batches = (
            0.0,
            0,
            0.0,
            0.0,
            0,
        )
        for left in range(0, len(order), TRAINING.batch_size):
            bins = order[left : left + TRAINING.batch_size]
            inputs = torch.from_numpy(
                protocol.rolling_batch(fold_data.normalized_features, bins)
            ).to(device)
            target = torch.from_numpy(targets[bins]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)[:, -1]
            loss = (prediction - target).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite loss: {session} fold {fold} epoch {epoch}"
                )
            loss.backward()
            gradient = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), TRAINING.gradient_clip, error_if_nonfinite=True
                )
            )
            optimizer.step()
            error_sum += float(loss.detach()) * len(bins)
            value_count += len(bins)
            gradient_sum += gradient
            gradient_max = max(gradient_max, gradient)
            batches += 1
        validation = protocol.evaluate_last(
            model,
            fold_data.normalized_features,
            fold_data.velocity,
            fold_data.validation_bins,
            fold_data.target_mean,
            fold_data.target_std,
            device,
            TRAINING.batch_size,
        )
        current = float(validation["normalized_loss"])
        if not math.isfinite(current):
            raise FloatingPointError(
                "Nonfinite validation loss; no checkpoint selected"
            )
        improved = current < best_loss
        if improved:
            best_loss, best_epoch, stale = current, epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        row = {
            "session": session,
            "fold": fold,
            "epoch": epoch,
            "temporal_head_lr": float(optimizer.param_groups[0]["lr"]),
            "encoder_stem_lr": float(optimizer.param_groups[1]["lr"]),
            "optimization_loss": error_sum / value_count,
            "gradient_mean_before_clip": gradient_sum / batches,
            "gradient_max_before_clip": gradient_max,
            "validation_loss": current,
            "validation_r2": validation["r2_mean"],
            "best": improved,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        protocol.write_json_atomic(epoch_path, history)
        print(
            f"{name} {session} fold {fold} epoch {epoch:02d}/20 | train={row['optimization_loss']:.5f} val={current:.5f} val_R2={validation['r2_mean']:+.4f}"
            + (" *best*" if improved else ""),
            flush=True,
        )
        scheduler.step()
        if stale >= TRAINING.patience:
            break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    return {
        "model_state": best_state,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
        "training_seconds": time.perf_counter() - started,
    }
