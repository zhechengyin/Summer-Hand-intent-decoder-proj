"""Fold-matched Midsize parameter reuse, with explicit GRU gates and causal lags."""

import torch

from .model import BASELINE, ScaledTCNGRU


def transfer_midsize_weights(model, source_state):
    """Copy compatible weights; leave every new parameter at its default init.

    This is a partial warm start, not a function-preserving network morphism:
    widening changes LayerNorm statistics and introduces new connections.
    The caller validates the session/fold, source SHA and preprocessing first.
    """
    with torch.device("meta"):
        expected = ScaledTCNGRU(BASELINE).state_dict()
    if set(source_state) != set(expected):
        raise ValueError(
            "Transfer source must have the complete original Midsize state"
        )
    for name, value in source_state.items():
        if value.shape != expected[name].shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid Midsize source tensor: {name}")
    destination = model.state_dict()
    records = {}
    with torch.no_grad():
        for name, source in source_state.items():
            target = destination[name]
            source = source.to(device=target.device, dtype=target.dtype)
            copied = 0
            if name.startswith("gru."):
                # GRU packs reset/update/new gates. Never copy 3*64 rows into
                # the first 192 rows of a 3*80 tensor; each gate has a new offset.
                hidden = model.architecture.decoder_hidden_size
                for gate in range(3):
                    src_rows = slice(gate * 64, (gate + 1) * 64)
                    dst_rows = slice(gate * hidden, gate * hidden + 64)
                    if source.ndim == 2:
                        target[dst_rows, : source.shape[1]].copy_(source[src_rows])
                    else:
                        target[dst_rows].copy_(source[src_rows])
                copied = source.numel()
            elif name.startswith("convolutions.") and name.endswith(".weight"):
                layer = int(name.split(".")[1])
                old_dilation = BASELINE.encoder_dilations[layer]
                new_dilation = model.architecture.encoder_dilations[layer]
                new_kernel = model.architecture.encoder_kernel_size
                for tap in range(3):
                    lag = (2 - tap) * old_dilation
                    if lag % new_dilation == 0 and lag // new_dilation < new_kernel:
                        dst_tap = new_kernel - 1 - lag // new_dilation
                        target[:64, :64, dst_tap].copy_(source[:, :, tap])
                        copied += source[:, :, tap].numel()
            else:
                slices = tuple(slice(0, size) for size in source.shape)
                target[slices].copy_(source)
                copied = source.numel()
            records[name] = {
                "source_parameters": source.numel(),
                "copied_parameters": copied,
            }
    copied = sum(row["copied_parameters"] for row in records.values())
    total = sum(p.numel() for p in model.parameters())
    return {
        "policy": "phase13_same_session_same_fold_overlap_v1",
        "source_parameters": sum(t.numel() for t in source_state.values()),
        "copied_parameters": copied,
        "new_random_parameters": total - copied,
        "unmapped_source_parameters": sum(t.numel() for t in source_state.values())
        - copied,
        "copied_fraction": copied / total,
        "all_parameters_trainable": all(p.requires_grad for p in model.parameters()),
        "function_preserving": model.architecture == BASELINE,
        "tensors": records,
    }
