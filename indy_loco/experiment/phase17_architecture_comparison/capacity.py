"""Analytical Phase-17 design budgets only: no model imports or training.

Run from any directory with Python to print the proposed capacity report.
Counts are conditional on the exact layers/biases described in README.md;
implementation must subsequently verify them with model.parameters().
"""

import json


def linear(inputs, outputs, bias=True):
    return inputs * outputs + (outputs if bias else 0)


def report():
    width = 64
    stem = linear(192, width)
    affine_norm = 2 * width
    head = linear(width, 2)
    shared = stem + affine_norm + head
    baseline_encoder = stem + affine_norm + 4 * (width * width * 3 + width)
    baseline_gru = 3 * (width * width + width * width + 2 * width)
    baseline = baseline_encoder + baseline_gru + head

    mingru_block = linear(width, 2 * width)
    mingru_ffn = linear(width, 128) + linear(128, width)
    mingru = shared + 3 * (mingru_block + mingru_ffn + 2 * affine_norm)

    inner, state, head_dim, groups, conv = 128, 64, 16, 1, 4
    heads = inner // head_dim
    projected = 2 * inner + 2 * groups * state + heads
    conv_channels = inner + 2 * groups * state
    mamba_core = (
        linear(width, projected, bias=False)
        + conv_channels * (conv + 1)
        + 3 * heads  # dt_bias, A_log, D (D_has_hdim=False)
        + inner  # gated RMSNorm scale, no bias
        + linear(inner, width, bias=False)
    )
    mamba = shared + 2 * (mamba_core + affine_norm)

    attention = linear(width, 3 * width) + linear(width, width)
    transformer_ffn = linear(width, 160) + linear(160, width)
    transformer = shared + 2 * (attention + transformer_ffn + 2 * affine_norm)

    rows = []
    for name, parameters in (
        ("midsize_tcn_gru", baseline),
        ("mingru_3x64_ff128", mingru),
        ("mamba2_2x64_state64", mamba),
        ("causal_transformer_2x64_ff160", transformer),
    ):
        rows.append(
            {
                "architecture": name,
                "parameters": parameters,
                "ratio_to_midsize": parameters / baseline,
                "fp32_weight_bytes": parameters * 4,
                "fp32_weight_kib": parameters * 4 / 1024,
                "status": "existing_baseline"
                if parameters == baseline
                else "implemented_untrained",
            }
        )
    return {
        "phase": 17,
        "status": "implemented_no_training",
        "count_method": "analytical layer shapes, not instantiated candidate models",
        "input_shape": [192, 50],
        "output_timestep": 49,
        "capacity": rows,
        "notes": [
            "Weights only; no activations, optimizer state, workspace or firmware.",
            "This analytical script runs no forward pass, training or test evaluation.",
            "Mamba-2: two blocks, expand=2, d_state=64, headdim=16, ngroups=1, d_conv=4.",
            "Proposed models replace the TCN and GRU; all use the same 192-feature input.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
