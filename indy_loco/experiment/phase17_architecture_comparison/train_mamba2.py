"""Train the fixed Phase-17 Mamba-2 model; no mamba-ssm/Triton install needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from indy_loco.experiment.phase17_architecture_comparison.run import main  # noqa: E402

if __name__ == "__main__":
    main(fixed_model="mamba2")
