"""Train the fixed Phase-17 causal Transformer on the original 30-fold protocol."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from indy_loco.experiment.phase17_architecture_comparison.run import main  # noqa: E402

if __name__ == "__main__":
    main(fixed_model="transformer")
