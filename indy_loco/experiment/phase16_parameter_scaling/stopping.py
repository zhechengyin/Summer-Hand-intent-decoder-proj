"""User-requested fast stopping; checkpoint selection still uses exact val MSE."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StopCriterion:
    min_epochs: int = 4
    patience: int = 3
    relative_improvement: float = 0.005
    max_epochs: int = 20

    def metadata(self):
        return {
            "metric": "validation_normalized_mse",
            **asdict(self),
            "checkpoint_selection": "exact_minimum_validation_mse",
            "test_used_for_stopping": False,
        }


class ValidationPlateau:
    def __init__(self, criterion=None):
        self.criterion = criterion or StopCriterion()
        self.anchor = None
        self.bad_epochs = 0
        self.last_epoch = 0

    def update(self, epoch, loss):
        if epoch != self.last_epoch + 1 or not math.isfinite(loss) or loss < 0:
            raise ValueError(
                "Stopping requires sequential epochs and finite nonnegative validation MSE"
            )
        self.last_epoch = epoch
        significant = self.anchor is None or (
            loss < self.anchor
            and (self.anchor - loss) / max(abs(self.anchor), 1e-12)
            >= self.criterion.relative_improvement
        )
        if significant:
            self.anchor = loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return epoch >= self.criterion.max_epochs or (
            epoch >= self.criterion.min_epochs
            and self.bad_epochs >= self.criterion.patience
        )
