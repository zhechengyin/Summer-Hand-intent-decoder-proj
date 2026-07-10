"""Stage 6 (support) -- the prosthetic/avatar state.

N2 is *state-injected*: the same neural intent maps to different commands
depending on this object. The state is deliberately high-level (a proof-of-concept
command interface) -- NOT a validated biomechanical model of a limb.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace


@dataclass
class ProstheticState:
    # Discrete, human-readable state used by the N2 rules.
    arm_position: str = "neutral"      # 'neutral' | 'extended'
    arm_raised: bool = False           # lifted above rest?
    hand_state: str = "open"           # 'open' | 'closed'
    wrist_angle: int = 0               # degrees, +/- around neutral
    holding_object: bool = False
    # Bounded continuous-ish levels so repeated commands saturate instead of
    # running away.
    extension_level: int = 0           # 0..max_extension
    height_level: int = 0              # 0..max_height
    grip_stability: float = 0.0        # 0..1, how secure the grasp is
    # Bookkeeping for smoothing / hysteresis.
    last_command: str = "none"
    last_confidence: float = 0.0

    # -- limits (not part of the mutable trajectory) -----------------------
    max_extension: int = field(default=3, repr=False)
    max_height: int = field(default=3, repr=False)

    def copy(self) -> "ProstheticState":
        return replace(self)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("max_extension", "max_height"):
            d.pop(k, None)
        return d

    def public_view(self) -> dict:
        """The compact 'current_state' block echoed in the N2 output."""
        return {
            "arm_position": self.arm_position,
            "arm_raised": self.arm_raised,
            "hand_state": self.hand_state,
            "wrist_angle": self.wrist_angle,
            "holding_object": self.holding_object,
        }

    def summary(self) -> str:
        return (f"arm={self.arm_position}{'/raised' if self.arm_raised else ''} "
                f"ext={self.extension_level} h={self.height_level} "
                f"hand={self.hand_state} wrist={self.wrist_angle}deg "
                f"holding={self.holding_object}")
