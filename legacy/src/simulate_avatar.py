"""Stage 9 (support) -- a minimal avatar / prosthetic-arm simulation.

This is a proof-of-concept command *interface*, not a validated controller. The
state is mutated by N2; this module only visualises it -- as ASCII for the CLI
demo, or as a 2-link arm + gripper for the matplotlib animation.
"""
from __future__ import annotations

import numpy as np

from .state import ProstheticState


class AvatarSimulator:
    """Render a ProstheticState. Geometry is illustrative, not biomechanical."""

    L1 = 1.0   # upper arm length
    L2 = 1.0   # forearm length

    # -- text -------------------------------------------------------------
    def render_ascii(self, state: ProstheticState) -> str:
        hand = "[==]" if state.hand_state == "closed" else "[  ]"
        arm = "-" * (2 + 3 * state.extension_level)
        elev = "^" if state.arm_raised else "_"
        return (f"shoulder{elev}{arm}o{hand}  "
                f"(wrist {state.wrist_angle:+d}deg, "
                f"{'holding' if state.holding_object else 'empty'})")

    # -- joint geometry ---------------------------------------------------
    def _joints(self, state: ProstheticState):
        # Shoulder elevation grows with height_level; forearm straightens with
        # extension_level.
        theta1 = np.deg2rad(20 + 25 * state.height_level)          # shoulder up
        theta2 = np.deg2rad(70 - 20 * state.extension_level)       # elbow open
        shoulder = np.array([0.0, 0.0])
        elbow = shoulder + self.L1 * np.array([np.cos(theta1), np.sin(theta1)])
        hand = elbow + self.L2 * np.array([np.cos(theta1 - theta2),
                                           np.sin(theta1 - theta2)])
        return shoulder, elbow, hand, theta1 - theta2

    # -- matplotlib -------------------------------------------------------
    def draw(self, ax, state: ProstheticState, title: str | None = None) -> None:
        ax.clear()
        shoulder, elbow, hand, hand_angle = self._joints(state)
        # arm links
        ax.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]],
                "-", lw=6, color="#4a6fa5", solid_capstyle="round")
        ax.plot([elbow[0], hand[0]], [elbow[1], hand[1]],
                "-", lw=6, color="#5b8def", solid_capstyle="round")
        ax.plot(*shoulder, "o", ms=10, color="#333")
        ax.plot(*elbow, "o", ms=8, color="#333")

        # gripper: two fingers, gap depends on hand_state; rotated by wrist_angle
        gap = 0.28 if state.hand_state == "open" else 0.06
        wrist = np.deg2rad(state.wrist_angle)
        base = hand_angle + wrist
        for sign in (+1, -1):
            perp = base + sign * np.pi / 2
            root = hand + gap * np.array([np.cos(perp), np.sin(perp)])
            tip = root + 0.35 * np.array([np.cos(base), np.sin(base)])
            ax.plot([root[0], tip[0]], [root[1], tip[1]], "-", lw=4,
                    color="#e07a3f")
        if state.holding_object:
            ax.plot(*(hand + 0.35 * np.array([np.cos(base), np.sin(base)])),
                    "s", ms=14, color="#6ab04c", alpha=0.8)

        ax.set_xlim(-0.5, 2.6)
        ax.set_ylim(-0.5, 2.6)
        ax.set_aspect("equal")
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=10)
        ax.text(-0.4, -0.4, state.summary(), fontsize=7, color="#555")
