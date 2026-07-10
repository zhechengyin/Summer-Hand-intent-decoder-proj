"""Stage 6 -- N2, the State-Injected Intent-to-Command Interpreter ("Mini AI Spine").

N2 turns N1's probability vector into a *safe, state-aware* prosthetic command.
It never maps a class label straight to an action -- it also reads the current
prosthetic/avatar state, smooths noisy predictions over time, applies confidence
and safety gates, and then updates the state.

    proba (from N1) + state  --N2-->  command  -->  new state

Honesty note: N2 is a command-interpretation state machine with safety heuristics.
It is NOT a validated biological model of the spinal cord, and the commands are
high-level (e.g. "close_hand"), not muscle activations or joint torques.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from .config import cfg_get
from .state import ProstheticState

# Naive class -> command mapping (what N1-only would do). N2 refines this.
BASE_COMMANDS = {
    "reach": "extend_arm_forward",
    "grasp": "close_hand",
    "lift": "raise_arm",
    "twist": "rotate_wrist",
}

# Commands that represent "did not act on new intent".
HOLD_COMMANDS = {"no_action", "hold_state", "request_more_evidence"}


@dataclass
class N2Output:
    intent: str                      # 'GRASP'
    confidence: float
    current_state: dict              # state used to make the decision
    prosthetic_action: str           # 'close_hand'
    command_vector: dict             # the N1 probability vector
    accepted: bool                   # did a physical action occur?
    reason: str                      # why deferred / modified
    next_state: ProstheticState      # state after applying the command

    def to_dict(self) -> dict:
        d = {
            "intent": self.intent,
            "confidence": round(self.confidence, 4),
            "current_state": self.current_state,
            "prosthetic_action": self.prosthetic_action,
            "command_vector": {k: round(v, 4) for k, v in self.command_vector.items()},
            "accepted": self.accepted,
            "reason": self.reason,
        }
        return d


class N2Interpreter:
    def __init__(self, cfg: dict):
        self.classes = list(cfg_get(cfg, "dataset.classes"))
        self.conf_thresh = float(cfg_get(cfg, "n2.confidence_threshold", 0.5))
        self.margin_thresh = float(cfg_get(cfg, "n2.margin_threshold", 0.15))
        self.win = int(cfg_get(cfg, "n2.smoothing_window", 5))
        self.min_agree = int(cfg_get(cfg, "n2.smoothing_min_agreement", 3))
        self.wrist_limit = int(cfg_get(cfg, "n2.wrist_limit_deg", 90))
        self.wrist_step = int(cfg_get(cfg, "n2.wrist_step_deg", 30))
        self.chance = 1.0 / len(self.classes)
        self.history: deque[str] = deque(maxlen=self.win)

    def reset(self) -> None:
        self.history.clear()

    # -- naive mapping (used by the N1-only baseline comparison) -----------
    def class_to_command(self, intent: str) -> str:
        return BASE_COMMANDS.get(intent.lower(), "no_action")

    # -- main step ---------------------------------------------------------
    def step(self, proba: dict[str, float], state: ProstheticState) -> N2Output:
        """One time step: interpret N1 proba against `state`, return N2Output."""
        # rank intents
        ranked = sorted(proba.items(), key=lambda kv: kv[1], reverse=True)
        top_intent, top_p = ranked[0]
        second_p = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_p - second_p
        self.history.append(top_intent)
        pre_state = state.public_view()

        # Gate 1: confidence / margin -----------------------------------
        if top_p < self.conf_thresh or margin < self.margin_thresh:
            if top_p < self.chance * 1.25:
                action, reason = "request_more_evidence", (
                    f"near-chance confidence ({top_p:.2f})")
            else:
                action, reason = "hold_state", (
                    f"low confidence/margin (p={top_p:.2f}, margin={margin:.2f})")
            ns = state.copy()
            ns.last_command, ns.last_confidence = action, top_p
            return N2Output(top_intent.upper(), top_p, pre_state, action,
                            dict(proba), False, reason, ns)

        # Gate 2: temporal smoothing (majority vote) ---------------------
        winner, votes = Counter(self.history).most_common(1)[0]
        if votes < self.min_agree:
            action, reason = "hold_state", (
                f"insufficient agreement ({votes}/{self.min_agree} for '{winner}')")
            ns = state.copy()
            ns.last_command, ns.last_confidence = action, top_p
            return N2Output(top_intent.upper(), top_p, pre_state, action,
                            dict(proba), False, reason, ns)

        committed = winner
        conf = float(proba.get(committed, top_p))

        # Gate 3: state-aware command synthesis --------------------------
        action, ns, reason = self._apply_state_rules(committed, conf, state)
        return N2Output(committed.upper(), conf, pre_state, action, dict(proba),
                        action not in HOLD_COMMANDS, reason, ns)

    # -- state transition rules -------------------------------------------
    def _apply_state_rules(self, intent: str, conf: float,
                           state: ProstheticState):
        s = state.copy()
        intent = intent.lower()
        reason = "state-consistent command"

        if intent == "reach":
            if s.extension_level >= s.max_extension:
                action, reason = "hold_reach", "arm already fully extended"
            else:
                s.extension_level += 1
                s.arm_position = "extended"
                action = "extend_arm_forward"

        elif intent == "grasp":
            if s.hand_state == "closed":
                if s.holding_object and s.grip_stability < 1.0:
                    s.grip_stability = min(1.0, s.grip_stability + 0.25)
                    action, reason = "increase_grip_stability", (
                        "already holding; stabilising grip")
                else:
                    action, reason = "maintain_grip", "hand already closed"
            else:
                s.hand_state = "closed"
                s.holding_object = s.arm_position == "extended"
                s.grip_stability = 0.5 if s.holding_object else 0.0
                action = "close_hand"

        elif intent == "lift":
            if s.arm_raised and s.height_level >= s.max_height:
                action, reason = "hold_position", "arm already fully raised"
            else:
                s.height_level = min(s.max_height, s.height_level + 1)
                s.arm_raised = True
                action = "raise_arm"

        elif intent == "twist":
            if abs(s.wrist_angle) >= self.wrist_limit:
                action, reason = "block_rotation", (
                    f"wrist at safe limit ({self.wrist_limit} deg)")
            elif abs(s.wrist_angle) + self.wrist_step > self.wrist_limit:
                s.wrist_angle = (self.wrist_limit if s.wrist_angle >= 0
                                 else -self.wrist_limit)
                action, reason = "limit_rotation", "clamped to safe wrist limit"
            else:
                s.wrist_angle += self.wrist_step
                action = "rotate_wrist"
        else:
            action, reason = "no_action", f"unknown intent '{intent}'"

        s.last_command, s.last_confidence = action, conf
        return action, s, reason
