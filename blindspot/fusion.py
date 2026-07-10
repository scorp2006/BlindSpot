"""
Fusion / trigger-severity engine - the heart of the multimodal claim.

THIS is the file that satisfies Track 04's strict rule ("2+ modalities that
actively influence each other"). Here, AUDIO changes how the VISUAL scene is
interpreted:

    a car you SEE + an engine/horn you HEAR + it's CLOSE  ->  HIGH severity  -> fire VLM + urgent warning
    a car you SEE + silence                               ->  LOW  severity  -> probably parked, stay silent

The engine takes vision detections + the latest audio result and decides ONE of:
    - "question"  : the user asked something  -> always fire the VLM
    - "hazard"    : fused danger detected     -> fire the VLM, speak urgently
    - "crowd"     : busy-environment cue       -> a gentle heads-up (no VLM needed)
    - "silent"    : nothing matters            -> SAY NOTHING  (the anti-overwhelm feature)

It also rate-limits VLM firing so we never spam the paid GPU.
"""

from __future__ import annotations
import time
from dataclasses import dataclass

import config
from blindspot.vision import Detection


@dataclass
class Decision:
    action: str            # "question" | "hazard" | "crowd" | "silent"
    fire_vlm: bool         # should we call the remote brain?
    urgent: bool           # should speech jump the queue?
    reason: str            # human explanation (for logs / debugging)
    quick_message: str     # something to say WITHOUT the VLM (may be "")
    question: str = ""      # the user's question, if any


def _matches(words: set[str], vocab: set[str]) -> bool:
    """True if any word overlaps the vocab (loose substring match too)."""
    for w in words:
        if w in vocab:
            return True
        for v in vocab:
            if v in w or w in v:
                return True
    return False


class FusionEngine:
    def __init__(self):
        self._last_vlm_fire = 0.0

    def _vlm_allowed(self) -> bool:
        return (time.time() - self._last_vlm_fire) >= config.VLM_MIN_INTERVAL_SECONDS

    def mark_vlm_fired(self):
        self._last_vlm_fire = time.time()

    def decide(
        self,
        detections: list[Detection],
        audio_labels: set[str],
        question: str = "",
    ) -> Decision:
        # 1) A user question ALWAYS wins - fire the VLM (respecting rate limit).
        if question:
            return Decision(
                action="question",
                fire_vlm=self._vlm_allowed(),
                urgent=True,
                reason="user asked a question",
                quick_message="",
                question=question,
            )

        # 2) Hazard fusion: is there a hazard OBJECT that is CLOSE, AND a
        #    hazard SOUND at the same time? That cross-modal AND is the magic.
        hazard_objs = [
            d for d in detections
            if d.label in config.HAZARD_OBJECTS
            and d.area_fraction >= config.HAZARD_CLOSE_AREA
        ]
        hazard_sound = _matches(audio_labels, config.HAZARD_SOUNDS)

        if hazard_objs and hazard_sound:
            d = hazard_objs[0]
            side = "" if d.horizontal == "ahead" else f" from your {d.horizontal}"
            msg = f"{d.label.capitalize()} approaching{side}. Please wait."
            return Decision(
                action="hazard",
                fire_vlm=self._vlm_allowed(),
                urgent=True,
                reason=f"saw {d.label} ({d.distance}) + heard hazard sound",
                quick_message=msg,   # spoken immediately; VLM refines if it fires
                question="",
            )

        # 3) Crowd cue: many people OR crowd sounds -> gentle heads-up, no VLM.
        people = sum(1 for d in detections if d.label == "person")
        crowd_sound = _matches(audio_labels, config.CROWD_SOUNDS)
        if people >= 3 or (people >= 1 and crowd_sound):
            return Decision(
                action="crowd",
                fire_vlm=False,
                urgent=False,
                reason=f"{people} people, crowd_sound={crowd_sound}",
                quick_message="You are entering a busy area.",
                question="",
            )

        # 4) Nothing important -> stay silent. This is a FEATURE, not a gap.
        return Decision(
            action="silent",
            fire_vlm=False,
            urgent=False,
            reason="nothing important",
            quick_message="",
            question="",
        )


# --------------------------------------------------------------------------
# Standalone test:  python -m blindspot.fusion
# Pure logic test with fake inputs - no camera, no mic, no models needed.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    def fake_car(area):
        # A wide, close-ish car box.
        w, h = 640, 480
        side = int((area * w * h) ** 0.5)
        return Detection("car", 0.9, (10, 10, 10 + side, 10 + side), w, h)

    eng = FusionEngine()

    print("car CLOSE + engine sound  ->", eng.decide([fake_car(0.10)], {"engine", "vehicle"}).__dict__)
    eng2 = FusionEngine()
    print("car CLOSE + SILENCE       ->", eng2.decide([fake_car(0.10)], set()).__dict__)
    eng3 = FusionEngine()
    print("question                  ->", eng3.decide([], set(), "what is this sign?").__dict__)
    eng4 = FusionEngine()
    print("empty scene               ->", eng4.decide([], set()).__dict__)
