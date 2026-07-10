"""
The Flag engine - the funnel that answers "what, if anything, do we do?"

THIS is the fix for the firehose problem (YOLO emits ~20 detections/second).
Every frame we run the cheap checks below, in strict priority order, and emit at
most ONE decision. Almost every frame the answer is "stay silent".

Severity tiers (highest priority first):

  🔴 DANGER  - a DANGER_OBJECT is approaching (box growing fast, from tracker).
               -> speak an INSTANT local warning (no VLM wait)
               -> AND flag the frame for a VLM follow-up (smart detail ~2s later)

  🟡 QUESTION - the user asked something (push-to-talk).
               -> fire the VLM to answer / read text

  🟢 PROACTIVE - the scene changed a lot and the calm cooldown passed.
               -> fire the VLM (it's allowed to reply "NOTHING")

  ⚪ SILENT  - none of the above. Say nothing. (the "don't overwhelm" feature)

The engine only DECIDES. The pipeline carries the decision out (speak / call VLM).
It also enforces the global VLM rate limit so the paid GPU is barely touched.
"""

from __future__ import annotations
import time
from dataclasses import dataclass

import config
from blindspot.vision import Detection
from blindspot.tracker import Tracker, ApproachEvent


@dataclass
class Decision:
    tier: str               # "danger" | "question" | "proactive" | "silent"
    speak_now: str          # something to say IMMEDIATELY, locally (may be "")
    urgent: bool            # should that speech jump the queue?
    fire_vlm: bool          # should we call the remote brain?
    question: str = ""       # the user's question, if any
    reason: str = ""         # for logs / the debug HUD


def _approach_phrase(ev: ApproachEvent) -> str:
    side = "" if ev.horizontal == "ahead" else f" on your {ev.horizontal}"
    if ev.horizontal == "ahead":
        return f"Careful, a {ev.label} is approaching ahead."
    return f"Careful, a {ev.label} is approaching{side}."


def _sound_matches(audio_words: set[str], vocab: set[str]) -> bool:
    for w in audio_words:
        for v in vocab:
            if v in w or w in v:
                return True
    return False


class FlagEngine:
    def __init__(self):
        self.tracker = Tracker()
        self._last_vlm_fire = 0.0
        self._last_proactive = 0.0
        self._last_scene_key = ""

    # -- rate limiting --
    def _vlm_allowed(self) -> bool:
        return (time.time() - self._last_vlm_fire) >= config.VLM_MIN_INTERVAL_SECONDS

    def mark_vlm_fired(self):
        self._last_vlm_fire = time.time()

    # -- cheap scene-change fingerprint (STAGE 1 of the funnel) --
    @staticmethod
    def _scene_key(dets: list[Detection]) -> str:
        return "|".join(f"{d.label}:{d.distance}:{d.horizontal}" for d in dets[:3])

    def decide(
        self,
        detections: list[Detection],
        audio_words: set[str] | None = None,
        question: str = "",
    ) -> Decision:
        audio_words = audio_words or set()

        # Always update the tracker (it needs every frame to measure growth).
        approaches = self.tracker.update(detections)
        # Keep only approaches of things that are actually dangerous.
        approaches = [a for a in approaches if a.label in config.DANGER_OBJECTS]

        # 🔴 TIER 1: DANGER - instant, local, highest priority.
        if approaches:
            ev = approaches[0]
            # Optional cross-modal boost: a matching sound makes it more urgent.
            sound_boost = _sound_matches(audio_words, config.HAZARD_SOUNDS)
            msg = _approach_phrase(ev)
            if sound_boost:
                msg = msg.rstrip(".") + " — I can hear it too."
            return Decision(
                tier="danger",
                speak_now=msg,           # spoken IMMEDIATELY, no VLM wait
                urgent=True,
                fire_vlm=self._vlm_allowed(),  # VLM refines if allowed
                reason=f"{ev.label} approaching (growth={ev.growth:.2f}"
                       f"{', +sound' if sound_boost else ''})",
            )

        # 🟡 TIER 2: QUESTION - user asked; always try the VLM.
        if question:
            return Decision(
                tier="question",
                speak_now="",            # let the VLM answer
                urgent=True,
                fire_vlm=self._vlm_allowed(),
                question=question,
                reason="user question",
            )

        # 🟢 TIER 3: PROACTIVE - calm scene changed; slow VLM tick.
        if config.PROACTIVE_ENABLED:
            key = self._scene_key(detections)
            changed = bool(key) and key != self._last_scene_key
            cooled = (time.time() - self._last_proactive) >= config.PROACTIVE_INTERVAL_SECONDS
            if changed and cooled and self._vlm_allowed():
                self._last_scene_key = key
                self._last_proactive = time.time()
                return Decision(
                    tier="proactive",
                    speak_now="",
                    urgent=False,
                    fire_vlm=True,       # VLM may reply "NOTHING"
                    reason="scene changed (proactive tick)",
                )
            # keep the fingerprint current so we don't fire on stale change
            if key:
                self._last_scene_key = key

        # ⚪ TIER 4: SILENT.
        return Decision(tier="silent", speak_now="", urgent=False,
                        fire_vlm=False, reason="nothing important")


# --------------------------------------------------------------------------
# Standalone logic test:  python -m blindspot.flag
# Pure logic - no camera, mic, or models.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    w, h = 640, 480

    def det(label, area, side="ahead"):
        s = int((area * w * h) ** 0.5)
        if side == "left":
            cx = w // 6
        elif side == "right":
            cx = 5 * w // 6
        else:
            cx = w // 2
        return Detection(label, 0.9, (cx - s // 2, 10, cx + s // 2, 10 + s), w, h)

    eng = FlagEngine()

    print("== growing car (approaching) ==")
    for a in [0.03, 0.05, 0.08, 0.13, 0.20]:
        d = eng.decide([det("car", a, "left")])
        print(f"  area={a:.2f} -> {d.tier:9} | {d.speak_now or d.reason}")

    print("\n== static chair (should stay silent) ==")
    eng2 = FlagEngine()
    for _ in range(5):
        d = eng2.decide([det("chair", 0.10)])
    print(f"  -> {d.tier} ({d.reason})")

    print("\n== user question ==")
    eng3 = FlagEngine()
    d = eng3.decide([], question="what does this sign say?")
    print(f"  -> {d.tier}, fire_vlm={d.fire_vlm}, q={d.question!r}")

    print("\n== empty scene ==")
    eng4 = FlagEngine()
    d = eng4.decide([])
    print(f"  -> {d.tier} ({d.reason})")
