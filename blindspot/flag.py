"""
The Flag engine - decides what BlindSpot does with each frame.

BlindSpot is a warm, constant companion: it keeps a blind user comfortable by
gently narrating what's around, warns instantly about real hazards, and answers
questions - without ever spamming or overwhelming.

Every frame, in priority order, it emits at most ONE decision:

  🔴 DANGER    - a trusted (stable) hazard object is approaching fast.
                 -> INSTANT local warning (no VLM wait) + VLM refine.
  🛑 OBSTRUCTED- the camera is blocked (a hand/too close).
                 -> say "camera blocked" ONCE, then stay quiet. (Fixes the old
                    "caution! caution!" spam when a hand covers the lens.)
  🟡 QUESTION  - user asked something -> always fire the VLM.
  🟢 PROACTIVE - fire the VLM for a rich, warm update, on scene-change OR a slow
                 timer. This is the "companion" richness.
  🗣️ NARRATE   - gentle, instant local narration from YOLO (fills the gaps so it
                 feels alive), de-duplicated by say-once memory.
  ⚪ SILENT    - nothing new worth saying. Stay quiet.

Two speeds blended: YOLO narrates instantly and constantly; the VLM layers in
richer detail periodically. Say-once memory means we never repeat the same idea.
"""

from __future__ import annotations
import time
from dataclasses import dataclass

import config
from blindspot.vision import Detection
from blindspot.tracker import Tracker, ApproachEvent, Track


@dataclass
class Decision:
    tier: str               # danger | obstructed | question | proactive | narrate | silent
    speak_now: str          # spoken IMMEDIATELY, locally (may be "")
    urgent: bool            # jump the speech queue?
    fire_vlm: bool          # call the remote brain?
    question: str = ""       # the user's question, if any
    reason: str = ""         # for logs / HUD


def _approach_phrase(ev: ApproachEvent) -> str:
    if ev.horizontal == "ahead":
        return f"Careful, a {ev.label} is approaching ahead."
    return f"Careful, a {ev.label} is approaching on your {ev.horizontal}."


def _narration_phrase(tracks: list[Track], max_items: int = 3) -> str:
    """A gentle, natural sentence from the stable tracks (the fast local voice).
    e.g. 'A person is close ahead, and a chair is on your right.'"""
    parts = []
    for t in tracks[:max_items]:
        art = "an" if t.label[0] in "aeiou" else "a"
        if t.horizontal == "ahead":
            parts.append(f"{art} {t.label} {t.distance} ahead")
        else:
            parts.append(f"{art} {t.label} on your {t.horizontal}")
    if not parts:
        return ""
    if len(parts) == 1:
        s = parts[0]
    else:
        s = ", ".join(parts[:-1]) + ", and " + parts[-1]
    return s[0].upper() + s[1:] + "."


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
        self._last_narration = 0.0
        self._last_scene_key = ""
        self._obstruction_announced = False
        # say-once memory: idea-string -> last time we said it
        self._said: dict[str, float] = {}

    # -- rate limiting --
    def _vlm_allowed(self) -> bool:
        return (time.time() - self._last_vlm_fire) >= config.VLM_MIN_INTERVAL_SECONDS

    def mark_vlm_fired(self):
        self._last_vlm_fire = time.time()

    # -- say-once memory: True if this idea is fresh enough to say --
    def _fresh(self, idea: str) -> bool:
        now = time.time()
        last = self._said.get(idea, 0.0)
        if now - last >= config.SAY_ONCE_SECONDS:
            self._said[idea] = now
            return True
        return False

    # -- scene fingerprint using STABLE tracks (identity, not jitter) --
    @staticmethod
    def _scene_key(tracks: list[Track]) -> str:
        # Use labels + coarse distance of the top stable tracks. Stable tracks
        # don't flicker, so this only changes on REAL change.
        return "|".join(f"{t.label}:{t.distance}" for t in tracks[:3])

    def decide(
        self,
        detections: list[Detection],
        audio_words: set[str] | None = None,
        question: str = "",
    ) -> Decision:
        audio_words = audio_words or set()
        now = time.time()

        # Update tracker (persistence, approach, obstruction all handled inside).
        result = self.tracker.update(detections)

        # 🛑 OBSTRUCTION - camera blocked. Say it ONCE, then stay quiet.
        if result.obstructed:
            if not self._obstruction_announced:
                self._obstruction_announced = True
                return Decision(
                    tier="obstructed",
                    speak_now="Your camera seems blocked.",
                    urgent=False, fire_vlm=False,
                    reason="camera obstructed (huge box)",
                )
            return Decision(tier="silent", speak_now="", urgent=False,
                            fire_vlm=False, reason="still obstructed (already said)")
        self._obstruction_announced = False  # reset once the view clears

        # 🔴 DANGER - a stable, dangerous object approaching fast. Instant.
        approaches = [a for a in result.approaches if a.label in config.DANGER_OBJECTS]
        if approaches:
            ev = approaches[0]
            sound_boost = _sound_matches(audio_words, config.HAZARD_SOUNDS)
            # Say-once per (object id + side) so we don't repeat every frame, but
            # DO re-warn if it changes side or a new object appears.
            idea = f"danger:{ev.track_id}:{ev.horizontal}"
            if self._fresh(idea):
                msg = _approach_phrase(ev)
                if sound_boost:
                    msg = msg.rstrip(".") + " — I can hear it too."
                return Decision(
                    tier="danger", speak_now=msg, urgent=True,
                    fire_vlm=self._vlm_allowed(),
                    reason=f"{ev.label} approaching (growth={ev.growth:.2f}"
                           f"{', +sound' if sound_boost else ''})",
                )
            # already warned about this exact approach recently -> don't repeat
            return Decision(tier="silent", speak_now="", urgent=False,
                            fire_vlm=False, reason="danger already warned")

        # 🟡 QUESTION - always fire the VLM (bypasses rate limit + busy-skip).
        if question:
            return Decision(
                tier="question", speak_now="", urgent=True, fire_vlm=True,
                question=question, reason="user question",
            )

        stable = result.stable_tracks
        scene_key = self._scene_key(stable)
        scene_changed = bool(scene_key) and scene_key != self._last_scene_key

        # 🟢 PROACTIVE - rich, warm VLM update on change OR slow timer.
        if config.PROACTIVE_ENABLED and stable:
            timer_due = (now - self._last_proactive) >= config.PROACTIVE_INTERVAL_SECONDS
            change_due = config.PROACTIVE_ON_CHANGE and scene_changed
            if (timer_due or change_due) and self._vlm_allowed():
                self._last_proactive = now
                self._last_scene_key = scene_key
                return Decision(
                    tier="proactive", speak_now="", urgent=False, fire_vlm=True,
                    reason=f"VLM update ({'change' if change_due else 'timer'})",
                )

        # 🗣️ NARRATE - gentle instant local narration (fills the gaps, feels alive).
        if config.NARRATION_ENABLED and stable:
            narration_due = (now - self._last_narration) >= config.NARRATION_MIN_INTERVAL_SECONDS
            if narration_due:
                sentence = _narration_phrase(stable)
                # say-once so we don't repeat the identical scene description
                if sentence and self._fresh(f"narrate:{scene_key}"):
                    self._last_narration = now
                    self._last_scene_key = scene_key
                    return Decision(
                        tier="narrate", speak_now=sentence, urgent=False,
                        fire_vlm=False, reason="steady narration",
                    )

        # keep fingerprint current so we don't fire on stale change later
        if scene_key:
            self._last_scene_key = scene_key

        # ⚪ SILENT
        return Decision(tier="silent", speak_now="", urgent=False,
                        fire_vlm=False, reason="nothing new")


# --------------------------------------------------------------------------
# Standalone logic test:  python -m blindspot.flag   (no camera/mic/models)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    w, h = 640, 480

    def det(label, area, side="ahead"):
        s = int((area * w * h) ** 0.5)
        cx = {"left": w // 6, "right": 5 * w // 6}.get(side, w // 2)
        return Detection(label, 0.9, (cx - s // 2, 10, cx + s // 2, 10 + s), w, h)

    print("== HAND over lens (huge box) -> 'blocked' ONCE, then silent ==")
    eng = FlagEngine()
    for i in range(4):
        d = eng.decide([det("person", 0.85)])
        print(f"  frame {i}: {d.tier:10} | {d.speak_now or d.reason}")

    print("\n== steady person ahead -> narrates ONCE, not every frame ==")
    eng2 = FlagEngine()
    for i in range(6):
        d = eng2.decide([det("person", 0.12)])
        print(f"  frame {i}: {d.tier:10} | {d.speak_now or d.reason}")

    print("\n== question always fires VLM ==")
    eng3 = FlagEngine()
    d = eng3.decide([], question="what does this say?")
    print(f"  -> {d.tier}, fire_vlm={d.fire_vlm}")

    print("\n== empty scene -> silent ==")
    eng4 = FlagEngine()
    print(f"  -> {eng4.decide([]).tier}")
