"""
The Flag engine - a thin reflex layer in front of an intelligent VLM.

Philosophy (this is the important part):
  - The VLM is the MIND. It decides what to say, how often, and when to stay
    quiet - guided by the user's standing instruction and its memory of what it
    already said. We don't script narration in Python anymore.
  - Python keeps only ONE hard rule: an instant DANGER reflex. If something is
    rushing at the user, we warn immediately and locally, because safety cannot
    wait ~2 seconds for a cloud round-trip. (Even humans flinch before they think.)

So every frame the engine emits at most one intent:

  🔴 DANGER   - a stable, dangerous object approaching fast -> instant local warning
                (+ the VLM may add detail).
  🟡 QUESTION - the user asked something -> the VLM answers.
  🛠️ SET_MODE - the user told us HOW to behave -> store it as the standing
                instruction (no VLM call; just acknowledge).
  🧠 OFFER    - periodically / on change, OFFER the VLM a frame; the VLM decides
                whether to speak or reply NOTHING. This is companion mode.
  ⚪ SILENT   - nothing to do this frame.

Obstruction (a hand fully covering the lens) is now persistence-gated and simply
suppresses the DANGER reflex - it does NOT spam "camera blocked". A close-up face
at desk distance no longer trips it.
"""

from __future__ import annotations
import time
from dataclasses import dataclass

import config
from blindspot.vision import Detection
from blindspot.tracker import Tracker, ApproachEvent, Track


@dataclass
class Decision:
    tier: str               # danger | question | set_mode | offer | silent
    speak_now: str          # spoken IMMEDIATELY, locally (may be "")
    urgent: bool
    fire_vlm: bool          # offer a frame to the VLM this turn?
    question: str = ""       # user question (for tier=question)
    instruction: str = ""    # new standing instruction (for tier=set_mode)
    reason: str = ""


def _approach_phrase(ev: ApproachEvent) -> str:
    if ev.horizontal == "ahead":
        return f"Careful, a {ev.label} is approaching ahead."
    return f"Careful, a {ev.label} is approaching on your {ev.horizontal}."


def _sound_matches(audio_words: set[str], vocab: set[str]) -> bool:
    for w in audio_words:
        for v in vocab:
            if v in w or w in v:
                return True
    return False


def _looks_like_mode(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in config.MODE_PHRASES)


class FlagEngine:
    def __init__(self):
        self.tracker = Tracker()
        self._last_vlm_fire = 0.0
        self._last_offer = 0.0
        self._last_scene_key = ""
        self._obstruction_streak = 0
        self._danger_said: dict[str, float] = {}
        self._obstacle_said: dict[str, float] = {}

    def _vlm_allowed(self) -> bool:
        return (time.time() - self._last_vlm_fire) >= config.VLM_MIN_INTERVAL_SECONDS

    def mark_vlm_fired(self):
        self._last_vlm_fire = time.time()
        self._last_offer = time.time()

    @staticmethod
    def _scene_key(tracks: list[Track]) -> str:
        return "|".join(f"{t.label}:{t.distance}" for t in tracks[:3])

    def _danger_fresh(self, key: str) -> bool:
        now = time.time()
        if now - self._danger_said.get(key, 0.0) >= config.SAY_ONCE_SECONDS:
            self._danger_said[key] = now
            return True
        return False

    def _obstacle_fresh(self, key: str) -> bool:
        now = time.time()
        if now - self._obstacle_said.get(key, 0.0) >= config.OBSTACLE_SAY_ONCE_SECONDS:
            self._obstacle_said[key] = now
            return True
        return False

    @staticmethod
    def _closest_obstacle(stable: list[Track]):
        """The nearest object that is close AND in the walking path (center of
        frame). Returns a Track or None. Category-agnostic - a bag counts."""
        best = None
        for t in stable:
            if t.label in config.OBSTACLE_IGNORE:
                continue
            if t.area_fraction < config.OBSTACLE_MIN_AREA:
                continue
            # is its center within the middle OBSTACLE_PATH_FRACTION of width?
            x1, _, x2, _ = t.box
            # frame width: recover from the box's frame ref via horizontal buckets
            # (tracker keeps horizontal; 'ahead' already means center third). We
            # accept 'ahead' OR a center-ish box as in-path.
            in_path = t.horizontal == "ahead"
            if in_path:
                if best is None or t.area_fraction > best.area_fraction:
                    best = t
        return best

    def decide(
        self,
        detections: list[Detection],
        audio_words: set[str] | None = None,
        question: str = "",
    ) -> Decision:
        audio_words = audio_words or set()
        now = time.time()
        result = self.tracker.update(detections)

        # Track obstruction persistence (but do NOT announce it - just note it so
        # we can suppress false danger while a hand covers the lens).
        if result.obstructed:
            self._obstruction_streak += 1
        else:
            self._obstruction_streak = 0
        obstructed = self._obstruction_streak >= config.OBSTRUCTION_MIN_FRAMES

        # 🟡/🛠️ USER SPOKE - highest priority (respond to the human).
        if question:
            if _looks_like_mode(question):
                return Decision(
                    tier="set_mode", speak_now="Okay, I'll do that.",
                    urgent=False, fire_vlm=False, instruction=question,
                    reason=f"set instruction: {question!r}",
                )
            return Decision(
                tier="question", speak_now="", urgent=True, fire_vlm=True,
                question=question, reason="user question",
            )

        if not obstructed:
            # 🔴 DANGER reflex - an approaching hazard (box growing fast). Most urgent.
            approaches = [a for a in result.approaches
                          if a.label in config.DANGER_OBJECTS]
            if approaches:
                ev = approaches[0]
                key = f"{ev.track_id}:{ev.horizontal}"
                if self._danger_fresh(key):
                    msg = _approach_phrase(ev)
                    if _sound_matches(audio_words, config.HAZARD_SOUNDS):
                        msg = msg.rstrip(".") + " — I can hear it too."
                    return Decision(
                        tier="danger", speak_now=msg, urgent=True,
                        fire_vlm=self._vlm_allowed(),
                        reason=f"{ev.label} approaching (growth={ev.growth:.2f})",
                    )

            # 🟠 OBSTACLE reflex - ANYTHING close and in the walking path, even
            # stationary, even a bag. A blind user can trip on it, so we MUST say
            # it (once). This is the core "don't let them walk into things" rule.
            obstacle = self._closest_obstacle(result.stable_tracks)
            if obstacle is not None:
                key = f"obs:{obstacle.id}"
                if self._obstacle_fresh(key):
                    where = "right in front of you" if obstacle.horizontal == "ahead" \
                            else f"ahead on your {obstacle.horizontal}"
                    art = "an" if obstacle.label[0] in "aeiou" else "a"
                    msg = f"Careful, {art} {obstacle.label} {where}."
                    return Decision(
                        tier="obstacle", speak_now=msg, urgent=True,
                        fire_vlm=False,
                        reason=f"{obstacle.label} in path ({obstacle.distance})",
                    )

        # 🧠 OFFER a frame to the VLM (companion mode). The VLM decides whether to
        # speak or reply NOTHING - we only gate HOW OFTEN we offer.
        if config.PROACTIVE_ENABLED:
            stable = result.stable_tracks
            scene_key = self._scene_key(stable)
            changed = bool(scene_key) and scene_key != self._last_scene_key
            timer_due = (now - self._last_offer) >= config.PROACTIVE_INTERVAL_SECONDS
            change_due = config.PROACTIVE_ON_CHANGE and changed
            # Offer even if the frame is obstructed/empty - the VLM can say
            # something helpful ("your camera looks covered") or NOTHING itself.
            if (timer_due or change_due) and self._vlm_allowed():
                self._last_offer = now
                self._last_scene_key = scene_key
                return Decision(
                    tier="offer", speak_now="", urgent=False, fire_vlm=True,
                    reason=f"offer to VLM ({'change' if change_due else 'timer'})",
                )
            if scene_key:
                self._last_scene_key = scene_key

        return Decision(tier="silent", speak_now="", urgent=False,
                        fire_vlm=False, reason="nothing to do")


# --------------------------------------------------------------------------
# Standalone logic test:  python -m blindspot.flag   (no camera/mic/models)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    w, h = 640, 480

    def det(label, area, side="ahead"):
        s = int((area * w * h) ** 0.5)
        cx = {"left": w // 6, "right": 5 * w // 6}.get(side, w // 2)
        return Detection(label, 0.9, (cx - s // 2, 10, cx + s // 2, 10 + s), w, h)

    print("== close-up face (0.55) should NOT be obstruction, should OFFER to VLM ==")
    e = FlagEngine()
    for i in range(6):
        d = e.decide([det("person", 0.55)])
        print(f"  f{i}: {d.tier:9} | fire_vlm={d.fire_vlm} | {d.reason}")

    print("\n== user sets a MODE ==")
    e2 = FlagEngine()
    d = e2.decide([], question="keep me company and describe everything calmly")
    print(f"  -> {d.tier}, instruction={d.instruction!r}, fire_vlm={d.fire_vlm}")

    print("\n== user asks a QUESTION ==")
    e3 = FlagEngine()
    d = e3.decide([], question="what does this sign say?")
    print(f"  -> {d.tier}, fire_vlm={d.fire_vlm}")

    print("\n== hand fully covers lens (0.9) for 5+ frames -> danger suppressed ==")
    e4 = FlagEngine()
    for i in range(7):
        d = e4.decide([det("person", 0.9)])
    print(f"  after covered: tier={d.tier} (offer/silent, NOT spamming 'blocked')")
