"""
Lightweight object tracker - persistence, approach detection, obstruction.

We do NOT need full multi-object tracking (DeepSORT etc.). We match detections
frame-to-frame by label + box overlap (IoU) and maintain a little state per
object so the rest of the system can be SMART instead of twitchy:

  - stable track id      -> so we can remember "already told the user about this"
  - frames_seen          -> ignore 1-frame flickers (a hand-blob, a YOLO misfire)
  - area growth history  -> detect something APPROACHING (box growing fast)
  - obstruction check    -> a box covering most of the frame is a hand/blocked
                            camera, NOT a person approaching (kills the
                            "caution! caution!" spam)

Everything here is cheap, local, and runs every frame.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import config
from blindspot.vision import Detection


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


@dataclass
class Track:
    id: int
    label: str
    box: tuple[int, int, int, int]
    area_fraction: float
    horizontal: str
    distance: str
    confidence: float
    history: list[float] = field(default_factory=list)  # recent area fractions
    frames_seen: int = 1     # how many consecutive frames we've matched it
    misses: int = 0

    def growth(self) -> float:
        """newest/oldest area ratio across the window (1.0 = no change)."""
        if len(self.history) < 2:
            return 1.0
        oldest, newest = self.history[0], self.history[-1]
        if oldest <= 0:
            return 1.0
        return newest / oldest

    def is_stable(self) -> bool:
        """Seen enough consecutive frames to be trusted (not a flicker)."""
        return self.frames_seen >= config.TRACK_MIN_FRAMES


@dataclass
class ApproachEvent:
    track_id: int
    label: str
    horizontal: str
    area_fraction: float
    growth: float


@dataclass
class TrackerResult:
    approaches: list[ApproachEvent] = field(default_factory=list)
    obstructed: bool = False       # camera blocked (hand / too close)
    stable_tracks: list[Track] = field(default_factory=list)  # trustworthy objs


class Tracker:
    """Matches detections frame-to-frame; reports approaches + obstruction."""

    def __init__(self):
        self._tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> TrackerResult:
        # --- 0) Obstruction check: is one box swallowing the whole frame? ---
        # A hand over the lens (or being pressed against something) shows up as a
        # single giant box. That's NOT an approaching hazard - it's a blocked
        # camera. Detect it and let the caller stay calm.
        obstructed = any(
            d.area_fraction >= config.OBSTRUCTION_AREA for d in detections
        )

        # --- 1) Match detections to existing tracks (same label, best IoU) ---
        used = set()
        for det in detections:
            best_i, best_iou = -1, 0.0
            for i, tr in enumerate(self._tracks):
                if i in used or tr.label != det.label:
                    continue
                score = _iou(tr.box, det.box)
                if score > best_iou:
                    best_i, best_iou = i, score

            if best_i >= 0 and best_iou >= config.TRACK_IOU_MATCH:
                tr = self._tracks[best_i]
                tr.box = det.box
                tr.area_fraction = det.area_fraction
                tr.horizontal = det.horizontal
                tr.distance = det.distance
                tr.confidence = det.confidence
                tr.history.append(det.area_fraction)
                if len(tr.history) > config.APPROACH_WINDOW_FRAMES:
                    tr.history.pop(0)
                tr.frames_seen += 1
                tr.misses = 0
                used.add(best_i)
            else:
                self._tracks.append(
                    Track(
                        id=self._next_id,
                        label=det.label,
                        box=det.box,
                        area_fraction=det.area_fraction,
                        horizontal=det.horizontal,
                        distance=det.distance,
                        confidence=det.confidence,
                        history=[det.area_fraction],
                    )
                )
                self._next_id += 1

        # --- 2) Age out tracks we didn't see this frame ---
        for i, tr in enumerate(self._tracks):
            if i not in used:
                tr.misses += 1
                tr.frames_seen = 0   # broke the consecutive streak
        self._tracks = [t for t in self._tracks if t.misses <= config.TRACK_MAX_MISSES]

        # If the camera is obstructed, don't emit approach spam - just report it.
        if obstructed:
            return TrackerResult(approaches=[], obstructed=True, stable_tracks=[])

        # --- 3) Stable tracks (trustworthy, not flickers) ---
        stable = [t for t in self._tracks if t.misses == 0 and t.is_stable()]

        # --- 4) Approaches: a STABLE object, big enough, growing fast, and NOT
        #        an obstruction-sized blob. ---
        events: list[ApproachEvent] = []
        for tr in stable:
            if tr.area_fraction < config.APPROACH_MIN_AREA:
                continue
            if tr.area_fraction >= config.OBSTRUCTION_AREA:
                continue  # too big -> treat as obstruction, not approach
            if tr.growth() >= config.APPROACH_GROWTH_RATIO:
                events.append(
                    ApproachEvent(
                        track_id=tr.id,
                        label=tr.label,
                        horizontal=tr.horizontal,
                        area_fraction=tr.area_fraction,
                        growth=tr.growth(),
                    )
                )
        events.sort(key=lambda e: e.growth, reverse=True)

        # Biggest (closest) stable tracks first - most relevant to narrate.
        stable.sort(key=lambda t: t.area_fraction, reverse=True)
        return TrackerResult(approaches=events, obstructed=False, stable_tracks=stable)


# --------------------------------------------------------------------------
# Standalone logic test:  python -m blindspot.tracker
# --------------------------------------------------------------------------
if __name__ == "__main__":
    w, h = 640, 480

    def det(label, area, side="center"):
        s = int((area * w * h) ** 0.5)
        cx = {"left": w // 6, "right": 5 * w // 6}.get(side, w // 2)
        return Detection(label, 0.9, (cx - s // 2, 10, cx + s // 2, 10 + s), w, h)

    print("== growing person (should approach only AFTER it's stable) ==")
    tr = Tracker()
    for i, a in enumerate([0.03, 0.05, 0.08, 0.13, 0.20]):
        r = tr.update([det("person", a)])
        tag = "APPROACH" if r.approaches else "-"
        print(f"  frame {i}: area={a:.2f} seen>=min={a>=config.APPROACH_MIN_AREA} -> {tag}")

    print("\n== HAND over lens (huge box) -> obstruction, NOT approach ==")
    tr2 = Tracker()
    for i in range(4):
        r = tr2.update([det("person", 0.85)])  # 85% of frame
        print(f"  frame {i}: obstructed={r.obstructed}, approaches={len(r.approaches)}")

    print("\n== one-frame flicker -> ignored (not stable) ==")
    tr3 = Tracker()
    r = tr3.update([det("person", 0.15)])
    print(f"  single frame: stable_tracks={len(r.stable_tracks)} (should be 0)")
    for _ in range(config.TRACK_MIN_FRAMES):
        r = tr3.update([det("person", 0.15)])
    print(f"  after {config.TRACK_MIN_FRAMES} frames: stable_tracks={len(r.stable_tracks)} (should be >=1)")
