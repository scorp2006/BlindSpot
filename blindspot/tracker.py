"""
Lightweight object tracker - just enough to detect "something is approaching".

We do NOT need full multi-object tracking (DeepSORT etc.). We only need to answer
one question cheaply, every frame:

    "Is any object's bounding box GROWING fast?"  (growing box == getting closer)

Method (simple + fast, pure Python/NumPy-free):
  - Each frame, match new detections to the previous frame's tracks by label +
    box overlap (IoU).
  - For a matched track, compare its current area to a short history.
  - If the area grew by more than APPROACH_GROWTH_RATIO over the window, and the
    box is at least APPROACH_MIN_AREA big, we call it "approaching".

This is the cheap, local, INSTANT danger signal. No VLM, no round-trip.
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
    label: str
    box: tuple[int, int, int, int]
    area_fraction: float
    horizontal: str
    # short history of recent area_fractions (oldest -> newest)
    history: list[float] = field(default_factory=list)
    misses: int = 0

    def growth(self) -> float:
        """How much the box grew across the history window.
        Returns newest/oldest ratio (1.0 = no change, 2.0 = doubled)."""
        if len(self.history) < 2:
            return 1.0
        oldest = self.history[0]
        newest = self.history[-1]
        if oldest <= 0:
            return 1.0
        return newest / oldest


@dataclass
class ApproachEvent:
    label: str
    horizontal: str          # left / ahead / right
    area_fraction: float
    growth: float


class Tracker:
    """Matches detections frame-to-frame and flags approaching objects."""

    def __init__(self):
        self._tracks: list[Track] = []

    def update(self, detections: list[Detection]) -> list[ApproachEvent]:
        # 1) Match each detection to an existing track (same label, best IoU).
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
                tr.history.append(det.area_fraction)
                if len(tr.history) > config.APPROACH_WINDOW_FRAMES:
                    tr.history.pop(0)
                tr.misses = 0
                used.add(best_i)
            else:
                # New object -> new track.
                self._tracks.append(
                    Track(
                        label=det.label,
                        box=det.box,
                        area_fraction=det.area_fraction,
                        horizontal=det.horizontal,
                        history=[det.area_fraction],
                    )
                )

        # 2) Age out tracks we didn't see this frame.
        for i, tr in enumerate(self._tracks):
            if i not in used:
                tr.misses += 1
        self._tracks = [t for t in self._tracks if t.misses <= config.TRACK_MAX_MISSES]

        # 3) Report which tracks are approaching (growing fast + big enough).
        events: list[ApproachEvent] = []
        for tr in self._tracks:
            if tr.misses != 0:
                continue
            if tr.area_fraction < config.APPROACH_MIN_AREA:
                continue
            g = tr.growth()
            if g >= config.APPROACH_GROWTH_RATIO:
                events.append(
                    ApproachEvent(
                        label=tr.label,
                        horizontal=tr.horizontal,
                        area_fraction=tr.area_fraction,
                        growth=g,
                    )
                )
        # Most-grown first (scariest).
        events.sort(key=lambda e: e.growth, reverse=True)
        return events


# --------------------------------------------------------------------------
# Standalone logic test:  python -m blindspot.tracker
# Simulates an object whose box grows over frames -> should fire an approach.
# No camera / models needed.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    w, h = 640, 480

    def det(label, area):
        side = int((area * w * h) ** 0.5)
        cx = w // 2
        return Detection(label, 0.9, (cx - side // 2, 10, cx + side // 2, 10 + side), w, h)

    tr = Tracker()
    print("Feeding a 'person' whose box grows each frame...")
    areas = [0.03, 0.045, 0.07, 0.11, 0.17]  # steadily growing
    for i, a in enumerate(areas):
        events = tr.update([det("person", a)])
        tag = "  APPROACHING!" if events else ""
        print(f"  frame {i}: area={a:.3f}{tag}",
              [f'{e.label} g={e.growth:.2f}' for e in events])

    print("\nNow a static 'chair' (should NOT approach)...")
    tr2 = Tracker()
    for i in range(5):
        events = tr2.update([det("chair", 0.10)])
        print(f"  frame {i}: static -> approaching={bool(events)}")
