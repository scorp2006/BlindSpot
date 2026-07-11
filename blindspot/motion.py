"""
Ego-motion detector - is the WEARER moving, or sitting still?

Why this exists: an obstacle caution ("careful, a chair in front of you") only
matters when the USER is moving toward things. Sitting at a desk, the laptop in
front of you is furniture, not a hazard - warning about it is noise.

How it works (cheap, local, no models): when the wearer walks or turns, the
ENTIRE camera frame shifts between consecutive frames (ego-motion), producing a
large global frame difference. Sitting still produces a tiny difference, even if
something small moves inside the scene. We threshold the mean absolute gray
difference and apply hysteresis so the state doesn't flap:

  - enter MOVING after MOTION_ON_FRAMES consecutive high-diff frames
  - return to STILL after MOTION_OFF_FRAMES consecutive low-diff frames
"""

from __future__ import annotations

import numpy as np

import config


class MotionEstimator:
    def __init__(self):
        self._prev = None
        self._moving_streak = 0
        self._still_streak = 0
        self._moving = False
        self.last_diff = 0.0   # exposed for the dashboard / debugging

    def update(self, frame_bgr) -> bool:
        """Feed the next frame; returns True if the user is currently moving."""
        import cv2
        small = cv2.cvtColor(cv2.resize(frame_bgr, (80, 60)),
                             cv2.COLOR_BGR2GRAY).astype(np.int16)
        if self._prev is None:
            self._prev = small
            return self._moving
        self.last_diff = float(np.mean(np.abs(small - self._prev)))
        self._prev = small

        if self.last_diff >= config.MOTION_DIFF_THRESHOLD:
            self._moving_streak += 1
            self._still_streak = 0
        else:
            self._still_streak += 1
            self._moving_streak = 0

        if not self._moving and self._moving_streak >= config.MOTION_ON_FRAMES:
            self._moving = True
        elif self._moving and self._still_streak >= config.MOTION_OFF_FRAMES:
            self._moving = False
        return self._moving


# --------------------------------------------------------------------------
# Standalone logic test:  python -m blindspot.motion   (no camera needed)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)

    m = MotionEstimator()
    print("== static frames (sitting) ==")
    for i in range(8):
        # tiny sensor noise only
        noisy = np.clip(base.astype(int) + rng.integers(-2, 3, base.shape), 0, 255).astype(np.uint8)
        print(f"  f{i}: moving={m.update(noisy)} diff={m.last_diff:.1f}")

    print("== walking (frame shifts each step) ==")
    frame = base
    for i in range(5):
        frame = np.roll(frame, 25, axis=1)   # global shift = ego-motion
        print(f"  f{i}: moving={m.update(frame)} diff={m.last_diff:.1f}")

    print("== stops again ==")
    for i in range(8):
        print(f"  f{i}: moving={m.update(frame)} diff={m.last_diff:.1f}")
