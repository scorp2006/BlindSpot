"""
Tier 1 - the live scene narrator.

The simplest end-to-end product: camera -> YOLO -> spoken description.
Runs the fast local loop, throttled to TARGET_FPS, and speaks a short scene
summary whenever the scene meaningfully changes (not every frame).

This alone is a guaranteed working demo. Every later tier layers on top of it.
"""

from __future__ import annotations
import time

import config
from blindspot.camera import Camera
from blindspot.vision import VisionModel, summarize


def _scene_key(detections) -> str:
    """A cheap 'fingerprint' of the scene so we only speak when it changes.
    Uses the top labels + their distance buckets, ignoring tiny jitter."""
    top = detections[:3]
    return "|".join(f"{d.label}:{d.distance}" for d in top)


def run(show_window: bool = True) -> None:
    vm = VisionModel()
    # Import here so Tier 1 can run even if speech deps are missing during a
    # very early bring-up (you'd just see printed sentences).
    from blindspot.speech import Voice
    voice = Voice()

    frame_interval = 1.0 / max(1, config.TARGET_FPS)
    last_key = ""
    print("[narrator] running. Press Ctrl+C (or 'q' in window) to stop.")

    try:
        with Camera() as cam:
            for frame in cam.frames():
                t0 = time.time()
                dets = vm.detect(frame)
                key = _scene_key(dets)

                if key and key != last_key:
                    sentence = summarize(dets)
                    if sentence:
                        print("[scene]", sentence)
                        voice.say(sentence)
                    last_key = key

                if show_window:
                    import cv2
                    for d in dets:
                        x1, y1, x2, y2 = d.box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame, d.label, (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        )
                    cv2.imshow("BlindSpot - narrator (Tier 1)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # Throttle to target FPS.
                dt = time.time() - t0
                if dt < frame_interval:
                    time.sleep(frame_interval - dt)
    except KeyboardInterrupt:
        print("\n[narrator] stopping.")
    finally:
        voice.shutdown()
        if show_window:
            import cv2
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
