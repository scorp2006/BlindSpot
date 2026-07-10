"""
Vision module - YOLOv8n object detection with spatial reasoning.

Turns a raw camera frame into a list of Detection objects, each carrying:
  - what it is        (label, e.g. "person")
  - how sure we are   (confidence)
  - where it is       (left / center / right, and near / mid / far by box size)

This "text facts" output is exactly what gets:
  1. spoken by the narrator (Tier 1),
  2. fed to the VLM as a hint (Tier 2),
  3. fused with sound to compute danger severity (Tier 4).

Everything here is fast + local. It runs every frame.
"""

from __future__ import annotations
from dataclasses import dataclass

import config


@dataclass
class Detection:
    label: str
    confidence: float
    # Bounding box in pixels: (x1, y1, x2, y2)
    box: tuple[int, int, int, int]
    frame_w: int
    frame_h: int

    @property
    def area_fraction(self) -> float:
        """Box area as a fraction of the whole frame (0-1). Proxy for closeness."""
        x1, y1, x2, y2 = self.box
        return max(0.0, (x2 - x1) * (y2 - y1)) / float(self.frame_w * self.frame_h)

    @property
    def horizontal(self) -> str:
        """Which third of the frame the object's center sits in."""
        x1, _, x2, _ = self.box
        cx = (x1 + x2) / 2.0
        third = self.frame_w / 3.0
        if cx < third:
            return "left"
        if cx > 2 * third:
            return "right"
        return "ahead"

    @property
    def distance(self) -> str:
        """Coarse distance bucket from box size."""
        af = self.area_fraction
        if af >= 0.25:
            return "very close"
        if af >= 0.08:
            return "close"
        if af >= 0.02:
            return "nearby"
        return "far"

    def phrase(self) -> str:
        """Human phrase, e.g. 'a person close on your left'."""
        article = "an" if self.label[0] in "aeiou" else "a"
        side = "" if self.horizontal == "ahead" else f" on your {self.horizontal}"
        pos = "ahead" if self.horizontal == "ahead" else side.strip()
        return f"{article} {self.label} {self.distance} {pos}".replace("  ", " ").strip()

    def as_fact(self) -> str:
        """Compact fact for the VLM prompt, e.g. 'person(close, ahead, 0.91)'."""
        return f"{self.label}({self.distance}, {self.horizontal}, {self.confidence:.2f})"


class VisionModel:
    """Loads YOLOv8n once, then .detect(frame) per frame."""

    def __init__(self, model_path=None, conf=None, device=None):
        # Import here (not at top) so importing this module is cheap and other
        # tiers don't pay the ultralytics import cost unless they detect.
        from ultralytics import YOLO
        import torch

        self.conf = config.YOLO_CONFIDENCE if conf is None else conf

        dev = device or config.YOLO_DEVICE
        if dev == "auto":
            dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = dev

        self.model = YOLO(model_path or config.YOLO_MODEL)
        print(f"[vision] YOLO loaded on device: {self.device}")

    def detect(self, frame) -> list[Detection]:
        h, w = frame.shape[:2]
        # verbose=False keeps the console clean; we do our own logging.
        results = self.model.predict(
            frame, conf=self.conf, device=self.device, verbose=False
        )
        dets: list[Detection] = []
        for r in results:
            names = r.names
            for b in r.boxes:
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                dets.append(
                    Detection(
                        label=names[int(b.cls[0])],
                        confidence=float(b.conf[0]),
                        box=(x1, y1, x2, y2),
                        frame_w=w,
                        frame_h=h,
                    )
                )
        # Biggest (closest) first - that's what matters most to the user.
        dets.sort(key=lambda d: d.area_fraction, reverse=True)
        return dets


def summarize(detections: list[Detection], max_items: int = 3) -> str:
    """One friendly sentence describing the scene, e.g.
    'A person close ahead, and a chair nearby on your right.'"""
    if not detections:
        return ""
    parts = [d.phrase() for d in detections[:max_items]]
    if len(parts) == 1:
        sentence = parts[0]
    else:
        sentence = ", ".join(parts[:-1]) + ", and " + parts[-1]
    return sentence[0].upper() + sentence[1:] + "."


def facts_line(detections: list[Detection], max_items: int = 8) -> str:
    """Compact fact string for the VLM, e.g.
    'person(close, ahead, 0.91); chair(nearby, right, 0.60)'."""
    return "; ".join(d.as_fact() for d in detections[:max_items])


# --------------------------------------------------------------------------
# Standalone test:  python -m blindspot.vision
# Live webcam + drawn boxes + printed scene sentence. Press 'q' to quit.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import cv2
    from blindspot.camera import Camera

    vm = VisionModel()
    print("[vision] Showing detections. Press 'q' to quit.")
    with Camera() as cam:
        last_sentence = ""
        for frame in cam.frames():
            dets = vm.detect(frame)
            for d in dets:
                x1, y1, x2, y2 = d.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"{d.label} {d.confidence:.2f}",
                    (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1,
                )
            sentence = summarize(dets)
            if sentence and sentence != last_sentence:
                print("[scene]", sentence)
                last_sentence = sentence
            cv2.imshow("BlindSpot vision test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
