"""
Camera source abstraction.

Gives the rest of the app a single, boring interface for "give me the next
frame" no matter whether the frames come from the laptop webcam, a phone over
IP Webcam, or a recorded video file. Swapping sources is a one-line config
change (config.CAMERA_SOURCE), never a code change.

Frames are returned as BGR numpy arrays (OpenCV's native format).
"""

from __future__ import annotations
import time

import cv2

import config


class Camera:
    """Thin wrapper around cv2.VideoCapture with reconnect + resize."""

    def __init__(self, source=None, width=None, height=None):
        self.source = config.CAMERA_SOURCE if source is None else source
        self.width = width or config.FRAME_WIDTH
        self.height = height or config.FRAME_HEIGHT
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        # CAP_DSHOW makes webcams start MUCH faster on Windows. Only use it for
        # integer (local) sources; URLs need the default backend.
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera source: {self.source!r}\n"
                "  - Webcam?  Is another app using it? Try config.CAMERA_SOURCE = 1\n"
                "  - Phone?   Is the IP Webcam URL right and on the same wifi?"
            )

        # Ask the driver for our target size (best-effort; we resize anyway).
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        """Return the next frame (BGR ndarray) or None if it failed."""
        if self.cap is None:
            self.open()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        # Normalise size so downstream models always get the same dimensions.
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        return frame

    def frames(self):
        """Generator that yields frames forever (skips transient read failures)."""
        if self.cap is None:
            self.open()
        misses = 0
        while True:
            frame = self.read()
            if frame is None:
                misses += 1
                if misses > 30:
                    raise RuntimeError("Camera stopped delivering frames.")
                time.sleep(0.05)
                continue
            misses = 0
            yield frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.release()


# --------------------------------------------------------------------------
# Standalone smoke test:  python -m blindspot.camera
# Opens a window showing the live feed. Press 'q' to quit.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[camera] opening source: {config.CAMERA_SOURCE!r}")
    with Camera() as cam:
        print("[camera] OK. Showing live feed. Press 'q' in the window to quit.")
        for frame in cam.frames():
            cv2.imshow("BlindSpot camera test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
    print("[camera] done.")
