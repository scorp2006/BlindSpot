"""
Full BlindSpot pipeline - all modalities wired together (severity-tiered funnel).

Two speeds, exactly as designed:

  FAST loop (every frame, local):
      camera -> YOLO vision -> tracker -> read latest audio -> flag.decide()
      - 🔴 DANGER: speaks an INSTANT local warning (no VLM wait)
      - narrates nothing on its own unless a tier says so

  SLOW brain (only when the flag engine says fire_vlm):
      send frame + vision facts + audio facts + question -> VLM -> speak answer
      - runs on a background thread so it NEVER stalls the fast loop
      - rate-limited so the paid GPU is barely touched
      - allowed to reply "NOTHING" (proactive mode) -> we stay silent

THE FUNNEL (why this doesn't drown in YOLO's ~20 detections/sec):
  flag.FlagEngine collapses the firehose into at most ONE decision per frame, and
  almost always that decision is "silent". Danger is instant + local; everything
  smart goes to the VLM, rarely.

GRACEFUL DEGRADATION: audio (YAMNet) and listen (Whisper) are OPTIONAL. If their
libraries aren't installed, the pipeline logs a note and runs vision-only.
"""

from __future__ import annotations
import threading
import time

import config
from blindspot.camera import Camera
from blindspot.vision import VisionModel, facts_line
from blindspot.speech import Voice
from blindspot.vlm import VLMBrain
from blindspot.flag import FlagEngine


class BlindSpot:
    def __init__(self, use_audio=True, use_stt=True, show_window=True):
        self.show_window = show_window

        # Always-on core.
        self.vision = VisionModel()
        self.voice = Voice()
        self.brain = VLMBrain()
        self.flag = FlagEngine()

        # Optional modalities - fail soft.
        self.audio = None
        if use_audio:
            try:
                from blindspot.audio import AudioScene
                self.audio = AudioScene()
                self.audio.start()
                print("[pipeline] audio (YAMNet) ON")
            except Exception as e:
                print(f"[pipeline] audio OFF ({e}). Vision-only.")

        self.ptt = None
        if use_stt:
            try:
                from blindspot.listen import PushToTalk
                self.ptt = PushToTalk()
                self.ptt.start()
                print("[pipeline] push-to-talk (Whisper) ON")
            except Exception as e:
                print(f"[pipeline] push-to-talk OFF ({e}). No spoken questions.")

        self._vlm_busy = threading.Lock()

    # -- run the VLM on a background thread so the fast loop never blocks --
    def _fire_vlm_async(self, frame, facts, sounds, question, urgent, tier):
        if not self._vlm_busy.acquire(blocking=False):
            return  # a VLM call is already in flight; skip this one
        self.flag.mark_vlm_fired()

        def work():
            try:
                answer = self.brain.ask(frame, facts=facts, sounds=sounds,
                                        question=question)
                if not VLMBrain.is_silent(answer):
                    print(f"[brain/{tier}] {answer}")
                    self.voice.say(answer, urgent=urgent)
            finally:
                self._vlm_busy.release()

        threading.Thread(target=work, daemon=True).start()

    def run(self):
        frame_interval = 1.0 / max(1, config.TARGET_FPS)
        print("\n[pipeline] BlindSpot is live. Ctrl+C (or 'q' in window) to stop.\n")
        try:
            with Camera() as cam:
                for frame in cam.frames():
                    t0 = time.time()

                    # --- FAST: vision ---
                    dets = self.vision.detect(frame)
                    v_facts = facts_line(dets)

                    # --- FAST: audio (latest cached result) ---
                    a_words: set[str] = set()
                    a_facts = ""
                    if self.audio is not None:
                        ar = self.audio.latest()
                        a_words = ar.label_set()
                        a_facts = ar.facts_line()

                    # --- FAST: any pending question? ---
                    question = ""
                    if self.ptt is not None:
                        q = self.ptt.get_question()
                        if q:
                            question = q

                    # --- THE FUNNEL: one decision for this frame ---
                    decision = self.flag.decide(dets, a_words, question)

                    if decision.tier != "silent":
                        print(f"[flag] {decision.tier}: {decision.reason}")

                    # 🔴 Speak the instant local warning NOW (no VLM latency).
                    if decision.speak_now:
                        self.voice.say(decision.speak_now, urgent=decision.urgent)

                    # Fire the brain if warranted (async, rate-limited).
                    # For a danger tier this is the SMART FOLLOW-UP after the
                    # instant warning; for question/proactive it's the main reply.
                    if decision.fire_vlm:
                        self._fire_vlm_async(
                            frame.copy(), v_facts, a_facts,
                            decision.question, decision.urgent, decision.tier,
                        )

                    # --- optional debug window ---
                    if self.show_window:
                        import cv2
                        danger = decision.tier == "danger"
                        for d in dets:
                            x1, y1, x2, y2 = d.box
                            color = (0, 0, 255) if danger else (0, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, d.label, (x1, max(15, y1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        hud = f"{decision.tier}  audio:{a_facts[:40]}"
                        cv2.putText(frame, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (255, 255, 0), 1)
                        cv2.imshow("BlindSpot - full pipeline", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                    # Throttle to target FPS.
                    dt = time.time() - t0
                    if dt < frame_interval:
                        time.sleep(frame_interval - dt)
        except KeyboardInterrupt:
            print("\n[pipeline] stopping.")
        finally:
            self.shutdown()

    def shutdown(self):
        self.voice.shutdown()
        if self.audio is not None:
            self.audio.stop()
        if self.ptt is not None:
            self.ptt.stop()
        if self.show_window:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    BlindSpot().run()
