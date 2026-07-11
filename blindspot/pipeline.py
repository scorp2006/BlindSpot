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

        self._vlm_busy = threading.Lock()   # guards background (offer) calls
        self._question_busy = False          # True while a question is being answered
        self._recent_frames: list = []       # small ring buffer for frame selection
        self._recent_said: list[str] = []    # memory of recent spoken lines (no-repeat)
        self._instruction = ""               # user's standing "behave like this" wish

    def _remember_said(self, text: str):
        self._recent_said.append(text)
        if len(self._recent_said) > config.MEMORY_LINES:
            self._recent_said.pop(0)

    # -- frame selection: keep a few recent frames, pick the sharpest --
    def _remember_frame(self, frame):
        self._recent_frames.append(frame)
        if len(self._recent_frames) > 4:
            self._recent_frames.pop(0)

    @staticmethod
    def _sharpness(frame) -> float:
        """Variance of Laplacian - higher = sharper (less motion blur)."""
        try:
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            return 0.0

    def _sharpest_recent(self, current):
        """Return the sharpest of the recent frames (or a copy of current)."""
        candidates = self._recent_frames + [current]
        if not candidates:
            return current.copy()
        best = max(candidates, key=self._sharpness)
        return best.copy()

    # -- run the VLM on a background thread so the fast loop never blocks --
    def _fire_vlm_async(self, frame, facts, sounds, question, urgent, tier):
        """Fire the VLM. QUESTIONS get a guaranteed lane (never dropped);
        background tiers (proactive/danger-refine) skip if one is already running.

        This fixes the bug where a running proactive description blocked the user's
        question - the user must always get an answer."""
        is_question = tier == "question"

        if not is_question:
            # Background call: skip if the VLM is already busy (no queue needed).
            if not self._vlm_busy.acquire(blocking=False):
                return
        else:
            # Question: it MUST run. Don't drop it. If a background call holds the
            # lock, we still proceed on a separate thread (the Space handles the
            # request; worst case two calls overlap briefly, which is fine).
            self._question_busy = True

        self.flag.mark_vlm_fired()
        # snapshot memory + instruction for this call
        recent = list(self._recent_said)
        instruction = self._instruction

        def work():
            try:
                answer = self.brain.ask(frame, facts=facts, sounds=sounds,
                                        question=question,
                                        instruction=instruction, recent=recent)
                if not VLMBrain.is_silent(answer):
                    # HARD no-repeat (don't trust the 7B to obey the prompt).
                    # Questions are never suppressed.
                    from blindspot.vlm import too_similar
                    if not is_question and too_similar(answer, self._recent_said):
                        print(f"[vlm] suppressed near-duplicate: {answer!r}")
                        return
                    print(f"[brain/{tier}] {answer}")
                    self._remember_said(answer)     # so the VLM won't repeat it
                    self.voice.say(answer, urgent=urgent)
            finally:
                if is_question:
                    self._question_busy = False
                else:
                    self._vlm_busy.release()

        threading.Thread(target=work, daemon=True).start()

    def run(self):
        frame_interval = 1.0 / max(1, config.TARGET_FPS)
        print("\n[pipeline] BlindSpot is live. Ctrl+C (or 'q' in window) to stop.\n")
        try:
            with Camera() as cam:
                for frame in cam.frames():
                    t0 = time.time()

                    # Keep a small buffer of recent frames for VLM frame-selection.
                    self._remember_frame(frame)

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

                    # 🛠️ User set a standing instruction ("keep me company", etc.)
                    if decision.tier == "set_mode" and decision.instruction:
                        self._instruction = decision.instruction
                        self._recent_said.clear()   # fresh start under new mode
                        print(f"[mode] instruction set: {decision.instruction!r}")

                    # 🔴 Speak the instant local warning NOW (no VLM latency).
                    if decision.speak_now:
                        self.voice.say(decision.speak_now, urgent=decision.urgent)
                        self._remember_said(decision.speak_now)

                    # Fire the brain if warranted (async, rate-limited).
                    # For a danger tier this is the SMART FOLLOW-UP after the
                    # instant warning; for question/proactive it's the main reply.
                    # FRAME SELECTION: send the sharpest of the last few frames,
                    # so the VLM never reasons over a motion-blurred image (better
                    # answers, fewer wasted calls).
                    if decision.fire_vlm:
                        best = self._sharpest_recent(frame)
                        vlm_q = decision.question
                        if decision.tier in ("danger", "obstacle") and decision.speak_now:
                            vlm_q = (f"Alert - I was just warned: '{decision.speak_now}' "
                                     "Quickly tell me what it is and how to avoid it.")
                        self._fire_vlm_async(
                            best, v_facts, a_facts,
                            vlm_q, decision.urgent, decision.tier,
                        )

                    # --- optional debug window ---
                    if self.show_window:
                        import cv2
                        # red for danger/obstruction, else green
                        alarm = decision.tier in ("danger", "obstructed")
                        box_color = (0, 0, 255) if alarm else (0, 255, 0)
                        for d in dets:
                            x1, y1, x2, y2 = d.box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(frame, d.label, (x1, max(15, y1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
                        hud = f"[{decision.tier}] {decision.reason}"
                        cv2.putText(frame, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (255, 255, 0), 2)
                        if a_facts:
                            cv2.putText(frame, f"heard: {a_facts[:45]}", (8, 46),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
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
