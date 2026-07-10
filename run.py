"""
BlindSpot launcher - one entry point for every tier.

Usage (from the repo root, with the venv active):

    python run.py narrator     # Tier 1: camera -> YOLO -> voice (no VLM, no audio)
    python run.py vlm-test     # Tier 2: one-shot VLM test on a webcam frame
    python run.py audio-test   # Tier 3: print what the mic hears (YAMNet)
    python run.py listen-test  # Tier 3: push-to-talk speech-to-text (Whisper)
    python run.py full         # Tier 4: everything fused together

Flags for `full`:
    --no-audio     run without YAMNet (if TF isn't installed)
    --no-stt       run without Whisper push-to-talk
    --no-window    headless (no OpenCV preview window)

This is just a thin dispatcher so your friend never has to remember module paths.
"""

import sys


def main():
    args = sys.argv[1:]
    cmd = args[0] if args else "full"
    flags = set(args[1:])

    if cmd in ("narrator", "tier1"):
        from blindspot.narrator import run
        run(show_window="--no-window" not in flags)

    elif cmd in ("vlm-test", "tier2"):
        from blindspot.vlm import __name__ as _  # noqa
        import runpy
        runpy.run_module("blindspot.vlm", run_name="__main__")

    elif cmd in ("audio-test",):
        import runpy
        runpy.run_module("blindspot.audio", run_name="__main__")

    elif cmd in ("listen-test",):
        import runpy
        runpy.run_module("blindspot.listen", run_name="__main__")

    elif cmd in ("full", "tier4"):
        from blindspot.pipeline import BlindSpot
        BlindSpot(
            use_audio="--no-audio" not in flags,
            use_stt="--no-stt" not in flags,
            show_window="--no-window" not in flags,
        ).run()

    else:
        print(__doc__)
        print(f"Unknown command: {cmd!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
