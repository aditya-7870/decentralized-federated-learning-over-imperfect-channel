"""
Single entry point for M1 / M2 / M3.

Examples:
    python run.py m1
    python run.py m1 --plot-only
    python run.py m2 --deltas 0.0 0.1
    python run.py m3
    python run.py plots
"""

from __future__ import annotations

import sys


USAGE = """\
Usage:
  python run.py m1 [args...]     Gossip FL under packet loss / asynchrony (trains)
  python run.py m2 [args...]     Event-triggered gossip (trains)
  python run.py m3 [args...]     Spectral-gap analysis (no training)
  python run.py plots            Regenerate M1+M2+M3 plots from saved JSON

Pass any extra flags through, e.g.:
  python run.py m1 --topologies ring --packet-loss 0.0 --num-rounds 20
  python run.py m2 --plot-only
"""


def _dispatch(module_name: str, extra_argv: list[str]) -> None:
    sys.argv = [module_name] + extra_argv
    if module_name == "main.py":
        from main import main
        main()
    elif module_name == "main_m2.py":
        from main_m2 import main
        main()
    elif module_name == "main_m3.py":
        from main_m3 import main
        main()
    else:
        raise ValueError(module_name)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(USAGE)
        return

    cmd = sys.argv[1].lower()
    extra = sys.argv[2:]

    if cmd in ("m1", "main"):
        _dispatch("main.py", extra)
    elif cmd in ("m2", "main_m2"):
        _dispatch("main_m2.py", extra)
    elif cmd in ("m3", "main_m3"):
        _dispatch("main_m3.py", extra)
    elif cmd in ("plots", "plot", "plot-only"):
        _dispatch("main.py", ["--plot-only"] + extra)
        _dispatch("main_m2.py", ["--plot-only"] + extra)
        _dispatch("main_m3.py", ["--plot-only"] + extra)
    else:
        print(f"Unknown command: {cmd}\n")
        print(USAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()
