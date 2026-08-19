"""
Convenience runner: executes all three steps in order and regenerates the
dashboard. Use this when you just want current results end to end; use
the individual 01/02/03 scripts when you actually want to pause at a
checkpoint and change something first (see WORKFLOW.md -- that's the
intended way to use this pipeline, this script is just the fast path).

Usage: python run_all.py   (from inside intraday_pairs_trading/, or from
the parent directory -- both work, see the path handling below)
"""

import os
import subprocess
import sys
import pathlib

STEPS = [
    "01_screen_intraday_pairs.py",
    "02_intraday_walk_forward.py",
    "03_generate_dashboard.py",
]


def main():
    here = pathlib.Path(__file__).parent.resolve()
    project_root = here.parent

    # Force unbuffered output on both this script and the children so
    # headers and child output interleave in the correct order even when
    # piped to a file or another command (e.g. `| tee log.txt`).
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    for step in STEPS:
        print("\n" + "#" * 70, flush=True)
        print(f"# Running {step}", flush=True)
        print("#" * 70, flush=True)
        result = subprocess.run(
            [sys.executable, str(here / step)],
            cwd=project_root,  # scripts read/write paths like "intraday_pairs_trading/..."
            env=env,
        )
        if result.returncode != 0:
            print(f"\n{step} failed (exit code {result.returncode}) -- stopping.")
            sys.exit(result.returncode)

    print("\n" + "#" * 70)
    print("# Done. Open intraday_pairs_trading/dashboard.html in a browser.")
    print("#" * 70)


if __name__ == "__main__":
    main()
