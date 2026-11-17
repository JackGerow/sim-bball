"""Simulate every NBA game on the day this script is run.

Usage:
    python scripts/simulate_today.py
    python scripts/simulate_today.py --date 2026-11-11 --sims 10000
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simbball.pipeline import main

if __name__ == "__main__":
    main()
