"""Puts the app (a module of the project, not of tests/) on the path."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
