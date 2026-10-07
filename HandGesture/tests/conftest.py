"""Stub the heavy module-scope imports so the scripts can be imported headlessly.

HandGesture.py and HandMagic.py only touch cv2/mediapipe/numpy inside main() and
the drawing helpers; at import time the names merely have to exist. This runs at
conftest import, which is before test modules are collected — a fixture would be
too late, since the test module imports the scripts at collection time.
"""

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _stub(name):
    m = types.ModuleType(name)
    m.__getattr__ = lambda attr: None        # any constant lookup resolves
    sys.modules.setdefault(name, m)
    return m


for _name in ("cv2", "numpy"):
    _stub(_name)

_mp = _stub("mediapipe")
_mp.solutions = types.SimpleNamespace(hands=types.SimpleNamespace(Hands=object))
